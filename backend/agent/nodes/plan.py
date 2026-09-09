"""Plan 节点：ICP 故障诊断意图分析、问题改写和工具规划。"""
import asyncio
import json
import logging
import os
import re
import time
import unicodedata
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from backend.agent.answer_contract import (
    ANSWER_FIELD_SPECS,
    ensure_intent_answer_fields,
    normalize_diagnostic_closure,
    normalize_answer_fields,
    normalize_response_mode,
)
from backend.agent.clarification_contract import (
    make_clarification_fallback,
    make_clarification_identity,
    make_clarification_reason,
    make_missing_input_requirement,
    normalize_clarification_identity,
    normalize_missing_input_requirements,
    unique_contract_items,
)
from backend.agent.user_context_binding import offered_user_inputs, validate_user_context_bindings
from backend.agent.model_runtime_policy import bounded_model_transport, remaining_seconds, TURN_TIMEOUT_SECONDS
from backend.core.llm_service import llm_service
from backend.agent.document_semantics import document_semantics_prompt
from backend.agent.model_role_service import agent_model_role_service
from backend.agent.memory import inject_memory, maybe_summarize
from backend.agent.query_understanding import (
    build_query_frame,
    normalize_tool_queries,
)
from backend.agent.tool_policy import (
    DOC_TYPE_BY_TOOL,
    QUERY_TOOL_NAMES,
    extract_hmc_components,
    normalize_calculation_authorizations,
    normalize_requested_sources as _parse_requested_sources,
    normalize_relation_expansion_contract,
    resolve_tool_policy,
)

logger = logging.getLogger(__name__)


def _positive_env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


PLAN_LLM_TIMEOUT_SECONDS = TURN_TIMEOUT_SECONDS
TARGET_LLM_TIMEOUT_SECONDS = TURN_TIMEOUT_SECONDS
# The planner returns a nested JSON object (targets, query frame and per-tool
# queries). 768 tokens regularly truncated otherwise valid Ollama JSON in the
# middle of the object, which silently forced every request onto broad recall.
# Keep this well below the answer budget, but large enough to finish the
# structured contract.
# The target now emits the whole turn contract, not a tiny intent label.
# A 512-token cap truncated JSON before context/anchor fields on every turn
# of a real 40-turn run, silently dropping all semantic safeguards.

_TARGET_INTENTS = {
    "exact_code",
    "symptom_diagnosis",
    "architecture_query",
    "entity_profile",
    "function_location",
    "cause_analysis",
    "impact_analysis",
    "maintenance_query",
    "reliability_query",
    "history_query",
    "hybrid",
}

# Open-ended discourse meaning belongs to the semantic target model.  Code
# validates the returned enum but must not try to recognize every possible
# Chinese paraphrase with an ever-growing keyword table.
_QUESTION_SEMANTICS = {
    "describe",
    "explain",
    "rank",
    "verify",
    "compare",
    "act",
    "unknown",
}
_RESPONSE_FOCUSES = {
    "importance",
    "immediate_impact",
    "count",
    "completeness_check",
    "first_action",
    "maintenance_action",
    "status_interface",
    "history_conditions",
    "recovery_verification",
    "exclusion_basis",
    "verification_criteria",
    "history_root_cause",
    "history_solution",
    "source_capability",
    "source_limitation",
    "exact_identity",
    "cause",
    "missing_input_requirements",
}
_RESPONSE_SCOPES = {"standalone", "incremental", "full_restatement", "unknown"}

# ``answer_mode`` remains in the state/SSE contract for saved agents and older
# consumers.  It must not recreate an intent-driven answer template, however:
# its compatibility value is derived only from the validated response/field
# contract that already crossed the model boundary.
_ARCHITECTURE_PRESENTATION_FIELDS = {
    "architecture_hierarchy",
    "architecture_function",
    "architecture_interface",
    "relation_path",
}
_HMC_PRESENTATION_FIELDS = {"hmc_identity", "hmc_symptom"}
_MAINTENANCE_PRESENTATION_FIELDS = {
    "maintenance_locator",
    "maintenance_action",
    "maintenance_api",
    "platform_test_status",
    "platform_test_result",
}


def _answer_mode_from_contract(query_frame: Mapping[str, Any]) -> str:
    """Return a legacy presentation label without consulting coarse intent."""

    fields = set(normalize_answer_fields(query_frame.get("required_answer_fields")))
    response = query_frame.get("response_mode")
    response_mode = (
        str(response.get("mode") or "unknown")
        if isinstance(response, Mapping)
        else "unknown"
    )
    closure = query_frame.get("diagnostic_closure")
    if response_mode == "diagnostic" or (
        isinstance(closure, Mapping) and closure.get("required") is True
    ):
        return "fault_diagnosis"
    if (
        fields & _HMC_PRESENTATION_FIELDS
        and query_frame.get("hmc_components_source") == "current_query"
    ):
        return "exact_code_lookup"
    if fields & _MAINTENANCE_PRESENTATION_FIELDS:
        return "maintenance_guidance"
    if fields & _ARCHITECTURE_PRESENTATION_FIELDS:
        return "architecture_knowledge"
    return "knowledge_explanation"

_TOPOLOGY_SUBJECT_KINDS = {
    "path",
    "channel",
    "branch",
    "supply_path",
    "power_path",
    "data_path",
    "signal_path",
    "redundant_leg",
}
_TOPOLOGY_QUESTION_ASPECTS = {
    "impact",
    "consequence",
    "continuity",
    "continued_operation",
    "mission_success",
}

_TYPED_RECORD_FIELD_SCHEMA = {
    "fmeca": {"failure_mode", "fault_cause", "function"},
}
_SUBJECT_BINDING_KINDS = {
    "component",
    "module",
    "function",
    *_TOPOLOGY_SUBJECT_KINDS,
}

# Open-language interpretation belongs to the target model.  These collections
# are protocol enums only: they bound what may cross the model/code boundary
# and never attempt to recognize wording in the user's question.
_ICP_DOCUMENT_TYPES = frozenset({
    "function_locator",
    "hmc",
    "fta",
    "rbd",
    "fmeca",
    "mfl",
    "fault_history",
})
_TOOL_BY_DOC_TYPE = {
    doc_type: tool for tool, doc_type in DOC_TYPE_BY_TOOL.items()
}
_COVERAGE_SCOPES = frozenset({
    "complete_collection",
    "focused_detail",
    "relation_path",
})
_COVERAGE_REFERENCE_MODES = frozenset({"current_query", "selected_anchor"})
_ARCHITECTURE_ROLES = frozenset({
    "source",
    "receiver",
    "converter",
    "processor",
    "validator",
    "upstream",
    "downstream",
    "destination",
    "responsible_component",
})
_COMPARISON_SUBJECT_KINDS = frozenset({
    "failure_mode",
    "component",
    "module",
    "path",
    "record",
})
_PAYLOAD_KINDS = frozenset({"data_flow", "signal_flow", "unknown"})
_PAYLOAD_DIRECTIONS = frozenset(
    {"external_to_system", "system_to_external", "internal", "unknown"}
)

TARGET_PROMPT = """你只负责识别用户这一轮明确询问的信息目标，不做检索规划，不回答问题。系统会在后续按目标补齐必要证据，因此这里不要把补充证据误写成用户目标。

目标定义：
- exact_code：明确给出或查询 HMC/LCN/HAMC/HSC 代码
- symptom_diagnosis：笼统要求诊断或列出候选故障，但没有单独限定原因、影响、维修等目标
- architecture_query：询问系统组成、层级、职责、接口、上下游或链路
- entity_profile：询问某个系统、模块、部件、功能或业务编号是什么、叫什么、属于什么类型、基本职责和所属位置
- function_location：不包含异常归因，只询问某功能/职责位于哪个系统、模块或部件
- cause_analysis：询问为什么发生、哪里出了问题、可能是什么问题
- impact_analysis：询问某状态会带来什么影响、后果、危害、严酷度/严重等级，或会牵连哪些能力
- maintenance_query：明确询问怎么检查、怎么修、换什么、维修步骤或 API
- reliability_query：询问某一路/通道/支路/供电路径失效后的能力边界、成功判据、冗余结构、可靠度或任务成功可能性；用户不需要知道 RBD、串并联或冗余等内部术语
- history_query：明确询问以前是否发生、过去如何处理或类似案例
- hybrid：确实无法归入以上目标

额外语义合同：
- diagnostic_closure：只有用户在描述当前、实际发生的异常并要求形成诊断闭环时才生成。required_dimensions 必须逐项使用 required_answer_fields 的规范字段，且只列这个闭环实际缺少的维度；basis 必须逐字摘录当前用户问题中的连续原话。不能用 intent 自动展开原因、影响、处置或历史套餐。
- response_mode：只描述本轮回答的组织宽度。mode 只能是 structure_only、focused、diagnostic、explanation、unknown，basis 必须逐字摘录当前用户问题中的连续原话。FTA/架构等只要求结构关系时用 structure_only；只回答明确字段用 focused；当前异常且用户要求诊断闭环用 diagnostic；解释概念/意义用 explanation；无法判断用 unknown。不得根据 intent 在代码侧补推。
- structural_cause_chain：用户问多个上游条件如何共同造成结果、故障如何沿结构传递、哪些底层条件会触发当前现象时，写入 required_answer_fields。
- 冗余切换只是现象时，不等于用户在问结构原因链。若用户描述一条支路在某工况下掉线、另一条支路接管，再问优先怀疑什么，只选择 cause_analysis / fault_cause；只有明确追问多路为何共同不可用或故障如何沿结构传播时才增加 structural_cause_chain。
- fault_location / maintenance_locator：用户问问题落在哪个模块、环节、SUR 板或功能子电路时，分别写入相应字段；不要求用户说 Function Locator 或 MFL。
- impact_propagation：用户问异常会先后牵连哪些功能、如何一路扩散或下游哪里受影响时写入；它要求跨文档关系补全，不等同于 RBD 可靠性计算。
- relation_expansion 是唯一可授权跨文档图查询的语义合同。needed=true 时 relation_kind 只能是 relation_path、structural_cause_chain、impact_propagation；reference_mode=current_query 时 basis 必须逐字摘录当前问题中的连续原话，selected_anchor 时除当前原话 basis 外还必须选择系统给出的 anchor_id。只写 needed、intent 或 required_answer_fields 均不授权图查询。无需关系查询时输出 needed=false，其余授权字段留空。
- question_semantics 只描述本轮要求的表达关系，从 describe、explain、rank、verify、compare、act、unknown 中选择一个；不要根据答案中应出现的某个词来判断。
- response_focus 是本轮必须直接回应的语义维度，可多选：importance、immediate_impact、count、completeness_check、first_action、maintenance_action、status_interface、history_conditions、recovery_verification、exclusion_basis、verification_criteria、history_root_cause、history_solution、source_capability、source_limitation、exact_identity、cause。它们是含义标签，不是要求答案照抄的标题或词汇。
- response_scope：只表示用户本轮要求的重述范围，必须附 response_scope_basis 逐字摘录当前原话。独立问题为 standalone；明确只补充新维度为 incremental；明确要求从头完整重述为 full_restatement；无法从当前原话判断时为 unknown 且 basis 留空。不得因 context_used=true 自动设为 incremental。
- calculation_authorizations 只表达本轮对 FTA/RBD 操作的语义方式，不填写算法参数。operation 只能是 fta_cut_sets、fta_probability、rbd_reliability；mode 只能是 calculate、explain、read_document；target 分别必须是 fta、fta、rbd；basis 必须逐字摘录当前用户问题中的一段连续原话。只有 mode=calculate 表示请求执行确定性算法，解释算法或读取资料已有数值分别使用 explain、read_document。
- requested_sources 表达用户明确要求回查的资料族。doc_type 只能是 function_locator、hmc、fta、rbd、fmeca、mfl、fault_history；question_role 只能是 evidence、capability、limitation、comparison；requested_fields 记录用户要求向该资料核实的规范字段，即使该资料通常不提供此字段，也必须保留问题，交给后续查询确认并解释边界。basis 必须逐字摘录当前用户问题中指向该资料的连续原话。业务对象属于哪类资料，与本轮要求向哪类资料求证是两件事；不能用当前对象、上轮来源或候选工具代替本轮指定的来源。用户只是询问业务对象而未要求指定资料时不要生成。 scope 默认 reference，表示必须参考而非排他；只有用户明确禁止其他资料时标 exclusive，并用 scope_basis 摘录排他要求。询问某资料本身能否证明某事用 capability/limitation，不等同于要求获取该事实。不得把“资料中”“刚才”误当成指定某一资料族。
- coverage_request 是独立的集合覆盖请求。scope 只能是 complete_collection、focused_detail、relation_path；member_kind 使用受控成员类型；reference_mode=current_query 时 subject 和 basis 都必须逐字来自当前问题，selected_anchor 时只能引用系统给出的 anchor_id。不要填写父ID、成员ID、权威计数等资料事实。
- 对真实组成、全部数量或组成是否完整的核对，需要 complete_collection 与 closed_collection_members，即使本轮是省略主体的短追问；这与仅统计上轮列了几项不同。追问主体由已选锚点消解时使用 selected_anchor，basis 仍取本轮原话。按所问层级选择成员类型，不把顶层功能域一律当成 module；无法确定层级时保留歧义，不把局部 Top-K 记录当成总体清单。
- architecture_roles 用于把用户当前明确命名的架构对象标成 source、receiver、converter、processor、validator、upstream、downstream、destination 或 responsible_component；subject 和 basis 都必须是当前问题中的连续原话。comparison_subjects 用于列出本轮需要并列比较的对象，kind 只能是 failure_mode、component、module、path、record，value 和 basis 都必须是当前问题中的连续原话。
- missing_input_requirements 只表示用户本轮在询问当前状态机尚缺输入；slot 写状态机槽名，basis 必须逐字来自当前问题。模型不负责臆测缺什么，最终清单会只从系统实际缺失槽生成。
- payload_contract 只描述本轮数据/信号载荷的方向相关性，不生成任何实体身份。kind 只能是 data_flow、signal_flow、unknown，direction 只能是 external_to_system、system_to_external、internal、unknown；非 unknown 时 source_role、destination_role 均须为当前原话对象，basis 是包含两个对象的当前问题连续原话。无法判断时使用 unknown 且角色可为空。
- 当用户已经把一个退化/故障模式作为前提，询问它为什么表现为多个功能异常、为何影响范围不是局部时，问题是在问该模式的局部与最终影响链，而不是反问该退化本身的起因；选择 impact_analysis，required_answer_fields 至少包含 local_impact、final_impact。若同一记录已直接给出这两项，不要误加 structural_cause_chain。

约束：
1. 只有用户同一句话明确提出多个目标时才能多选。
2. “哪里出了问题/可能是什么问题”属于 cause_analysis，不是 maintenance_query。
3. 禁止以“用户隐含需要”“可能还需要”“更全面”为由增加目标。约束和交付目标要区分：用户限定资料来源、适用条件或参数可信度，不等于另要求逐项列出所有来源或参数；保留这些条件，但不要把它们改写为新增分问。
4. 对话历史仅用于消解“它/这个问题/这种异常”等指代和省略，不能把助手之前主动提供的内容变成本轮目标或已确认事实。
5. 用户先描述异常现象，再问“优先怀疑哪里、先查哪部分、可能是哪一段”，属于 cause_analysis；功能定位只是检索步骤，不能把答案目标降成 function_location。
6. 明确命名一条路、通道、支路或供电/数据/信号路径已经失效，并询问影响、后果或还能否工作时，后果同时取决于故障影响记录和剩余拓扑的成功判据，因此同时选择 impact_analysis 和 reliability_query；用户不必主动提到另一条路、备份、RBD、串并联或冗余。
7. 询问某模块/部件“是否重要、有多关键、为什么重要”，以 impact_analysis 为主目标并增加 architecture_query：需要说明其职责/依赖以及失效影响。只有用户还问单路失效、备份或任务继续能力时才增加 reliability_query。
8. contextualized_query 必须把本轮省略的对象补全成可独立理解的问题，保留每个分问、条件和当前纠正；不得增加历史中没有的代码、模块或现象。requested_facets 用自然语言写本轮实际要解决的分问，不受规范字段或 response_focus 目录限制；解释、比较、反事实、验证等关系不能被压缩成字段名。required_answer_fields 仅映射本轮可能需要读取的证据字段，不是回答范围或措辞白名单；无法映射的分问仍必须保留在 contextualized_query 和 requested_facets 中，不得为了完整扩展用户目标。
9. 本轮问题可独立理解时 context_status=standalone；结合历史可唯一消解时为 resolved；存在多个同类对象无法判断时为 ambiguous；历史中没有所指对象时为 missing。
10. 证据锚点只表示上轮回答实际引用过某条记录；需要沿用时必须在 selected_anchor_ids 中选择其 anchor_id，不得生成新 ID。新一轮仍需回到权威存储重新查询。
11. 覆盖义务只由带 basis/已选锚点的 coverage_request 产生：普通局部说明为 focused_detail，穷尽集合为 complete_collection，链路/多跳关系为 relation_path。裸 coverage_scope、collection_subject 或 collection_member_kind 只是旧格式审计值，不能授权闭集、扩展窗口或增加 closed_collection_members。
12. coverage_request.scope=complete_collection 时，subject 必须是需要闭合核验的父主体，member_kind 写成员类型；current_query 的 subject 和 basis 必须都来自当前原话，selected_anchor 必须选系统提供的 anchor_id。主体无法唯一确定时设 context_status=ambiguous，不能猜测。
13. 询问 FMECA、MFL、故障历史等业务记录“是什么”仍属于 entity_profile，但 required_answer_fields 应使用 record_identity，并由相应结构化资料回答；不要把所有业务 ID 都当成架构组件，也不要另造“ID查询”意图。
14. 使用 topology_dependency 输出上述判断的结构依据。只有同时满足“明确命名的路径类对象 + 已描述失效状态 + 询问影响/连续工作能力”时 required=true；subject_kind 只能是 path、channel、branch、supply_path、power_path、data_path、signal_path、redundant_leg，question_aspects 只能从 impact、consequence、continuity、continued_operation、mission_success 中选择。basis 必须是同时包含 subject 与 failure_state 的当前问题连续原话；若不存在一段共同原话，可分别填写 subject_basis 与 failure_state_basis，且二者都必须是当前问题连续原话并包含对应值。
15. 普通元件状态或单个模块的功能影响不自动等于路径可靠性。例如接插件接触电阻增大、温度传感器漂移、某信号调理器件参数漂移，只要没有把对象描述为承担一条路/通道/支路/供电路径，也没有询问剩余系统能否工作，topology_dependency.required=false，不增加 reliability_query。
16. “这个办法现场真用过吗、处理后是否复发、是否完成验证闭环”属于 history_query，并按问题选择 history_time/history_steps/history_result/history_preventive_action；不要求用户说“历史案例”。
17. “这个异常最终落在哪块 SUR 板、哪一级功能子电路”“不用拆机能否从状态量定位到哪块板”属于 maintenance_query，required_answer_fields 分别包含 maintenance_locator 或 maintenance_api。
18. 用户把多个当前对象放在同一个问题中（例如 A、B 两路供电/两个通道/多个模块）时，subjects 必须逐个保留用户已命名的对象；不得为了简短把它们收缩成一个旧会话中的单对象，也不得凭空补造新的业务 ID。后续会分别用已有权威身份查询并合并结果。
19. 用户问多个已命名的路、通道或模块为何共同不可用、哪些上游条件让它们一起失效时，属于 cause_analysis；required_answer_fields 包含 structural_cause_chain，并生成带当前原话 basis 的 relation_expansion 合同。只有另外问剩余路径能否继续承担任务时，才增加 reliability_query。
20. typed_record_fields 只做当前原话到结构化表字段的类型标注：值必须是用户本轮完整写出的连续原文，不得改写、补全或从历史答案猜测。subject_bindings 同样必须给出当前原话 basis；指代追问通过 selected_anchor_ids 延续，不得为“它/这个”补造绑定值。
21. 用户描述当前异常现象并询问判断方向、优先怀疑、如何检查或处置时，问题同时要求原因判断和可执行的维修依据；cause_analysis 类目标必须保留 maintenance_action（现象稳定复现下再加 maintenance_api），不得把维修维度从 required_answer_fields 中删去。这是读取维度的合同，不是回答范围扩张。

示例：
- “高负载连续运行时机箱过热，采样值偶发越限，优先怀疑哪里” → cause_analysis
- “网络接入与路由功能属于哪个模块” → function_location
- “MOD-PWR-PSUA 是什么”或“电源 A 模块是什么” → entity_profile
- “FM-DEMO-001 是什么” → entity_profile，required_answer_fields 为 record_identity；它是业务记录身份，不默认当成架构模块
- “外部维护指令经过哪些环节才生效” → architecture_query
- “两路中一路异常、另一路正常，继续运行有什么风险” → impact_analysis + reliability_query
- “两路供电为什么会一起不可用” → cause_analysis，required_answer_fields 包含 structural_cause_chain，relation_expansion=true；不因没有询问剩余能力而自动变成可靠性数值问题
- “若采集信号支路一路断开，会有什么影响” → impact_analysis + reliability_query；topology_dependency.required=true，subject_kind=branch；不要求用户补说冗余路径或 RBD
- “某通信通道断开后，设备还能工作吗” → impact_analysis + reliability_query；topology_dependency.required=true，subject_kind=channel
- “接插件接触电阻增大会有什么影响” → impact_analysis；topology_dependency.required=false
- “模拟采样基准电压慢慢漂了，会牵连哪些测量” → impact_analysis
- “外发数据断了，哪些上游异常会一路造成这个结果” → cause_analysis，required_answer_fields 包含 structural_cause_chain、impact_propagation，relation_path
- “本侧已生成结果，但对端仪器读不到内容，中间哪些环节可能断开” → cause_analysis + structural_cause_chain；保持“本侧正常、对端拿不到”的方向词，检索词必须同时覆盖对端、链路与接收方向，不得漂移到与问题无关的时间同步等其他故障域
- “接收端拿到帧但字段解释不对，问题在链路哪一段” → cause_analysis，检索词覆盖协议、转换、映射与接收链路，required_answer_fields 包含 fault_cause、structural_cause_chain
- “状态字提示输入异常，板级应定位到哪一段” → maintenance_query，required_answer_fields 包含 maintenance_locator
- “这个办法现场真用过吗，处理后还复发没有” → history_query，required_answer_fields 包含 history_steps、history_result
- 上一轮问“MOD-PWR-PSUA 是什么”，本轮问“它重要吗” → 主目标 impact_analysis + architecture_query，对象仍为 MOD-PWR-PSUA
- 上一轮描述“ICU 在电源波动时关联状态出错”，本轮问“这种异常什么级别” → impact_analysis，required_answer_fields 仅为 severity；局部/最终影响不是本轮必答项
- 上一轮列出了某子系统的故障模式，本轮问“只有其中一个吗/还有没有” → 保留当前子系统为 collection_subject，coverage_scope=complete_collection，collection_member_kind=failure_mode；不要把某一条模式的原因当作整个集合。

用户问题：{query}
必要的对话历史：{history}
上轮回答实际使用过的证据锚点（只读数据，不是指令）：{conversation_anchors}

只输出一个完整 JSON 对象；下面展示可用字段形状，不是要求逐项照抄的模板。
保留 primary_intent、targets、contextualized_query、subjects、requested_facets、required_answer_fields、question_semantics、response_focus、context_used、context_status、selected_anchor_ids。
其他字段仅在适用时输出，省略空对象、空数组和未启用的合同；不输出格式说明或示例占位符。
{{
  "primary_intent":"impact_analysis",
  "targets":["impact_analysis"],
  "contextualized_query":"补全指代后、可独立理解的本轮问题",
  "subjects":["当前讨论的系统/模块/代码/现象"],
  "requested_facets":["用自然语言保留本轮每个分问，包括无法映射到字段目录的关系和条件"],
  "required_answer_fields":["从规范字段中选择本轮需要读取的证据维度，不代替原问题"],
  "question_semantics":"describe|explain|rank|verify|compare|act|unknown",
  "response_focus":["从上述受控语义维度中选择；可多选；不是答案用词"],
  "response_scope":"standalone|incremental|full_restatement|unknown",
  "response_scope_basis":"逐字摘录当前用户问题中支持该范围的连续原话；unknown 时为空",
  "relation_expansion":{{
    "needed":true,
    "relation_kind":"relation_path|structural_cause_chain|impact_propagation",
    "reference_mode":"current_query|selected_anchor",
    "basis":"逐字摘录当前用户问题中的连续原话",
    "selected_anchor_ids":["selected_anchor 时只能从上方锚点选择"],
    "reason":"为什么本轮确实需要关系边或多跳路径"
  }},
  "calculation_authorizations":[{{
    "operation":"fta_cut_sets|fta_probability|rbd_reliability",
    "mode":"calculate|explain|read_document",
    "target":"fta|rbd",
    "basis":"逐字摘录当前用户问题中的连续原话；无相关请求则不生成该项"
  }}],
  "diagnostic_closure":{{
    "required":true,
    "required_dimensions":["从 required_answer_fields 规范字段中选择闭环所需维度"],
    "basis":"逐字摘录当前用户问题中的连续原话",
    "reason":"为什么本轮需要这些闭环维度；可为空字符串"
  }},
  "response_mode":{{
    "mode":"structure_only|focused|diagnostic|explanation|unknown",
    "basis":"逐字摘录当前用户问题中的连续原话"
  }},
  "topology_dependency":{{
    "required":false,
    "subject":"明确命名的路径类对象；无则为空字符串",
    "subject_kind":"path|channel|branch|supply_path|power_path|data_path|signal_path|redundant_leg|unknown",
    "failure_state":"用户已经描述的失效状态；无则为空字符串",
    "question_aspects":["impact"],
    "basis":"同时包含 subject 与 failure_state 的当前用户问题连续原话；不能共同摘录时为空",
    "subject_basis":"basis 为空时，包含 subject 的当前原话连续片段",
    "failure_state_basis":"basis 为空时，包含 failure_state 的当前原话连续片段"
  }},
  "typed_record_fields":{{
    "fmeca":{{"failure_mode":"仅当用户本轮原话完整给出该字段值时填写，否则省略"}}
  }},
  "subject_bindings":[{{
    "doc_type":"rbd",
    "kind":"component|module|function|path|channel|branch|supply_path|power_path|data_path|signal_path|redundant_leg",
    "value":"用户本轮明确写出的工程对象",
    "basis":"包含该对象的用户本轮原话片段"
  }}],
  "requested_sources":[{{
    "doc_type":"function_locator|hmc|fta|rbd|fmeca|mfl|fault_history",
    "basis":"逐字摘录当前用户问题中的连续原话",
    "requested_fields":["该资料下本轮明确要求的规范字段；没有字段要求则为空数组"],
    "question_role":"evidence|capability|limitation|comparison",
    "scope":"reference|exclusive",
    "scope_basis":"仅 exclusive 时填写：用户明确排除其他来源的连续原话"
  }}],
  "coverage_request":{{
    "scope":"complete_collection|focused_detail|relation_path",
    "subject":"当前问题明确写出的集合主体，或所选锚点中的主体",
    "member_kind":"failure_mode|cause|component|module|subsystem|record",
    "reference_mode":"current_query|selected_anchor",
    "basis":"逐字摘录当前用户问题中的连续原话",
    "selected_anchor_ids":["reference_mode=selected_anchor 时只能从上方锚点选择"]
  }},
  "architecture_roles":[{{
    "role":"source|receiver|converter|processor|validator|upstream|downstream|destination|responsible_component",
    "subject":"用户本轮明确写出的对象",
    "basis":"逐字摘录当前用户问题中的连续原话"
  }}],
  "comparison_subjects":[{{
    "kind":"failure_mode|component|module|path|record",
    "value":"用户本轮明确写出的比较对象",
    "basis":"逐字摘录当前用户问题中的连续原话"
  }}],
  "missing_input_requirements":[{{
    "slot":"状态机槽名",
    "basis":"逐字摘录当前用户问题中的连续原话"
  }}],
  "payload_contract":{{
    "kind":"data_flow|signal_flow|unknown",
    "direction":"external_to_system|system_to_external|internal|unknown",
    "source_role":"当前原话中的载荷来源对象；unknown 可为空",
    "destination_role":"当前原话中的载荷去向对象；unknown 可为空",
    "basis":"包含非空角色的当前用户问题连续原话；unknown 可为空"
  }},
  "coverage_scope":"仅兼容审计；执行范围以通过校验的 coverage_request 为准",
  "collection_subject":"仅兼容审计；不单独产生集合义务",
  "collection_member_kind":"仅兼容审计；不单独产生集合义务",
  "context_used":true,
  "context_status":"standalone|resolved|ambiguous|missing",
  "selected_anchor_ids":["只能从上方证据锚点的 anchor_id 中选择"]
}}

required_answer_fields 规范字段：
{answer_field_catalog}。
计算结果/所需参数使用 calculation_result/calculation_parameters，具体资料和算法由 calculation_authorizations 决定；不要为凑字段把 FTA 割集映射成 FMECA 的 failure_probability 或 RBD 的 rbd_missing_parameters。没有字段可映射的分问仍保留在自然语言目标中。
topology_dependency.required=true 时，required_answer_fields 至少包含 rbd_single_failure_boundary 和 rbd_success_criterion；询问影响/后果时同时保留用户所需的 local_impact/final_impact。
"""


COLLECTION_TARGET_REPAIR_PROMPT = """你负责修正一个未通过结构校验的“完整集合”问答合同，不回答用户问题，也不规划工具。

用户问题：{query}
必要对话历史：{history}
第一次分类结果：{decision}

该结果已判断 coverage_scope=complete_collection，但集合合同不完整。请重新理解用户实际要穷尽的集合，并只输出 JSON：
{{
  "primary_intent":"architecture_query|entity_profile|cause_analysis|impact_analysis|symptom_diagnosis|hybrid",
  "targets":["与本轮明确问题一致的目标"],
  "contextualized_query":"补全指代后的独立问题",
  "subjects":["本轮集合主体"],
  "requested_facets":["用户实际要求的维度"],
  "required_answer_fields":["closed_collection_members","仅当用户明确要求时才增加的其他规范字段"],
  "coverage_request":{{
    "scope":"complete_collection",
    "subject":"被枚举成员所属的父系统、模块、功能或记录",
    "member_kind":"failure_mode|cause|component|module|subsystem|record",
    "reference_mode":"current_query|selected_anchor",
    "basis":"逐字摘录当前用户问题中支持完整集合的连续原话",
    "selected_anchor_ids":["selected_anchor 时只能使用已提供锚点"]
  }},
  "context_used":false,
  "context_status":"standalone|resolved|ambiguous|missing",
  "selected_anchor_ids":[]
}}

校验规则：
1. coverage_request.subject、member_kind、basis 和 closed_collection_members 均不能为空；不要把集合中的某一成员写成父主体。
2. required_answer_fields 只保留用户明确要求展示的内容。仅要求“全部有哪些成员”时只填 closed_collection_members；不要自动增加原因、影响、严酷度、检测、维修或历史。
3. 若主体确实无法从问题或历史唯一确定，context_status=ambiguous；不得猜测另一个对象。
4. 成员类型按语义判断，不依赖用户是否使用英文 ID 或内部文档术语。
"""

PLAN_PROMPT = """你是 ICP 检索规划器。只把已解析任务转换为查询，不重新解释用户目标。
当前原问题：{query}
唯一任务合同（对象、分问、来源和授权由任务理解阶段负责）：{dialogue_context}
会话材料（只用于核对条件，不能新增对象或授权）：{history}
记忆（不是工程事实）：{memory}
系统设定：{system_prompt}

可用查询工具：
function_locator（架构/功能）、hmc_query（报码）、fmeca_query（模式/原因/影响/检测）、
fta_query（原因结构）、rbd_query（成功判据/结构）、mfl_query（维修/状态接口）、
fault_history_query（历史条件/处理/验证）、hybrid_search（语义发现）。
工具执行范围仍由程序根据任务合同校验。

只输出 JSON：
{{
  "entity_keywords": ["任务中对象的检索短语"],
  "type_keywords": [],
  "query_frame": {{
    "symptoms": ["用户实际描述的现象"],
    "operating_stage": "未知",
    "operating_stage_basis": "",
    "temporal_pattern": "未知",
    "temporal_pattern_basis": "",
    "constraints": ["用户给定的条件"],
    "normalized_terms": ["不改变对象和含义的检索短语"]
  }},
  "tool_queries": {{"hybrid_search": ["带对象、现象和本轮分问的完整检索句"]}},
  "tools_plan": [{{"tool": "hybrid_search", "reason": "需要找到支持本轮分问的材料"}}],
  "reasoning": "简短说明检索选择"
}}

不要输出 intent、query_targets、requested_facets、实体绑定、来源约束或计算/图授权；它们由唯一任务合同提供。
只选择支持本轮分问的最小查询集，不因为资料有某列就扩展问题。不得猜测 ID、报码、根因或案例。
每条查询均保留已消解对象、故障现象和区分条件；不要把“相近案例”变成泛化的历史检索。
工况及频次 basis 只能引用当前用户原话或任务合同中已选用户输入；部件老化不等于异常持续发生。
没有有效任务合同时只用原问题进行 hybrid_search，不自行恢复另一套语义合同。
"""


def _format_history(messages, max_turns=None):
    from backend.agent.answer_context_binding import conversation_review_context
    context = conversation_review_context({"messages": messages}, max_messages=None if max_turns is None else max_turns * 2)
    for message in context["messages"]:
        namespace = message["message_id"]
        message["content"] = re.sub(
            r"\[\^(\d+)\]", lambda match: f"〔历史引用 {namespace}:{match.group(1)}〕",
            message["content"],
        )
    return json.dumps(context["messages"], ensure_ascii=False) if context["messages"] else "无"


def _offered_conversation_anchors(
    messages: Any,
    *,
    limit: int = 12,
) -> List[Dict[str, Any]]:
    """Offer bounded used identities across the server-provided history window.

    An unrelated/capability turn carries no engineering anchor and must not
    evict an older identity while the anchor budget still has room. This does
    not recover messages already outside the bounded server history window.
    """

    if not isinstance(messages, list):
        return []
    collected: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for raw_anchor in message.get("conversation_anchors") or []:
            if not isinstance(raw_anchor, dict):
                continue
            anchor_id = str(raw_anchor.get("anchor_id") or "").strip()
            if not anchor_id or anchor_id in seen:
                continue
            seen.add(anchor_id)
            anchor = dict(raw_anchor)
            origin_message_id = str(
                message.get("message_id") or message.get("id") or ""
            ).strip()
            if origin_message_id:
                anchor["origin_message_id"] = origin_message_id
            collected.append(anchor)
            if len(collected) >= limit:
                return collected
    return collected


_ANCHOR_SUBJECT_FIELDS = (
    "component_id", "component_name", "abbreviation", "function_id",
    "function_name", "module_id", "module_name", "system_name",
    "subsystem_name", "interface_id", "hmc_code", "record_code",
    "mfl_code", "case_id", "failure_mode", "fault_mode", "product_flag",
    "diagram_key", "diagram_id", "event_id", "fault_symptom",
    "fault_description", "collection_parent_id", "collection_parent_name",
    "collection_member_id", "collection_member_name",
    # FTA citations persist their graph identity as event fields; a branch
    # follow-up ("树里输出侧那条分支的维修清单") must be able to carry the
    # exact event wording into retrieval terms.  Only the name-like parts of
    # bounded structured entries are used, never their prose descriptions.
    "top_event", "basic_events", "intermediate_events",
)


def _anchor_subject_values(value: Any) -> List[Any]:
    """Flatten one persisted answer-field value into lexical identities.

    Structured fields may store plain strings or bounded mappings such as
    ``{"name": ..., "description": ...}`` (FTA events, collection members).
    Descriptions stay inside this turn's evidence; only stored name/ID keys
    become the next turn's retrieval subjects.
    """

    entries = value if isinstance(value, (list, tuple, set)) else [value]
    values: List[Any] = []
    for entry in entries:
        candidate: Any = entry
        if isinstance(candidate, str):
            text = candidate.strip()
            if (text.startswith("{") and text.endswith("}")) or (
                text.startswith("[") and text.endswith("]")
            ):
                try:
                    candidate = json.loads(text)
                except (ValueError, TypeError):
                    values.append(text)
                    continue
            else:
                values.append(text)
                continue
        if isinstance(candidate, (list, tuple, set)):
            # A JSON-array string re-enters as a sequence; flatten it with the
            # same name-only rule instead of treating it as one opaque subject.
            values.extend(_anchor_subject_values(candidate))
            continue
        if isinstance(candidate, Mapping):
            values.extend(
                candidate.get(key)
                for key in ("name", "event_name", "member_name", "code", "id")
            )
            continue
        values.append(candidate)
    return values


def _anchor_subjects(anchors: List[Dict[str, Any]]) -> List[str]:
    """Extract only citation-persisted identities for a selected follow-up.

    This is deliberately not entity resolution. It merely makes the exact,
    already public identifiers in selected answer citations available as the
    next turn's retrieval terms, so a long previous answer cannot hide the
    current subject behind its prose layout.
    """

    values: List[Any] = []
    for anchor in anchors:
        if not isinstance(anchor, dict):
            continue
        fields = anchor.get("answer_fields")
        if isinstance(fields, dict):
            for name in _ANCHOR_SUBJECT_FIELDS:
                values.extend(_anchor_subject_values(fields.get(name)))
        values.append(anchor.get("record_id"))
        values.append(anchor.get("title"))
    return _merge_text_lists(values, limit=20, item_limit=300)


_COLLECTION_MEMBER_KINDS = {
    "failure_mode", "cause", "component", "module", "subsystem", "record",
}


def _downgrade_unrequested_collection_contract(
    decision: Dict[str, Any],
    *,
    query: Any,
) -> Dict[str, Any]:
    """Turn an accidental N-of-N plan into a normal bounded explanation.

    This preserves the model's actual topic and requested fields while
    removing only the unsupported completeness promise.  It is intentionally
    based on discourse quantifiers, not ICP terminology or a fault synonym
    table: “什么会导致电源不可用” remains a cause-analysis question, whereas
    “列出电源模块的所有故障模式” may use a closed collection if PostgreSQL
    can prove its parent scope.
    """

    del query  # compatibility parameter; raw wording is never an authority.
    request = decision.get("coverage_request")
    if str(decision.get("coverage_scope") or "") != "complete_collection":
        return decision
    if (
        isinstance(request, Mapping)
        and str(request.get("scope") or "") == "complete_collection"
        and str(request.get("subject") or "").strip()
        and str(request.get("member_kind") or "") in _COLLECTION_MEMBER_KINDS
    ):
        return decision
    downgraded = dict(decision)
    downgraded["coverage_scope"] = "focused_detail"
    downgraded["collection_subject"] = ""
    downgraded["collection_member_kind"] = "unknown"
    downgraded["required_answer_fields"] = [
        field
        for field in normalize_answer_fields(
            decision.get("required_answer_fields"),
            requested_facets=decision.get("requested_facets") or [],
        )
        if field != "closed_collection_members"
    ]
    downgraded["requested_facets"] = [
        value
        for value in (decision.get("requested_facets") or [])
        if str(value or "").strip() not in {"完整性", "完整清单", "全部"}
    ]
    return downgraded


def _extract_json_object(text: Any) -> Optional[Dict[str, Any]]:
    value = str(text or "").strip()
    if value.startswith("```json"):
        value = value[7:]
    if value.startswith("```"):
        value = value[3:]
    if value.endswith("```"):
        value = value[:-3]
    value = value.strip()
    if not value.startswith("{"):
        start = value.find("{")
        end = value.rfind("}")
        if start >= 0 and end > start:
            value = value[start : end + 1]
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _bounded_text_list(value: Any, *, limit: int | None = None, item_limit: int | None = None) -> List[str]:
    values = value if isinstance(value, list) else []
    result: List[str] = []
    seen: set[str] = set()
    for raw_value in values:
        text = " ".join(str(raw_value or "").split())
        identity = text.casefold()
        if not text or identity in seen:
            continue
        seen.add(identity)
        result.append(text)
        if limit is not None and len(result) >= limit:
            break
    return result


def _merge_text_lists(*values: Any, limit: int, item_limit: int = 240) -> List[str]:
    flattened: List[Any] = []
    for value in values:
        if isinstance(value, (list, tuple, set)):
            flattened.extend(value)
        elif value is not None:
            flattened.append(value)
    return _bounded_text_list(flattened, limit=limit, item_limit=item_limit)


def _literal_identity_text(value: Any, *, limit: int = 500) -> str:
    """Normalize presentation only, preserving identity punctuation."""

    return " ".join(
        unicodedata.normalize("NFKC", str(value or "")).split()
    )


def _typed_record_field_shape(value: Any) -> Dict[str, Dict[str, str]]:
    """Bound the model-authored typed-record shape before grounding it."""

    if not isinstance(value, Mapping):
        return {}
    result: Dict[str, Dict[str, str]] = {}
    for raw_doc_type, raw_fields in value.items():
        doc_type = str(raw_doc_type or "").strip()
        allowed_fields = _TYPED_RECORD_FIELD_SCHEMA.get(doc_type)
        if not allowed_fields or not isinstance(raw_fields, Mapping):
            continue
        fields: Dict[str, str] = {}
        for raw_field, raw_value in raw_fields.items():
            field = str(raw_field or "").strip()
            rendered = _literal_identity_text(raw_value)
            if field in allowed_fields and rendered:
                fields[field] = rendered
        if fields:
            result[doc_type] = fields
    return result


def _ground_typed_record_fields(
    value: Any, *, raw_user_query: Any
) -> Dict[str, Dict[str, str]]:
    """Keep only typed table values literally present in the current turn."""

    query = _literal_identity_text(raw_user_query, limit=4000).casefold()
    if not query:
        return {}
    result: Dict[str, Dict[str, str]] = {}
    for doc_type, fields in _typed_record_field_shape(value).items():
        grounded = {
            field: rendered
            for field, rendered in fields.items()
            if len(rendered) >= 2 and rendered.casefold() in query
        }
        if grounded:
            result[doc_type] = grounded
    return result


def _subject_binding_shape(value: Any) -> List[Dict[str, str]]:
    """Validate the closed RBD subject-binding schema without trusting it."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    result: List[Dict[str, str]] = []
    for raw in value[:12]:
        if not isinstance(raw, Mapping):
            continue
        doc_type = str(raw.get("doc_type") or "").strip().casefold()
        kind = str(raw.get("kind") or "").strip().casefold()
        rendered = _literal_identity_text(
            raw.get("value") or raw.get("subject") or raw.get("name"),
            limit=300,
        )
        basis = _literal_identity_text(raw.get("basis"), limit=500)
        if (
            doc_type == "rbd"
            and kind in _SUBJECT_BINDING_KINDS
            and rendered
            and basis
        ):
            result.append(
                {
                    "doc_type": "rbd",
                    "kind": kind,
                    "value": rendered,
                    "basis": basis,
                }
            )
    return result


def _ground_subject_bindings(
    value: Any, *, raw_user_query: Any
) -> List[Dict[str, Any]]:
    """Authorize subject identity only from a verbatim current-turn span."""

    query = _literal_identity_text(raw_user_query, limit=4000).casefold()
    if not query:
        return []
    result: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in _subject_binding_shape(value):
        rendered = item["value"].casefold()
        basis = item["basis"].casefold()
        if basis not in query or rendered not in basis:
            continue
        identity = (item["kind"], rendered)
        if identity in seen:
            continue
        seen.add(identity)
        result.append(
            {
                **item,
                "grounded": True,
                "source": "current_query",
            }
        )
    return result


def _parse_topology_dependency(
    value: Any,
    *,
    raw_user_query: Any,
) -> Dict[str, Any]:
    """Validate the target model's topology-dependent impact classification.

    This deliberately consumes a structured semantic decision instead of
    re-classifying raw user wording with aliases.  The subject, failed state and
    requested aspect must all be present, preventing a generic component-impact
    question from being promoted to an RBD query merely because it contains a
    word such as “路” in another context.
    """

    if not isinstance(value, dict):
        return {}
    requested = value.get("required") is True
    subject = " ".join(str(value.get("subject") or "").split())
    subject_kind = str(value.get("subject_kind") or "").strip().casefold()
    if subject_kind not in _TOPOLOGY_SUBJECT_KINDS:
        subject_kind = "unknown"
    failure_state = " ".join(str(value.get("failure_state") or "").split())
    shared_basis = _literal_current_span(value.get("basis"), raw_user_query)
    subject_basis = _literal_current_span(
        value.get("subject_basis"), raw_user_query
    )
    failure_state_basis = _literal_current_span(
        value.get("failure_state_basis"), raw_user_query
    )
    grounded_subject = bool(
        subject
        and (
            subject in shared_basis
            or subject in subject_basis
        )
    )
    grounded_failure = bool(
        failure_state
        and (
            failure_state in shared_basis
            or failure_state in failure_state_basis
        )
    )
    aspects: List[str] = []
    for raw_aspect in _bounded_text_list(
        value.get("question_aspects"), limit=5, item_limit=48
    ):
        aspect = str(raw_aspect).strip().casefold()
        if aspect in _TOPOLOGY_QUESTION_ASPECTS and aspect not in aspects:
            aspects.append(aspect)
    required = bool(
        requested
        and grounded_subject
        and subject_kind in _TOPOLOGY_SUBJECT_KINDS
        and grounded_failure
        and aspects
    )
    result = {
        "required": required,
        "subject": subject,
        "subject_kind": subject_kind,
        "failure_state": failure_state,
        "question_aspects": aspects,
    }
    if required:
        if shared_basis and subject in shared_basis and failure_state in shared_basis:
            result["basis"] = shared_basis
        else:
            result["subject_basis"] = subject_basis
            result["failure_state_basis"] = failure_state_basis
    return result


def _parse_diagnostic_closure(
    value: Any,
    *,
    raw_user_query: Any,
) -> Dict[str, Any]:
    """Validate the target model's current-incident closure contract.

    This is deliberately independent from the primary intent: an observed
    abnormality may be phrased as a cause, impact or maintenance question.
    The contract never creates facts or authorizes a calculator; it only keeps
    the bounded evidence-completion path from being mistaken for a scalar
    knowledge lookup.
    """

    return normalize_diagnostic_closure(
        value,
        raw_user_query=raw_user_query,
    )


def _parse_response_mode(
    value: Any,
    *,
    raw_user_query: Any,
) -> Dict[str, str]:
    """Validate the target model's response-breadth protocol."""

    return normalize_response_mode(value, raw_user_query=raw_user_query)


def _literal_current_span(value: Any, raw_user_query: Any, *, limit: int = 500) -> str:
    """Return an exact current-turn span, or ``""`` when provenance fails."""

    span = str(value or "").strip()
    query = str(raw_user_query or "")
    return span if span and span in query else ""


def _parse_response_scope(
    value: Any,
    basis_value: Any,
    *,
    raw_user_query: Any,
) -> tuple[str, str]:
    """Validate continuation breadth without inferring it from history use."""

    scope = str(value or "").strip().casefold()
    if scope == "unknown":
        return "unknown", ""
    basis = _literal_current_span(basis_value, raw_user_query)
    if scope not in _RESPONSE_SCOPES or not basis:
        return "unknown", ""
    return scope, basis


def _parse_coverage_request(
    value: Any,
    *,
    raw_user_query: Any,
    offered_anchor_ids: Any,
) -> Optional[Dict[str, Any]]:
    """Validate a model-authored coverage contract and discard fact claims."""

    if not isinstance(value, Mapping):
        return None
    scope = str(value.get("scope") or "").strip()
    member_kind = str(value.get("member_kind") or "").strip()
    reference_mode = str(value.get("reference_mode") or "").strip()
    subject = str(value.get("subject") or "").strip()
    basis = _literal_current_span(value.get("basis"), raw_user_query)
    if (
        scope not in _COVERAGE_SCOPES
        or member_kind not in _COLLECTION_MEMBER_KINDS
        or reference_mode not in _COVERAGE_REFERENCE_MODES
        or not subject
        or not basis
    ):
        return None

    request: Dict[str, Any] = {
        "scope": scope,
        "subject": subject,
        "member_kind": member_kind,
        "reference_mode": reference_mode,
        "basis": basis,
    }
    if reference_mode == "current_query":
        if not _literal_current_span(subject, raw_user_query, limit=300):
            return None
        return request

    offered = {
        str(item or "").strip()
        for item in offered_anchor_ids or []
        if str(item or "").strip()
    }
    selected = [
        item
        for item in _bounded_text_list(
            value.get("selected_anchor_ids"), limit=12, item_limit=160
        )
        if item in offered
    ]
    if not selected:
        return None
    request["selected_anchor_ids"] = selected
    return request


def _parse_architecture_roles(
    value: Any,
    *,
    raw_user_query: Any,
) -> List[Dict[str, str]]:
    """Validate role labels and literal object provenance only."""

    if not isinstance(value, list):
        return []
    result: List[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw in value[:24]:
        if not isinstance(raw, Mapping):
            continue
        role = str(raw.get("role") or "").strip()
        subject = _literal_current_span(raw.get("subject"), raw_user_query, limit=300)
        basis = _literal_current_span(raw.get("basis"), raw_user_query)
        if (
            role not in _ARCHITECTURE_ROLES
            or not subject
            or not basis
            or subject not in basis
        ):
            continue
        identity = (role, subject)
        if identity in seen:
            continue
        seen.add(identity)
        result.append({"role": role, "subject": subject, "basis": basis})
    return result


def _parse_comparison_subjects(
    value: Any,
    *,
    raw_user_query: Any,
) -> List[Dict[str, str]]:
    """Validate explicitly enumerated comparison subjects."""

    if not isinstance(value, list):
        return []
    result: List[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw in value[:16]:
        if not isinstance(raw, Mapping):
            continue
        kind = str(raw.get("kind") or "").strip()
        subject = _literal_current_span(raw.get("value"), raw_user_query, limit=300)
        basis = _literal_current_span(raw.get("basis"), raw_user_query)
        if (
            kind not in _COMPARISON_SUBJECT_KINDS
            or not subject
            or not basis
            or subject not in basis
        ):
            continue
        identity = (kind, subject)
        if identity in seen:
            continue
        seen.add(identity)
        result.append({"kind": kind, "value": subject, "basis": basis})
    return result


def _parse_missing_input_requirements(
    value: Any,
    *,
    raw_user_query: Any,
) -> List[Dict[str, str]]:
    """Validate only the model's semantic request; state owns missing slots."""

    if not isinstance(value, list):
        return []
    result: List[Dict[str, str]] = []
    seen: set[str] = set()
    for raw in value[:16]:
        if not isinstance(raw, Mapping):
            continue
        slot = str(raw.get("slot") or "").strip()[:160]
        basis = _literal_current_span(raw.get("basis"), raw_user_query)
        if not slot or not basis or slot in seen:
            continue
        seen.add(slot)
        result.append({"slot": slot, "basis": basis})
    return result


def _parse_payload_contract(
    value: Any,
    *,
    raw_user_query: Any,
) -> Optional[Dict[str, str]]:
    """Validate a direction contract without turning roles into identities."""

    if not isinstance(value, Mapping):
        return None
    kind = str(value.get("kind") or "").strip()
    direction = str(value.get("direction") or "").strip()
    if kind not in _PAYLOAD_KINDS or direction not in _PAYLOAD_DIRECTIONS:
        return None
    source_role = str(value.get("source_role") or "").strip()
    destination_role = str(value.get("destination_role") or "").strip()
    basis = str(value.get("basis") or "").strip()
    if kind == "unknown" and direction == "unknown":
        if not source_role and not destination_role and not basis:
            return {
                "kind": kind,
                "direction": direction,
                "source_role": "",
                "destination_role": "",
                "basis": "",
            }
    if (
        not source_role
        or not destination_role
        or not _literal_current_span(basis, raw_user_query)
        or source_role not in basis
        or destination_role not in basis
    ):
        return None
    return {
        "kind": kind,
        "direction": direction,
        "source_role": source_role,
        "destination_role": destination_role,
        "basis": basis,
    }


def _machine_missing_input_requirements(
    model_requirements: Any,
    state: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Materialize a missing-input answer solely from state-machine slots.

    The target model supplies only the discourse decision that this turn asks
    about missing inputs.  It cannot add a slot: at least one grounded model
    item must name a currently missing slot, after which every current machine
    slot is returned so the answer cannot silently omit one.
    """

    modeled = [
        item for item in (model_requirements or []) if isinstance(item, Mapping)
    ]
    if not modeled:
        return []

    machine: List[tuple[str, str, Any]] = []
    seen: set[str] = set()
    for source in ("missing_parameters", "unresolved_identity_slots"):
        raw_values = state.get(source)
        values = raw_values if isinstance(raw_values, (list, tuple, set)) else []
        for raw in values:
            if isinstance(raw, Mapping):
                slot = str(raw.get("slot") or raw.get("name") or "").strip()
            else:
                slot = str(raw or "").strip()
            slot = slot[:160]
            if not slot or slot in seen:
                continue
            seen.add(slot)
            machine.append((slot, source, raw))
    if not machine:
        return []

    modeled_by_slot = {
        str(item.get("slot") or "").strip(): str(item.get("basis") or "").strip()
        for item in modeled
        if str(item.get("slot") or "").strip()
        and str(item.get("basis") or "").strip()
    }
    if not set(modeled_by_slot).intersection(slot for slot, _, _ in machine):
        return []
    common_basis = next(iter(modeled_by_slot.values()))
    requirements: List[Dict[str, Any]] = []
    for slot, source_name, raw in machine:
        kind = (
            "calculation_parameter"
            if source_name == "missing_parameters"
            else "identity_slot"
        )
        raw_identity = raw.get("identity") if isinstance(raw, Mapping) else None
        identity = normalize_clarification_identity(raw_identity)
        if identity is None:
            identity = make_clarification_identity(
                fields={"parameter" if kind == "calculation_parameter" else "slot": slot}
            )
        requirements.append(
            make_missing_input_requirement(
                kind=kind,
                slot=slot,
                source_kind="state",
                source_name=source_name,
                identity=identity,
                basis=modeled_by_slot.get(slot) or common_basis,
            )
        )
    return unique_contract_items(requirements)


def _parse_target_response(
    text: Any,
    *,
    offered_anchor_ids: Any = (),
    raw_user_query: Any = "",
    offered_user_context: Any = (),
) -> Optional[Dict[str, Any]]:
    """Parse the model-only goal and conversational-reference classification.

    ``query_targets`` is accepted as a compatibility shape so mocked planner
    responses and older model prompts remain usable. No target is inferred
    from the raw wording in this function.
    """

    parsed = _extract_json_object(text)
    if not parsed:
        return None
    raw_targets = parsed.get("targets")
    if not isinstance(raw_targets, list):
        raw_targets = parsed.get("query_targets")
    targets: List[str] = []
    for item in raw_targets if isinstance(raw_targets, list) else []:
        value = item.get("intent") if isinstance(item, dict) else item
        intent = str(value or "").strip()
        if intent in _TARGET_INTENTS and intent not in targets:
            targets.append(intent)

    topology_dependency = _parse_topology_dependency(
        parsed.get("topology_dependency"),
        raw_user_query=raw_user_query,
    )
    diagnostic_closure = _parse_diagnostic_closure(
        parsed.get("diagnostic_closure"),
        raw_user_query=raw_user_query,
    )
    response_mode = _parse_response_mode(
        parsed.get("response_mode"),
        raw_user_query=raw_user_query,
    )
    if topology_dependency.get("required"):
        # A named path failure with an impact/continuity question has two answer
        # dimensions: recorded effects and the RBD success boundary.  Preserve
        # both even when an older/smaller target model emitted only one of them.
        for target in ("impact_analysis", "reliability_query"):
            if target not in targets:
                targets.append(target)

    primary = str(parsed.get("primary_intent") or parsed.get("intent") or "").strip()
    topology_aspects = set(topology_dependency.get("question_aspects") or [])
    if topology_dependency.get("required"):
        if topology_aspects & {"impact", "consequence"}:
            primary = "impact_analysis"
        elif primary not in {"impact_analysis", "reliability_query"}:
            primary = "reliability_query"
    if primary not in _TARGET_INTENTS:
        primary = targets[0] if targets else ""
    if primary and primary not in targets:
        targets.insert(0, primary)
    if not targets:
        return None
    if primary in targets:
        targets = [primary, *(target for target in targets if target != primary)]
    else:
        primary = targets[0]
    contextualized_query = " ".join(
        str(parsed.get("contextualized_query") or "").split()
    )
    raw_context_used = parsed.get("context_used")
    context_used = raw_context_used is True or str(raw_context_used or "").strip().casefold() in {
        "1", "true", "yes",
    }
    context_status = str(parsed.get("context_status") or "").strip().casefold()
    if context_status not in {"standalone", "resolved", "ambiguous", "missing"}:
        context_status = "resolved" if context_used else "standalone"
    offered = {
        str(value or "").strip()
        for value in offered_anchor_ids or []
        if str(value or "").strip()
    }
    selected_anchor_ids = [
        value
        for value in _bounded_text_list(
            parsed.get("selected_anchor_ids"), limit=12, item_limit=160
        )
        if value in offered
    ]
    coverage_request = _parse_coverage_request(
        parsed.get("coverage_request"),
        raw_user_query=raw_user_query,
        offered_anchor_ids=offered,
    )
    if coverage_request:
        selected_anchor_ids = _merge_text_lists(
            selected_anchor_ids,
            coverage_request.get("selected_anchor_ids") or [],
            limit=12,
            item_limit=160,
        )
    relation_expansion_present = "relation_expansion" in parsed
    relation_expansion = normalize_relation_expansion_contract(
        parsed.get("relation_expansion"),
        raw_user_query=raw_user_query,
        offered_anchor_ids=offered,
    )
    if relation_expansion_present and not relation_expansion:
        # A malformed positive object is an explicit fail-closed decision. This
        # also prevents the later retrieval planner from replacing it with an
        # ungrounded graph request.
        relation_expansion = {
            "needed": False,
            "relation_kind": "",
            "reference_mode": "",
            "basis": "",
            "selected_anchor_ids": [],
            "reason": "",
        }
    if relation_expansion.get("needed") is True:
        selected_anchor_ids = _merge_text_lists(
            selected_anchor_ids,
            relation_expansion.get("selected_anchor_ids") or [],
            limit=12,
            item_limit=160,
        )
    # Legacy scalar fields are retained only in the raw model trace.  They are
    # not authority for a closed set or an expanded answer window: that
    # requires a grounded ``coverage_request`` object.
    coverage_scope = "unknown"
    collection_member_kind = "unknown"
    if coverage_request:
        coverage_scope = str(coverage_request["scope"])
        collection_member_kind = str(coverage_request["member_kind"])
    requested_facets = _bounded_text_list(
        parsed.get("requested_facets")
    )
    required_fields = normalize_answer_fields(
        parsed.get("required_answer_fields"),
        requested_facets=requested_facets,
    )
    if not coverage_request or coverage_request.get("scope") != "complete_collection":
        required_fields = [
            field for field in required_fields
            if field != "closed_collection_members"
        ]
    question_semantics = str(
        parsed.get("question_semantics") or "unknown"
    ).strip().casefold()
    if question_semantics not in _QUESTION_SEMANTICS:
        question_semantics = "unknown"
    response_focus: List[str] = []
    for raw_focus in parsed.get("response_focus") or []:
        focus = str(raw_focus or "").strip().casefold()
        if focus in _RESPONSE_FOCUSES and focus not in response_focus:
            response_focus.append(focus)
    response_scope, response_scope_basis = _parse_response_scope(
        parsed.get("response_scope"),
        parsed.get("response_scope_basis"),
        raw_user_query=raw_user_query,
    )
    if topology_dependency.get("required"):
        required_fields = normalize_answer_fields(
            [
                *required_fields,
                "rbd_single_failure_boundary",
                "rbd_success_criterion",
            ]
        )
    required_fields = ensure_intent_answer_fields(
        required_fields,
        intents=targets,
    )
    # Typed answer fields are semantic output contracts.  When a compact
    # target classifier expresses a board/API/history/structural requirement
    # through one of those fields, retain its owning information target even
    # if it chose a neighbouring primary label such as function_location.
    # This is not raw-text keyword routing and never authorizes calculators.
    target_by_field = {
        "structural_cause_chain": "cause_analysis",
        "impact_propagation": "impact_analysis",
        "maintenance_locator": "maintenance_query",
        "maintenance_api": "maintenance_query",
        "platform_test_status": "maintenance_query",
        "platform_test_result": "maintenance_query",
        "history_time": "history_query",
        "history_root": "history_query",
        "history_steps": "history_query",
        "history_solution": "history_query",
        "history_result": "history_query",
        "history_personnel": "history_query",
        "history_preventive_action": "history_query",
    }
    for field, target in target_by_field.items():
        if field in required_fields and target not in targets:
            targets.append(target)
    if primary in targets:
        targets = [primary, *(target for target in targets if target != primary)]
    required_fields = ensure_intent_answer_fields(
        required_fields,
        intents=targets,
    )
    if (
        coverage_request
        and coverage_scope == "complete_collection"
        and str(coverage_request.get("subject") or "").strip()
        and collection_member_kind in _COLLECTION_MEMBER_KINDS
        and "closed_collection_members" not in required_fields
    ):
        # The coverage/member schema itself establishes an N-of-N obligation.
        # Persist it as an answer field even when an older model response omits
        # the redundant field name; no domain semantics are inferred here.
        required_fields = normalize_answer_fields(
            [*required_fields, "closed_collection_members"]
        )
    normalized_query_targets: List[Dict[str, Any]] = [
        {
            "intent": target,
            "source": "semantic_classifier",
            "reason": (
                "命名路径失效的后果同时需要故障影响与RBD能力边界"
                if topology_dependency.get("required")
                and target in {"impact_analysis", "reliability_query"}
                else "语义模型识别的本轮明确信息目标"
            ),
        }
        for target in targets
    ]
    if normalized_query_targets and required_fields:
        normalized_query_targets[0]["required_fields"] = list(required_fields)
    result: Dict[str, Any] = {
        "primary_intent": primary,
        "query_targets": normalized_query_targets,
        "contextualized_query": contextualized_query,
        "subjects": _bounded_text_list(parsed.get("subjects")),
        "requested_facets": requested_facets,
        "required_answer_fields": required_fields,
        "question_semantics": question_semantics,
        "response_focus": response_focus,
        "response_scope": response_scope,
        "topology_dependency": topology_dependency,
        "typed_record_fields": _typed_record_field_shape(
            parsed.get("typed_record_fields")
        ),
        "subject_bindings": _subject_binding_shape(
            parsed.get("subject_bindings")
        ),
        # Grounding against the current raw turn happens only when this target
        # decision is reconciled with the independent retrieval planner.
        "calculation_authorizations": normalize_calculation_authorizations(
            parsed.get("calculation_authorizations")
        ),
        # This is an LLM-authored discourse contract, not a keyword
        # classifier. Later nodes only use it to decide whether an
        # authoritative query can prove a closed set.
        "coverage_scope": coverage_scope,
        "collection_subject": (
            str(coverage_request.get("subject") or "")
            if coverage_request
            else ""
        ),
        "collection_member_kind": collection_member_kind,
        "context_used": context_used,
        "status": context_status,
        "selected_anchor_ids": selected_anchor_ids,
    }
    if context_used and context_status == "resolved":
        user_bindings = validate_user_context_bindings(parsed.get("user_context_bindings"), offered_user_context)
        if user_bindings:
            result["user_context_bindings"] = user_bindings
    if diagnostic_closure:
        result["diagnostic_closure"] = diagnostic_closure
    if response_mode:
        result["response_mode"] = response_mode
    if response_scope_basis:
        result["response_scope_basis"] = response_scope_basis
    if "requested_sources" in parsed:
        result["requested_sources"] = _parse_requested_sources(
            parsed.get("requested_sources"), raw_user_query=raw_user_query
        )
    if "coverage_request" in parsed and coverage_request:
        result["coverage_request"] = coverage_request
    if relation_expansion_present:
        result["relation_expansion"] = relation_expansion
    if "architecture_roles" in parsed:
        result["architecture_roles"] = _parse_architecture_roles(
            parsed.get("architecture_roles"), raw_user_query=raw_user_query
        )
    if "comparison_subjects" in parsed:
        result["comparison_subjects"] = _parse_comparison_subjects(
            parsed.get("comparison_subjects"), raw_user_query=raw_user_query
        )
    if "missing_input_requirements" in parsed:
        result["missing_input_requirements"] = _parse_missing_input_requirements(
            parsed.get("missing_input_requirements"),
            raw_user_query=raw_user_query,
        )
    if "payload_contract" in parsed:
        payload_contract = _parse_payload_contract(
            parsed.get("payload_contract"), raw_user_query=raw_user_query
        )
        if payload_contract:
            result["payload_contract"] = payload_contract
    return result


def _parse_json_response(text: str) -> dict:
    try:
        value = text.strip()
        if value.startswith("```json"):
            value = value[7:]
        if value.startswith("```"):
            value = value[3:]
        if value.endswith("```"):
            value = value[:-3]
        value = value.strip()
        if not value.startswith("{"):
            start = value.find("{")
            end = value.rfind("}")
            if start >= 0 and end > start:
                value = value[start:end + 1]
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("规划结果不是 JSON 对象")
        parsed["_planner_failed"] = False
        return parsed
    except Exception as exc:
        logger.warning("Plan JSON parse failed: %s; raw=%s", exc, text[:200])
        # B1 降级计数：规划模型 JSON 解析失败退回宽召回。
        try:
            from backend.agent.tracing import record_degradation

            record_degradation("planner_json", detail=str(exc)[:200])
        except Exception:
            pass
        return {
            # A failed semantic planner must not be replaced by a keyword
            # classifier. Broad recall is slower but does not silently omit a
            # document family just because the user used unexpected wording.
            "intent": "hybrid",
            "query_targets": [
                {
                    "intent": "hybrid",
                    "source": "fallback",
                    "reason": "规划模型输出不可用，扩大到七类资料召回",
                }
            ],
            "rewritten_query": "",
            "entity_keywords": [],
            "type_keywords": [],
            "query_frame": {},
            "tool_queries": {},
            "calculation_authorizations": [],
            "tools_plan": [
                {"tool": "hybrid_search", "reason": "从七类文档执行语义召回"},
            ],
            "reasoning": "规划模型不可用，采用七类资料的安全降级召回",
            "_planner_failed": True,
        }


def _prepare_plan_model(model):
    """给规划阶段设置独立的小预算；领域策略仍由确定性白名单校验。"""
    if not model:
        return model
    prepared = dict(model)
    if prepared.get("provider") == "ollama":
        # 规划只需稳定的结构化 JSON；关闭长思考并使用确定性采样，避免
        # 同一问题因格式漂移退回宽召回。
        prepared["reasoning"] = False
        prepared["temperature"] = 0.0
        prepared["format"] = "json"
    return bounded_model_transport(prepared, PLAN_LLM_TIMEOUT_SECONDS)


def _prepare_target_model(model):
    """Reserve enough bounded output for the complete turn contract."""

    if not model:
        return model
    prepared = dict(model)
    if prepared.get("provider") == "ollama":
        prepared["reasoning"] = False
        prepared["temperature"] = 0.0
        prepared["format"] = "json"
    prepared["stream"] = False
    return bounded_model_transport(prepared, TARGET_LLM_TIMEOUT_SECONDS)


def _valid_complete_collection_contract(decision: Any) -> bool:
    if not isinstance(decision, dict):
        return False
    request = decision.get("coverage_request")
    if not isinstance(request, Mapping):
        return False
    if str(request.get("scope") or "") != "complete_collection":
        return False
    if str(decision.get("coverage_scope") or "") != "complete_collection":
        return False
    if not str(request.get("subject") or "").strip():
        return False
    if str(request.get("member_kind") or "") not in _COLLECTION_MEMBER_KINDS:
        return False
    return "closed_collection_members" in normalize_answer_fields(
        decision.get("required_answer_fields"),
        requested_facets=decision.get("requested_facets") or [],
    )


async def _repair_complete_collection_contract(
    *,
    query: str,
    history: str,
    decision: Dict[str, Any],
    offered_anchor_ids: Sequence[str],
    model: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    prompt = COLLECTION_TARGET_REPAIR_PROMPT.format(
        query=query,
        history=history,
        decision=json.dumps(decision, ensure_ascii=False, default=str),
    )
    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(
                llm_service.call_model,
                [{"role": "user", "content": prompt}],
                _prepare_target_model(model),
            ),
            timeout=remaining_seconds(),
        )
    except asyncio.TimeoutError:
        logger.warning("Complete-collection contract repair exceeded %.1fs", TARGET_LLM_TIMEOUT_SECONDS)
        return None
    except Exception as exc:
        logger.warning("Complete-collection contract repair failed: %s", exc)
        return None
    repaired = _parse_target_response(
        response,
        offered_anchor_ids=offered_anchor_ids,
    )
    return repaired if _valid_complete_collection_contract(repaired) else None


def _prior_turn_texts(
    messages: Any, current_query: str
) -> Tuple[str, str]:
    """Return the user query and assistant answer of the previous turn."""

    if not isinstance(messages, list):
        return "", ""
    items = [item for item in messages if isinstance(item, Mapping)]
    current = " ".join(str(current_query or "").split()).strip("，。！？?!")
    while items:
        last = items[-1]
        if (
            str(last.get("role") or "") == "user"
            and " ".join(str(last.get("content") or "").split())
            .strip("，。！？?!")
            == current
        ):
            items.pop()
            continue
        break
    prior_user = ""
    prior_assistant = ""
    for item in reversed(items):
        role = str(item.get("role") or "")
        if role == "user" and not prior_user:
            prior_user = str(item.get("content") or "")
        elif role == "assistant" and not prior_assistant:
            prior_assistant = str(item.get("content") or "")
        if prior_user and prior_assistant:
            break
    return prior_user, prior_assistant


_TOOL_BY_DOC_TYPE = {
    str(doc_type): tool for tool, doc_type in DOC_TYPE_BY_TOOL.items()
}


def _prior_answer_doc_types(messages: Any) -> List[str]:
    """Doc types actually cited by the nearest prior assistant answer."""

    if not isinstance(messages, list):
        return []
    result: List[str] = []
    for message in reversed(messages):
        if not isinstance(message, Mapping) or str(message.get("role") or "") != "assistant":
            continue
        seen: set[str] = set()
        for anchor in message.get("conversation_anchors") or []:
            if not isinstance(anchor, Mapping):
                continue
            doc_type = str(anchor.get("doc_type") or "").strip()
            if doc_type and doc_type not in seen:
                seen.add(doc_type)
                result.append(doc_type)
        break
    return result


def _prior_answer_anchor_subjects(messages: Any) -> List[str]:
    """Public identity terms of the nearest prior assistant answer's citations."""

    if not isinstance(messages, list):
        return []
    anchors: List[Mapping[str, Any]] = []
    for message in reversed(messages):
        if not isinstance(message, Mapping) or str(message.get("role") or "") != "assistant":
            continue
        anchors = [
            dict(anchor)
            for anchor in message.get("conversation_anchors") or []
            if isinstance(anchor, Mapping)
        ]
        break
    return _anchor_subjects(anchors)


def _prior_delivered_fields(messages: Any) -> List[str]:
    """Answer fields the immediately preceding assistant answer actually
    delivered (required − remaining).  The chat router persists this list on
    the assistant message; the values are stable answer-contract field names
    only, never prose, so they can be subtracted from the next turn's
    obligation set without reusing old conclusions as current facts.
    """

    if not isinstance(messages, list):
        return []
    for message in reversed(messages):
        if not isinstance(message, Mapping) or str(message.get("role") or "") != "assistant":
            continue
        delivered = message.get("delivered_answer_fields")
        if not isinstance(delivered, list):
            return []
        fields: List[str] = []
        seen: set[str] = set()
        for value in delivered:
            normalized = str(value or "").strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                fields.append(normalized)
        return fields
    return []


def _selected_anchor_delivery_contracts(
    messages: Any,
    dialogue_context: Mapping[str, Any],
    selected_anchors: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Read natural goal progress for the selected evidence and source turn.

    Legacy field manifests are read-only compatibility. Progress describes
    what was delivered or left unresolved, not an engineering fact or a rule
    to suppress a deeper question about an already discussed dimension.
    """

    if (
        not isinstance(messages, list)
        or not isinstance(dialogue_context, Mapping)
        or not bool(dialogue_context.get("context_used"))
        or not selected_anchors
    ):
        return []

    selected_evidence_ids = {
        str(anchor.get("evidence_id") or anchor.get("anchor_id") or "").strip()
        for anchor in selected_anchors
        if str(anchor.get("evidence_id") or anchor.get("anchor_id") or "").strip()
    }
    selected_record_ids = {
        str(anchor.get("record_id") or "").strip().upper()
        for anchor in selected_anchors
        if str(anchor.get("record_id") or "").strip()
    }
    selected_origin_ids = {
        str(anchor.get("origin_message_id") or "").strip()
        for anchor in selected_anchors
        if str(anchor.get("origin_message_id") or "").strip()
    }
    selected_subject_ids: set[str] = set(selected_record_ids)
    for anchor in selected_anchors:
        fields = anchor.get("answer_fields")
        if not isinstance(fields, Mapping):
            continue
        for key, raw in fields.items():
            normalized_key = str(key or "").strip().lower()
            if not (
                normalized_key.endswith("_id")
                or normalized_key.endswith("_code")
                or normalized_key in {"abbreviation", "hmc_code", "record_code"}
            ):
                continue
            values = raw if isinstance(raw, (list, tuple, set)) else [raw]
            selected_subject_ids.update(
                str(value or "").strip().upper()
                for value in values
                if str(value or "").strip()
            )

    result: List[Dict[str, Any]] = []
    seen_entries: set[tuple[str, str]] = set()
    for message in reversed(messages):
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        message_id = str(message.get("message_id") or message.get("id") or "").strip()
        if selected_origin_ids and message_id not in selected_origin_ids:
            continue
        manifest = message.get("answer_delivery_manifest")
        if not isinstance(manifest, list):
            continue
        for raw_entry in manifest:
            if not isinstance(raw_entry, Mapping):
                continue
            is_goal = raw_entry.get("kind") == "goal_delivery"
            field = str(raw_entry.get("goal" if is_goal else "field") or "").strip()
            entry_key = (message_id, field)
            if not field or entry_key in seen_entries:
                continue
            if is_goal and raw_entry.get("status") not in {"answered", "partial", "unanswered"}:
                continue
            evidence_ids = {
                str(value or "").strip()
                for value in (raw_entry.get("evidence_ids") or [])
                if str(value or "").strip()
            }
            record_ids = {
                str(value or "").strip().upper()
                for value in (raw_entry.get("record_ids") or [])
                if str(value or "").strip()
            }
            subject_ids = {
                str(value or "").strip().upper()
                for value in (raw_entry.get("subject_ids") or [])
                if str(value or "").strip()
            }
            matches = bool(evidence_ids & selected_evidence_ids)
            if is_goal:
                # An unresolved goal can have no engineering citation. Keep
                # it only from the very turn selected now, never as a fact.
                matches = matches or (not evidence_ids and message_id in selected_origin_ids)
            else:
                matches = matches or bool(record_ids & selected_record_ids or subject_ids & selected_subject_ids)
            if not matches:
                continue
            entry = dict(raw_entry)
            if message_id:
                entry["origin_message_id"] = message_id
            result.append(entry)
            seen_entries.add(entry_key)
    return result


def _contextual_prior_delivered_fields(
    messages: Any,
    dialogue_context: Mapping[str, Any],
    selected_anchors: Sequence[Mapping[str, Any]] | None = None,
) -> List[str]:
    """Return prior delivery metadata only for a resolved continuation.

    Field names shape a follow-up but do not identify an engineering object by
    themselves. Carrying them into an unrelated new question could suppress a
    legitimately requested field merely because the preceding turn happened
    to answer the same dimension.
    """

    if not isinstance(dialogue_context, Mapping) or not bool(
        dialogue_context.get("context_used")
    ):
        return []
    if selected_anchors is not None:
        return [
            str(entry.get("field") or "").strip()
            for entry in _selected_anchor_delivery_contracts(
                messages, dialogue_context, selected_anchors
            )
            if str(entry.get("field") or "").strip()
        ]
    # Compatibility for pre-manifest callers and stored sessions. Production
    # planning always supplies ``selected_anchors`` and therefore never applies
    # a field-only delivery to a different object.
    return _prior_delivered_fields(messages)


def _continuation_source_tools(
    *,
    query: str,
    messages: Any,
    prior_doc_types: Sequence[str] = (),
    selected_anchors: Sequence[Mapping[str, Any]] = (),
    context_used: bool = False,
    response_focus: Sequence[str] = (),
    current_requested_sources: Sequence[Mapping[str, Any]] = (),
) -> List[str]:
    """Source tools for a boundary follow-up on explicitly selected citations.

    The nearest answer can cite several unrelated families. Only anchors the
    target stage selected for this turn may be re-read; the prior answer's whole
    citation set, prose, and document list are never execution authority.
    """

    # Compatibility parameters remain for saved tests/callers, but neither raw
    # prose nor an unscoped prior document list grants source access.
    del query, messages, prior_doc_types
    # 用户本轮的结构化来源合同优先，不叠加延续来源。
    if current_requested_sources:
        return []
    if context_used is not True or not selected_anchors:
        return []
    # The target model, rather than a fixed phrase list, decides whether this
    # is a source-capability/boundary follow-up.  A plain change of dimension
    # (history, repair, impact, ...) must not drag the old source family along.
    focuses = {
        str(value or "").strip() for value in response_focus
        if str(value or "").strip()
    }
    if not focuses & {"source_capability", "source_limitation"}:
        return []
    cited_tools = [
        _TOOL_BY_DOC_TYPE.get(str(anchor.get("doc_type") or "").strip())
        for anchor in selected_anchors
        if isinstance(anchor, Mapping)
        and any(
            str(anchor.get(key) or "").strip()
            for key in ("anchor_id", "record_id", "evidence_id")
        )
    ]
    cited_tools = list(dict.fromkeys(tool for tool in cited_tools if tool))
    return cited_tools


async def _classify_explicit_targets(
    *,
    query: str,
    history: str,
    conversation_anchors: List[Dict[str, Any]],
    model: Optional[Dict[str, Any]],
    user_context: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    prompt = TARGET_PROMPT.format(
        query=query,
        history=history,
        answer_field_catalog="、".join(f"{key}（{spec['label']}）" for key, spec in ANSWER_FIELD_SPECS.items()),
        conversation_anchors=json.dumps(
            conversation_anchors, ensure_ascii=False, default=str
        ),
    )
    prompt += "\n\n" + document_semantics_prompt()
    if user_context:
        prompt += ("\n\n可引用的先前用户原始输入（未确认、不是证据或计算授权）：\n"
                   + json.dumps(user_context, ensure_ascii=False)
                   + '\n当本轮仍指向这些输入时，可设置context_used=true、context_status=resolved，并输出user_context_bindings=[{"message_id":"给出的消息ID","quote":"相关用户消息中的连续原文"}]。已有证据锚点时也须保留仍适用的用户现象和工况；锚点只证明先前引用过记录，不证明候选已成为当前故障。'
                   "只选择本轮仍适用的输入；不选择助手结论、不复制旧计算授权、不跨话题自动继承。"
                   "用户追问还需哪些原始资料时，仍保留原待查对象，而非改查不相干故障。无法明确指代时保持缺少或歧义状态。")
    offered_anchor_ids = [
        str(item.get("anchor_id") or "") for item in conversation_anchors
        if isinstance(item, dict) and str(item.get("anchor_id") or "")
    ]
    decision = None
    deadline = time.monotonic() + remaining_seconds()
    try:
        for attempt in range(2):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise asyncio.TimeoutError
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    llm_service.call_model,
                    [{"role": "user", "content": prompt}],
                    bounded_model_transport(_prepare_target_model(model), remaining),
                    template_name="explicit_target_contract",
                ),
                timeout=remaining,
            )
            decision = _parse_target_response(
                response, offered_anchor_ids=offered_anchor_ids, raw_user_query=query,
                offered_user_context=user_context or [],
            )
            if decision:
                break
            _record_target_failure(f"invalid_contract; response_chars={len(str(response))}")
            raw_response = str(response or "").strip()
            if attempt or not raw_response.startswith("{"):
                break
            # Retry malformed form within the original budget, never infer a
            # permission from planner prose or a vocabulary match.
            prompt += ("\n合同校正：上次输出没有有效信息目标。请重新返回完整JSON，"
                       "primary_intent和targets的值必须来自：" + ", ".join(sorted(_TARGET_INTENTS))
                       + "。字段名称、计算操作名称不是意图值，不能放入targets。"
                       "计算请求只在calculation_authorizations中按operation/mode/target/basis表达；"
                       "指代选择仍只从上方anchor_id选取。省略未使用的可选对象，避免复述规则以免截断。"
                       "仍只依据同一个原始问题和同一组候选锚点判断，不增加用户未要求的计算授权。")
    except asyncio.TimeoutError:
        logger.warning("Explicit target classification exceeded %.1fs", TARGET_LLM_TIMEOUT_SECONDS)
        _record_target_failure("timeout")
        return None
    except Exception as exc:
        logger.warning("Explicit target classification failed: %s", exc)
        _record_target_failure(type(exc).__name__)
        return None
    if decision:
        # The small target model is allowed to recognize information goals,
        # but it cannot invent an exhaustive-answer obligation from a broad
        # cause question.  Apply this before the collection-repair call so an
        # accidental `complete_collection` is never promoted into a stricter
        # contract by a second model pass.
        decision = _downgrade_unrequested_collection_contract(
            decision,
            query=query,
        )
    if (
        decision
        and str(decision.get("coverage_scope") or "") == "complete_collection"
        and not _valid_complete_collection_contract(decision)
    ):
        repaired = await _repair_complete_collection_contract(
            query=query,
            history=history,
            decision=decision,
            offered_anchor_ids=offered_anchor_ids,
            model=model,
        )
        if repaired:
            decision = repaired
        else:
            # Preserve the user's exhaustive coverage request even when the
            # semantic repair cannot identify a safe parent/kind.  Later
            # stages may retrieve candidates, but cannot claim closure without
            # a PostgreSQL collection identity.
            decision["required_answer_fields"] = normalize_answer_fields(
                [
                    *(decision.get("required_answer_fields") or []),
                    "closed_collection_members",
                ]
            )
    return decision


def _record_target_failure(detail: str) -> None:
    logger.warning("Target contract unavailable: %s", detail)
    try:
        from backend.agent.tracing import record_degradation
        record_degradation("target_contract", detail=detail)
    except Exception:
        logger.debug("Unable to record target-contract degradation", exc_info=True)


async def _generate_plan_response(prompt: str, model, stream_queue) -> str:
    messages = [{"role": "user", "content": prompt}]
    if stream_queue is None:
        return await asyncio.to_thread(llm_service.call_model, messages, model)

    response = ""
    async for token in llm_service.stream_model(messages, model):
        response += token
        try:
            await stream_queue.put({"type": "thinking_delta", "source": "plan", "content": token})
        except Exception:
            pass
    return response


def _bind_retrieval_plan(parsed, target, dialogue, query):
    """Project retrieval hints onto the single validated semantic contract.

    Empty target fields are meaningful: a query writer cannot fill them with
    different goals, identities or permissions. Its observation slots still
    undergo raw-user provenance validation in build_query_frame.
    """
    if not target:
        fallback = _parse_json_response("")
        fallback["rewritten_query"] = query
        if parsed.get("_planner_failed") and parsed.get("reasoning"):
            fallback["reasoning"] = parsed["reasoning"]
        return fallback
    result = {key: parsed.get(key) for key in (
        "tool_queries", "tools_plan", "type_keywords", "reasoning", "_planner_failed",
    )}
    raw_frame = parsed.get("query_frame")
    raw_frame = raw_frame if isinstance(raw_frame, Mapping) else {}
    frame = {key: raw_frame[key] for key in (
        "symptoms", "operating_stage", "operating_stage_basis", "temporal_pattern",
        "temporal_pattern_basis", "constraints", "normalized_terms",
    ) if key in raw_frame}
    for key in (
        "requested_facets", "required_answer_fields", "question_semantics",
        "response_focus", "response_scope", "response_scope_basis",
        "coverage_request", "coverage_scope", "collection_subject",
        "collection_member_kind", "topology_dependency", "diagnostic_closure",
        "response_mode", "typed_record_fields", "subject_bindings",
        "requested_sources", "architecture_roles", "comparison_subjects",
        "missing_input_requirements", "payload_contract",
    ):
        if key in dialogue:
            frame[key] = dialogue[key]
    frame["entities"] = list(dialogue.get("subjects") or [])
    result.update(
        intent=target["primary_intent"], query_targets=target["query_targets"],
        rewritten_query=dialogue.get("contextualized_query") or query,
        entity_keywords=list(dialogue.get("subjects") or []),
        query_frame=frame,
        relation_expansion=dialogue.get("relation_expansion") or {},
        calculation_authorizations=target.get("calculation_authorizations") or [],
    )
    return result


async def plan_node(state, config=None) -> Dict:
    agent_id = state.get("agent_id", "")
    query = state.get("query", "")
    system_prompt = state.get("system_prompt", "")

    memory_context = state.get("memory_context", "")
    if not memory_context and state.get("memory_enabled", False):
        try:
            result = await inject_memory(state, config or {})
            memory_context = result.get("memory_context", "")
        except Exception:
            memory_context = ""

    messages = state.get("messages", [])
    summary_threshold = int(state.get("summary_threshold", 20) or 20)
    keep_recent = int(state.get("keep_recent_turns", 8) or 8)
    if len(messages) >= summary_threshold:
        try:
            await maybe_summarize(
                state,
                summary_threshold=summary_threshold,
                keep_recent=keep_recent,
                agent_id=agent_id,
            )
        except Exception as exc:
            logger.warning("Conversation summary failed: %s", exc)

    chat_model_id = state.get("chat_model_id", "")
    if chat_model_id:
        from backend.storage.postgres_client import postgres_client
        model = postgres_client.get_model(chat_model_id)
    else:
        model = agent_model_role_service.get_model_by_role(agent_id, "plan") if agent_id else None
    model = agent_model_role_service.apply_agent_llm_params(state, model)
    model = _prepare_plan_model(model)

    formatted_history = _format_history(messages)
    conversation_anchors = _offered_conversation_anchors(messages)
    explicit_target_decision = await _classify_explicit_targets(
        query=query,
        history=formatted_history,
        conversation_anchors=conversation_anchors,
        model=model,
        user_context=offered_user_inputs(state),
    )
    dialogue_context: Dict[str, Any] = {
        "contextualized_query": query,
        "subjects": [],
        "requested_facets": [],
        "required_answer_fields": [],
        "question_semantics": "unknown",
        "response_focus": [],
        "response_scope": "unknown",
        "response_scope_basis": "",
        "topology_dependency": {},
        "typed_record_fields": {},
        "subject_bindings": [],
        "coverage_scope": "unknown",
        "collection_subject": "",
        "collection_member_kind": "unknown",
        "context_used": False,
        "status": "standalone",
        "selected_anchor_ids": [],
    }
    if explicit_target_decision:
        contextualized_query = str(
            explicit_target_decision.get("contextualized_query") or query
        ).strip()
        dialogue_context = {
            "contextualized_query": contextualized_query,
            "subjects": list(explicit_target_decision.get("subjects") or []),
            "requested_facets": list(
                explicit_target_decision.get("requested_facets") or []
            ),
            "required_answer_fields": list(
                explicit_target_decision.get("required_answer_fields") or []
            ),
            "question_semantics": str(
                explicit_target_decision.get("question_semantics") or "unknown"
            ),
            "response_focus": list(
                explicit_target_decision.get("response_focus") or []
            ),
            "response_scope": str(
                explicit_target_decision.get("response_scope") or "unknown"
            ),
            "response_scope_basis": str(
                explicit_target_decision.get("response_scope_basis") or ""
            ),
            "topology_dependency": dict(
                explicit_target_decision.get("topology_dependency") or {}
            ),
            "typed_record_fields": dict(
                explicit_target_decision.get("typed_record_fields") or {}
            ),
            "subject_bindings": list(
                explicit_target_decision.get("subject_bindings") or []
            ),
            "coverage_scope": str(
                explicit_target_decision.get("coverage_scope") or "unknown"
            ),
            "collection_subject": str(
                explicit_target_decision.get("collection_subject") or ""
            ),
            "collection_member_kind": str(
                explicit_target_decision.get("collection_member_kind") or "unknown"
            ),
            "context_used": bool(explicit_target_decision.get("context_used")),
            "status": str(explicit_target_decision.get("status") or "standalone"),
            "selected_anchor_ids": list(
                explicit_target_decision.get("selected_anchor_ids") or []
            ),
        }
        for key in (
            "requested_sources",
            "coverage_request",
            "architecture_roles",
            "comparison_subjects",
            "missing_input_requirements",
            "payload_contract",
            "diagnostic_closure",
            "response_mode",
            "relation_expansion",
            "user_context_bindings",
        ):
            if key not in explicit_target_decision:
                continue
            value = explicit_target_decision[key]
            dialogue_context[key] = (
                dict(value) if isinstance(value, Mapping) else list(value or [])
            )
    # Selected citation IDs come only from the target model and are already
    # intersected with the server-offered set. Do not complete a diagram family
    # from a count phrase in the user's prose: any additional FTA/RBD member
    # must itself be named by ``selected_anchor_ids`` (or a future typed group
    # contract with explicit origin/document/cardinality validation).
    selected_anchor_ids = set(dialogue_context["selected_anchor_ids"])
    selected_conversation_anchors = [
        dict(item)
        for item in conversation_anchors
        if str(item.get("anchor_id") or "") in selected_anchor_ids
    ]
    if selected_conversation_anchors:
        # A selected citation contributes only its persisted identity fields,
        # never its old prose or conclusion. The live query still re-reads the
        # authoritative record in this turn before using it as evidence.
        dialogue_context["subjects"] = _merge_text_lists(
            dialogue_context.get("subjects"),
            _anchor_subjects(selected_conversation_anchors),
            limit=20,
        )
        if dialogue_context.get("status") == "standalone":
            dialogue_context["status"] = "resolved"
            dialogue_context["context_used"] = True

    prompt = PLAN_PROMPT.format(
        query=query,
        history=formatted_history,
        dialogue_context=json.dumps(dialogue_context, ensure_ascii=False),
        memory=memory_context or "无",
        system_prompt=system_prompt or "ICP 故障诊断",
    )
    prompt += "\n\n" + document_semantics_prompt()

    stream_queue = None
    if config and isinstance(config, dict):
        stream_queue = config.get("configurable", {}).get("stream_queue")

    try:
        response = await asyncio.wait_for(
            _generate_plan_response(prompt, model, stream_queue),
            timeout=remaining_seconds(),
        )
        parsed = _parse_json_response(response)
    except asyncio.TimeoutError:
        logger.warning("Plan node exceeded %.1fs; using broad retrieval fallback", PLAN_LLM_TIMEOUT_SECONDS)
        parsed = _parse_json_response("")
        parsed["rewritten_query"] = query
        parsed["reasoning"] = "规划模型超过时间预算，扩大到七类资料语义召回"
    except Exception as exc:
        logger.error("Plan node failed: %s", exc)
        parsed = _parse_json_response("")
        parsed["rewritten_query"] = query
        parsed["reasoning"] = f"规划模型不可用，采用 ICP 默认混合检索流程：{exc}"

    # Only retrieval hints survive the query writer. Semantic ownership never
    # falls through to a second model, even when target parsing is unavailable.
    parsed = _bind_retrieval_plan(parsed, explicit_target_decision, dialogue_context, query)

    final_frame = (
        dict(parsed.get("query_frame"))
        if isinstance(parsed.get("query_frame"), dict)
        else {}
    )
    validated_coverage_request = _parse_coverage_request(
        final_frame.get("coverage_request"),
        raw_user_query=query,
        offered_anchor_ids=[
            str(item.get("anchor_id") or "")
            for item in conversation_anchors
            if isinstance(item, Mapping) and str(item.get("anchor_id") or "")
        ],
    )
    if validated_coverage_request:
        final_frame["coverage_request"] = validated_coverage_request
        final_frame["coverage_scope"] = str(validated_coverage_request["scope"])
        final_frame["collection_subject"] = str(
            validated_coverage_request["subject"]
        )
        final_frame["collection_member_kind"] = str(
            validated_coverage_request["member_kind"]
        )
        if validated_coverage_request["scope"] == "complete_collection":
            final_frame["required_answer_fields"] = normalize_answer_fields(
                [
                    *(final_frame.get("required_answer_fields") or []),
                    "closed_collection_members",
                ]
            )
    else:
        final_frame.pop("coverage_request", None)
        final_frame["coverage_scope"] = "unknown"
        final_frame["collection_subject"] = ""
        final_frame["collection_member_kind"] = "unknown"
        final_frame["required_answer_fields"] = [
            field
            for field in normalize_answer_fields(
                final_frame.get("required_answer_fields") or []
            )
            if field != "closed_collection_members"
        ]
    parsed["query_frame"] = final_frame

    # Keep the planner's frame coherent with its structured targets before the
    # focused-tool policy runs.  In particular, an older planner may preserve a
    # reliability target while returning only FMECA impact fields; without this
    # repair the field contract would incorrectly remove ``rbd_query``.
    policy_frame = (
        dict(parsed.get("query_frame"))
        if isinstance(parsed.get("query_frame"), dict)
        else {}
    )
    # Retrieval-planner JSON is untrusted even when the target stage is absent.
    # Keep only field-level closure and response breadth contracts grounded in
    # this exact turn; invalid legacy flags do not cross the policy boundary.
    validated_diagnostic_closure = normalize_diagnostic_closure(
        policy_frame.get("diagnostic_closure"),
        raw_user_query=query,
    )
    validated_response_mode = normalize_response_mode(
        policy_frame.get("response_mode"),
        raw_user_query=query,
    )
    policy_frame.pop("diagnostic_closure", None)
    policy_frame.pop("response_mode", None)
    if validated_diagnostic_closure:
        policy_frame["diagnostic_closure"] = validated_diagnostic_closure
    if validated_response_mode:
        policy_frame["response_mode"] = validated_response_mode
    # Open discourse dimensions come from the structured target classifier.
    # They are validated enums, not a keyword match over the user's wording.
    response_focus = [
        str(value or "").strip()
        for value in (
            policy_frame.get("response_focus")
            or dialogue_context.get("response_focus")
            or []
        )
        if str(value or "").strip() in _RESPONSE_FOCUSES
    ]
    question_semantics = str(
        policy_frame.get("question_semantics")
        or dialogue_context.get("question_semantics")
        or "unknown"
    ).strip().casefold()
    if question_semantics not in _QUESTION_SEMANTICS:
        question_semantics = "unknown"
    response_scope, response_scope_basis = _parse_response_scope(
        policy_frame.get("response_scope")
        or dialogue_context.get("response_scope"),
        policy_frame.get("response_scope_basis")
        or dialogue_context.get("response_scope_basis"),
        raw_user_query=query,
    )
    # ``response_focus`` is an answer-organization contract. It must not be
    # converted into evidence fields here: the target model owns the explicit
    # ``required_answer_fields`` list, and diagnostic additions must arrive via
    # grounded ``diagnostic_closure.required_dimensions``.
    if response_focus:
        policy_frame["response_focus"] = response_focus
    policy_frame["question_semantics"] = question_semantics
    policy_frame["response_scope"] = response_scope
    if response_scope_basis:
        policy_frame["response_scope_basis"] = response_scope_basis
    else:
        policy_frame.pop("response_scope_basis", None)
    policy_intents = [
        str(item.get("intent") or "")
        for item in (parsed.get("query_targets") or [])
        if isinstance(item, dict) and str(item.get("intent") or "")
    ]
    primary_policy_intent = str(parsed.get("intent") or "").strip()
    if primary_policy_intent and primary_policy_intent not in policy_intents:
        policy_intents.insert(0, primary_policy_intent)
    policy_frame["required_answer_fields"] = ensure_intent_answer_fields(
        policy_frame.get("required_answer_fields") or [],
        intents=policy_intents,
    )
    parsed["query_frame"] = policy_frame

    selected_relation_anchor_ids = [
        str(anchor.get("anchor_id") or "").strip()
        for anchor in selected_conversation_anchors
        if isinstance(anchor, Mapping) and str(anchor.get("anchor_id") or "").strip()
    ]
    # Retrieval hints never supply missing graph authority.
    raw_relation_expansion = dialogue_context.get("relation_expansion")
    relation_expansion = normalize_relation_expansion_contract(
        raw_relation_expansion,
        raw_user_query=query,
        offered_anchor_ids=selected_relation_anchor_ids,
    )
    if not relation_expansion:
        relation_expansion = {
            "needed": False,
            "relation_kind": "",
            "reference_mode": "",
            "basis": "",
            "selected_anchor_ids": [],
            "reason": "",
        }

    # One semantic owner, followed by raw-turn provenance and domain checks.
    # The query writer neither grants nor vetoes a calculation request.
    calculation_authorizations = normalize_calculation_authorizations(
        (
            explicit_target_decision.get("calculation_authorizations")
            if isinstance(explicit_target_decision, dict)
            else []
        ),
        raw_user_query=query,
    )

    policy = resolve_tool_policy(
        intent=parsed.get("intent", "hybrid"),
        proposed_plan=parsed.get("tools_plan", []),
        query=f"{query} {parsed.get('rewritten_query') or ''}",
        type_keywords=parsed.get("type_keywords", []),
        query_targets=parsed.get("query_targets", []),
        # Only the user's original wording may authorize deterministic
        # FTA/RBD calculations.  Rewrites are retrieval hints, not authority.
        raw_user_query=query,
        planner_failed=bool(parsed.get("_planner_failed")),
        coverage_scope=(
            (parsed.get("query_frame") or {}).get("coverage_scope", "unknown")
            if isinstance(parsed.get("query_frame"), dict)
            else "unknown"
        ),
        collection_member_kind=(
            (parsed.get("query_frame") or {}).get("collection_member_kind", "unknown")
            if isinstance(parsed.get("query_frame"), dict)
            else "unknown"
        ),
        required_answer_fields=(
            (parsed.get("query_frame") or {}).get("required_answer_fields", [])
            if isinstance(parsed.get("query_frame"), dict)
            else []
        ),
        diagnostic_closure=(
            (parsed.get("query_frame") or {}).get("diagnostic_closure", {})
            if isinstance(parsed.get("query_frame"), dict)
            else {}
        ),
        relation_expansion=relation_expansion,
        selected_anchor_ids=selected_relation_anchor_ids,
        selected_anchor_doc_types=[
            str(anchor.get("doc_type") or "")
            for anchor in selected_conversation_anchors
            if isinstance(anchor, Mapping) and str(anchor.get("doc_type") or "")
        ],
        calculation_authorizations=calculation_authorizations,
        requested_sources=policy_frame.get("requested_sources") or [],
        user_context_state={
            **state,
            "query": query,
            "dialogue_context": dialogue_context,
            "query_frame": {
                "user_input_query": query,
                "user_context_bindings": dialogue_context.get("user_context_bindings") or [],
            },
        },
    )
    query_frame = build_query_frame(
        query=query,
        rewritten_query=parsed.get("rewritten_query") or query,
        entity_keywords=parsed.get("entity_keywords", []),
        query_targets=policy["query_targets"],
        planner_frame=parsed.get("query_frame"),
    )
    identity_frame = (
        parsed.get("query_frame")
        if isinstance(parsed.get("query_frame"), Mapping)
        else {}
    )
    typed_record_fields = _ground_typed_record_fields(
        identity_frame.get("typed_record_fields"),
        raw_user_query=query,
    )
    if typed_record_fields:
        query_frame["typed_record_fields"] = typed_record_fields

    subject_candidates = list(identity_frame.get("subject_bindings") or [])
    topology_for_identity = dialogue_context.get("topology_dependency")
    if (
        isinstance(topology_for_identity, Mapping)
        and topology_for_identity.get("required") is True
        and str(topology_for_identity.get("subject") or "").strip()
    ):
        # Topology classification may guide retrieval, but its object becomes
        # an identity selector only when the value is literally grounded in
        # this user turn.  The common validator below adds the provenance flag.
        subject_candidates.append(
            {
                "doc_type": "rbd",
                "kind": topology_for_identity.get("subject_kind"),
                "value": topology_for_identity.get("subject"),
                "basis": topology_for_identity.get("subject"),
            }
        )
    subject_bindings = _ground_subject_bindings(
        subject_candidates,
        raw_user_query=query,
    )
    if subject_bindings:
        query_frame["subject_bindings"] = subject_bindings

    # These objects have already passed enum and current-turn/anchor
    # provenance validation in the target stage.  The generic query-frame
    # builder intentionally knows nothing about their open-language meaning.
    for key in ("requested_sources", "architecture_roles", "comparison_subjects"):
        if key in dialogue_context:
            query_frame[key] = list(dialogue_context.get(key) or [])
    coverage_request = policy_frame.get("coverage_request")
    if isinstance(coverage_request, Mapping):
        query_frame["coverage_request"] = dict(coverage_request)
    payload_contract = dialogue_context.get("payload_contract")
    if isinstance(payload_contract, Mapping):
        query_frame["payload_contract"] = dict(payload_contract)
    if validated_diagnostic_closure:
        query_frame["diagnostic_closure"] = dict(validated_diagnostic_closure)
    else:
        query_frame.pop("diagnostic_closure", None)
    if validated_response_mode:
        query_frame["response_mode"] = dict(validated_response_mode)
    if "missing_input_requirements" in dialogue_context:
        query_frame["missing_input_requirements"] = _machine_missing_input_requirements(
            dialogue_context.get("missing_input_requirements"), state
        )

    current_hmc_components = extract_hmc_components(query)
    if current_hmc_components:
        # This is a deterministic parse of the raw turn, not a model-created
        # identity.  Downstream nodes verify the source tag and round trip.
        query_frame["hmc_components"] = current_hmc_components
        query_frame["hmc_components_source"] = "current_query"
    if dialogue_context.get("user_context_bindings"):
        query_frame["user_context_bindings"] = list(dialogue_context["user_context_bindings"])
        query_frame["user_input_query"] = query
    # A user can explicitly ask us to check one named document family while
    # asking a different semantic question (for example “why did it fail—go
    # check RBD”).  Keep this as a source-coverage obligation rather than
    # falsifying the semantic intent.  The answer contract then requires a
    # compact, evidence-backed contribution from that source when it exists.
    source_coverage_tools = list(policy.get("requested_source_tools") or [])
    source_coverage_fields = list(policy.get("source_coverage_fields") or [])
    # 续轮来源延续只读取目标阶段本轮明确选择的证据锚点。最近一答可能同时
    # 引用多个资料族；未选引用不得因“哪些能说/不能说”的开放语义而获得执行
    # 权限、成为检索词或扩大回答来源边界。
    continued_tools = _continuation_source_tools(
        query=query,
        messages=messages,
        selected_anchors=selected_conversation_anchors,
        context_used=bool(dialogue_context.get("context_used")),
        response_focus=response_focus,
        current_requested_sources=policy.get("requested_sources") or [],
    )
    continued_subjects: List[str] = []
    if continued_tools:
        boundary_focuses = {"source_capability", "source_limitation"}
        pure_source_boundary = bool(response_focus) and set(
            response_focus
        ).issubset(boundary_focuses)
        if pure_source_boundary:
            # This turn asks what the explicitly selected citations can or
            # cannot establish. A retrieval proposal based only on the broad
            # entity/profile intent must not pull in an unselected neighbour
            # source from the previous answer (or from planner preference).
            # Re-read the selected PostgreSQL owner(s) only; an independently
            # requested business dimension is not a pure boundary turn and is
            # therefore handled by the normal field/source contracts above.
            selected_source_set = set(continued_tools)
            removed_query_tools = {
                tool
                for tool in policy.get("authorized_tools") or []
                if tool in QUERY_TOOL_NAMES and tool not in selected_source_set
            }
            policy["allowed_tools"] = [
                tool
                for tool in policy.get("allowed_tools") or []
                if tool not in QUERY_TOOL_NAMES or tool in selected_source_set
            ]
            policy["authorized_tools"] = [
                tool
                for tool in policy.get("authorized_tools") or []
                if tool not in QUERY_TOOL_NAMES or tool in selected_source_set
            ]
            policy["tools_plan"] = [
                item
                for item in policy.get("tools_plan") or []
                if not isinstance(item, Mapping)
                or str(item.get("tool") or "") not in QUERY_TOOL_NAMES
                or str(item.get("tool") or "") in selected_source_set
            ]
            policy["authorized_doc_types"] = list(dict.fromkeys(
                DOC_TYPE_BY_TOOL[tool]
                for tool in policy["authorized_tools"]
                if tool in DOC_TYPE_BY_TOOL
            ))
            policy["hybrid_doc_types"] = []
            policy["rejected_tools"] = list(dict.fromkeys([
                *(policy.get("rejected_tools") or []),
                *sorted(removed_query_tools),
            ]))
        continued_subjects = _anchor_subjects(selected_conversation_anchors)
        for tool in continued_tools:
            if tool in source_coverage_tools:
                continue
            source_coverage_tools.append(tool)
            if tool not in policy["allowed_tools"]:
                policy["allowed_tools"].append(tool)
            if tool not in policy["authorized_tools"]:
                policy["authorized_tools"].append(tool)
            doc_type = DOC_TYPE_BY_TOOL.get(tool)
            if doc_type and doc_type not in policy["authorized_doc_types"]:
                policy["authorized_doc_types"].append(doc_type)
            if doc_type and "hybrid_search" in policy["allowed_tools"]:
                policy["hybrid_doc_types"] = list(dict.fromkeys([
                    *(policy.get("hybrid_doc_types") or []), doc_type,
                ]))
            if not any(
                item.get("tool") == tool for item in (policy.get("tools_plan") or [])
            ):
                policy["tools_plan"].append({
                    "tool": tool,
                    "reason": "延续上一轮已引用资料来源，保持结论边界与同一资料族",
                })
        policy["notes"] = [
            *(policy.get("notes") or []),
            "延续上一轮已引用来源：" + "、".join(
                DOC_TYPE_BY_TOOL.get(tool, tool) for tool in continued_tools
            ),
        ]
    if source_coverage_fields:
        query_frame["required_answer_fields"] = normalize_answer_fields([
            *(query_frame.get("required_answer_fields") or []),
            *source_coverage_fields,
        ])
    query_frame["requested_source_tools"] = source_coverage_tools
    if source_coverage_tools:
        # Keep source coverage as facet-scoped obligations.  It is tempting to
        # turn a source-boundary question into a whole-turn document lock, but
        # that drops a second, explicitly requested goal in questions such as
        # “按 FMECA 说明影响，同时给出历史复测结果”.  Downstream nodes may
        # narrow a *pure* source-meta turn, while mixed turns retain every
        # independently requested field and its own evidence family.
        obligations = [
            dict(item)
            for item in (policy.get("source_obligations") or [])
            if isinstance(item, Mapping)
        ]
        obligated_tools = {
            str(item.get("tool") or "").strip() for item in obligations
        }
        for tool in continued_tools:
            if tool in obligated_tools or not DOC_TYPE_BY_TOOL.get(tool):
                continue
            obligations.append(
                {
                    "tool": tool,
                    "doc_type": DOC_TYPE_BY_TOOL[tool],
                    "fields": [],
                    "question_role": "capability",
                    "basis": "selected_anchor",
                }
            )
        query_frame["source_obligations"] = obligations
        if set(response_focus) & {"source_capability", "source_limitation"}:
            explicit_target_fields = set(
                normalize_answer_fields(
                    dialogue_context.get("required_answer_fields") or []
                )
            )
            source_fields = set(source_coverage_fields)
            permitted_fields = explicit_target_fields | source_fields
            query_frame["required_answer_fields"] = normalize_answer_fields(
                field
                for field in (query_frame.get("required_answer_fields") or [])
                if field in permitted_fields
            )
    query_frame["target_contract_status"] = "valid" if explicit_target_decision else "unavailable"
    # Prior natural goals stay turn-bound conversational progress. They do
    # not rewrite this turn's goals, remove retrieval fields or grant tools.
    prior_delivery_contracts = _selected_anchor_delivery_contracts(
        messages, dialogue_context, selected_conversation_anchors
    )
    prior_delivered_fields = _contextual_prior_delivered_fields(
        messages, dialogue_context, selected_conversation_anchors
    )
    if prior_delivery_contracts:
        query_frame["prior_delivery_contracts"] = prior_delivery_contracts
    if prior_delivered_fields:
        query_frame["prior_delivered_fields"] = prior_delivered_fields
    if question_semantics != "unknown":
        query_frame["question_semantics"] = question_semantics
    if response_focus:
        query_frame["response_focus"] = response_focus
    query_frame["response_scope"] = response_scope
    if response_scope_basis:
        query_frame["response_scope_basis"] = response_scope_basis
    else:
        query_frame.pop("response_scope_basis", None)
    # 边界/元问题（“RBD 能证明什么”“哪些结论不能说”）把回答依据限定在
    # 上一轮已批准来源族。answer 端据此把答案窗口锁到这些 doc_type，
    # 防止 function_locator 等强语义邻居跨域顶替上一轮对象自身的记录。
    # 仅追问轮（上轮确有引用记录）且命中边界表述时锁定；用户本轮显式点名
    # 新来源时以用户为准（requested_source_tools 会使其失去锁定效果）。
    locked_doc_types = list(dict.fromkeys(
        DOC_TYPE_BY_TOOL.get(tool, "")
        for tool in continued_tools
        if DOC_TYPE_BY_TOOL.get(tool, "")
    ))
    if (
        locked_doc_types
        and bool(set(response_focus) & {"source_capability", "source_limitation"})
    ):
        query_frame["evidence_domain_locked"] = True
        query_frame["locked_doc_types"] = locked_doc_types
    topology_dependency = dialogue_context.get("topology_dependency")
    if isinstance(topology_dependency, dict) and topology_dependency.get("required"):
        # ``build_query_frame`` intentionally exposes a closed set of generic
        # fields.  Re-attach this already-validated domain sub-contract here
        # rather than teaching the generic builder to interpret raw wording.
        query_frame["topology_dependency"] = dict(topology_dependency)
    tool_queries = normalize_tool_queries(parsed.get("tool_queries"))
    query_tools = {
        "function_locator", "hmc_query", "fmeca_query", "fta_query",
        "rbd_query", "mfl_query", "fault_history_query", "hybrid_search",
    }
    tool_queries = {
        tool: values
        for tool, values in tool_queries.items()
        if tool in query_tools and tool in policy["allowed_tools"]
    }
    fallback_query = str(parsed.get("rewritten_query") or query).strip()
    for tool in policy["allowed_tools"]:
        if tool not in query_tools or not fallback_query:
            continue
        if dialogue_context["context_used"]:
            tool_queries[tool] = _merge_text_lists(
                [fallback_query], tool_queries.get(tool), limit=6, item_limit=1200
            )
        elif tool not in tool_queries:
            # This is a literal fallback query, not a rule-derived synonym.
            tool_queries[tool] = [fallback_query]
    if continued_subjects:
        # 延续来源的检索词以上一轮已公开的记录身份为主：边界元问句本身
        # 没有可用的主题词，直接用上一轮锚点身份回查同一资料族。
        for tool in continued_tools:
            if tool not in policy["allowed_tools"]:
                continue
            tool_queries[tool] = _merge_text_lists(
                continued_subjects, tool_queries.get(tool),
                limit=6, item_limit=1200,
            )
    reasoning = str(parsed.get("reasoning") or "").strip()
    if policy.get("notes"):
        reasoning = f"{reasoning}；策略校验：{'；'.join(policy['notes'])}".strip("；")

    # The policy boundary re-validates the object and returns the exact
    # executable contract. Do not reconstruct it from selected tools, coarse
    # intent, or answer fields after that boundary.
    relation_expansion = dict(policy.get("relation_expansion") or {
        "needed": False,
        "relation_kind": "",
        "reference_mode": "",
        "basis": "",
        "selected_anchor_ids": [],
        "reason": "",
    })
    task_profile = {
        "primary_goal": policy["intent"],
        "goals": [
            str(item.get("intent") or "")
            for item in policy["query_targets"]
            if isinstance(item, dict) and str(item.get("intent") or "")
        ],
        "entities": list(query_frame.get("entities") or []),
        "requested_facets": list(query_frame.get("requested_facets") or []),
        "required_answer_fields": list(
            query_frame.get("required_answer_fields") or []
        ),
        "question_semantics": str(
            query_frame.get("question_semantics") or "unknown"
        ),
        "response_focus": list(query_frame.get("response_focus") or []),
        "response_scope": str(
            query_frame.get("response_scope") or "unknown"
        ),
        "response_mode": str(
            (query_frame.get("response_mode") or {}).get("mode") or "unknown"
        ) if isinstance(query_frame.get("response_mode"), Mapping) else "unknown",
        "coverage_scope": str(query_frame.get("coverage_scope") or "unknown"),
        "collection_subject": str(query_frame.get("collection_subject") or ""),
        "collection_member_kind": str(
            query_frame.get("collection_member_kind") or "unknown"
        ),
        "relation_required": relation_expansion.get("needed") is True,
        "diagnostic_closure_required": bool(
            (query_frame.get("diagnostic_closure") or {}).get("required")
            if isinstance(query_frame.get("diagnostic_closure"), dict)
            else False
        ),
        "dialogue_context_used": bool(dialogue_context["context_used"]),
        "context_status": str(dialogue_context["status"]),
        "selected_anchor_count": len(selected_conversation_anchors),
        "requested_source_tools": source_coverage_tools,
    }

    context_status = str(dialogue_context.get("status") or "standalone")
    missing_requirements = normalize_missing_input_requirements(
        query_frame.get("missing_input_requirements")
    )
    unresolved_identity_slots = [
        item
        for item in missing_requirements
        if item.get("kind") == "identity_slot"
    ]
    clarification_reasons: List[Dict[str, Any]] = []
    if context_status in {"ambiguous", "missing"}:
        clarification_reasons.append(
            make_clarification_reason(
                kind=(
                    "ambiguous_identity"
                    if context_status == "ambiguous"
                    else "unresolved_identity"
                ),
                source_kind="query_frame",
                source_name="dialogue_context",
                blocked_claims=["resolved_conversation_subject"],
            )
        )
    clarification_required = bool(missing_requirements or clarification_reasons)
    clarification_fallback = make_clarification_fallback(
        required=bool(clarification_reasons and not missing_requirements)
    )
    if clarification_required:
        # A structured gap must not become a broad FMECA/OpenSearch scan.  Keep
        # the normal answer path, though: the answer model owns the natural
        # question and its only model-failure fallback is domain-neutral.
        policy["authorized_tools"] = []
        policy["allowed_tools"] = []
        policy["authorized_doc_types"] = []
        policy["hybrid_doc_types"] = []
        policy["tools_plan"] = []
        policy["calculation_authorizations"] = []
        policy["calculation_requests"] = []
        tool_queries = {}
        relation_expansion = {
            "needed": False,
            "relation_kind": "",
            "reference_mode": "",
            "basis": "",
            "selected_anchor_ids": [],
            "reason": "",
        }
        task_profile["relation_required"] = False

    # Keep the legacy state key for downstream compatibility, but derive it
    # from the validated model contract.  Intent is deliberately absent: it is
    # an execution ceiling, not authority to widen the answer.
    answer_mode = _answer_mode_from_contract(query_frame)

    return {
        "scope_route": "knowledge_base",
        "clarification_kind": "",
        "intent": policy["intent"],
        "query_targets": policy["query_targets"],
        "calculation_authorizations": policy["calculation_authorizations"],
        "calculation_requests": policy["calculation_requests"],
        "relation_expansion": relation_expansion,
        "answer_mode": answer_mode,
        "rewritten_query": parsed.get("rewritten_query") or query,
        "entity_keywords": parsed.get("entity_keywords", []),
        "type_keywords": parsed.get("type_keywords", []),
        "query_frame": query_frame,
        "requested_source_tools": source_coverage_tools,
        "dialogue_context": dialogue_context,
        "conversation_anchors": conversation_anchors,
        "selected_conversation_anchors": selected_conversation_anchors,
        "task_profile": task_profile,
        "normalized_terms": query_frame.get("normalized_terms", []),
        "tool_queries": tool_queries,
        # evidence_synthesis 是固定内部阶段，不作为 LLM 规划的外部工具展示。
        "tools_plan": [
            item for item in policy["tools_plan"] if item.get("tool") != "evidence_synthesis"
        ],
        "authorized_tools": policy["authorized_tools"],
        "allowed_tools": policy["allowed_tools"],
        "authorized_doc_types": policy["authorized_doc_types"],
        "hybrid_doc_types": policy["hybrid_doc_types"],
        "plan_reasoning": reasoning,
        "missing_input_requirements": missing_requirements,
        "unresolved_identity_slots": unresolved_identity_slots,
        "clarification_reasons": unique_contract_items(clarification_reasons),
        "clarification_fallback": clarification_fallback,
        "clarification_question": "",
        "memory_context": memory_context,
        "iteration": state.get("iteration", 0) + 1,
    }


def route_after_plan(state: Dict[str, Any]) -> str:
    """Route legacy clarification states; new plan gaps use the answer path."""

    return (
        "direct_response"
        if str(state.get("scope_route") or "") == "clarification"
        else "icp_diagnosis"
    )
