# ICP 七类文档解析与证据智能体详细设计

## 1. 目的与边界

本文档是 ICP-RAG 七类工程文档处理和初步诊断智能体的实施基线。设计目标不是适配一组固定测试文件，而是在字段名称、表头层数、章节样式、分页方式和图片布局发生变化时，仍能做到：

- 原件、原始值和来源位置不丢失；
- 可确定解析的事实进入 PostgreSQL；
- 可检索副本进入 OpenSearch，并能回查 PostgreSQL 或原件；
- 七类文档都从 PostgreSQL 权威事实自动生成可重建 HugeGraph 投影；
- 自动图像抽取只形成 `provisional` 候选，不能冒充正式工程拓扑；
- 架构、FTA、RBD 采用版本化自动校验；冲突或残缺结构自动隔离，不等待人工门禁；
- 智能体只通过受限工具读取数据，不能在问答过程中修改知识；
- 证据不足、版本冲突或参数缺失时返回候选和追问，不生成唯一诊断或猜测数值。

系统仍严格只有七类用户文档，不增加第八类产品入口，不恢复图片画廊，不改变 Dify 兼容接口。

## 2. 数据所有权

```text
ArtifactStore + LocalFS  原件、抽取图片、可复核证据
          │
          ▼
PostgreSQL               文档状态、结构化事实、图候选/修订、来源链（权威）
          │
          ├────────────► OpenSearch  BM25、BGE-M3 向量、过滤字段（可重建）
          │
          └─ explicit/validated ─► HugeGraph  显式关系与分级工程拓扑（可重建）
```

当存储内容冲突时，以 PostgreSQL 记录及其 `source_ref` 指向的原件证据为准。OpenSearch 和 HugeGraph 不能反向覆盖 PostgreSQL。

## 3. 统一标识、来源和状态

### 3.1 文档和记录

所有正式记录至少包含：

| 字段 | 含义 |
|---|---|
| `doc_id` | 本次上传文档 ID |
| `doc_type` | 七类类型之一 |
| `record_id` | PostgreSQL 行 ID |
| `business_id` | HMC、FMECA、MFL、Function、Case、Diagram 等业务 ID |
| `source_ref` | `文件#章节/表/行/图片` 的稳定定位 |
| `raw_data` | 原始字段和值，不因规范化而覆盖 |
| `normalization_warnings` | 继承单元格、模糊表头、非法枚举、缺失字段等告警 |
| `document_metadata` | 系统、构型、文档编号、版本和数据状态 |

业务 ID 不应跨文档版本覆盖记录。数据库唯一键通常以 `doc_id + business_id` 为主；
HMC 允许同一码在不同物理来源行保留不同说法，因此使用 `doc_id + variant_id`
（由规范码和稳定 `source_ref` 生成），代码查询可返回多个证据变体。跨版本查询通过业务 ID 和适用范围显式完成。

### 3.2 图修订

一个工程图逻辑对象由 `doc_id + diagram_key` 标识。表格、正文表达式和图片分析是同一个图修订的多个证据源，不生成多套正式图。

人工审核（兼容字段）、自动校验、投影和生命周期是正交状态，不能压成一个字段：

```text
review_status:     pending | conflicted | confirmed | rejected
validation_status: pending | auto_validated | provisional | quarantined | failed | manual_confirmed
projection_status: not_projected | publishing | published | failed | stale
lifecycle_status:  active | archived
```

解析器产生候选和不可变修订，随后由确定性校验器推进 `validation_status`：表格边或受控表达式通过类型约束后为 `auto_validated`；仅来自视觉且结构完整、置信度达标时为 `provisional`；拓扑冲突、悬空边、环路、非法参数、低置信或抽取歧义为 `quarantined`。`confirmed/manual_confirmed` 只为兼容显式人工操作，不是运行所必需。`published` 必须在 HugeGraph 写入后按 ID 和数量读回成功才可设置；来源已从一次成功重解析中消失时，图对象进入 `archived`，旧发布修订进入 `stale`。LLM 不得直接推进任何状态。

图修订至少保存：

```text
diagram_id / diagram_key / revision_no
diagram_type
canonical_graph
review_status / validation_status / validation_tier / projection_status
graph_hash
source_refs
scope(system/configuration/version)
validator_version / validation_report / quarantine_reason / published_at
```

图证据独立保存：

```text
evidence_id / diagram_id
source_kind(table/image/text_expression)
source_ref / artifact_key / image_url / mime_type
extracted_graph / extraction_warnings / extraction_confidence
```

## 4. 解析流水线

```text
1. ArtifactStore 原子保存原件并计算 SHA-256
2. PostgreSQL 登记 icp_documents
3. 提取封面元数据和适用范围
4. 生成保序文档结构：章节、段落、表格、图片、图题、来源位置
5. ParserRegistry 按 doc_type 调用专用解析器
6. 字段识别、原始值保留、规范化和行级校验
7. PostgreSQL 幂等写入权威事实
8. 生成确定性 OpenSearch 投影 ID，批量 embedding/upsert
9. 架构/FTA/RBD 形成图候选及多来源对照
10. 七类生成 Document 来源节点和显式结构关系；架构/FTA/RBD 再执行自动校验、分级投影或隔离
```

解析重试必须幂等。同一 `doc_id` 重复解析时，以确定性记录键 upsert；相同来源哈希、投影器版本和读回 ID 集合可直接复用，来源变化则生成或更新待自动校验修订。

OpenSearch 写入失败不能把 PostgreSQL 事实回滚为不存在；应记录
`projection_rebuild_required/projection_failed`，诊断回填仍以 PostgreSQL 为准。
PostgreSQL、图候选、图片证据或生命周期写入失败则整个解析任务进入 `failed`；该
`doc_id` 的旧行和任何中间行都不能由诊断工具服务，失败后的 OpenSearch 中间投影必须清理或保持不可用。

## 5. 通用表格解析策略

真实 Word/Excel 表格不依赖固定表号或固定首行：

1. 扫描前若干行和 1～6 层候选表头；
2. 对合并表头执行受控的横向继承，优先识别叶子表头；
3. 表头名称经过空白、标点、全半角和大小写规范化后匹配别名；
4. 使用必填字段组和字段覆盖率评分选取表头，而不是单个关键字；
5. 跳过跨页重复表头、页脚、说明行和全空行；
6. 对真实的纵向合并单元格仅在配置允许的字段上继承；
7. 每行保留 `table/sheet/row` 来源和原始单元格；
8. 字段校验失败进入 `failed_rows` 且不静默丢弃；数据库写入或快照收敛失败必须使整份文档失败；
9. 未知列保留在 `raw_data`，便于以后扩展别名而无需重新获得原件；
10. 字段别名表与业务规范化器分离，便于适配供应商格式。

禁止依靠 LLM 判断普通结构化表格字段。LLM 只可作为无法用确定性规则解析的图片候选抽取手段。
旧 `.xls` 读取路径无法可靠恢复单元格显示格式、公式缓存和合并元数据，必须输出限制告警；
数值型代码字段因无法证明前导零而不得晋升为权威事实，优先要求转换为保留格式的 `.xlsx`。

DOCX 正文、表格单元格、文本框/VML 中的图片均保存原始 MIME 与来源定位。TIFF、SVG、
EMF、WMF 或未知二进制若当前视觉模型链不支持，只保留 Artifact/PG 证据并明确跳过视觉，
不得改扩展名或伪报为 PNG。

## 6. 七类文档入库设计

### 6.1 系统架构与功能说明

确定性来源包括功能表、子系统表、模块表、接口表、数据流表和明确的结构表达式。保存：

```text
icp_components: System / Subsystem / Module
icp_functions: Function、父功能、实现模块、说明、别名、失效表现
icp_interfaces: source、target、输入/输出、协议/信号、方向
diagram revision: 图片关系候选（自动标为 provisional 或 quarantined）
```

表格中的明确组件 ID、父组件 ID、功能 ID、实现模块 ID、接口 ID 和端点 ID 自动投影；图片识别出的 Function 不直接成为权威 `icp_functions` 行，也不能覆盖确定性结构。普通段落写检索投影，不推断图边。

### 6.2 HMC

支持完整 HMC 列或 `LCN + HAMC + HSC` 组合。三码全部按字符串保存，保留前导零和 `**`。原文没有完整码时才按受控规则生成规范码。

```text
诊断投影 = HMC + 三码 + 故障现象
维修投影 = HMC + 故障现象 + 分析/维修方法
```

同一故障的不同说法仍保存为不同权威行；实体归并只在有明确业务 ID 或后续人工审核时进行，不由向量相似度自动合并。
完整 HMC 与同一行的 `LCN/HAMC/HSC` 冲突时保留两组原值并阻止该行晋升，不能静默覆盖。
HugeGraph 中精确 HMC 代码形成 `HMCCode` 节点，并按源行显式连接故障现象与维修方法文本节点；自由文本只在同一 `doc_id + 字段类型 + 原文` 范围内去重。

### 6.3 FMECA

一条故障模式一行。严重度、概率等级和数值概率分开保存：

```text
severity_raw / severity_normalized
probability_raw / probability_level / probability_value
```

字母等级、定性描述和数值不得混写到单一 NUMERIC 字段。原始宽表值、空值和补偿措施必须保留。
数值概率以 Decimal 安全字符串写入 PostgreSQL `NUMERIC(30,18)`；科学计数原文始终保存在
`probability_raw`，OpenSearch 浮点投影不作为精度真源。
HugeGraph 中每个明确 FMECA 记录代码形成 `FMECARecord`，与故障模式、原因、局部/最终影响、检测方法和措施建立字段来源明确的边；严酷度和概率保留为记录属性。

### 6.4 MFL

保存模块/SUR/子电路故障、API 名称、调用方法、使用方法和维修处置。布尔字段同时保留原值：

```text
api_reportable_raw + api_reportable
platform_test_raw + platform_test_status
```

`已完成/部分完成/待验证/未开展` 是状态枚举，不能压缩成布尔值。
HugeGraph 中 `MFLRecord` 按源列连接模块/SUR/子电路故障文本；明确 API 名称形成可跨文档按精确 ID 合并的 `API` 节点，并连接调用方法和使用方法。

### 6.5 FTA

表格确定性提取：故障树编号、顶事件、关联功能、假设、基本事件 ID、故障模式 ID、模块、原因和概率。最小割集表作为文档声明保存，但正式最小割集仍由 `auto_validated/manual_confirmed` 拓扑重新计算并比对。

图片识别负责候选事件、逻辑门和边。表格与图片通过事件 ID、规范化名称和所在章节对齐：

- 完全一致：标记 `matched`；
- 表格可补充图片参数：合并属性但保留两份证据；
- 节点、门或边冲突：标记 `conflicted`；
- 只有表格基本事件而无拓扑：允许检索，不允许图计算；
- 只有图片且结构、置信度达标：`provisional`，仅浏览；否则 `quarantined`。

### 6.6 RBD

保存模型名称、成功判据、任务时间、模块参数、原始计算结果表。正文中的明确 `SERIES/PARALLEL/...` 表达式使用确定性语法解析为候选拓扑，再与图片候选对照。

结构表达式、图片和模块参数属于同一图修订。没有完整拓扑或参数时，只返回缺失项，不计算可靠度。
数值解析只接受无歧义标量及受控单位；科学计数法、`×10^n`、可靠度和修复率可规范化，
范围、不等式和未知单位只保留原值并告警。缺少或损坏拓扑时，模块参数和声明结果仍保存为
不可确认、不可计算的证据候选，不据此生成边。

### 6.7 故障历史

同时支持：

- 标题中的案例 ID；
- `字段：值` 段落；
- 两列表单式案例表；
- 一行一个案例的宽表；
- `第1步`、`步骤1`、`1.`、`(1)` 等步骤格式。

保存任务阶段/工作模式、验证结果和原始关联索引。案例、现象、排故方法、根因、方案和预防措施按字段关系入图。每个有序步骤生成文档内独立的 `FaultHistoryStep` 实例：案例通过 `has_diagnosis_step` 指向步骤实例，步骤实例通过 `has_step_content` 指向按原文去重的文本节点，相邻步骤实例通过 `next_step` 表达先后顺序。关联 HMC/FMECA/FTA/RBD/MFL 只有在源字段明确给出代码/ID 时建立 `references_*` 边，不把它升级为因果关系。

## 7. HugeGraph 投影边界

所有七类文档都自动投影，但只投影来源能够直接证明的关系：

| 文档类型 | 节点 | 允许的边 |
|---|---|---|
| 全部 | `Document` | 各源实体 `extracted_from` 文档 |
| Function Locator | `Component`、`Function`、`Interface` | 明确父子组件、模块实现功能、接口源/目标、显式输入输出 |
| HMC | `HMCCode`、字段文本 | `describes_fault`、`has_repair_method` |
| FMECA | `FMECARecord`、字段文本 | 故障模式、原因、影响、检测和措施对应的列关系 |
| MFL | `MFLRecord`、`API`、字段文本 | 模块/SUR/子电路故障、API 上报、调用/使用方法 |
| 故障历史 | `FaultHistoryCase`、步骤/字段文本、显式代码引用 | 案例字段、步骤顺序、`references_hmc/fmeca/fta/rbd/mfl` |
| FTA | 顶/中间/基本事件、逻辑门 | 图中明确的原因链和逻辑门连接 |
| RBD | START/END、可靠性块 | 明确串联、并联、冗余、旁路和成功路径 |

业务节点只有完全一致的代码/ID 才可跨文档共享。没有业务 ID 的文本节点 ID 必须包含 `doc_id + value_kind + exact_text`，不得跨文档或跨字段做语义归并。表格与图片指向同一 FTA/RBD 时，确定性表格/表达式为规范来源，图片候选单独隔离，避免重复投影同一拓扑。

结构化边的规范身份是 `(doc_id, relation, source, target)`。同一文档中相同关系、起点和终点即使在多行或多字段重复出现，也只生成一条 HugeGraph 边；每次出现的行号、字段名等来源限定合并保存在 `occurrence_qualifiers`，`source_refs` 同样累积保留。这样既符合 HugeGraph `SINGLE` 边语义，又不会因去重丢失证据来源。

禁止投影：

- 段落语义相似关系；
- 向量近邻；
- LLM 推测的因果关系；
- `quarantined/failed` 视觉结果；
- 仅因名称相似产生的实体合并。

HugeGraph 顶点和边 ID 必须由业务 ID 或 `doc_id + 类型 + stable_key` 确定生成。结构化投影只清理自己的 `icp_structured` 命名空间，不能误删同文档的 diagram revision；共享业务节点重建时只裁剪当前文档贡献。每次发布执行写后按 ID/数量读回校验。

## 8. 智能体工具和证据契约

### 8.1 工具组

```text
定位/检索：function_locator, hybrid_search, hmc_query, fmeca_query,
          mfl_query, fault_history_query, fta_query, rbd_query
跨文档关系：graph_traversal
计算：     fta_analyzer, rbd_calculator
证据融合： fault_reasoning
```

以上名称来自当前 `backend/agent/tools/registry.py`。`icp_diagnosis` 是首轮领域工具编排器，
会按规划结果调用七类专用查询；`adaptive_tools`、`adaptive_retrieval`、`evidence_gate`
和 `answer` 是 LangGraph 工作流节点，不作为面向 Planner 的领域查询工具注册。
领域能力应是独立、可测试的函数，不能再把多个工具名都映射成一次全库扫描。

### 8.2 ToolResult

```text
schema_version / tool / call_id
status: ok | partial | empty | failed | blocked
params / duration_ms / data
evidence[]
diagnostics: backend, fallbacks, warnings, errors
safety: scope_status, allowed_claims, blocked_claims, clarification_questions
```

`empty` 与 `failed` 必须区分。

### 8.3 Evidence

```text
evidence_id
source_kind: postgres | opensearch | hugegraph | artifact | calculation
authority: authoritative_record | search_projection | confirmed_relation | validated_relation |
           source_artifact | unconfirmed_extraction | deterministic_derivation
level: L1_DIRECT | L2_CORROBORATED | L3_INFERRED | L4_UNRESOLVED
doc_id / doc_type / record_id / business_id / source_ref
scope / review_status / derived_from / limitations / conflicts_with
retrieval_score / rerank_score
```

检索分数只能表示相关性。候选字段使用 `diagnostic_support`，不使用容易被理解为工程概率的 `diagnostic_confidence`。

## 9. LangGraph 编排

```text
scope_router
  → plan
  → icp_diagnosis（按授权工具执行首轮定向查询）
  → adaptive_tools（判断是否需要一批受限补查或多跳关系查询）
  → adaptive_retrieval（按需）
  → graph_traversal（按需，可由 adaptive_tools 直接进入，也可在补查后进入）
  → icp_calculation
  → evidence_gate
  → answer
```

`scope_router` 对能力说明、知识库范围外问题和必要澄清直接转入
`direct_response`，不启动知识检索。知识库问题进入 `plan`；首轮查询后，
`adaptive_tools` 最多追加一批受限查询。无需补查或查图时直接进入
`icp_calculation`，该节点仅执行用户明确授权的 FTA/RBD 确定性计算，
没有计算请求时透传现有证据。

Planner 只生成建议，确定性策略负责最终白名单：

- 完整 HMC 强制先查询 PostgreSQL；
- 架构问答不进入故障候选评分；
- 维修/API 问题默认不查询 FTA/RBD；
- 数值计算必须由用户意图明确触发；
- 图查询可展示 `provisional` 但必须标为待核验上下文；计算只接受 `auto_validated/manual_confirmed`、当前 active、published 且适用范围匹配的图修订；
- OpenSearch 命中必须回填 PostgreSQL 后才能升级为权威事实；
- 未知意图先做受限跨类召回，再按命中文档类型查询，不直接遍历全图。

## 10. 证据门禁

回答前由确定性代码执行：

1. 系统、构型、版本冲突的证据分组，不合并结论；
2. 没有 `doc_id + record_id/source_ref` 的内容不能作为 L1 正式引用；
3. HMC 精确命中证明记录存在，不自动证明唯一根因；
4. `provisional` 只能作 L4 上下文，不能驱动正式拓扑或计算；`quarantined/failed` 完全移出推理上下文；
5. 历史案例只能作为经验佐证；
6. 维修动作必须有适用版本 HMC/MFL 等直接资料；
7. 参数缺失时计算器只返回缺失列表；
8. 最终引用必须来自证据注册表，模型生成的未知引用被拒绝；
9. 唯一诊断要求作用域兼容、直接证据、无未解决冲突，并且替代候选已排除。

## 11. 降级策略

| 故障 | 允许继续 | 必须限制 |
|---|---|---|
| Embedding 不可用 | BM25 + PG | 标记缺少语义召回 |
| Rerank 不可用 | RRF 顺序 | 标记 `partial` |
| OpenSearch 不可用 | PG 精确/结构化查询 | 不宣称完成宽泛召回 |
| PostgreSQL 不可用 | 展示 OS 线索 | 阻断权威诊断和计算 |
| HugeGraph 不可用 | PG 中 auto_validated/manual_confirmed 图可用于确定性计算 | 不宣称完成图数据库多跳遍历 |
| 图为 provisional | 展示候选和原图 | 禁止正式图结论/计算 |
| 图被 quarantined | 展示隔离原因和原图 | 禁止进入推理与计算 |
| 原件不可读 | 展示 PG 记录并标记不可复核 | 阻断高风险维修建议 |

## 12. 实施与验收顺序

1. 通用结构/表格解析与纯函数测试；
2. 七类字段规范化、原值/来源保存和 PG 幂等 schema；
3. 七类显式关系构建、架构/FTA/RBD 候选与版本化自动校验；
4. OpenSearch 投影、回填和重建审计；
5. 独立领域查询、工具白名单和统一证据契约；
6. 只读工程拓扑工具和分级图证据计算门禁；
7. Evidence Gate、答案引用校验和降级策略；
8. 合成样例回归、格式扰动测试、服务联调和前端生产构建。

格式扰动测试至少覆盖：表头上移/下移、别名、两至六层表头、合并单元格、重复表头、列顺序变化、附加未知列、空行/说明行、Word 标题样式缺失、图片在标题前后、图题缺失、表格与图片冲突、部分字段缺失、同业务 ID 不同文档版本。

## 13. 当前实现基线与已知边界

本轮实现已经形成可继续迭代的基线：七类专用解析器共享文档中间表示、表头识别和来源定位；结构化事实写入 PostgreSQL，检索副本写入 OpenSearch，原件与图片证据写入 ArtifactStore；七类均自动生成 Document/显式关系投影，架构/FTA/RBD 使用候选、修订、自动校验、分级发布、异常隔离和 HugeGraph 读回校验；智能体通过工具白名单、统一证据契约和 Evidence Gate 约束回答。合成样例用于验证字段和结构契约，不能被描述为真实工程资料联调。

仍需明确保留以下边界：

1. 真实文档出现新的表头别名、跨页布局、嵌套对象或损坏文件时，应保留原件、失败行和来源位置，再以脱敏样例扩充解析规则，不能用 LLM 猜出缺失字段；
2. 七类权威事实、图片/图候选和生命周期收敛目前都不是一个跨表、跨存储原子快照；表格类仍有“清理旧行后逐行写入”的窗口。重解析进入 `parsing` 或 `failed` 后，旧权威快照即使仍有残留行也不可对外服务；当前选择可用性降级而不是错误地回退旧数据。若要求失败后继续服务完整上一版，必须增加七类 staging 表/候选暂存区并执行原子 swap；
3. Function Locator 的 PostgreSQL 明确组件/功能/接口关系会自动入图，架构图片只能作为 provisional/quarantined 候选，不能反向覆盖权威表；
4. FTA/RBD 图片自动抽取始终不是正式事实。没有可用视觉模型时，表格和明确表达式仍可确定性解析，图片只保存为证据，不能因此伪造图节点或关系；
5. OpenSearch 或模型调用失败会留下可重建/重试标记，PostgreSQL 权威记录不因检索投影失败被降级为搜索结果；
6. 不同机型、构型、软硬件版本的真实隔离仍需持续用工程样例验证；作用域不完整或冲突时，智能体必须返回候选与补充问题，不能输出唯一诊断。
