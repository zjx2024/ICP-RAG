# ICP-RAG 工程质量修缮路线图

> 适用分支：`refactor/icp-diagnosis-specialization`
> 编写日期：2026-09-03
> 状态：待执行（步骤 0 与 P0-A 可立即开工）

本文档记录一次完整代码审查的结论、对外部建议方案的核实结果，以及合并后的分阶段修缮计划。文档中所有 `文件:行号` 引用均经实际命令核实，核实方法见 [附录 A](#附录-a引用核实方法)。

---

## 1. 审查基线

### 1.1 体量

| 指标 | 数值 |
|---|---:|
| 后端 Python | 119 文件 / **74,090 行** |
| 前端 TS/TSX | 29 文件 / **9,073 行** |
| 测试 | 74 文件 / 34,157 行 / **949 用例** |
| 评测集 | 5 个 JSON / **164 例**（本次新增 100 例多轮） |
| 未提交改动 | 291 项（其中 135 项 untracked） |

`backend/agent/nodes/answer.py` 在本次审查期间由 14,538 行增长至 **14,644 行**，说明该文件仍在被持续追加。

### 1.2 实际执行过的验证

| 检查项 | 命令 | 结果 |
|---|---|---|
| 后端编译 | `python -m compileall -q backend` | 通过 |
| 后端测试 | `.venv/bin/python -m unittest discover -s tests` | **949 全绿（约 20s）** |
| 评测集加载 | 评测器 `_load_cases` 校验 3 份新文件 | 100 例 / 200 轮全部合法 |
| 评测集溯源 | 245 个 `expected_records` 与 `test-docs/` 比对 | 零缺失 |
| Dify 路由 | 静态检查 `api/__init__.py` + `routers/dify.py` | 两接口完整保留 |

### 1.3 未能验证的项（诚实声明）

- **前端构建**：本机无 `npm`（`npm: 没有那个文件或目录`），`npm ci && npm run build` 未实际执行，相关结论均来自静态审查。
- **真实服务联调**：PostgreSQL / OpenSearch / HugeGraph / LLM 在本环境不可用，因此**新评测集从未实际跑过一次**，其断言是待校准的假设而非基线。
- **CI 实际状态**：`.github/workflows/ci.yml` 的判定见 2.1 节，但 GitHub 上的真实运行结果未获取。

---

## 2. 总体判断

这是一个**领域建模深度远超工程成熟度**的项目。

| 维度 | 评分 | 说明 |
|---|:---:|---|
| 领域设计与安全性 | 8.5 / 10 | 罕见地严谨，且落到了代码而非文档 |
| 测试有效性 | 7.5 / 10 | 量大、隔离性好，但缺治理（无 conftest、无 lint） |
| 工程规范与可维护性 | 4 / 10 | 明显短板，巨型文件已到临界点 |
| 前端质量 | 5 / 10 | 功能完整，零测试、零 ESLint、巨型组件 |
| 交付就绪度 | 5 / 10 | 291 项未提交、CI 实际失败 |

**核心矛盾**：安全架构的正确性正在被代码组织的失控所侵蚀。现在拆分是 1–2 周的工程，再往后就是重写。

### 2.1 阻塞交付的头号问题：CI 是假绿的

`.github/workflows/ci.yml:22-30` 的 backend job **没有 `pip install -r requirements.txt`**。用系统 Python 复现的结果：

```
Ran 770 tests
FAILED (failures=1, errors=17)   # ModuleNotFoundError: No module named 'langgraph'
```

换用 `.venv/bin/python` 后才是 949 全绿。含义是：**CI 绿灯不等于测试通过**，17 个测试文件根本没被加载，测试集被"能导入就跑、不能导入就算了"静默削弱。

这条必须最先修——它使后续所有度量（含评测闭环）都建立在残缺的测试集上。

---

## 3. 已经做对的事（禁止回退）

以下设计是本项目最核心的资产，任何"追赶成熟框架"的重构都不得破坏它们。

### 3.1 确定性优先、LLM 后置

| 机制 | 位置 | 说明 |
|---|---|---|
| FTA/RBD 纯确定性计算 | `backend/agent/tools/icp_calculation.py`（640 行） | 全文件**无** `llm` / `prompt` / `ainvoke` |
| 共享基本事件 fail-closed | `backend/icp/reliability.py:242-255` | 宁可报错也不编数 |
| 路径超限拒绝出数值 | `backend/icp/reliability.py:374-377` | 同上 |
| 图关系重授权 fail-closed | `backend/agent/graph.py:125-139` | 回调导入失败时降级为拒绝，注释明确写了"缺失不意味着放宽安全" |

### 3.2 工程图证据分级

`backend/icp/diagram_validation.py:51` 定义 `VISION_CONFIDENCE_THRESHOLD = 0.75`：

- 视觉结果最高只到 `provisional`（`462-463`）
- 低于阈值或缺少置信度 → 隔离（`468-471`）
- `backend/icp/image_analysis.py:63-72` 解析失败时返回 `confidence=0.0`，**已被下游正确隔离**，链路是通的

### 3.3 拒绝无证据唯一诊断

- `backend/agent/tools/icp_diagnosis.py:901-932` `_exact_identity_rows` 规范化全等匹配优先
- 语义命中必须回 PostgreSQL 验真（`:3276` 显式区分"PostgreSQL 权威记录 / OpenSearch 检索投影"）
- HMC 手册明确要求：部分代码查询应返回候选列表并标注缺失段，**不得用向量距离猜唯一代码**

### 3.4 其他应保留的强度

- 37 处 `CREATE TABLE` **全部带 `IF NOT EXISTS`**，幂等成立
- 949 用例零外部依赖，20 秒内跑完，70% 使用替身——这是刻意且正确的工程决策
- `artifact_store.py` 路径逃逸防护完整；`opensearch_client.py` 字段名经正则白名单校验
- `AGENTS.md` 的红线（无画廊、统一上传、Dify 保留、七类不动）基本落地

---

## 4. 问题清单

### P0 — 阻塞交付与度量

| # | 问题 | 位置 | 影响 |
|---|---|---|---|
| P0-1 | **CI 缺 `pip install`** | `.github/workflows/ci.yml:22-30` | 17 个测试文件静默未加载，绿灯失真 |
| P0-2 | **图谱客户端假成功** | `backend/storage/hugegraph_client.py:683/704/709/746/789/864/871/1086/1541/1544` | `[]` / `None` / `{"vertices":[]}` 既表示"真空"也表示"故障"，上层无法区分 |
| P0-3 | 97 处 `print()` | 同上文件 | 应改 logger |
| P0-4 | 裸吞异常 | `hugegraph_client.py:847-848/870-871`、`icp/diagram_repository.py:143-144`、`icp/document_types.py:127-128` | 诊断链路静默降级 |
| P0-5 | **死解析器注册覆盖隐患** | `backend/core/parsers/text.py:38-40`、`backend/core/parsers/fault_tree.py:25-27` | 两模块从未被 import，装饰器不执行；**一旦有人 import 就静默覆盖**正在使用的 `DiagramParser` / `FaultHistoryParser`，无任何告警 |
| P0-6 | 部署安全基线 | `backend/main.py:26` CORS `["*"]` + `allow_credentials=True`；`backend/config/config.py:34` 默认口令 `"password"`；`Dockerfile` 无 `USER`（默认 root） | — |
| P0-7 | 16 处 `_call_model` 私有调用 | `answer.py` 内 7 处；`plan.py`、`scope_router.py`、`adaptive_tools.py`、`reflect.py`、`memory.py`×2、`chat_router.py` | tracing 无处统一挂钩 |

### P1 — 影响正确性与可维护性

| # | 问题 | 位置 |
|---|---|---|
| P1-1 | `answer.py` **14,644 行**；最大单函数约 1,665 行 | `backend/agent/nodes/answer.py` |
| P1-2 | `icp_diagnosis.py` 5,398 行、`plan.py` 2,086 行（`plan_node` 单函数 621 行）、`adaptive_tools.py` 1,502 行 | `backend/agent/` |
| P1-3 | **生产文件内嵌测试替身**，`main.py:46` 用 `.__class__.__name__ == "MockPostgres"` 字符串判断 | `backend/storage/postgres_client.py:2132/2154`、`backend/main.py:46` |
| P1-4 | 60 处 `except Exception: ... pass` | 后端全量，集中在 `icp_diagnosis.py`、`plan.py`、`db.py` |
| P1-5 | SQL 标识符 f-string 拼接 | `backend/agent/db.py:623`、`:745`（当前值走 `%s` 绑定，暂无注入，但缺白名单断言） |
| P1-6 | 迭代上限双轨不一致 | `graph.py:99` 硬编码 `iteration >= 4` vs `reflect.py` + `state.py:153` 的 `max_reflections=3` |
| P1-7 | 构建产物入库 | `.gitignore` 未忽略 `static/dist`；66 个构建 JS 已 git 跟踪 |
| P1-8 | 前端零质量门 | 无 ESLint / Prettier、无测试、57 处 `any`（`GraphCanvas.tsx` 独占 51 处）、`Diagnosis.tsx` 2,109 行含 23 个 `useState` |
| P1-9 | 零开发依赖、零 lint 配置 | `requirements.txt` 26 条依赖全为运行时；无 `pyproject.toml` / `ruff.toml` / `mypy.ini` / pre-commit |
| P1-10 | 死代码 | `api/kb.ts:32`（第二套上传封装）、`pages/TypeLogs.tsx`（无路由）、`hooks/` 3 个 SWR hook + `store/useVisualizeStore` 零引用、`openTaskStream` 在 `kb.ts:122` 与 `icp.ts:200` 完全重复 |

### P2 — 领域能力差距

对应 `AGENTS.md` 第 13 节自认的差距：知识版本化与冲突语义、人机复核闭环、检索精修、延迟与成本。

---

## 5. 对外部建议方案（CLM5.3flash）的核实结论

外部方案的方向判断是对的，但**约三分之一的具体引用是错的或凭空编造的**。执行前必须区分这两部分。

### 5.1 属实、可直接采纳

- 工作区约 290 项未提交、核心模块全 untracked —— 实际 291 项 / 135 untracked，属实
- `answer.py` 巨型文件、`_history_detail_present` 与 `_target_facet_is_present` 雷同 —— 属实
- 排序权重 `0.55/0.45` 硬编码 —— 属实，`backend/agent/tools/icp_diagnosis.py:3446`
- `diagnostic_confidence = diagnostic_support` 兼容 hack —— 属实，`backend/agent/nodes/adaptive_tools.py:1460`，`state.py:103` 注释已自认
- 三处重复围栏剥除解析 —— 属实，`plan.py:763/1076`、`adaptive_tools.py:784`
- 别名表多重编码 —— 属实，但**是 4 份不是 3 份**：`_ANSWER_FIELD_ALIASES`(adaptive_tools:106)、`_ROUTE_ALIASES`+`_DOCUMENT_SOURCE_ALIASES`(scope_router:46,58)、`_EXACT_FACET_ALIASES`(answer_contract:256)、`tool_policy._EXPLICIT_SOURCE_TOOL_PATTERNS`
- `_call_model` 私有调用 —— 属实，但**是 16 处不是 6 处**

### 5.2 引用错误或凭空编造（执行时会找不到目标）

| 声称 | 核实结果 |
|---|---|
| `answer.py:14047` 的无条件调试日志 | **不存在**。该行位于 `_render_architecture_answer` 正常函数体内 |
| `/tmp/answer_debug.log` | **全仓零命中**（grep `answer_debug`、`/tmp/`、`open(.*debug`） |
| `13834-13879` 整块复制两遍 | **不成立**。`evidence_indexes = {` 全文件仅出现 1 次。真实问题是 `approved_ids = {...}` 模式散落 **13 处** |
| evals 共 54 例 | 实为 **64 例**，漏了 `icp_multiturn_dialogue_cases_additional.json`（10 例） |

> **教训**：后续任何规划都要求"引用必附可复现的 grep 命令"，否则执行时会反复出现"找不到目标—误判已完成"。

### 5.3 该方案遗漏、且会破坏它自身方案的四项

1. CI 缺 `pip install`（P0-1）—— 它主张"评测接入 unittest 发现机制"，但 CI 里 17 个文件根本没被加载
2. 图谱客户端假成功（P0-2）—— 它主张"tracing 覆盖每次工具调用"，但工具自己分不清成功与失败
3. 死解析器覆盖隐患（P0-5）—— 它主张"每次移动一批函数跑一次评测"，若评测期间撞上覆盖，指标会莫名回退
4. `static/dist` 已入库（P1-7）—— 直接影响它主张的"步骤 0 拆聚焦提交"

---

## 6. 修缮路线图

**总原则**：先把测量工具修准，再谈测量；先让回滚有锚点，再动大文件。

> 与外部方案的**最大分歧**在于：它主张"先建仪表盘再动刀"，这没错；但当前这台机器的**传感器本身是坏的**。在 CI 漏装依赖、图谱客户端返回假成功、解析器注册表有覆盖隐患的情况下建 tracing 和评测，测出的是失真数据，会给后续重构发错误信号。
>
> 因此唯一真正必须先做的是 **P0-A**，不是 P0-B。

### 步骤 0：落地工作区（约 0.5 天）

1. `.gitignore` 增加 `static/dist/`、`.pytest_cache/`、`.mypy_cache/`、`.coverage`，并执行 `git rm -r --cached static/dist`
2. 按功能域拆分为聚焦提交：

```
refactor(icp): 七类文档注册表归一化与 document_types
feat(agent):   scope_router 范围路由与 direct_response
feat(agent):   evidence_gate 证据门禁
feat(icp):     graph_authority 图关系重授权
feat(parsers): document_ir 文档中间表示
feat(agent):   answer 四域答案策略
...
```

**验收**：`compileall backend` + `unittest` 全绿 + `npm ci && npm run build` 通过；`git status` 干净。

---

### P0-A：修复失真的地基（约 2–3 天）

在做任何 tracing / 评测之前，先让现有信号变可信。

| # | 动作 | 理由 |
|---|---|---|
| A1 | CI 补 `pip install -r requirements.txt`，并让导入失败**计为 ERROR** | 否则 P0-B 的评测接进去也跑不全 |
| A2 | `hugegraph_client.py` 引入 `HugeGraphUnavailable` 异常类型，区分"空结果"与"故障"；97 处 `print` → logger；`847-848`/`870-871` 补日志 | trace 记录的工具有效性取决于此 |
| A3 | 删除 `text.py` / `fault_tree.py` 的死注册，或改为显式 `register_parser()` 调用，消除 registry 的导入顺序依赖 | 防止评测期间被静默覆盖 |
| A4 | CORS 收紧、`Dockerfile` 加 `USER`、默认口令改强制环境变量 | 与重构无关但不能拖 |
| A5 | `llm_service` 加正式调用入口 `call_model(...)`，收敛 16 处 `_call_model` | **这是 P0-B tracing 统一挂钩的前提**，必须单列 |

**验收**：

- CI 在真实环境下 949 用例全绿
- `grep -c 'print(' backend/storage/hugegraph_client.py` 为 0
- 无外部模块调用 `_call_model`

---

### P0-B：度量基础（约 1–2 周）

#### B1 全链路 tracing

- 以会话为单位生成 `trace_id`，从 `backend/agent/api/chat_router.py` 进入后贯穿 `scope_router → plan → icp_diagnosis → adaptive_tools → (graph_traversal) → icp_calculation → evidence_gate → answer`
- 记录：LLM 调用（角色模型 / 耗时 / token / prompt 模板名）、工具调用（工具名 / **授权来源** / 返回证据数 / 耗时）
- JSON-lines 写可配置目录 + 按 `trace_id` 回放的查询接口
- 挂钩点统一在 A5 建立的 `llm_service.call_model` 入口
- ~~删除 `answer.py:14047` 调试日志~~ —— 该项不存在，动作取消

#### B2 评测闭环

- **评分脚本优先于用例数量**：先让脚本能出分，再批量加用例。反过来会造出一堆没有评分标准的用例
- 现有 164 例已覆盖十三个情形（见第 7 节），可先跑通再按需增补
- 确定性指标：

| 指标 | 判定方式 |
|---|---|
| 引用有效率 | `[^n]` 是否指向 `approved_evidence_ids` |
| 候选命中率 | 正确故障代码是否进 Top-K |
| 字段完整率 | `answer_contract` 必答字段在场 |
| 澄清触发正确率 | 歧义输入是否走 `direct_response` |
| 越权率 | 模型提出的工具是否被白名单过滤 |

- 接入 unittest 发现机制，让别名表 / prompt / 权重改动必须过评测

#### B3 静默失败可见化

三类降级各加计数进 trace：规划器 JSON 解析失败降宽召回、检索投影失败只进 stats、adaptive fallback 触发。

**验收**：能回答"每次提问实际几次模型调用、降级率多少、解析失败率多少"。

---

### P1：控制实现复杂度（约 3–4 周）

| # | 动作 | 要点 |
|---|---|---|
| C1 | 拆分 `answer.py` | 四个答案域 `fault / history / target / calculation` 各成策略模块；抽共享「字段在场断言」模块消除 `_history_detail_present`(11215) 与 `_target_facet_is_present`(5774) 的雷同；公共后处理抽成显式 pipeline 组合器，去重目标是**散落 13 处的 `approved_ids` 模式**（不是外部方案错误指认的 13834-13879）。每移动一批函数跑一次 B2 评测，指标不回退才继续 |
| C2 | 护栏语义单一来源 | `answer_contract.py` 作为唯一权威（字段→工具映射、必答字段、聚焦契约）；prompt 中对应规则改为**由契约生成**；排序权重（`:3446`）进配置文件并标注"待评测调优"。收敛范围是 **4 份别名表** |
| C3 | 结构化输出契约化 | `plan` / `adaptive_tools` / `scope_router` 改为 schema 约束输出 + 校验失败带错误重试一次，替代 3 处重复围栏解析（`plan.py:763/1076`、`adaptive_tools.py:784`）。解析失败率应在 B3 指标里看到下降 |
| C4 | AgentState 治理 | 80+ 字段先出「字段生命周期表」（写/读/废弃）；**同时直接清掉 `diagnostic_confidence`**（`adaptive_tools.py:1460`）——它是唯一有确凿证据的废弃字段，不必等整表重构 |
| C5 | 巨型文件止血 | `postgres_client.py` 拆 schema_migrations / kb_repo / document_repo，**移除内嵌 Mock**，`main.py:46` 字符串判断改显式配置开关；60 处静默异常按链路分级，诊断链路内必须上抛或计入 B3；`_query_rows(table=...)` 加白名单断言 |

---

### P2：领域能力补强（约 4–6 周，按依赖与性价比重排）

| 顺序 | 内容 | 理由 |
|---|---|---|
| D1 | 延迟与成本 | 唯一能直接消费 B1 trace 数据的项，收益最快：完整性编辑器仅在必答字段缺失时触发；精确 HMC（代码确定、无歧义、PG 命中）加短 TTL 缓存；目标把精确代码查询压到 1–2 次模型调用 |
| D2 | 检索精修 | 需先有稳定的改写评测集。优先表格行 contextual enrichment；硬编码别名表停止加长 |
| D3 | 知识版本化与冲突语义 | 检索增加机型/构型/软件版本维度；同一业务 ID 新版本结论不同时标记 supersede 并向答案暴露"两版记载不一致"。数据模型变更放 `backend/icp/repository.py`，保持幂等建表 |
| D4 | 人机复核闭环 | 最重，放最后。provisional 候选逐个确认 → 转 `auto_validated`；确认样本自动沉淀为 `test_diagram_extraction.py` 回归用例，让"确定性抽取覆盖率"成为可度量增长指标 |

---

## 7. 有意不做的事

以下是设计上的正确取舍，不得在"追赶成熟框架"的名义下回退。

**来自外部方案，完全同意：**

- **不加自主反思循环 / 自由 ReAct** —— 单批补查 + 线性图是安全特性。若未来确需第二轮补查，用确定性上限触发（必答字段缺口 > 阈值），不恢复 LLM 自主循环
- **不开长期记忆** —— 诊断结论不应跨会话污染
- **不加第八类知识卡片、不恢复画廊 / 通用 KB / 独立上传**
- **不为用社区检测等流行技术而合并实体** —— "完全一致业务 ID 才合并"的纪律比图的丰富度重要

**本次补充：**

- **不在 P0-A 完成前启动 P0-B** —— 传感器坏了先修传感器
- **不追求 300 例评测集** —— 先让评分脚本能出分，164 例足够启动飞轮
- **不一次性重排 `AgentState` 的 80+ 字段** —— 除已确认废弃的 `diagnostic_confidence` 外，其余等生命周期表出来再决定
- **不引入 Langfuse 等重型平台** —— 内网环境自建轻量 JSON-lines 方案更稳妥（外部方案说"可选"，此处明确排除）

---

## 8. 本次已完成的交付

### 8.1 100 例多轮评测集

| 文件 | 用例 | 轮次 | 覆盖 |
|---|---:|---:|---|
| `evals/icp_multiturn_regression_positioning.json` | 33 | 66 | 精确代码查询+跨文档补全(10)、部分代码候选收窄(1)、模糊现象+追问收窄(12)、具体定位+现场验证(10) |
| `evals/icp_multiturn_regression_analysis.json` | 35 | 70 | 原因追问+处置闭环(9)、后果影响(6)、维修与API上报能力(8)、历史排故(6)、架构与指代消解(6) |
| `evals/icp_multiturn_regression_boundary.json` | 32 | 64 | FTA确定性(5)、RBD确定性(5)、范围外连续拒答(8)、相关但无记载(8)、歧义澄清+恢复(6) |

**多轮相较单轮，刻意多测四个维度**（这些正对应 `plan` 节点"当前对象 + 本轮所问维度"的机制）：

1. **跨文档补全**：首轮给 HMC 代码，次轮问该故障模式在 FMECA 中的原因与影响——测的是把"这个"绑定到上一轮实体，而非泛化到整个子系统
2. **拒答后恢复**：`MT-OOS-03/04/07/08` 首轮范围外，次轮用户真正转向知识库——验证拒答没有把后续正常问题一起挡掉（拒答逻辑最典型的过冲缺陷）
3. **澄清后恢复**：`MT-CLAR-*` 首轮歧义走 `clarification`，次轮补充对象后应恢复 `knowledge_base` 并正确绑定
4. **换问法诱导**：`MT-OOS-01` 首轮问天气，次轮"天气对设备运行有没有影响"——不得因为出现"设备"二字就转入检索

**已验证**：三份文件全部通过评测器 `_load_cases` 加载（未知断言名拒绝、正则编译、turn id 唯一性）；245 个 `expected_records` 与 `test-docs/` 逐条比对零缺失。

**待校准**：`MT-NOEV-*`（8 例无记载正则）、`MT-OOS-*` 的 route 判定、`MT-CLAR-*` 的 route 判定，首次实跑大概率需要调整。这属于预期内——它们正是系统当前最可能不达标的地方。

### 8.2 待办

删除 `evals/icp_regression_cases.json`（第一版单轮文件，删除操作未获批准，留着会与三份多轮文件重复）。

---

## 9. 每步必须执行的验证

```bash
# 后端
python -m compileall -q backend
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'

# 前端（本机无 npm，需在有 Node 的环境执行）
cd frontend && npm ci && npm run build

# 评测（需服务可用：PG / OpenSearch / HugeGraph / LLM）
.venv/bin/python scripts/evaluate_icp_paraphrases.py \
  evals/icp_multiturn_regression_positioning.json \
  --concurrency 4 -o /tmp/pos.json
```

不向 `main` 直接合并；提交信息遵循 `feat(icp):` / `fix(frontend):` / `refactor(agent):` / `test(icp):` 约定。

---

## 附录 A：引用核实方法

本文档所有 `文件:行号` 均经实际命令核实。**后续修改本文档或制定任何规划时，必须遵循同一纪律：引用必附可复现的命令。**原因见 5.2 节——外部方案中约三分之一的引用指向了不存在的目标。

核实示例：

```bash
# 行号确认
sed -n '3446p' backend/agent/tools/icp_diagnosis.py

# 死注册确认（grep 全仓是否被 import）
grep -rn "parsers.text\|parsers import text\|fault_tree" backend --include=*.py

# 计数类断言
wc -l backend/agent/nodes/answer.py
grep -c 'print(' backend/storage/hugegraph_client.py

# 评测集溯源（HMC 代码跨单元格，必须按表格结构解析）
# 详见 8.1 节的验证脚本思路：拼 LCN + HSC 后再比对
```

无法用命令核实的判断，本文档一律标注为"推断"或"待验证"，不写成既成事实。
