# ICP 评测修复执行记录

> 本文记录多轮问答修复、工程质量优化、100 例评测集整改和正式复测的可复现执行状态。
> 所有结果仅针对仓库中的合成测试资料，不代表真实工程资料或适航维修结论。

> 2026-09-05 续接：当前实施顺序与验收以 [ICP整改实施计划.md](./ICP整改实施计划.md) 为准。下文保留历史状态；阶段 2 的“实现完成”不表示通过验收，复核已发现数值身份关联等缺口。基础设施测试已有部分补齐，工程阶段仍未整体验收。

## 执行原则

- 生产代码不得引用评测用例 ID、固定问题全文或合成记录到答案的映射。
- 开放语义由模型输出受控结构；代码只校验结构、身份、证据、数值和安全边界。
- 自然语言答案不要求出现固定措辞。精确代码、API 名称、业务 ID 和确定性数值除外。
- PostgreSQL 是权威事实来源；旧答案和 OpenSearch 相似命中不能直接授权新结论。
- 先保存可重建基线，再修改代码；每个修复必须同时有正例、干扰邻居和歧义闭锁测试。

## 阶段状态

| 阶段 | 状态 | 说明 |
|---|---|---|
| 0. 冻结基线 | 已完成 | 已保存代码、文档、评测集、服务、模型和现状评测基线；记录正式复测前风险 |
| 1. 当前问题修复 | 进行中 | 结构化多轮骨架已存在，正在移除关键词驱动和正常路径固定模板 |
| 2. 评测器升级 | 实现完成，待集成验收 | 已增加回答义务、交付字段、精确证据身份、容差数值与独立语义裁判；100 例尚未迁移 |
| 3. 100 例整改 | 未开始 | 修正 gold、扩展 3～5 轮和补充跨轮场景 |
| 4. 工程质量优化 | 未开始 | 在行为护栏下解除循环依赖并补基础设施测试 |
| 5. 正式复测 | 未开始 | 19 例 canary 后执行三份 100 例，并重复运行评估稳定性 |

## 阶段 0 基线

快照日期：2026-09-04（Asia/Shanghai）

### 代码与工作区

- 分支：`refactor/icp-diagnosis-specialization`
- HEAD：`fe1acaad426ac6406985970ed7bd9fcd5d4855de`
- 上游：`origin/refactor/icp-diagnosis-specialization`，ahead 7 / behind 0
- 工作区：30 个已跟踪文件修改、6 个未跟踪文件、0 个 staged 文件
- 已跟踪补丁 SHA256：`f31af0a0bbf16bb1a888da9feec8b89f93f3bf2c30631fdcbbfc26ea7cfafe17`
- 未跟踪归档 SHA256：`f7d0a378bd1c30909ef84d1c5282b10d08f1955eaf1fef966278e901add66918`

本地可重建材料位于：

```text
.artifacts/evals/stage0/worktree-tracked.patch
.artifacts/evals/stage0/worktree-untracked.tar.gz
```

### 输入资料与评测集

- 七类 `test-docs` 均只有一份，ArtifactStore 中的源文件与仓库文件逐一 SHA256 相同。
- 三份 100 例评测文件合计 100 例、200 轮；全部是两轮对话。
- 当前 19 例 canary 为 19 例、40 轮。
- 完整逐文件哈希保存在机器可读 manifest：

```text
.artifacts/evals/stage0/baseline-current.json
SHA256: 9b4e6dbc2e159c877ea5177e1ca1d4a743b4128af0877111d2280d2ac0e56d7c
```

### 服务与数据

- FastAPI `/readyz`：PostgreSQL、OpenSearch、HugeGraph REST、ArtifactStore 均为 ready。
- PostgreSQL 实际端口为 5434；七类权威记录数量为：Functions 30、HMC 100、FMECA 72、MFL 72、历史案例 24 / 步骤 144、FTA 8、RBD 4。
- OpenSearch：green，alias `knowledge_graph_default` 指向 `knowledge_graph_default_v1`，共 494 条，与七类解析向量数一致。
- HugeGraph：1.7.0，2126 个顶点、4683 条边、7 个 Document 来源节点。
- FTA 8 个和 RBD 4 个当前 revision 均为 revision 2、`auto_validated`、`deterministic_source`、`published`。
- 默认模型：`qwen3.6:latest`；Embedding：`bge-m3:latest`；Reranker：`bge-reranker-v2-m3`。
- Agent 参数：temperature 0.2、top_p 0.8、max_tokens 4096、retrieval_top_k 12、长期记忆关闭。

### 现状 19 例基线

报告：

```text
.artifacts/evals/stage0/canary19-current.json
SHA256: 844c9ba86592eb5b0602d5bf12f1f79b79ea33b9f709ed502396597d7298a676
```

结果：

- 用例：11/19（57.89%）
- 轮次：31/40（77.5%）
- 无运行错误、无清理失败
- 失败用例：MT-ARCH-02、MT-FAULT-01、MT-RBD-01、MT-RBD-02、MT-RBD-03、MT-HISTORY-01、MT-MFL-01、MT-SOURCE-01

本次 canary 已经使用增强后的来源白名单、记录身份和回答长度断言，因此不能与旧的弱断言成绩直接横向比较。

### 已确认风险

1. HugeGraph REST 健康，但 Gremlin `/gremlin` 的简单遍历返回 400/500；正式图遍历评测前必须修复或让工具明确降级。
2. Function Locator 的结构化事实和投影已成功，但文档主状态仍为 `kg_pending`；其 4 张视觉图中 1 张 provisional/published、2 张 quarantined、1 张没有 revision。
3. MFL 结构化投影有 2 条 `explicit_id_property_conflict` warning，需要确认是否为合法跨来源限定或真实冲突。
4. ArtifactStore 有一个未登记的旧 HMC 测试文件；评测必须始终按当前七个 `doc_id` 隔离。
5. 当前 RBD 源文件与仓库完全一致。MT-RBD-03 的 `0.99888052` 仍是显示输出模型十小时确定性结果；失败来自模型选择和四项结果交付不完整，不是源文件漂移。

## 阶段 1 当前修改顺序

1. 贯通回答义务和交付 manifest，删除未接线的第二套契约。
2. 正常增量回答交还模型组织语言，固定模板只用于明确的安全降级。
3. 自然语言相似度不得产生 FMECA 精确身份。
4. 将历史案例父记录绑定作为工具到回答的唯一结构化结果。
5. HMC 仅接受明确代码字段，停止从任意数字猜测 LCN/HSC；无显式关系时不把 MFL 邻居归属于该报码。
6. RBD 使用结构化对象和来源边界；未绑定模型时不得保留或计算全部召回模型。
7. 编辑器首次回显内部 manifest 后直接安全降级，停止重复相同提示造成重试风暴。
8. 修复 Gremlin 可用性预检和 Function Locator 终态表达，再进入正式集成评测。

### 非特化约束横向复审

在第一轮定向修复通过后，又对 plan、adaptive tools、answer 及各领域后处理做了
一次横向审计。审计确认阶段 1 仍未达到验收条件，原因不是某个评测样例未命中，
而是旧实现中还存在会普遍影响任意问法的合同越权：

- 原始问句正则仍可决定 FTA 的回答模式；
- 粗粒度 intent 仍可自动扩张成 FMECA、HMC、MFL、历史全套餐补查；
- fault、target、history 后处理仍可从 intent 或答案表面文字新增回答义务并覆盖健康模型正文；
- 草稿中出现多个成员或数量仍可反向激活完整集合义务；
- `reliability_query` 仍可在模型未指定字段时自动添加 RBD 成功判据；
- RBD 成功判据仍可能通过自然语言正则和名称相似度被推成某组的单失效能力。

这些问题按“模型生成带当前轮原话依据的闭集合同，代码只校验枚举、依据、身份、
证据、数值与 deny-only 安全边界”的统一原则继续整改。完成前不进入阶段 3，也不把
中间 canary 分数当作效果结论。

### 阶段 1 中间金丝雀（非验收结果）

报告：

```text
.artifacts/evals/stage1/canary19-after-stage1.json
SHA256: dec73debe174bc0dfac1f8c62da3b2a4493f0ae8c1be0be2d5b66aab11bbe6ad
```

这次运行发生在阶段 1 完成前，且 19 例仍混用旧的固定词/弱正则断言，因此只作诊断采样，不作为修复效果验收分数。

- 用例：9/19（47.37%）
- 轮次：28/40（70%）
- 无清理失败；有 1 轮空答案，已纳入阶段 1 回归检查
- 主要暴露的通用问题：结构化 RBD 对象未绑定导致计算空转；来源边界追问被误降级成架构缺失；个别追问虽内容正确但仍被旧的来源/固定措辞断言判失败。

### 阶段 2 已完成的评测器能力

- SSE 报告已保留 `completion_status`、`unconfirmed_fields`、`delivered_answer_fields` 和 `answer_delivery_manifest`。
- 可断言回答义务状态、必须/禁止/允许交付字段，以及精确记录身份。
- 数值断言支持绝对/相对容差，并可绑定模型或业务身份。
- 开放语义使用独立 JSON 裁判，裁判器可见当前问题、最近历史与最终答案；调用或解析失败时闭锁失败。
- 固定正则仅保留为旧评测兼容能力，不得用于新增的开放语义 gold。

## 阶段 3 前置审计（只读）

三份文件可被当前评测器正常加载，用例 ID 和轮次 ID 无重复：

| 文件 | 用例 | 轮次 |
|---|---:|---:|
| `icp_multiturn_regression_positioning.json` | 33 | 66 |
| `icp_multiturn_regression_analysis.json` | 35 | 70 |
| `icp_multiturn_regression_boundary.json` | 32 | 64 |
| 合计 | 100 | 200 |

当前结论：题型覆盖面较广，可作 smoke/baseline，但尚不符合修复后的质量验收要求。

- 100 例全部只有 2 轮；3～5 轮、非相邻回指均为 0。
- 已有模糊现象 12 例、具体故障 10 例、精确代码 10 例、范围外 8 例、无证据 8 例、澄清恢复 6 例和话题切换 4 例。
- 200 轮中 `semantic_criteria=0`、`numeric_expectations=0`、`required_used_record_ids=0`；157 轮依赖 `answer_regex`，其中 141 轮含宽松 OR。
- 49 轮依赖旧 facet；非范围外/澄清轮中，只有 37 轮约束了任何记录身份。
- metadata 里的 `expected_records` / `expected_value` 当前不是实际断言。
- 10 个 FTA/RBD 数值 gold 已与当前合成 `test-docs` 核对一致，但必须迁移到带容差和模型身份的 `numeric_expectations`。

已确认的 P0 gold/设计错误包括：`MT-VAGUE-04`、`MT-HIST-04/05`、`MT-CAUSE-09`、`MT-MAINT-02/04/07`、`MT-FTA-01/03`、`MT-RBD-02/03/04`、`MT-ARCH-06`、`MT-IMPACT-02`、`MT-OOS-01/06/08`、`MT-CLAR-02/06`。阶段 3 将先修正这些 gold，再迁移断言。

扩轮方案保持用例总数 100 不变：给 14 个现有用例追加轮次，总轮次预计从 200 增至 232，覆盖非相邻指代、旧结论撤回、跨话题返回、数值参数修正、邻居记录隔离和证据边界。
