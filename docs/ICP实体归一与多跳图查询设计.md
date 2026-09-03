# ICP 实体归一与多跳图查询设计

> 状态：实现基线
>
> 本体版本：`1.0.0`
>
> 适用范围：ICP-RAG 七类工程文档
>
> 权威真源：PostgreSQL
>
> 图投影：HugeGraph
>
> 语义候选：OpenSearch + BGE-M3

## 1. 目标与边界

本设计用于把当前“七类文档分别检索、回答阶段再汇总”的方式，扩展为可追溯的跨文档多跳查询：

```text
用户现象
→ 定位候选规范实体
→ 沿正式关系查询原因、影响、模块、功能、维修和历史路径
→ 回 PostgreSQL 校验路径中的每一个节点和关系
→ 生成逐跳可引用的证据
→ 由 LLM 整理为用户答案
```

需要解决的核心问题是：

1. 七类文档使用同一套稳定的实体类型和关系类型；
2. 同一业务对象在不同文档中能够确定性地指向同一个规范实体；
3. 相似文本可以帮助发现候选，但不能自动伪造成同一实体；
4. HugeGraph 只保存可由 PostgreSQL 和原始证据重建的关系投影；
5. 多跳路径中的每一跳都有明确的来源、适用范围和验证状态；
6. FTA/RBD 的视觉候选、正式拓扑和数值计算保持严格隔离。

本设计不改变以下产品约束：

- 用户侧仍然只有 `function_locator`、`hmc`、`fta`、`rbd`、`fmeca`、`mfl`、`fault_history` 七类知识文档，不增加第八类卡片；
- PostgreSQL 始终是权威事实来源；
- OpenSearch 只负责全文、向量召回和候选归一检索；
- HugeGraph 是可删除重建的关系投影，不反向修改 PostgreSQL；
- ArtifactStore + LocalFS 保存原文、表格、图片等证据；
- 不恢复独立图片画廊；
- 不破坏 Dify 的现有接口；
- LLM 不得直接写入正式实体、正式关系或把候选归一升级为正式归一；
- FTA/RBD 数值必须继续由确定性 Python 算法计算。

## 2. 总体架构

```text
原始 Word / Excel / 图片
        │
        ├── ArtifactStore + LocalFS：原件和图片证据
        │
        ▼
七类专用解析器
        │
        ├── PostgreSQL 领域表：HMC、FMECA、MFL、Function、历史、图修订
        ├── PostgreSQL 身份层：提及、规范实体、归一结果、关系断言
        └── OpenSearch：文本/向量索引与语义候选
                         │
                         ▼
                  确定性投影器
                         │
                         ▼
                  HugeGraph 正式关系图
                         │
                         ▼
              graph_traversal 受限遍历
                         │
                         ▼
             PostgreSQL 逐节点/逐边回查
                         │
                         ▼
                   Evidence Gate
                         │
                         ▼
                 LLM 整理最终答案
```

数据模型分为三层：

```text
规范业务实体 CanonicalEntity
        ↑ resolved_to
文档实体提及 EntityMention
        ↓ source_ref
原始段落、表格单元格、图片区域 SourceFragment
```

关系同样分为两个层次：

- `RelationAssertion`：PostgreSQL 中按来源文档保存的权威关系断言；
- HugeGraph 物理边：为遍历而聚合的投影，可同时由多条断言支持。

## 3. 本体版本管理

### 3.1 版本格式

本体使用 SemVer：

```text
MAJOR.MINOR.PATCH
```

- `PATCH`：不改变身份和关系语义的字段、别名、说明或校验修正；
- `MINOR`：向后兼容地新增实体类型、关系类型或可选属性；
- `MAJOR`：改变实体身份元组、关系方向、关系含义或删除类型，必须完整重建。

第一版固定为 `1.0.0`。本体版本和投影器版本分别管理，不能混为一个字段。

### 3.2 版本规则

每条以下记录必须保存 `ontology_version`：

- 规范实体；
- 实体提及；
- 归一结果和候选；
- 关系断言；
- HugeGraph 投影清单；
- `graph_traversal` 查询结果。

同一时刻只允许一个版本作为写入版本；旧版本可在迁移期间只读。跨 `MAJOR` 版本禁止直接混合遍历。

建议维护：

```text
icp_ontology_versions
  version
  status: draft | active | read_only | retired
  schema_hash
  projector_version
  activated_at
```

## 4. 核心实体定义

### 4.1 工程结构实体

| 实体类型 | 含义 | 正式身份依据 |
|---|---|---|
| `System` | ICP 系统 | 系统代码 + 构型范围 |
| `Subsystem` | 子系统 | 子系统代码 + 所属系统 |
| `Module` | 模块/板卡/功能单元 | 模块 ID + 所属系统/构型 |
| `Component` | 模块内部明确部件 | 部件 ID + 所属模块 |
| `Function` | 系统或模块功能 | 功能 ID + 所属系统 |
| `Interface` | 明确接口或信号接口 | 接口 ID；无 ID 时仅文档内身份 |
| `API` | 可调用或上报 API | API 规范名称 + 所属系统 |

`System`、`Subsystem`、`Module` 和 `Component` 不依靠名称长度或层级序号猜测类型。类型必须来自结构化字段、明确表头或通过类型校验的工程图节点。

### 4.2 故障与维修实体

| 实体类型 | 含义 | 正式身份依据 |
|---|---|---|
| `FaultMode` | 故障模式 | 故障模式代码；无代码时仅文档内身份 |
| `Symptom` | 可观察故障现象 | 现象代码；纯文本默认仅文档内身份 |
| `FaultCause` | 文档明确记载的原因 | 原因代码；纯文本默认仅文档内身份 |
| `FaultEffect` | 局部或最终影响 | 影响代码；纯文本默认仅文档内身份 |
| `DetectionMethod` | 检测方式或判据 | 检测方法代码；纯文本默认仅文档内身份 |
| `MaintenanceAction` | 检查、隔离、维修或更换动作 | 动作代码；纯文本默认仅文档内身份 |

`FaultMode`、`Symptom`、`FaultCause` 和 `FaultEffect` 必须保持类型区别。语义接近不能成为合并理由，例如“滤波电容老化”“输入纹波增大”“滤波性能退化”分别可能是原因、影响和故障模式。

### 4.3 七类记录实体

| 实体类型 | 含义 | 正式身份依据 |
|---|---|---|
| `HMCCode` | HMC 代码记录 | 系统范围 + LCN + HAMC + HSC |
| `FMECARecord` | 一条 FMECA 记录 | 系统范围 + FMECA 代码 |
| `MFLRecord` | 一条 MFL 记录 | 系统范围 + MFL 代码 |
| `FaultHistoryCase` | 一条历史案例 | 案例编号 |
| `FaultHistoryStep` | 案例中的有序步骤实例 | 案例编号 + 步骤序号 + 文档内出现序号 |

HMC 的 LCN、HAMC、HSC 始终按字符串处理，保留 HSC 前导零和 HAMC 的 `**` 通配语义。

### 4.4 FTA/RBD 实体

| 实体类型 | 含义 | 正式身份依据 |
|---|---|---|
| `FTADiagram` | 一棵故障树的逻辑对象 | `diagram_id` |
| `FTAEvent` | 顶事件、中间事件或基本事件 | `revision_id + source_node_id` |
| `FTAGate` | AND/OR/XOR 等逻辑门 | `revision_id + source_gate_id` |
| `RBDDiagram` | 一张可靠性框图 | `diagram_id` |
| `RBDBlock` | START、END 或功能块 | `revision_id + source_block_id` |
| `RBDJunction` | 并联分支和汇合点 | `revision_id + source_junction_id` |

图修订内的节点身份默认不能跨修订自动合并。只有节点明确带有模块 ID、功能 ID或故障模式代码时，才通过 `represents_*` 关系连接到规范业务实体。

### 4.5 来源实体

| 实体类型 | 含义 | 身份依据 |
|---|---|---|
| `Document` | 原始文档记录 | `doc_id` |
| `SourceFragment` | 段落、表格行/单元格、图片区域 | `doc_id + source_ref + content_hash` |

`SourceFragment` 主要保存在 PostgreSQL/ArtifactStore 证据层，默认不全部投影成 HugeGraph 顶点，避免图膨胀。HugeGraph 边通过 `assertion_ids` 回查来源片段。

### 4.6 保持为属性的数据

以下内容默认是实体或记录属性，而不是独立节点：

- 文档版本、适用系统、构型和有效期；
- FMECA 严酷度、概率等级和原始概率；
- RBD 可靠度、故障率、修复率、任务时间；
- FTA 事件概率或故障率；
- 布尔状态、处理状态、投影状态；
- 页码、表号、行列号和图片坐标。

只有未来出现受控业务代码且确有跨文档遍历价值时，才通过本体 `MINOR` 版本将其提升为实体。

## 5. 核心关系定义

所有关系使用固定方向和类型约束。正式图中禁止使用没有业务语义的泛化 `related_to`。

### 5.1 系统结构与功能

| 关系 | 方向 | 含义 |
|---|---|---|
| `contains` | System/Subsystem/Module → Subsystem/Module/Component | 明确的组成层级 |
| `contains_function` | Function → Function | 明确的功能分解 |
| `implements` | Module/Component → Function | 模块或部件实现功能 |
| `exposes` | Module/Component → Interface | 端点提供接口 |
| `connects_to` | Interface → Module/Component/Interface | 明确的接口目标 |
| `input_via` | Function → Interface | 功能的明确输入接口 |
| `output_via` | Function → Interface | 功能的明确输出接口 |

### 5.2 故障、原因、影响与维修

| 关系 | 方向 | 建立条件 |
|---|---|---|
| `has_fault_mode` | Module/Component/Function → FaultMode | 同一结构化行明确提供对象 ID 和故障模式 |
| `manifests_as` | FaultMode → Symptom | 同一权威记录明确给出故障模式和现象 |
| `caused_by` | FaultMode → FaultCause | FMECA/历史根因等明确字段关系 |
| `results_in` | FaultMode → FaultEffect | FMECA 局部/最终影响字段 |
| `affects` | FaultEffect → Function/Module/System | 影响对象具有明确 ID |
| `detected_by` | FaultMode → DetectionMethod/API | 明确检测方法或 API 引用 |
| `repaired_by` | FaultMode/HMCCode/MFLRecord → MaintenanceAction | 明确维修/处置字段 |

`caused_by` 只能表示来源明确记载的因果关系。LLM 推断、向量相似或 FTA 图片低置信结果不能创建该关系。

### 5.3 文档记录关系

| 关系 | 方向 |
|---|---|
| `describes_symptom` | HMCCode → Symptom |
| `describes_fault_mode` | HMCCode → FaultMode |
| `recommends_action` | HMCCode → MaintenanceAction |
| `analyzes_fault_mode` | FMECARecord → FaultMode |
| `records_cause` | FMECARecord → FaultCause |
| `records_effect` | FMECARecord → FaultEffect |
| `records_detection` | FMECARecord → DetectionMethod |
| `locates_fault_mode` | MFLRecord → FaultMode |
| `locates_at` | MFLRecord → Module/Component |
| `reports_via_api` | MFLRecord → API |
| `observed_symptom` | FaultHistoryCase → Symptom |
| `confirmed_cause` | FaultHistoryCase → FaultCause |
| `resolved_by` | FaultHistoryCase → MaintenanceAction |
| `has_diagnosis_step` | FaultHistoryCase → FaultHistoryStep |
| `has_step_content` | FaultHistoryStep → SourceFragment/文档内文本实体 |
| `next_step` | FaultHistoryStep → FaultHistoryStep |
| `references_hmc` | FaultHistoryCase/其他记录 → HMCCode |
| `references_fmeca` | FaultHistoryCase/其他记录 → FMECARecord |
| `references_fta` | FaultHistoryCase/其他记录 → FTADiagram |
| `references_rbd` | FaultHistoryCase/其他记录 → RBDDiagram |
| `references_mfl` | FaultHistoryCase/其他记录 → MFLRecord |

`confirmed_cause` 只在故障历史的“根因/确认原因”等明确字段存在时建立；普通处理步骤不能反推为根因。

### 5.4 FTA

| 关系 | 方向 | 说明 |
|---|---|---|
| `decomposed_by` | FTAEvent → FTAGate | 一个事件由逻辑门分解 |
| `has_input_event` | FTAGate → FTAEvent | 逻辑门的输入事件 |
| `represents_fault_mode` | FTAEvent → FaultMode | 事件明确带故障模式代码时建立 |

不能把 FTA 简化成一组普通 `causes` 边，因为 AND、OR、XOR 的含义必须由逻辑门节点保留。由基本事件追溯到顶事件时，遍历上述关系的反方向。

### 5.5 RBD

| 关系 | 方向 | 说明 |
|---|---|---|
| `next_block` | RBDBlock/RBDJunction → RBDBlock/RBDJunction | 按 START 到 END 的成功路径方向连接 |
| `represents_module` | RBDBlock → Module | 功能块明确标注模块 ID |
| `implements_function` | RBDBlock → Function | 功能块明确标注功能 ID |

串联、并联和冗余由 `next_block` 拓扑及 Junction 表达，不使用仅凭名称生成的 `parallel_with`。

### 5.6 来源关系

| 关系 | 方向 | 说明 |
|---|---|---|
| `extracted_from` | 任一源实体 → Document | 记录实体来自哪个文档 |
| `supported_by` | 关系断言/实体提及 → SourceFragment | 存在于 PostgreSQL 证据层，不要求全部物理入图 |

## 6. 规范实体 ID

### 6.1 ID 算法

规范实体使用 UUIDv5 确定性生成：

```text
identity_tuple = canonical_json({
  ontology_major,
  entity_type,
  identity_scope,
  normalized_business_key
})

entity_id = uuid5(ICP_ENTITY_NAMESPACE, identity_tuple)
HugeGraph vertex id = "icp:ce:v1:" + entity_id
```

PostgreSQL 同时保存：

- `entity_id`：稳定 UUID；
- `canonical_key`：可读的规范键；
- `identity_hash`：规范身份元组 SHA-256；
- `identity_scope`：`global_business`、`scoped_business`、`document_text`、`diagram_revision` 或 `document_occurrence`；
- `identity_components`：生成 ID 的原始规范字段 JSONB。

### 6.2 各类身份元组

示例：

```text
System:
  (system_code, configuration_scope)

Module:
  (system_code, configuration_scope, module_id)

HMCCode:
  (system_code, configuration_scope, lcn, hamc, hsc)

FMECARecord:
  (system_code, configuration_scope, fmeca_code)

FTAEvent:
  (revision_id, source_node_id)

无业务代码的 FaultCause:
  (doc_id, value_kind, exact_original_text)
```

无业务代码的纯文本实体必须保留 `doc_id + 字段类型 + 精确原文` 边界，不能仅用名称哈希生成全局 ID。

### 6.3 确定性格式归一

可直接执行的格式归一包括：

- Unicode NFKC；
- 全角字符转半角；
- 去除首尾空白、合并展示性连续空白；
- 代码字段按各自规范统一分隔符；
- API 和模块代码按注册规范处理大小写；
- HSC 始终作为字符串并保留前导零；
- HAMC 的 `**` 原样保留，不替换成空值或任意具体值。

格式归一只改变表示，不改变业务含义。原始值必须另行保存。

## 7. 实体归一

### 7.1 两条独立通道

实体归一分为正式通道和候选通道：

```text
正式通道：业务代码/显式引用/权威别名 → resolved_to → 规范实体
候选通道：字面/向量/上下文相似 → candidate_semantic → 只用于召回
```

两条通道不得共用一个模糊的 `confidence` 字段，也不能按阈值把候选通道自动升级为正式通道。

### 7.2 可自动成为正式归一的情况

只允许：

1. 相同实体类型、相同适用范围且规范业务代码完全一致；
2. 文档字段明确写出目标代码或 ID；
3. 内部权威映射表明确给出别名到唯一业务 ID 的映射；
4. 图节点明确标注业务 ID，且图修订通过相应类型与状态校验；
5. 兼容保留的 `manual_confirmed` 历史结果。

正式归一方法枚举：

```text
resolved_exact_id
resolved_explicit_reference
resolved_authoritative_alias
resolved_manual_legacy
```

内部权威映射表不是第八类用户文档入口；它是代码/数据库中的主数据配置，必须有版本、来源和冲突检测。

### 7.3 相似度候选

相似度仍然必须计算，用于发现候选实体和找到图查询起点。候选特征包括：

- `lexical_score`：规范化字面重合、编辑距离、代码片段匹配；
- `embedding_score`：BGE-M3 向量相似度；
- `context_score`：所属系统、构型、模块、功能、字段类型一致性；
- `structure_score`：已确定邻接实体的重合程度；
- `rerank_score`：可选 BGE-Reranker-v2-M3 相关性分数。

硬性门禁先于打分：

- 实体类型不兼容时不进入候选；
- 系统、构型或版本明确冲突时不进入候选；
- 已知代码不一致时不允许被名称相似覆盖；
- `FaultCause`、`FaultMode`、`Symptom`、`FaultEffect` 不可跨类型合并。

综合分只表示“值得进一步检索”，不表示“同一实体的工程概率”。权重、候选阈值、Top-K、模型名和模型版本必须配置化并写入记录。建议每个提及最多保留 10 个候选。

候选状态：

```text
candidate_semantic
candidate_ambiguous
candidate_conflicted
rejected_by_scope
rejected_by_type
```

在当前没有人工审核的产品条件下，所有纯相似度结果永久停留在候选层。候选不写入正式 HugeGraph，不支持正式多跳结论和数值计算。

### 7.4 别名和冲突

权威别名必须满足：

- 实体类型相同；
- 适用范围相同；
- 一个规范化别名在同一范围内只指向一个活动实体；
- 别名来源可追溯。

若同一别名指向多个实体，则整组别名标记为 `ambiguous`，只用于候选召回。不能选择相似度最高者自动覆盖冲突。

### 7.5 不允许的归一操作

- 仅凭 BGE-M3 分数创建 `same_as`；
- 仅凭名称、缩写或 LLM 判断合并实体；
- 将不同类型实体合并；
- 把候选关系进行传递闭包；
- 因一份新文档的相似描述而重写旧实体 ID；
- 把 `provisional` 图节点合并到正式规范实体；
- 在 HugeGraph 中直接完成实体归一并反写 PostgreSQL。

## 8. 七类文档映射

### 8.1 Function Locator

正式实体：

- `System`、`Subsystem`、`Module`、`Component`；
- `Function`、`Interface`；
- `Document`。

正式关系：

- 明确父 ID 生成 `contains`；
- 明确模块/部件 ID 和功能 ID 生成 `implements`；
- 明确接口端点生成 `exposes`、`connects_to`；
- 明确功能输入输出接口生成 `input_via`、`output_via`。

普通段落只进入 OpenSearch 和实体提及表，不从“上级、负责、连接”等自然语言自动制造正式关系。架构图片遵守图修订分级：确定性表格/表达式可 `auto_validated`；高置信视觉结果最多为 `provisional`。

### 8.2 HMC

正式实体：

- `HMCCode`；
- 文档内 `Symptom` 或 `FaultMode`；
- 文档内 `MaintenanceAction`；
- 只有存在明确模块 ID 时才连接 `Module`。

正式关系：

- `describes_symptom` / `describes_fault_mode`；
- `recommends_action`；
- 有明确模块 ID 时建立 `locates_at` 或受类型约束的等价关系。

不同 HMC 行对同一故障的不同说法保持独立来源。诊断向量仍为“HMC + 故障现象”，维修向量仍为“HMC + 故障现象 + 维修方法”。

### 8.3 FMECA

正式实体：

- `FMECARecord`；
- `FaultMode`、`FaultCause`、`FaultEffect`、`DetectionMethod`、`MaintenanceAction`；
- 产品/功能标志存在明确业务 ID 时连接相应 `Module` 或 `Function`。

若这些字段只有自由文本，则实体身份限制在 `doc_id + 字段类型 + 原文`。同一行字段仍可确定性建立：

```text
FMECARecord → analyzes_fault_mode → FaultMode
FaultMode   → caused_by          → FaultCause
FaultMode   → results_in         → FaultEffect
FaultMode   → detected_by        → DetectionMethod
```

这些是表格列语义直接表达的关系，不需要 LLM 推断。严酷度、概率和任务阶段保留为记录或关系限定属性。

### 8.4 MFL

正式实体：

- `MFLRecord`；
- 文档内 `FaultMode`、`MaintenanceAction`；
- 有明确 ID 的 `Module`、`Component`、`API`。

正式关系：

- `locates_fault_mode`；
- 明确模块/子电路 ID 时建立 `locates_at`；
- `reports_via_api`；
- 明确维修动作建立 `repaired_by`。

“模块故障模式、SUR 板故障模式、功能子电路级故障模式”保持字段级类型和来源，不因文本相似自动归一。

### 8.5 FTA

正式实体和关系来自当前活动修订：

- `FTADiagram`、`FTAEvent`、`FTAGate`；
- `decomposed_by`、`has_input_event`；
- 事件明确引用故障模式代码时建立 `represents_fault_mode`。

允许进入正式图遍历的修订必须同时满足：

```text
lifecycle_status = active
projection_status = published
validation_status in (auto_validated, manual_confirmed)
```

`provisional` 可以独立展示和作为待核验上下文返回，但不能生成正式跨文档路径、唯一诊断或 FTA 数值输入。`quarantined/failed/pending` 不进入推理图。

### 8.6 RBD

正式实体和关系：

- `RBDDiagram`、`RBDBlock`、`RBDJunction`；
- `next_block`；
- 明确模块/功能 ID 时建立 `represents_module`、`implements_function`。

正式状态要求与 FTA 相同。可靠度、故障率、修复率和任务时间保持 PostgreSQL 精度真源；HugeGraph 只支持路径定位，不成为数值真源。

### 8.7 故障历史

正式实体：

- `FaultHistoryCase`；
- `FaultHistoryStep`；
- 文档内 `Symptom`、`FaultCause`、`MaintenanceAction`；
- 源字段明确给出的 HMC/FMECA/FTA/RBD/MFL 引用实体。

正式关系：

- `observed_symptom`；
- 根因字段建立 `confirmed_cause`；
- 最终方案/明确处理动作建立 `resolved_by`；
- `has_diagnosis_step`、`has_step_content`、`next_step`；
- `references_hmc/fmeca/fta/rbd/mfl`。

显式引用只表示“该案例引用了某记录”，不能自动升级为 `caused_by` 或 `repaired_by`。

## 9. PostgreSQL 数据模型

新表采用幂等初始化或正式迁移脚本创建，不修改现有七类领域表的权威含义。

### 9.1 `icp_canonical_entities`

```text
entity_id UUID PRIMARY KEY
ontology_version VARCHAR NOT NULL
entity_type VARCHAR NOT NULL
canonical_key TEXT NOT NULL
identity_hash CHAR(64) NOT NULL
identity_scope VARCHAR NOT NULL
identity_components JSONB NOT NULL
canonical_name TEXT
attributes JSONB NOT NULL DEFAULT '{}'
scope JSONB NOT NULL DEFAULT '{}'
lifecycle_status active | retired | quarantined
created_at / updated_at / retired_at
UNIQUE (ontology_version, entity_type, identity_hash)
```

### 9.2 `icp_entity_mentions`

```text
mention_id UUID PRIMARY KEY
ontology_version VARCHAR NOT NULL
doc_id UUID NOT NULL REFERENCES icp_documents ON DELETE CASCADE
doc_type VARCHAR NOT NULL
entity_type VARCHAR NOT NULL
record_type VARCHAR
record_id TEXT
field_name TEXT
source_ref TEXT NOT NULL
raw_text TEXT NOT NULL
normalized_text TEXT NOT NULL
explicit_business_id TEXT
scope JSONB NOT NULL DEFAULT '{}'
extractor_name / extractor_version
raw_data JSONB NOT NULL DEFAULT '{}'
content_hash CHAR(64) NOT NULL
created_at / updated_at
UNIQUE (doc_id, entity_type, source_ref, content_hash)
```

### 9.3 `icp_entity_resolutions`

只保存正式归一：

```text
mention_id UUID PRIMARY KEY REFERENCES icp_entity_mentions ON DELETE CASCADE
entity_id UUID NOT NULL REFERENCES icp_canonical_entities
ontology_version VARCHAR NOT NULL
resolution_method VARCHAR NOT NULL
resolution_status resolved | quarantined | retired
resolution_basis JSONB NOT NULL
resolver_version VARCHAR NOT NULL
scope_check JSONB NOT NULL
created_at / updated_at
```

数据库约束必须禁止 `candidate_semantic` 写入此表。

### 9.4 `icp_entity_resolution_candidates`

```text
candidate_id UUID PRIMARY KEY
mention_id UUID NOT NULL REFERENCES icp_entity_mentions ON DELETE CASCADE
candidate_entity_id UUID NOT NULL REFERENCES icp_canonical_entities
ontology_version VARCHAR NOT NULL
candidate_status VARCHAR NOT NULL
lexical_score DOUBLE PRECISION
embedding_score DOUBLE PRECISION
context_score DOUBLE PRECISION
structure_score DOUBLE PRECISION
rerank_score DOUBLE PRECISION
combined_score DOUBLE PRECISION
feature_snapshot JSONB NOT NULL
embedding_model / embedding_version
rerank_model / rerank_version
resolver_version VARCHAR NOT NULL
created_at / updated_at
UNIQUE (mention_id, candidate_entity_id, resolver_version)
```

候选表没有触发正式图写入的数据库触发器。

### 9.5 `icp_authoritative_aliases`

```text
alias_id UUID PRIMARY KEY
ontology_version VARCHAR NOT NULL
entity_type VARCHAR NOT NULL
normalized_alias TEXT NOT NULL
scope_hash CHAR(64) NOT NULL
entity_id UUID NOT NULL REFERENCES icp_canonical_entities
source_kind code_registry | explicit_alias_declaration | legacy_confirmed
source_ref TEXT NOT NULL
alias_status active | ambiguous | retired
created_at / updated_at
```

同一 `entity_type + normalized_alias + scope_hash` 若存在多个目标，全部转为 `ambiguous`，不能自动选择。

### 9.6 `icp_relation_assertions`

```text
assertion_id UUID PRIMARY KEY
ontology_version VARCHAR NOT NULL
relation_type VARCHAR NOT NULL
source_entity_id UUID NOT NULL REFERENCES icp_canonical_entities
target_entity_id UUID NOT NULL REFERENCES icp_canonical_entities
doc_id UUID NOT NULL REFERENCES icp_documents ON DELETE CASCADE
doc_type VARCHAR NOT NULL
assertion_method structured_field | explicit_reference |
                 validated_diagram | visual_candidate | legacy_manual
assertion_status formal | provisional | quarantined | retired
scope JSONB NOT NULL DEFAULT '{}'
properties JSONB NOT NULL DEFAULT '{}'
created_at / updated_at / retired_at
UNIQUE (doc_id, relation_type, source_entity_id, target_entity_id)
```

该唯一键严格遵守 `(doc_id, relation, source, target)` 的规范身份。同一文档多行重复支持同一关系时不生成平行断言。

### 9.7 `icp_relation_occurrences`

```text
occurrence_id UUID PRIMARY KEY
assertion_id UUID NOT NULL REFERENCES icp_relation_assertions ON DELETE CASCADE
record_id TEXT
field_name TEXT
source_ref TEXT NOT NULL
qualifier JSONB NOT NULL DEFAULT '{}'
raw_excerpt TEXT
content_hash CHAR(64) NOT NULL
created_at
UNIQUE (assertion_id, source_ref, content_hash)
```

此表完整保留同一关系在多行、多字段的来源，避免 HugeGraph `SINGLE` 边去重导致证据丢失。

### 9.8 `icp_graph_projection_manifests`

```text
projection_id UUID PRIMARY KEY
ontology_version VARCHAR NOT NULL
projector_version VARCHAR NOT NULL
namespace VARCHAR NOT NULL
source_snapshot_hash CHAR(64) NOT NULL
status running | published | failed | stale
expected_vertex_count / expected_edge_count
actual_vertex_count / actual_edge_count
vertex_ids JSONB
edge_ids JSONB
started_at / completed_at
error TEXT
UNIQUE (ontology_version, projector_version, source_snapshot_hash)
```

现有 `icp_structured_graph_projections` 和 `icp_graph_projection_runs` 在迁移阶段继续兼容；新清单不应破坏旧图修订发布记录。

### 9.9 索引

至少建立：

- 规范实体：`(entity_type, identity_hash)`、`canonical_name`；
- 提及：`(doc_id, entity_type)`、`(record_type, record_id)`；
- 候选：`(mention_id, combined_score DESC)`；
- 断言：`source_entity_id`、`target_entity_id`、`relation_type`、`doc_id`；
- 断言范围：必要的 GIN `scope` 索引；
- 关系出现：`assertion_id`。

所有 SQL 查询必须使用参数绑定，不接受用户提供的原始 SQL。

## 10. HugeGraph 投影

### 10.1 命名空间

新正式图使用独立命名空间：

```text
kb_id = icp_ontology_v1
```

旧 `icp_structured` 在迁移和验收期间保留只读，不能原地破坏式重写。

### 10.2 顶点

物理顶点使用 HugeGraph `Entity` label，业务类型放入 `entity_type`，避免在线动态创建大量不兼容 schema。建议属性：

```text
entity_id
ontology_version
entity_type
canonical_name
canonical_key
identity_scope
scope_json
lifecycle_status
kb_id
source_doc_ids
updated_at
```

顶点只投影 `active` 正式实体；`candidate_semantic` 不生成顶点身份合并。

### 10.3 边

PostgreSQL 中跨文档的多个正式断言可能支持同一条物理边。物理边身份为：

```text
(ontology_version, relation_type, source_entity_id, target_entity_id)
```

属性包含：

```text
assertion_ids
doc_ids
source_refs 摘要或引用计数
scope_union
assertion_methods
support_count
projection_hash
```

逻辑断言仍按 `(doc_id, relation, source, target)` 独立保存，物理聚合不能覆盖来源差异。

### 10.4 正式与候选图隔离

- `formal` 断言进入 `icp_ontology_v1`；
- `provisional` 图只保留在 PostgreSQL 图修订和现有浏览投影中，不进入正式命名空间；
- `quarantined/retired` 不投影；
- 相似度候选保存在 PostgreSQL/OpenSearch，不投影 `same_as`；
- 若未来增加候选图层，必须使用独立 namespace、虚线 UI 和不可推理标记。

### 10.5 写入与重建

投影过程：

```text
读取某一 PostgreSQL 一致性快照
→ 计算确定性顶点/边 ID 和 source_snapshot_hash
→ 写入 shadow namespace
→ 按 ID 和数量读回
→ 比较顶点、边、断言 ID 集合和属性哈希
→ 成功后标记 published
→ 失败则保留旧 published 投影并记录错误
```

按文档重解析时，只裁剪该 `doc_id` 对实体和物理边的贡献；仍有其他文档支持的顶点和边不得删除。删除文档后若实体无任何正式来源，则先标记 `retired`，由重建任务清理投影。

## 11. `graph_traversal` 工具契约

### 11.1 工具职责

`graph_traversal` 只做受限、可审计的图遍历：

- 规范实体邻域展开；
- 两个规范实体之间的路径查询；
- 原因链、影响链、维修链；
- FTA 基本事件到顶事件的多层路径；
- RBD START 到 END 的成功路径定位；
- 系统、子系统、模块、功能和接口层级展开。

工具不接受用户编写的 Gremlin，不在查询时生成实体或关系，不执行 FTA/RBD 数值计算。

### 11.2 输入

```json
{
  "schema_version": "1.0",
  "ontology_version": "1.0.0",
  "mode": "cause_chain",
  "start_entity_ids": ["uuid"],
  "target_entity_ids": [],
  "target_entity_types": ["FaultCause", "FTAEvent"],
  "direction": "both",
  "max_hops": 3,
  "relation_allowlist": ["caused_by", "represents_fault_mode", "has_input_event", "decomposed_by"],
  "scope": {
    "system_code": "SYN-ICP-9000",
    "configuration": "CONFIG-A",
    "document_version": "V1.0"
  },
  "formal_only": true,
  "include_evidence": true,
  "max_paths": 20,
  "max_vertices": 100
}
```

约束：

- `start_entity_ids` 必须先由 PostgreSQL 精确查询或实体解析器产生；
- 默认 `max_hops=3`，硬上限为 5；
- `relation_allowlist` 由确定性工具策略按 `mode` 生成，Planner 只能提出建议；
- 默认且正式回答中必须 `formal_only=true`；
- `max_paths`、`max_vertices` 和超时均有后端硬上限；
- 使用白名单模板生成 Gremlin，禁止拼接用户原始关系名或语句。

### 11.3 查询模式

| mode | 允许的典型关系 |
|---|---|
| `neighborhood` | 按实体类型生成受限一至三跳关系集合 |
| `architecture` | `contains`、`implements`、`exposes`、`connects_to` |
| `cause_chain` | `has_fault_mode`、`caused_by`、FTA 逻辑门关系 |
| `impact_chain` | `results_in`、`affects`、`manifests_as` |
| `maintenance_chain` | HMC/MFL/历史记录关系、`repaired_by`、`resolved_by` |
| `history_chain` | 显式代码引用、步骤和处置关系 |
| `fta_ancestors` | `decomposed_by`、`has_input_event` 的受控反向遍历 |
| `rbd_paths` | `next_block`，只返回正式发布修订 |
| `shortest_path` | 两个已解析实体间的白名单最短路径 |

### 11.4 输出

```json
{
  "schema_version": "1.0",
  "tool": "graph_traversal",
  "status": "ok",
  "backend": "hugegraph",
  "ontology_version": "1.0.0",
  "paths": [
    {
      "path_id": "stable-hash",
      "hop_count": 3,
      "path_authority": "formal",
      "vertices": [
        {
          "entity_id": "uuid",
          "entity_type": "FaultMode",
          "name": "输入滤波性能退化",
          "scope": {}
        }
      ],
      "edges": [
        {
          "relation": "caused_by",
          "source_entity_id": "uuid",
          "target_entity_id": "uuid",
          "assertion_ids": ["uuid"],
          "evidence_ids": ["ev-1"]
        }
      ],
      "limitations": []
    }
  ],
  "evidence": [],
  "diagnostics": {
    "visited_vertices": 12,
    "truncated": false,
    "warnings": []
  }
}
```

不得把 HugeGraph 的原始 JSON/Gremlin 响应直接展示给用户。

### 11.5 PostgreSQL 回查

HugeGraph 返回路径后必须批量回查：

1. 每个顶点对应活动的 `icp_canonical_entities`；
2. 每条边至少对应一条活动 `formal` 断言；
3. 每条断言存在来源 occurrence 和可访问的 `source_ref`；
4. 查询范围与实体/断言适用范围不冲突；
5. FTA/RBD 修订仍是 active、published、auto_validated/manual_confirmed；
6. 本体版本和投影清单仍匹配。

任何一跳回查失败，该路径从正式结果中删除；部分路径仍有效时返回 `partial`，全部失效时返回 `empty` 或 `blocked`，不能把过期投影当事实。

### 11.6 HugeGraph 不可用

默认返回 `failed` 并明确标注图遍历不可用。只有在已经实现并通过同等回归测试的 PostgreSQL `WITH RECURSIVE` 有界回退后，才允许返回：

```text
backend = postgres_fallback
```

不能把内存 Mock 或单库关键词检索声称为多跳图查询成功。

## 12. 智能体调用流程

```text
用户问题
→ scope_router 判定产品路径
  ├→ agent_capability / out_of_scope / clarification
  │    → direct_response（不调用知识工具，不产生证据）→ END
  └→ knowledge_base
       → plan 整理当前对象、本轮目标、必需字段和最小工具计划
          ├→ 仍需澄清时 direct_response → END
          └→ icp_diagnosis 执行策略授权后的首批定向查询
               → PostgreSQL/OpenSearch 返回结构化记录和检索证据
               → 身份层绑定正式业务锚点，语义相似只保留候选锚点
               → adaptive_tools 检查目标字段缺口
                  ├→ adaptive_retrieval（最多一批已授权补查）
                  ├→ graph_traversal（多跳有价值且已绑定锚点）
                  └→ icp_calculation（仅用户原始问题明确授权的确定性计算）
               → PostgreSQL 逐节点、逐边重新授权
               → evidence_gate → answer → END
```

相似度候选只负责“可能从哪个实体开始走”。从候选起点走出的边仍必须全部是正式关系，答案必须表达为：

```text
若你描述的“输入滤波单元”指的是模块 ICU，则文档关系显示……
```

不能省略入口实体尚未确定的限制。

当前 LangGraph 已实现为：

```text
scope_router
  ├→ direct_response → END
  └→ plan
       ├→ direct_response → END
       └→ icp_diagnosis
            → adaptive_tools
               ├→ adaptive_retrieval
               │    ├→ graph_traversal → icp_calculation
               │    └→ icp_calculation
               ├→ graph_traversal → icp_calculation
               └→ icp_calculation
                    → evidence_gate → answer → END
```

`scope_router` 只分流，不规划工具也不回答工程问题。`direct_response`
统一处理智能体能力说明、范围外问题和无法消解的指代；该分支清空工具、
证据和引用结果，不得降级为七类资料宽召回。只有 `knowledge_base` 路径进入
`plan`；`plan` 若仍判定缺少必要对象，也可再路由到 `direct_response`。

Planner 的 `tools_plan` 经授权校验后就是首批实际执行集，不再被意图固定
套餐覆盖。确定性强制只保留结构化 HMC 精确查询、用户原话明确授权
的 FTA/RBD 计算，以及规划失败/空计划时的最小安全召回。

图查询是系统完成回答的内部手段，不是必须由用户明确提出的信息目标。Planner
通过 `relation_expansion` 判断完整答案是否需要跨文档关系补全；用户不需要说出
“关系、路径、关联、图谱”或任何数据库术语。原因分析、模糊故障定位和架构定位
默认具有图扩展价值；影响、维修、历史和可靠性问题在需要串联多个来源时也会自动
触发。实际遍历仍必须等直接查询绑定业务实体后才执行。

以下语义默认需要图遍历；自然问题不必采用这些固定说法：

- “这个故障可能由哪些原因逐层导致”；
- “会向哪些模块和功能传播”；
- “从现象到维修定位的关系是什么”；
- “某基本事件会通向哪些顶事件”；
- “某模块位于哪些 RBD 成功路径”；
- “某 HMC、FMECA、MFL 和历史案例之间有什么明确关联”。

简单精确 HMC 查询、单字段 FMECA 解释不必强行查询图。

## 13. 证据门禁

### 13.1 可批准为正式路径的条件

路径中的每一跳必须同时满足：

- 实体状态为 `active`；
- 实体归一方法为 exact ID、explicit reference、authoritative alias 或兼容 manual confirmed；
- 关系断言状态为 `formal`；
- 来源文档和 `source_ref` 可访问；
- 适用系统、构型、文档版本和有效期不冲突；
- 图修订符合正式状态要求；
- PostgreSQL 回查与 HugeGraph 投影一致。

### 13.2 候选和 provisional 的处理

- 语义候选锚点：路径等级最高为 `L3_INFERRED`，必须说明入口未确认；
- `provisional` 图：仅作 `unconfirmed_extraction` 上下文，不进入正式路径；
- `quarantined`：完全不返回给回答模型；
- OpenSearch 命中：回填 PostgreSQL 后才可升级为权威事实；
- 检索分数和相似度不能称为诊断概率。

### 13.3 路径证据

每条边至少生成一个 Evidence：

```text
evidence_id
source_kind = hugegraph（路径定位） + postgres（权威回查）
authority = confirmed_relation | validated_relation
level = L1_DIRECT | L2_CORROBORATED | L3_INFERRED
doc_id / doc_type / record_id / business_id
source_ref
assertion_id
path_id / hop_index
scope
limitations
derived_from
```

回答引用的 `[1]`、`[2]` 必须能定位到具体段落、表格行或图片，而不能只显示“FTA结构化关系证据”。路径摘要可作为一个组合证据，但必须列出各跳的子证据。

### 13.4 阻断条件

以下情况禁止输出唯一诊断或正式因果链：

- 路径依赖候选 `possible_same_as`；
- 路径任一跳缺少来源断言；
- 构型或版本冲突；
- 视觉图只有 provisional；
- HugeGraph 与 PostgreSQL 不一致；
- 起点只有多个相似候选且无法区分；
- 唯一依据是向量 Top-1。

## 14. 迁移与重建方案

迁移必须可回滚、幂等且不删除用户现有数据。

### 阶段 1：契约和表结构

1. 固化本体 `1.0.0`、类型约束和关系白名单；
2. 幂等创建身份层和关系断言表；
3. 增加规范 ID、范围归一和关系断言单元测试；
4. 保持现有解析、检索和图页面继续工作。

### 阶段 2：七类回填

按 `doc_id` 从现有 PostgreSQL 权威表回填：

```text
领域记录
→ EntityMention
→ 确定性 CanonicalEntity/Resolution
→ RelationAssertion/Occurrence
```

FTA/RBD/架构图只读取活动修订，并严格按验证状态分类。每个文档保存回填水位、来源哈希、解析器版本和本体版本；相同输入重复执行不得产生重复实体或断言。

### 阶段 3：语义候选

1. 为未正式归一的提及写入 OpenSearch 候选索引；
2. 使用本地 BGE-M3 生成候选；
3. 可选使用 BGE-Reranker-v2-M3 重排；
4. 结果只写 `icp_entity_resolution_candidates`；
5. 验证不存在由相似度触发的正式实体合并。

该候选索引是内部检索投影，不是新的用户知识文档类型。

### 阶段 4：Shadow HugeGraph 投影

1. 在 `icp_ontology_v1` 构建全量 shadow 投影；
2. 按 ID、数量、断言集合和哈希读回校验；
3. 随机抽样多跳路径并回 PostgreSQL 复核；
4. 旧 `icp_structured` 保持只读可恢复；
5. 验收完成后通过配置切换读取 namespace。

### 阶段 5：智能体接入

1. 增加 `graph_traversal` 工具和统一 ToolResult/Evidence；
2. 在工具策略中按意图开放，不允许 Planner 任意遍历；
3. 接入 PostgreSQL 路径回查和 Evidence Gate；
4. 前端执行过程展示查询模式、起点、跳数和命中路径数；
5. 右侧证据展示逐跳原文/表格/图片定位。

### 阶段 6：旧投影退役

只有在下列条件满足后才允许退役旧命名空间：

- 新投影全量读回一致；
- 七类抽样和自动化测试通过；
- 智能体多跳引用可追溯；
- 已保留投影清单和可重复重建命令；
- 用户明确批准清理旧图投影。

退役只删除可重建的 HugeGraph 投影，不删除 PostgreSQL 权威记录和 ArtifactStore 原件。

## 15. 当前实现落点

本设计已落在下列现有模块，不再以尚未存在的
`ontology.py`、`graph_repository.py` 或 `ontology_projection.py` 作为当前实现路径：

```text
backend/icp/entity_resolution.py
  实体/关系类型白名单、确定性业务身份、文档内文本身份，
  以及规范实体、提及、语义候选、正式关系断言的 PostgreSQL 仓储

backend/icp/entity_candidates.py
  字面/BGE-M3/上下文候选发现；结果只写候选层，不晋升正式归一或关系

backend/icp/structured_graph.py
  从七类结构化行构建确定性节点/边和稳定 ID；
  区分普通结构化文档与正式 FTA/RBD 图修订，本模块不直接写 HugeGraph

backend/icp/structured_projection.py
  按 doc_id 读取 PostgreSQL 权威行，调用 structured_graph 和
  entity_resolution.sync_structured_graph，执行 HugeGraph 文档级幂等投影、状态记录和读回校验

backend/icp/graph_authority.py
  将 HugeGraph 返回的关系逐边回查 PostgreSQL 断言、文档状态和
  FTA/RBD 当前正式修订；缺失、过期、歧义或后端错误均 fail closed

backend/storage/hugegraph_client.py
  HugeGraph 底层读写客户端；只操作可重建投影，不承担业务权威判定

backend/agent/tools/graph_traversal.py
  受限图遍历、深度/节点/边/路径硬上限、投影来源保留和 ToolResult/Evidence；
  通过 graph_authority 回调重新授权，不接受用户 Gremlin

backend/agent/tool_policy.py
  确定性限定 Planner 的工具授权上限和图查询前置条件

backend/agent/nodes/scope_router.py
backend/agent/nodes/direct_response.py
  完成知识库/能力说明/范围外/澄清分流，非知识分支不调用工具或证据

backend/agent/nodes/plan.py
backend/agent/nodes/adaptive_tools.py
  产生首批最小计划，并在必需字段缺失时最多安排一批受限补查或按需查图

backend/agent/nodes/evidence_gate.py
  对结构化记录和多跳路径证据执行回答前门禁

backend/agent/graph.py
  编排 scope_router/direct_response、plan、首批查询、最多一批自适应补查、
  按需 graph_traversal、确定性计算、evidence_gate 和 answer 的条件边
```

`structured_graph.py` 和 `structured_projection.py` 是当前七类明确关系投影主链路，
不是仅供迁移的旧兼容实现。PostgreSQL 仍是权威真源；HugeGraph 中任何用于
回答的关系都必须经 `graph_authority.py` 当轮重新授权。新 ICP 领域 SQL
应继续收敛在 `backend/icp` 下的专用模块，避免无边界扩张
`backend/storage/postgres_client.py`。

## 16. 验收标准

### 16.1 实体身份

1. 相同系统、构型和业务代码在不同文档中生成相同 `entity_id`；
2. 不同构型的同名模块不会合并；
3. HSC `0001` 不会变成 `1`；
4. HAMC `**` 不会丢失或被编造成具体值；
5. 不同实体类型的相似文本不会合并；
6. 无业务代码文本不会跨 `doc_id` 合并；
7. 重复解析不增加重复实体、断言或 occurrence。

### 16.2 关系

1. 每条正式关系都可回到至少一个 `source_ref`；
2. 同一文档相同 `(relation, source, target)` 只存在一条断言；
3. 多行重复来源完整保存在 occurrence；
4. 删除一份文档不会删除其他文档仍支持的实体和物理边；
5. 普通段落、相似度和 LLM 不会创建正式 `caused_by`；
6. 历史显式引用不会被错误升级为因果关系。

### 16.3 FTA/RBD

1. FTA 支持超过三层的事件/逻辑门遍历；
2. AND/OR/XOR 门不会被压平成普通因果边；
3. RBD 可以沿 START 到 END 返回串并联路径；
4. `provisional` 不进入正式路径和数值计算；
5. 计算仍使用 PostgreSQL 正式 `canonical_graph` 和确定性 Python 算法。

### 16.4 多跳查询

1. 默认 1～3 跳、硬上限 5 跳生效；
2. 循环图不会造成无限遍历；
3. 路径、顶点和超时上限生效并返回 `truncated`；
4. 每条路径的每一跳都有 assertion 和 Evidence；
5. HugeGraph 脏边或过期边会被 PostgreSQL 回查剔除；
6. 候选起点在答案中明确标记“入口未确认”；
7. 不接受原始 Gremlin 注入。

### 16.5 端到端

至少覆盖以下问题：

```text
某故障现象影响哪些功能和模块？
某故障模式的明确原因和最终影响是什么？
某 HMC 对应哪些 FMECA、MFL 和历史案例？
某 FTA 基本事件经过哪些逻辑门到达顶事件？
某模块位于哪些 RBD 成功路径？
某维修动作有哪些直接来源，是否存在历史验证？
```

答案必须逐跳引用具体原文、表格行或图片定位；没有正式跨文档连接时应明确返回“当前只有分散证据，没有可证明的跨文档路径”，不能由 LLM 补齐。

### 16.6 项目回归

实现完成后必须通过：

```bash
python -m compileall -q backend
python -m unittest discover -s tests -p 'test_*.py' -v
cd frontend && npm ci && npm run build
git diff --check
```

并确认：

- Dify 两个接口路径仍存在；
- 首页仍是统一上传入口；
- 用户侧仍然只有七类文档；
- `/gallery` 没有恢复；
- `/images` 工程图证据仍可访问；
- HugeGraph 不可用时不会把 Mock 或单库搜索冒充多跳成功；
- 无证据候选不会被输出为唯一诊断。

## 17. 最终原则

本方案的核心不是“把尽可能多的内容连成图”，而是：

> 用相似度找到可能的起点，用明确业务 ID 和来源字段建立正式身份，用有证据的关系完成多跳遍历，再由 PostgreSQL 对路径逐跳验真。

因此，正式图可以相对稀疏，但每条边必须可信、可复核、可重建。语义候选可以丰富，但必须与正式身份和正式关系永久隔离。只有坚持这一边界，HugeGraph 才能从展示层升级为真正可用于智能体诊断的多跳关系查询层，而不会成为错误关系的放大器。
