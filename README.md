# ICP-RAG：ICP 故障诊断知识库与智能体

ICP-RAG 是面向 ICP 航空电子系统的专用故障知识平台。系统接收七类 Word、Excel 和工程图文档，按文档类型执行差异化预处理，将结构化事实、检索投影、图关系和原始证据统一提供给故障诊断智能体。

> **安全说明**：系统输出属于知识检索与辅助诊断结果，不替代适用构型、有效版本的正式维修手册、放行程序和人工复核。合成测试数据不得作为实际航空维修依据。

## 1. 七类知识文档

| 文档类型 | 主要内容 | 主存储与处理方式 |
|---|---|---|
| 系统架构与功能说明（Function Locator） | 系统、子系统、模块、功能、接口、典型失效表现 | PostgreSQL 功能结构 + OpenSearch + 显式组件/功能关系投影；视觉图仅作 provisional 候选 |
| HMC | LCN、HAMC、HSC、故障现象、分析/维修方法 | PostgreSQL 权威记录 + 诊断/维修双向量索引 + 代码到现象/方法显式关系 |
| FMECA | 故障模式、原因、影响、严酷度、检测及补偿措施 | PostgreSQL 宽表记录 + OpenSearch + 记录字段显式关系 |
| FTA | 顶事件、中间事件、基本事件、AND/OR 逻辑 | PostgreSQL 节点边 JSON + OpenSearch + HugeGraph 投影 |
| RBD | 功能块、串并联/冗余结构、成功判据和可靠性参数 | PostgreSQL 节点边 JSON + OpenSearch + HugeGraph 投影 + 确定性可靠度计算 |
| MFL | 模块/SUR 板/功能电路故障、API 上报和维修定位 | PostgreSQL 权威记录 + OpenSearch + MFL/API 显式关系 |
| 故障历史 | 发生时间、现象、排故方法、根因、步骤、方案和关联知识 | PostgreSQL 案例/步骤 + OpenSearch + 步骤和显式代码引用关系 |

## 2. 系统架构

```text
Word / Excel / 图片
        │
        ├─ 七类专用解析器
        │    ├─ 多级表头与表格记录
        │    ├─ 章节/段落/案例结构
        │    └─ FTA、RBD、系统架构图像理解
        │
        ├──────────────┬────────────────┬────────────────┐
        ▼              ▼                ▼                ▼
 PostgreSQL         OpenSearch     ArtifactStore      HugeGraph
 权威结构数据       BM25 + 向量索引    LocalFS 原文/证据   图关系投影
        └──────────────┴────────────────┴────────────────┘
                               │
                               ▼
                ICP 故障诊断 LangGraph
 范围路由 → 对话对象/信息目标 → 受限工具规划 → 多源检索 → 证据门禁 → LLM整理答案
```

数据库职责：

- PostgreSQL 是权威事实来源；
- OpenSearch 在同一版本化索引中负责 CJK BM25、精确过滤和 BGE-M3 向量召回；
- ArtifactStore + LocalFS 保存原始文档及图片证据，宿主机持久化目录为 `volumes/artifacts`；
- HugeGraph 保存七类文档的可重建显式关系投影：每份文档都有 Document 来源节点，五类结构化文档按明确字段/代码/步骤连边，架构/FTA/RBD 按自动校验等级投影；
- 向量和图谱均为可重建派生数据。

解析完成后系统不再等待人工确认。确定性表格或表达式拓扑通过类型约束后标记为 `auto_validated`；高置信视觉结果只标记为 `provisional`，可浏览但不能用于正式拓扑结论或可靠性计算；冲突、残缺和低置信结果进入 `quarantined`。跨文档实体只按完全一致的业务 ID/代码合并，普通文字不做相似度合并，也不由 LLM 补造关系。

结构化边以 `(doc_id, relation, source, target)` 为规范身份；同一文档内相同端点和关系的多次字段/行来源合并为一条边，来源限定保存在 `occurrence_qualifiers`。故障历史中的每个步骤使用独立 `FaultHistoryStep` 实例，通过 `has_diagnosis_step`、`has_step_content` 和 `next_step` 分别表达案例归属、步骤原文与先后顺序。

运行栈固定为 PostgreSQL 15、OpenSearch、ArtifactStore + LocalFS、HugeGraph、
Ollama + Qwen3.8 27B、BGE-M3 和 BGE-Reranker-v2-M3。当前方案不部署独立向量库：
OpenSearch 同时承担全文和 1024 维向量检索。

## 3. 诊断流程

用户无需知道具体模块或 HMC，只需描述现象、发生阶段、原始告警及已执行操作。默认流程如下：

```text
用户问题
  → 区分工程知识、能力说明、范围外与必要澄清
  → 结合近期对话解析“当前对象 + 本轮所问维度”
  → 上轮实际引用记录只作为有界锚点，本轮回 PostgreSQL 重新核验
  → Function Locator 定位系统/模块/功能
  → OpenSearch 分别执行 BM25 与 BGE-M3 向量召回
  → 加权 RRF 融合、正文去重、BGE-Reranker-v2-M3 全候选重排
  → HMC/FMECA/FTA/RBD/MFL/历史结构化校验
  → 候选故障排序和证据支持度判断（不是工程概率）
  → 信息不足时提出区分度较高的补充问题
  → 输出带来源的诊断与检查建议
```

有完整 `LCN + HAMC + HSC` 时，系统优先使用 PostgreSQL 精确查询，不以向量近似匹配代替代码匹配。

## 4. ICP 专用工具

- `function_locator`：功能与模块定位；
- `hybrid_search`：七类知识混合召回；
- `hmc_query`：HMC 精确/条件查询；
- `fmeca_query`：原因、影响、严酷度和检测方法；
- `fta_query` / `fta_analyzer`：原因链、最小割集和顶事件概率；
- `rbd_query` / `rbd_calculator`：成功路径与系统可靠度；
- `mfl_query`：维修清单和 API；
- `fault_history_query`：历史排故经验；
- `graph_traversal`：在已绑定业务 ID 且跨文档多跳有信息增益时查询关系，返回边逐条回 PostgreSQL 再授权；
- `fault_reasoning`：多源证据融合、候选排序和追问判断。

FTA/RBD 数值由确定性 Python 算法计算。缺少概率、故障率、可靠度或任务时间时，系统明确报告缺失参数，不生成虚构数值。

FTA 最小割集只依赖完整逻辑结构，不依赖事件概率。当前确定性算法支持 AND/OR；XOR 仍可查询其显式结构，但由于尚无带否定事件的割集表示和明确的多输入约定，计算会明确拒绝，不按 OR 替代。`provisional` 图仍不得用于正式计算。

候选相关性权重可用 `ICP_CANDIDATE_RANKING_WEIGHTS` JSON 显式覆盖（见 `.env.example`）；默认权重不变，实际权重记录在 `evidence_synthesis.params.ranking_profile`。权重不是故障概率，也不能改变精确身份、权威来源优先级、证据门禁或计算授权。

## 5. 前端页面

- **七类文档知识库**：上传、解析、查看原文/向量/结构化记录，以及自动入图状态、隔离原因和图投影；
- **ICP 故障诊断**：会话式诊断、实时工具调用、候选证据和证据支持度；
- **系统设置**：本地 Qwen3.8 27B、BGE-M3 和 BGE-Reranker-v2-M3 的角色与端点配置；原 Qwen3.6 保留，可在模型设置中回切。

独立“图片画廊”和旧通用知识库页面已从产品入口移除。图片抽取与静态服务仍保留，用于 FTA、RBD、系统架构原图及证据查看。

## 6. Dify 外部知识库兼容

Dify 接口保持不变：

```text
GET  /api/v1/dify/server-info
POST /api/v1/dify/retrieval
```

Dify 中的外部知识库地址填写：

```text
http://<host>:19999/api/v1/dify
```

当前 Dify 接口继续兼容原有 `knowledge_bases` 检索链路，七类 ICP 文档链路独立演进，避免破坏已有外部调用。

## 7. 启动方式

### 7.1 开发环境

启动基础设施：

```bash
./start.sh
```

默认端口：

| 服务 | 宿主机端口 |
|---|---:|
| PostgreSQL | 5433 |
| OpenSearch | 9201 |
| HugeGraph / Hubble | 8082 / 18089 |
| FastAPI | 19999（源码热启动）或 19999→容器 9999 |

安装后端并热启动：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./start_server.sh
```

启动前端开发服务器：

```bash
cd frontend
npm ci
npm run dev
```

模型身份和端点在“系统设置”页面写入 `llm_models` 表，不再依赖 `.env` 指定模型。
Embedding 必须通过 Ollama `/api/embed` 返回 1024 维 BGE-M3 向量；rerank 服务须兼容
`/v1/rerank`、`/api/rerank` 或 `/rerank` 之一。应用运行在容器内时，模型地址必须是
容器可达的局域网地址，不能填写宿主机语义下的 `localhost`。

### 7.2 Docker 完整构建

Dockerfile 使用 Node 多阶段构建，自动编译当前 React 源码：

```bash
cp .env.example .env  # 可选；生产环境应修改数据库密码和 APP_UID/APP_GID
mkdir -p volumes/artifacts/images
docker compose -p icp up -d --build
```

也可只重建应用：

```bash
./restart-app.sh
```

### 7.3 历史迁移说明

项目早期使用 MinIO 存储对象，已通过官方 API 完成向 ArtifactStore(LocalFS) 的整体迁移，
迁移工具（`migrate_minio_to_localfs.py`、`requirements-migration.txt`）已随迁移完成移除。
如需了解当时的迁移方案与回滚设计，见
[`docs/ArtifactStore本地文件存储方案.md`](docs/ArtifactStore本地文件存储方案.md)。

## 8. 测试

后端语法与确定性可靠性测试：

```bash
python -m compileall -q backend
python -m unittest discover -s tests -p 'test_*.py' -v
```

前端构建：

```bash
cd frontend
npm ci
npm run build
```

GitHub Actions 会执行以上检查，并额外确认 Dify 路由仍被挂载。

## 9. 主要目录

```text
backend/
  agent/                 LangGraph、会话和 ICP 专用工具
  api/routers/           文档、模型、Dify、ICP 领域 API
  core/parsers/          七类文档解析器
  icp/                   领域仓储、图像解析和可靠性算法
  storage/               PostgreSQL、OpenSearch、ArtifactStore、HugeGraph 客户端
frontend/src/
  pages/Diagnosis.tsx    ICP 故障诊断工作台
  pages/Type*.tsx        七类文档管理
  api/                   前端 API 客户端
tests/
  test_icp_reliability.py
```

## 10. 当前边界

- 图像解析质量取决于所配置 Vision/Chat 模型和原图清晰度；
- 视觉抽取的 FTA/RBD/架构图始终是 provisional 或 quarantined；只有确定性来源通过类型约束形成的 auto_validated 图可用于正式结构分析和确定性计算；
- 不同机型、构型、软件版本的知识必须通过文档版本和适用范围隔离；
- 对真实维修流程的集成测试需要可访问的 PostgreSQL、OpenSearch、持久化 LocalFS、HugeGraph 和模型服务。

## 11. 当前维护文档

- [智能体架构整改方案](docs/ICP智能体架构整改方案.md)：按因果对照、证据/任务职责、发布链路替换和全量验收推进；不是已完成能力声明；
- [部署指南](docs/部署指南.md)：本地热启动、Docker、备份、恢复和迁移验收；
- [开源本地化完整实施方案](docs/开源本地化完整实施方案.md)：数据分层、检索、投影、回滚及验收；
- [七类文档解析与证据智能体详细设计](docs/ICP七类文档解析与证据智能体详细设计.md)：解析、工具、工作流与证据门禁；
- [实体归一与多跳图查询设计](docs/ICP实体归一与多跳图查询设计.md)：实体身份、关系断言、图投影和多跳查询；
- [ArtifactStore 本地文件存储方案](docs/ArtifactStore本地文件存储方案.md)：文件契约、一致性审计及旧 MinIO 迁移。

上述文档描述当前实现或仍受支持的迁移流程；已结束的通用知识库、旧 ES/MinIO 主运行时和人工入图方案不再保留为活跃说明。
