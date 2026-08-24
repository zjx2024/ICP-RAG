# AGENTS.md — ICP-RAG 本地 Codex 项目说明

本文件是本仓库的**编码智能体工作说明**。本地 Codex 在修改代码前应先阅读本文件，再阅读与任务直接相关的源码和 `README.md`。

---

## 1. 项目定位

ICP-RAG 是面向 **ICP 航空电子系统故障诊断**的专用知识库与智能体项目，不再是通用知识库搭建平台。

系统接收七类原始工程文档，按文档类型执行不同的预处理，将结果保存为：

- PostgreSQL 中的权威结构化事实；
- Elasticsearch 中的关键词和向量检索索引；
- MinIO 中的原始文档与图片证据；
- HugeGraph 中按需生成的图关系投影；
- LangGraph 智能体可调用的领域工具。

最终目标是让用户只描述大概故障现象，系统就能完成：

```text
现象理解
→ 功能/模块定位
→ 七类文档混合检索
→ HMC/FMECA/FTA/RBD/MFL/历史证据校验
→ 候选故障排序
→ 必要时追问
→ 输出有来源、可复核的辅助诊断结果
```

> 诊断输出仅用于知识检索和辅助分析，不替代适用构型、有效版本的正式维修手册、放行程序和人工复核。

---

## 2. 当前开发分支与协作规则

当前专用化改造分支：

```text
refactor/icp-diagnosis-specialization
```

对应 Draft PR：

```text
https://github.com/zjx2024/ICP-RAG/pull/1
```

工作规则：

1. 不要直接向 `main` 强推或自动合并。
2. 修改前先确认本地位于上述分支，或从该分支再创建更小的任务分支。
3. 每次改动应尽量聚焦一个问题，避免顺手重写无关模块。
4. 修改后必须运行后端测试和前端生产构建。
5. 不要声称完成真实环境联调，除非 PostgreSQL、ES、MinIO、HugeGraph 和模型服务均实际可用且测试通过。

---

## 3. 不可破坏的产品约束

以下约束属于当前项目的核心产品边界。

### 3.1 知识文档必须严格保持七类

项目只面向以下七类原始文档：

| `doc_type` | 中文名称 | 主要形态 |
|---|---|---|
| `function_locator` | 系统架构与功能说明 | 架构图、功能图、段落、少量表格 |
| `hmc` | HMC 故障代码手册 | 大量表格、少量说明文字 |
| `fta` | FTA 故障树分析 | 故障树图、节点说明、分析段落 |
| `rbd` | RBD 可靠性框图分析 | 框图、成功判据、参数和计算说明 |
| `fmeca` | FMECA 故障模式影响及危害性分析 | 大量宽表 |
| `mfl` | MFL 维修故障清单 | 大量表格、API 和维修定位信息 |
| `fault_history` | 故障历史查询文档 | 案例段落、案例表格、处理步骤 |

不要新增“数据字典”“追溯矩阵”“术语词典”“评测问题集”等第八类产品文档入口。它们可以作为内部开发资料存在，但不能成为用户侧知识库卡片。

七类注册表位于：

```text
backend/icp/document_types.py
```

### 3.2 上传只能有一个统一入口

首页是唯一用户上传入口：

```text
七类文档知识库首页
→ 点击“上传文档”
→ 下拉选择文档类型
→ 选择或拖拽文件
→ 按 doc_type 上传
→ 自动刷新并显示在对应类型卡片和文档列表
```

关键前端文件：

```text
frontend/src/pages/TypeHome.tsx
frontend/src/components/UnifiedUploadModal.tsx
frontend/src/api/icp.ts
```

各类型页面不应重新出现独立上传控件。类型页面只负责：

- 文档管理；
- 解析；
- 内容查看；
- 日志；
- FTA/RBD 图结构确认和可视化。

旧 `/type/:docType/upload` 路径仅作为兼容跳转，不应恢复上传页面。

### 3.3 不得恢复独立图片画廊

已经移除的产品功能：

```text
图片画廊导航
Gallery 页面
全局图片列表页面
前端 images API
/gallery 独立页面
```

但是下面这些图片能力必须保留：

- Word 内嵌图片抽取；
- FTA、RBD、系统架构图视觉解析；
- MinIO 图片保存；
- `/images` 证据静态访问；
- 文档详情中的原图证据；
- 智能体引用相关工程图。

原则：

> 删除“图片集中浏览产品”，保留“工程图解析与证据链能力”。

### 3.4 Dify 接口必须保留

不得删除或修改现有接口路径：

```text
GET  /api/v1/dify/server-info
POST /api/v1/dify/retrieval
```

关键文件：

```text
backend/api/routers/dify.py
backend/api/__init__.py
```

当前 Dify 仍兼容旧 `knowledge_bases` 检索链路。因此，即使通用知识库前端已删除，不能在未完成迁移和兼容验证前粗暴删除所有通用后端文档/检索代码。

### 3.5 PostgreSQL 是权威事实来源

数据库职责：

- PostgreSQL：正式结构化知识和业务状态；
- Elasticsearch：BM25、向量召回和检索副本；
- MinIO：原始 Word、Excel、图片证据；
- HugeGraph：图关系投影，不是唯一事实来源。

向量、图谱和标准化文本均应可由 PostgreSQL 和原始文件重建。

### 3.6 不允许无证据的唯一诊断

用户通常只会描述模糊现象。系统在证据不足时必须：

- 返回多个候选；
- 说明匹配依据；
- 区分“资料明确记载”“跨文档关联推断”“仍需确认”；
- 提出能区分候选的补充问题；
- 不得把向量 Top-1 当成唯一故障结论。

有完整 HMC 或 `LCN + HAMC + HSC` 时，优先精确查询 PostgreSQL，不以向量近似替代代码匹配。

---

## 4. 七类文档处理方式

### 4.1 Function Locator / 系统架构与功能说明

目标结构：

```text
系统 → 子系统 → 模块 → 功能 → 接口/输入输出 → 典型失效表现
```

主要表：

```text
icp_functions
```

主要解析文件：

```text
backend/core/parsers/function_architecture.py
backend/core/parsers/function_architecture_indexed.py
backend/icp/image_analysis.py
backend/icp/repository.py
```

Function Locator 应返回标准化候选，如：

```json
{
  "function_id": "...",
  "system_name": "...",
  "module_name": "...",
  "function_name": "...",
  "confidence": 0.82,
  "source_ref": "..."
}
```

不能只返回一段没有 ID 和来源的自由文本。

### 4.2 HMC

真实 HMC 可能使用多级合并表头：

```text
HMC代码
├─ LCN
├─ HAMC
└─ HSC
```

解析要求：

- 支持 Word/Excel；
- 支持多级表头和跨页重复表头；
- LCN、HAMC、HSC 按字符串保存；
- 保留 HSC 前导零；
- 保留 `**` 通配语义；
- 原表没有完整 HMC 列时，由 LCN/HAMC/HSC 生成规范代码；
- 故障现象与维修方法分开处理。

当前向量设计：

```text
诊断向量：HMC + 故障现象
维修向量：HMC + 故障现象 + 维修方法
```

不要重新把大量重复的“更换某模块”与症状完全混成唯一诊断向量。

主要文件：

```text
backend/core/parsers/excel_row.py
backend/storage/postgres_client.py
```

### 4.3 FMECA

需要提取并保留：

```text
产品/功能标志
功能
故障模式
故障原因
任务阶段与工作方式
局部影响
最终影响
严酷度
故障概率
检测方法
设计改进措施
使用补偿措施
```

真实文档可能是横向宽表、多级表头、单元格长文本和部分空值。

不得只把整张表转换成一个大文本块；一条故障模式应形成可检索、可回溯的独立记录。

### 4.4 MFL

需要提取：

```text
MFL代码
模块故障模式
SUR板故障模式
功能子电路级故障模式
API是否可直接上报
API名称
API调用方法
API使用方法
平台级摸底实验状态
```

布尔值需要归一化，例如：

```text
是 / 支持 / √ / Y / true → true
否 / N / false           → false
```

### 4.5 FTA

FTA 结构语义：

```text
顶事件
中间事件
基本事件
AND / OR / XOR 门
事件与逻辑门连接
概率或故障率参数
```

主要文件：

```text
backend/core/parsers/diagram.py
backend/icp/image_analysis.py
backend/icp/reliability.py
backend/agent/tools/icp_calculation.py
```

FTA 数值和最小割集必须由确定性 Python 算法完成，不得由 LLM 猜测。

自动视觉抽取的图结构应视为“待确认数据”；真实工程使用前需要人工复核节点、边和逻辑门。

### 4.6 RBD

RBD 结构语义：

```text
START / END
功能块
串联路径
并联分支
冗余结构
旁路
成功判据
可靠度、故障率、修复率和任务时间
```

RBD 与 FTA 不能继续共用同一种故障树语义。

可靠度由：

```text
backend/icp/reliability.py
```

确定性计算。缺少模块可靠度、故障率或任务时间时，应明确返回缺失参数，而不是生成数值。

### 4.7 故障历史

需要结构化保存：

```text
故障编号
发生时间
排故人员
设备类型
故障现象
关联模块
排故方法
根因
处理步骤
最终方案
预防措施
关联 HMC/FMECA/FTA/MFL
```

主要表：

```text
fault_history_cases
fault_history_steps
```

主要文件：

```text
backend/core/parsers/fault_history.py
backend/icp/repository.py
```

---

## 5. 端到端数据流

### 5.1 上传与解析

```text
首页统一上传
→ 选择 doc_type
→ POST /api/v1/document-types/{docType}/documents/upload
→ MinIO 保存原文件
→ icp_documents 登记
→ 对应卡片文档数刷新
→ 用户进入文档管理并启动解析
→ ParserRegistry 根据 doc_type 路由专用解析器
→ PostgreSQL 写结构化真源
→ Elasticsearch 写文本/向量索引
→ FTA/RBD 可进入图结构确认和 HugeGraph 投影
```

解析器注册入口：

```text
backend/core/parsers/registry.py
```

### 5.2 ICP 故障诊断

默认 LangGraph：

```text
plan
→ icp_diagnosis
→ icp_calculation
→ answer
```

主要文件：

```text
backend/agent/graph.py
backend/agent/nodes/plan.py
backend/agent/tools/icp_diagnosis.py
backend/agent/tools/icp_calculation.py
backend/agent/nodes/answer.py
backend/agent/state.py
```

默认专用 Agent 在后端启动时幂等创建/刷新：

```text
backend/agent/icp_defaults.py
```

默认关闭长期记忆，避免旧诊断判断污染新会话。

### 5.3 Dify

```text
Dify
→ /api/v1/dify/retrieval
→ 旧 knowledge_bases 检索兼容链路
```

在完成专门迁移前，不要让七类 ICP 改造破坏该链路。

---

## 6. ICP 智能体工具

当前工具定义集中在：

```text
backend/agent/tools/registry.py
```

工具职责：

| 工具 | 职责 |
|---|---|
| `function_locator` | 从模糊现象定位系统、模块、功能 |
| `hybrid_search` | 在七类文档中执行关键词和向量混合召回 |
| `hmc_query` | HMC 精确或条件查询 |
| `fmeca_query` | 查询原因、影响、严酷度和检测方法 |
| `fta_query` | 查询顶事件、原因链和逻辑门 |
| `fta_analyzer` | 最小割集、顶事件概率等确定性分析 |
| `rbd_query` | 查询可靠性框图、路径和参数 |
| `rbd_calculator` | 确定性可靠度计算 |
| `mfl_query` | 查询维修清单、功能电路和 API |
| `fault_history_query` | 检索历史排故案例 |
| `fault_reasoning` | 合并证据、排序候选并决定是否追问 |

LLM 负责：

- 理解用户表达；
- 规划工具；
- 解释证据；
- 组织答案；
- 提出补充问题。

LLM 不负责：

- 凭空生成故障代码；
- 猜测维修程序；
- 自行计算 FTA/RBD 数值；
- 将向量近似结果当作唯一事实。

---

## 7. 主要目录与文件

```text
backend/
  main.py                         FastAPI 入口、启动初始化、SPA、图片证据静态服务
  agent/
    api/                           Agent、会话、SSE 对话接口
    graph.py                       LangGraph 编排
    icp_defaults.py                默认 ICP Agent 配置
    nodes/                         plan / answer 等节点
    tools/                         ICP 诊断、FTA/RBD计算和兼容工具
  api/routers/
    types.py                       七类文档上传、解析、查看、图谱接口
    icp_domain.py                  Function/历史结构查看和 FTA/RBD计算接口
    dify.py                        Dify 兼容接口，必须保留
    models.py                      模型管理
  core/parsers/
    registry.py                    七类解析器注册表
    excel_row.py                   HMC/FMECA/MFL 表格解析
    diagram.py                     FTA/RBD 图文解析
    function_architecture.py       系统架构与功能解析
    fault_history.py               故障历史解析
  icp/
    document_types.py              七类类型定义
    repository.py                  ICP 领域结构化仓储
    image_analysis.py              架构/FTA/RBD视觉结构处理
    reliability.py                 FTA/RBD确定性计算
  storage/
    postgres_client.py             现有大型 PG 客户端与兼容逻辑
    es_client.py                   ES 索引与检索
    minio_client.py                原始文件和证据图片
    hugegraph_client.py            图谱投影

frontend/src/
  App.tsx                          当前前端路由
  layouts/MainLayout.tsx           顶部品牌与三项一级导航
  layouts/TypeLayout.tsx           类型页侧栏
  pages/TypeHome.tsx               七类卡片、统计、统一上传入口
  components/UnifiedUploadModal.tsx统一上传窗口
  pages/TypeDocs.tsx               类型内文档管理与解析，不负责上传
  pages/DocDetail.tsx              单文档详情
  pages/TypeView.tsx               类型内容查看
  pages/TypeGraph.tsx              FTA/RBD 图结构确认和图谱生成
  pages/TypeVisualize.tsx          图谱可视化
  pages/Diagnosis.tsx              ICP 故障诊断工作台
  pages/Models.tsx                 模型设置
  api/icp.ts                       七类文档与 ICP 领域 API

static/dist/                       前端构建产物；不要手工修改

tests/
  test_icp_reliability.py          FTA/RBD 确定性算法测试
```

---

## 8. 当前前端路由

```text
/                                  七类文档知识库首页
/diagnosis                         ICP 故障诊断
/models                            系统设置
/type/:docType/documents           某类型文档管理与解析
/type/:docType/view                内容查看
/type/:docType/logs                日志管理
/type/:docType/graph               FTA/RBD 图结构与图谱生成
/type/:docType/visualize           FTA/RBD 图谱可视化
/type/:docType/doc/:docId          单文档详情
```

兼容跳转：

```text
/agent                 → /diagnosis
/type/:docType/upload  → documents
/gallery               → /
/kb/*                   → /
```

不要重新增加 `/gallery` 或旧通用知识库页面。

---

## 9. 开发与运行

### 9.1 基础设施

```bash
./start.sh
```

默认宿主机端口：

| 服务 | 端口 |
|---|---:|
| PostgreSQL | 5433 |
| Elasticsearch | 9201 |
| MinIO API | 9002 |
| MinIO Console | 9003 |
| HugeGraph | 8082 |
| Hubble | 18089 |
| FastAPI 热启动 | 19999 |
| Vite | 5174 |

### 9.2 后端热启动

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./start_server.sh
```

`start_server.sh` 会使用 `backend.main:app`，并在需要时构建前端。

### 9.3 前端开发

```bash
cd frontend
npm ci
npm run dev
```

Vite 会代理：

```text
/api    → http://localhost:19999
/images → http://localhost:19999
```

### 9.4 Docker

```bash
docker compose -p icp up -d --build
```

Dockerfile 使用 Node + Python 多阶段构建。不要依赖仓库中旧的 `static/dist` 作为最新源码证明。

---

## 10. 修改后必须执行的检查

后端：

```bash
python -m compileall -q backend
python -m unittest discover -s tests -p 'test_*.py' -v
```

前端：

```bash
cd frontend
npm ci
npm run build
```

附加建议：

```bash
git diff --check
```

最低验收项：

1. TypeScript 生产构建通过；
2. Python compileall 通过；
3. FTA/RBD 单元测试通过；
4. Dify 路由仍挂载；
5. 首页统一上传可选七类类型；
6. 上传后对应卡片文档数刷新；
7. 类型页没有独立上传入口；
8. `/gallery` 没有恢复；
9. `/images` 工程图证据仍可访问；
10. 没有把无证据候选写成唯一诊断。

---

## 11. 编码约定

### 11.1 通用原则

- 保持中文界面和中文领域术语；
- 代码注释说明“为什么”，避免重复描述代码表面行为；
- 优先小步修改，不要为一个 UI 问题重写整个数据层；
- 新增字段或表必须幂等，重复启动不能报错或重复创建数据；
- 所有跨存储记录必须保留 `doc_id`、业务 ID 和来源信息；
- 解析失败要记录原因，不能静默丢行；
- 原始值和标准化值应尽量分开保留；
- HMC、HSC、API 名称等代码字段必须按字符串处理；
- 不要把合成测试数据描述为真实适航知识。

### 11.2 后端

- 新 ICP 领域 SQL 优先放在 `backend/icp/repository.py`；
- 不要继续无边界扩张 `postgres_client.py`，除非是兼容现有通用接口所必需；
- 查询必须使用参数绑定，不拼接用户输入；
- ES 中只保存检索副本和 metadata，最终权威详情优先回 PostgreSQL；
- FTA/RBD 算法保持纯函数或可测试函数，避免把计算隐藏在 prompt 中；
- 后台任务必须有失败、取消和恢复状态；
- API 返回结构应明确区分空结果、缺少参数和服务器错误。

### 11.3 前端

- TypeScript 保持严格可构建；
- 不要直接修改 `static/dist`；
- 上传入口只维护在 `UnifiedUploadModal`；
- 类型卡片点击进入 `/type/:docType/documents`；
- 页面删除前先全仓搜索 import、路由和 API 引用；
- 保持工程图证据展示，不新增全局画廊；
- 对异步上传、解析、删除和 SSE 任务给出明确状态与错误提示；
- 不要在前端根据文件名自动猜文档类型，类型必须由用户下拉明确选择。

---

## 12. 常见任务定位

### 修改首页、卡片、统一上传

```text
frontend/src/pages/TypeHome.tsx
frontend/src/components/UnifiedUploadModal.tsx
frontend/src/api/icp.ts
```

### 修改 Logo、顶部导航和整体布局

```text
frontend/src/layouts/MainLayout.tsx
```

### 修改类型页面菜单和路由

```text
frontend/src/App.tsx
frontend/src/layouts/TypeLayout.tsx
frontend/src/pages/TypeDocs.tsx
```

### 修改七类文档定义

```text
backend/icp/document_types.py
backend/api/routers/types.py
frontend/src/types/index.ts
```

变更七类定义前必须先确认不会引入第八类用户知识卡片。

### 修改表格解析

```text
backend/core/parsers/excel_row.py
```

重点回归：多级表头、合并单元格、重复表头、前导零、`**`、空单元格。

### 修改 FTA/RBD 图解析

```text
backend/core/parsers/diagram.py
backend/icp/image_analysis.py
backend/icp/reliability.py
```

必须同时检查 FTA 与 RBD 的语义没有被混淆。

### 修改诊断工作流

```text
backend/agent/graph.py
backend/agent/tools/icp_diagnosis.py
backend/agent/tools/icp_calculation.py
backend/agent/nodes/plan.py
backend/agent/nodes/answer.py
frontend/src/pages/Diagnosis.tsx
```

### 修改 Dify

```text
backend/api/routers/dify.py
backend/api/__init__.py
```

任何修改都必须保持原接口路径兼容，并增加回归测试。

---

## 13. 当前已知边界和后续重点

以下能力仍属于持续完善范围，不要在没有证据时宣称已经彻底解决：

1. 真实 Word 文档格式差异很大，解析器仍需用脱敏真实样例迭代；
2. FTA/RBD/架构图视觉抽取需要人工确认闭环；
3. 跨 HMC、FMECA、FTA、MFL、历史文档的统一实体 ID 尚需继续增强；
4. 不同机型、构型、软件版本和文档版本的隔离需要更完整的数据模型；
5. PostgreSQL 与 ES 的一致性、失败重试和重建流程仍需完善；
6. Dify 当前继续走旧知识库兼容链路，尚未完全切换到七类 ICP 专用检索；
7. 检索效果必须通过真实问题集评估，不能只凭对话观感判断；
8. 工程图自动解析结果不能未经确认就用于正式可靠性结论。

---

## 14. Codex 推荐工作流程

处理任何新任务时按以下顺序：

1. 先复述任务边界，确认是否影响七类文档、统一上传、Dify、图片证据或诊断安全；
2. 搜索相关路由、组件、API、数据库方法和测试，不只看单一文件；
3. 找到最小可行修改面；
4. 修改代码；
5. 增加或更新测试；
6. 运行 Python 和前端构建；
7. 检查是否误恢复画廊、通用知识库页面或类型内上传；
8. 总结修改文件、行为变化、测试结果和仍未验证的环境项；
9. 不自动合并到 `main`。

提交说明建议使用：

```text
feat(icp): ...
fix(frontend): ...
fix(parser): ...
refactor(agent): ...
test(icp): ...
docs: ...
```

---

## 15. 最终判断标准

一次改动是否符合项目方向，可以用下面的问题判断：

- 它是否让七类文档更容易正确入库、检索和追溯？
- 它是否有助于用户从模糊现象逐步定位，而不是制造一个看似确定的答案？
- 它是否保留了 PostgreSQL 权威事实、原始文档和工程图证据？
- 它是否没有破坏 Dify？
- 它是否没有恢复图片画廊、通用知识库搭建器或每类独立上传入口？
- 它是否经过可重复的构建和测试验证？

若任何答案是否定的，应在提交前重新评估设计。
