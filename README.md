# ICP-RAG：ICP 故障诊断知识库与智能体

ICP-RAG 是面向 ICP 航空电子系统的专用故障知识平台。系统接收七类 Word 原始文档，按文档类型执行差异化预处理，将结构化事实、语义索引、图关系和原始证据统一提供给故障诊断智能体。

> **安全说明**：系统输出属于知识检索与辅助诊断结果，不替代适用构型、有效版本的正式维修手册、放行程序和人工复核。合成测试数据不得作为实际航空维修依据。

## 1. 七类知识文档

| 文档类型 | 主要内容 | 主存储与处理方式 |
|---|---|---|
| 系统架构与功能说明（Function Locator） | 系统、子系统、模块、功能、接口、典型失效表现 | PostgreSQL 功能结构 + Elasticsearch 语义索引 + 架构图关系 |
| HMC | LCN、HAMC、HSC、故障现象、分析/维修方法 | PostgreSQL 权威记录 + 诊断/维修双向量索引 |
| FMECA | 故障模式、原因、影响、严酷度、检测及补偿措施 | PostgreSQL 宽表记录 + Elasticsearch |
| FTA | 顶事件、中间事件、基本事件、AND/OR 逻辑 | PostgreSQL 节点边 JSON + Elasticsearch + HugeGraph 投影 |
| RBD | 功能块、串并联/冗余结构、成功判据和可靠性参数 | PostgreSQL 节点边 JSON + 确定性可靠度计算 |
| MFL | 模块/SUR 板/功能电路故障、API 上报和维修定位 | PostgreSQL 权威记录 + Elasticsearch |
| 故障历史 | 发生时间、现象、排故方法、根因、步骤、方案和关联知识 | PostgreSQL 案例/步骤 + Elasticsearch |

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
 PostgreSQL       Elasticsearch       MinIO          HugeGraph
 权威结构数据       BM25 + 向量索引     原文/图片证据    图关系投影
        └──────────────┴────────────────┴────────────────┘
                               │
                               ▼
                ICP 故障诊断 LangGraph
   plan → 多源检索/功能定位 → FTA/RBD确定性计算 → answer
```

数据库职责：

- PostgreSQL 是权威事实来源；
- Elasticsearch 负责模糊故障现象、专业关键词和向量语义召回；
- MinIO 保存原始文档及图片证据；
- HugeGraph 保存需要多跳查询的图关系投影；
- 向量和图谱均为可重建派生数据。

## 3. 诊断流程

用户无需知道具体模块或 HMC，只需描述现象、发生阶段、原始告警及已执行操作。默认流程如下：

```text
模糊故障现象
  → Function Locator 定位系统/模块/功能
  → 七类文档 BM25 + 向量混合召回
  → HMC/FMECA/FTA/RBD/MFL/历史结构化校验
  → 候选故障排序和置信度判断
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
- `fault_reasoning`：多源证据融合、候选排序和追问判断。

FTA/RBD 数值由确定性 Python 算法计算。缺少概率、故障率、可靠度或任务时间时，系统明确报告缺失参数，不生成虚构数值。

## 5. 前端页面

- **七类文档知识库**：上传、解析、查看原文/向量/结构化记录、确认 FTA/RBD 图结构；
- **ICP 故障诊断**：会话式诊断、实时工具调用、候选证据和置信度；
- **系统设置**：Chat、Vision、Embedding 和 Rerank 模型配置。

独立“图片画廊”和旧通用知识库页面已从产品入口移除。图片抽取与静态服务仍保留，用于 FTA、RBD、系统架构原图及证据查看。

## 6. Dify 外部知识库兼容

Dify 接口保持不变：

```text
GET  /api/v1/dify/server-info
POST /api/v1/dify/retrieval
```

Dify 中的外部知识库地址填写：

```text
http://<host>:9999/api/v1/dify
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
| Elasticsearch | 9201 |
| MinIO API / Console | 9002 / 9003 |
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

### 7.2 Docker 完整构建

Dockerfile 使用 Node 多阶段构建，自动编译当前 React 源码：

```bash
docker compose -p icp up -d --build
```

也可只重建应用：

```bash
./restart-app.sh
```

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
  storage/               PostgreSQL、ES、MinIO、HugeGraph 客户端
frontend/src/
  pages/Diagnosis.tsx    ICP 故障诊断工作台
  pages/Type*.tsx        七类文档管理
  api/                   前端 API 客户端
tests/
  test_icp_reliability.py
```

## 10. 当前边界

- 图像解析质量取决于所配置 Vision/Chat 模型和原图清晰度；
- FTA/RBD 自动提取结果应在图结构页面人工确认后再用于正式分析；
- 不同机型、构型、软件版本的知识必须通过文档版本和适用范围隔离；
- 对真实维修流程的集成测试需要可访问的 PostgreSQL、Elasticsearch、MinIO、HugeGraph 和模型服务。
