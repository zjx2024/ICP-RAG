# ICP 七类专业文档知识库技术方案 v2.4（类型卡片首页 + 类型侧边栏导航 + 差异化操作流程）

> 在 v2.3 基础上再次改版。v2.4 核心变化：
> 1. **类型页恢复"侧边栏导航"布局（对齐旧知识库体验）**：点卡片进入后为类型管理框架页 `TypeLayout`，左侧侧边栏按 `data_form` 差异化：
>    - **全部类型**：文档上传（upload）/ 内容查看（view）/ 日志管理（logs）；
>    - **图类（fta/rbd）额外**：知识图谱生成（graph）/ 知识图谱可视化（visualize）——图类为两阶段流程（解析三元组 → 人工确认生成图谱 → 可视化），故需这两个独立入口；
> 2. **路由调整**：`/type/:docType` 改为嵌套路由框架页（默认重定向到 `upload`）；原单页 TypeDocs 挂到 `/type/:docType/upload`；文档四视图保持 `/type/:docType/doc/:docId`（在侧边栏内渲染，高亮"内容查看"）；
> 3. **新增页面**：TypeLayout（侧边栏框架）、TypeView（内容查看）、TypeLogs（日志管理）、TypeGraph（图谱生成·仅图类）、TypeVisualize（图谱可视化·仅图类）。

> 历史版本（v2.3，本版继承的设计）：
> 1. **去掉"知识库"概念**：首页直接展示 7 个文档类型卡片，点击卡片进入该类型的文档管理页；
> 2. **类型即顶层组织单位**：`doc_type` 承担原"知识库"的隔离、组织、过滤职责，解析、存储、检索、前端全部以类型为第一路由键；
> 3. **三类文档的"解析流程"与"查看流程"差异化设计**：
>    - **文本类**（fault_history / function_locator）：只需分块向量化，查看"原文结构 + 向量化内容"；
>    - **表格类**（hmc / fmeca / mfl）：解析即入 PG + 重要字段向量化，查看"原文 + 向量化内容 + PG 存储结构"；
>    - **图类**（fta / rbd）：沿用"先解析三元组 → 人工确认 → 生成知识图谱"，查看"原文图片 + 三元组 + 图谱可视化"。
>
> 技术栈约定：后端 FastAPI + PostgreSQL + Elasticsearch + HugeGraph + MinIO；前端 React 18 + TypeScript + Vite + Ant Design 5（现有项目栈）。旧知识库模块保持不动，新能力走类型体系。

---

## 1. 背景与改版目标

### 1.1 v2.1 遗留问题

v2.1 确立了"类型驱动解析"，但仍保留"知识库"作为中间层（创建 KB 时选类型）。带来的问题是：

| 问题 | 表现 |
|------|------|
| 多一层无意义的概念 | 知识库只是为了容纳类型而存在，类型本身就是分类 |
| 操作路径长 | 创建知识库 → 选类型 → 进库 → 上传，多一步 |
| 首页信息量低 | 知识库列表无法直接表达七类业务文档 |

### 1.2 本次改版目标

1. **首页 = 7 个文档类型卡片**（替代知识库列表），每类一张卡片（名称 / 说明 / 格式 / 统计）。
2. **卡片点击进入该类型的文档管理页**：上传、解析、查看、删除全部在此完成。
3. **数据与检索全面去 kb 化**：文档挂在类型下，ES 检索按 `doc_type` 过滤。

---

## 2. 总体架构

```
                 ┌────────────────────────────────────────┐
                 │   首页：7 个文档类型卡片（TypeHome）      │
                 │   hmc / fmeca / mfl / fta / rbd /       │
                 │   fault_history / function_locator      │
                 └───────────────┬────────────────────────┘
                                 │ 点击卡片（doc_type）
                                 ▼
                  ┌──────────────────────────────────┐
                  │  类型文档管理页（TypeDocs）         │
                  │  上传(类型已定) / 列表 / 解析 / 预览 │
                  └───────────────┬──────────────────┘
                                  │ doc_type
                                  ▼
                  ParserRegistry 路由（策略模式）
         ┌──────────────┬──────────────┬──────────────┐
         │ 表格类       │ 图类         │ 文本类        │
         │ hmc/fmeca/mfl│ fta/rbd      │ fault_history │
         │              │              │ function_locator
         ▼              ▼              ▼
    Excel 行级解析   vision 故障树/   专用文本分块
                    框图结构识别
         │              │              │
         ▼              ▼              ▼
  PG 真源表          故障树 JSON      document_chunks
  fault_hmc/fmea/mfl (document_fault_trees)
         │              │              │
         └──────┬───────┴──────┬───────┘
                ▼              ▼
          embedding(模型配置)  手动生成图谱(fta/rbd)
                │              │
                ▼              ▼
   ES 统一索引 knowledge_graph_{tenant}   HugeGraph(convert_fault_tree_to_hugegraph)
   （doc_type_kwd 过滤，不再依赖 kb_id）
                │
                ▼
  检索/问答（doc_type 过滤 + 业务过滤） ──> agent 工具
```

**设计原则**：

1. **类型即顶层**：`doc_type` 承担原知识库的隔离与组织职责，首页即类型清单。
2. **一个索引**：ES 统一索引 + `doc_type_kwd` 过滤。
3. **一张真源表**：表格类 PG 建新表，ES 存检索视图。
4. **解析即契约**：类型决定格式校验、解析器、入库、预览组件。
5. **复用优先**：图谱、删除、SSE 任务、模型管理复用现有组件；旧知识库模块不动。

---

## 3. 文档类型体系

### 3.1 七类类型定义

| doc_type | 中文名 | 数据形式 | 输入格式 | 解析器 | PG 真源 | ES 标记 | 图谱 |
|----------|--------|----------|----------|--------|---------|---------|------|
| `hmc` | HMC 码查询 | 表格 | .xlsx/.xls/.docx | ExcelRowParser | `fault_hmc` | doc_type=hmc | 无 |
| `fmeca` | FMECA 故障影响 | 表格 | .xlsx/.xls/.docx | ExcelRowParser | `fmea` | doc_type=fmeca | 无 |
| `mfl` | MFL 清单 | 表格 | .xlsx/.xls/.docx | ExcelRowParser | `mfl` | doc_type=mfl | 无 |
| `fta` | FTA 故障树 | 图片 | .docx(内嵌图)/图片 | FaultTreeParser | `document_fault_trees` | doc_type=fta | ✅ |
| `rbd` | RBD 可靠性框图 | 图片 | .docx(内嵌图)/图片 | FaultTreeParser | `document_fault_trees` | doc_type=rbd | ✅ |
| `fault_history` | 故障历史 | 文本 | .docx/.txt | TextParser | `document_chunks` | doc_type=fault_history | 可选 |
| `function_locator` | 功能定位 | 文本 | .docx/.txt | TextParser | `document_chunks` | doc_type=function_locator | 可选 |

### 3.2 类型注册表（PG 新表 `document_types`）

类型即配置：前端首页从 `GET /api/v1/document-types` 拉取卡片（名称/说明/格式/统计），后端用其驱动解析分派与格式校验。

```sql
CREATE TABLE IF NOT EXISTS document_types (
    doc_type      VARCHAR(32) PRIMARY KEY,
    name          VARCHAR(64) NOT NULL,
    description   TEXT,
    data_form     VARCHAR(16) NOT NULL,         -- table / image / text
    input_formats VARCHAR(128) NOT NULL,
    parser_key    VARCHAR(64) NOT NULL,
    pg_source     VARCHAR(64),
    es_doc_type   VARCHAR(32) NOT NULL,
    needs_kg      BOOLEAN DEFAULT FALSE,
    is_active     BOOLEAN DEFAULT TRUE,
    sort_order    INTEGER DEFAULT 0,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO document_types (doc_type, name, data_form, input_formats, parser_key, pg_source, es_doc_type, needs_kg, sort_order) VALUES
('hmc',   'HMC 码查询',       'table', '.xlsx,.xls,.docx', 'excel_row', 'fault_hmc', 'hmc',   FALSE, 1),
('fmeca', 'FMECA 故障影响',   'table', '.xlsx,.xls,.docx', 'excel_row', 'fmea',      'fmeca', FALSE, 2),
('mfl',   'MFL 清单',         'table', '.xlsx,.xls,.docx', 'excel_row', 'mfl',       'mfl',   FALSE, 3),
('fta',   'FTA 故障树',       'image', '.docx,.png,.jpg', 'fault_tree', 'document_fault_trees', 'fta', TRUE, 4),
('rbd',   'RBD 可靠性框图',   'image', '.docx,.png,.jpg', 'fault_tree', 'document_fault_trees', 'rbd', TRUE, 5),
('fault_history',    '故障历史', 'text', '.docx,.txt', 'text_chunk', 'document_chunks', 'fault_history', FALSE, 6),
('function_locator', '功能定位', 'text', '.docx,.txt', 'text_chunk', 'document_chunks', 'function_locator', FALSE, 7);
```

> 首页卡片统计（文档数、已索引记录数）由 `icp_documents` 按 `doc_type` 聚合返回，见 §6.5 API。

---

## 4. 解析架构（类型驱动，沿用 v2.1）

### 4.1 ParserRegistry（策略模式）

```python
# backend/core/parsers/registry.py（新增）
class ParserRegistry:
    _parsers: Dict[str, Parser] = {}

    @classmethod
    def register(cls, doc_type: str):
        def deco(parser_cls):
            cls._parsers[doc_type] = parser_cls()
            return parser_cls
        return deco

    @classmethod
    def get(cls, doc_type: str) -> Parser: ...
```

解析器统一接口：

```python
class Parser(ABC):
    doc_type: str
    @abstractmethod
    def validate(self, file_path: str) -> List[str]: ...
    @abstractmethod
    def parse(self, file_path: str, ctx: ParseContext) -> ParseResult: ...
```

### 4.2 解析器清单

| 解析器 | doc_type | 职责 |
|--------|----------|------|
| `ExcelRowParser` | hmc / fmeca / mfl | 读 Excel 或 docx 内嵌表格 → 表头校验 → 逐行结构化 → upsert PG 真源表 → 重要字段拼接向量化 + 过滤标签字段 → ES → 行统计 |
| `FaultTreeParser` | fta / rbd | 提取图片 → 复用现有 vision 分析（`analyze_image`）→ 三元组 JSON 存 `document_fault_trees` → **整图描述 + 节点/三元组向量化** 写 ES → 节点统计 |
| `TextParser` | fault_history / function_locator | 专用分块 → 写 `document_chunks` → 抽取业务标识（fault_code / device_code）→ embedding → ES |

### 4.3 统一解析编排

```
1. 取 icp_documents.doc_type
2. parser = ParserRegistry.get(doc_type)
3. errors = parser.validate(file)        # 格式/表头校验
4. result = parser.parse(file, ctx)      # 分类型入库（PG/ES/embedding）
5. SSE 推送进度与结果统计（行/节点/分块）
6. 更新文档状态 indexed；fta/rbd 追加 kg_pending
```

### 4.4 各类解析细节

**表格类（hmc/fmeca/mfl）——每行一条：**

| 步骤 | 说明 |
|------|------|
| 输入来源 | .xlsx/.xls（读表）或 .docx 内嵌表格（复用现有表格提取 `parse_table_to_json`） |
| 表头校验 | 对比模板列（HMC：HMC代码/LCN/HAMC/HSC/故障描述现象/维修方法；FMECA 13 列；MFL 9 列），缺失列报错 |
| 行级 upsert | `ON CONFLICT (业务主键) DO UPDATE`，主键：hmc=(hmc_code,lcn,hamc,hsc)、fmeca=code、mfl=mfl_code |
| 重要字段向量化 | `content`=该行重要字段拼接（故障描述/维修方法/故障影响/检测措施等），embed 写 ES |
| 过滤标签 | 关键字段写 keyword 过滤列（三码/severity/api_reportable/business_id/pg_id），供精确过滤与聚合 |
| 结果统计 | 成功行 / 失败行（行号+原因），失败行不入库 |

**图类（fta/rbd）——参照现有故障树处理链路：**

| 步骤 | 说明 |
|------|------|
| 图片提取 | docx 内嵌图片（复用现有 `_extract_images_and_tables`）或直接上传图片 |
| vision 识别 | 复用现有 `llm_service.analyze_image`：返回 `description`（**整图中文描述**）、`image_type`、`fault_tree_graph`（**三元组：vertices 节点 + edges 关系**，节点含名称/类型 OR·AND·BASIC/概率/描述） |
| 三元组存储 | `fault_tree_graph` JSON 存 `document_fault_trees`（复用现有表，kb_id 列语义改为存 doc_type） |
| **整图描述向量化** | `图片: {标题}\n{description}` embed 写 ES（content_type=fault_tree_image，doc_type=fta/rbd）——与现有图片描述向量化逻辑一致 |
| **三元组向量化** | 节点/边逐条展平：节点 `content`=名称+描述+父节点+门类型+概率，边 `content`=关系+两端节点，embed 写 ES（content_type=fault_tree_node，`title`=顶事件名，`chapter`=层级路径，business_id=节点名） |
| 图谱入库 | 手动"生成知识图谱"时复用 `convert_fault_tree_to_hugegraph(graph, doc_id, doc_type, image_index)` 入 HugeGraph（顶点 Fault/Cause + 边 causes，含 gate_type/probability） |

> 说明：现有通用解析中，故障树图片的整图描述已随 `img_content` 向量化入 ES（[document_service.py](file:///home/kdfy/test_code/ICP_RAG/backend/services/document_service.py#L259-L275)）。v2.2 在其基础上**升级**：`fault_tree_graph` 三元组（vertices/edges）逐条向量化，使"顶事件/中间事件/底事件"及"任意节点的故障原因"均可被语义召回，再配合 HugeGraph 图遍历给出完整故障路径。

**文本类（fault_history/function_locator）：**

| 步骤 | 说明 |
|------|------|
| 专用分块 | 按标题/编号分块（复用标题抽取），写 `document_chunks` |
| 业务标识 | 正则/LLM 抽取 fault_code / device_code，写 ES `business_id_kwd` |
| 向量化 | 每块 embed 写 ES，doc_type 标记 |

---

## 5. 数据结构设计（去 kb 化）

### 5.1 新增表

| 表 | 用途 | 关键字段 |
|----|------|----------|
| `document_types` | 类型注册表 | doc_type, name, data_form, input_formats, parser_key, pg_source, es_doc_type, needs_kg, sort_order |
| `icp_documents` | **类型文档主表（替代 documents）** | doc_id(PK), doc_type, file_name, minio_path, status, parse_count, last_result JSONB, created_at, updated_at |
| `fault_hmc` / `fmea` / `mfl` | 表格类真源表 | doc_id 关联，**无 kb_id**（类型即 kb），详见附录 9.1 |

```sql
-- 类型文档主表（新文档生命周期挂在此表，与旧 knowledge_bases/documents 无耦合）
CREATE TABLE IF NOT EXISTS icp_documents (
    doc_id      VARCHAR(36) PRIMARY KEY,
    doc_type    VARCHAR(32) NOT NULL REFERENCES document_types(doc_type),
    file_name   VARCHAR(255) NOT NULL,
    minio_path  VARCHAR(512),
    status      VARCHAR(32) DEFAULT 'uploaded',   -- uploaded/parsing/indexed/kg_pending/kg_done/failed
    parse_count INTEGER DEFAULT 0,
    last_result JSONB,        -- {rows_ok, rows_failed, failed_rows:[{row,reason}], nodes, edges, chunks, kg_pending}
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_icp_docs_type ON icp_documents (doc_type);
CREATE INDEX IF NOT EXISTS idx_icp_docs_status ON icp_documents (status);
```

> 设计说明：不再创建"知识库"表；`icp_documents.doc_type` 承担原 `documents.kb_id` 的组织职责。`document_meta` 与 `kb_config` 不再需要（类型与解析结果直接内联到 `icp_documents`）。

### 5.2 复用现有从属表

| 表 | 关联方式 | 说明 |
|----|----------|------|
| `document_chunks` | doc_id | 文本类分块 / 表格类预览（可选） |
| `document_images` | doc_id / chunk_id | 图类原图 |
| `document_tables` | doc_id / chunk_id | 表格类输入来源（docx 内嵌表解析）与预览原始数据 |
| `document_fault_trees` | doc_id | 图类结构化 JSON；`kb_id` 列**语义改为存 doc_type**（兼容列，不新增） |

### 5.3 ES 统一索引扩展

新增字段：`doc_type_kwd / business_id_kwd / pg_id_kwd / hmc_code_kwd / lcn_kwd / hamc_kwd / hsc_kwd / severity_kwd / product_kwd / api_reportable_kwd / platform_test_kwd`（模板新增 + 存量 put_mapping，只增不改）。

检索过滤维度从 `kb_id` 切换为 `doc_type_kwd`：`es_client` 新增 `search_structured(doc_type, query, filters, ...)`（在 `hybrid_search` 基础上把 kb 过滤换成 doc_type + 业务过滤）。

### 5.4 删除流程

按 doc_id 执行（8 步扩展）：

```
1. 删 HugeGraph（按 doc_id，含共享裁剪逻辑）
2. 删 ES 记录（按 doc_id）
3. 删 fault_hmc / fmea / mfl 中 doc_id 记录
4. 删 document_fault_trees / document_images / document_tables / document_chunks
5. 删解析/KG 日志（保留留痕）
6. 删 MinIO 文件
7. 删 icp_documents 记录
```

---

## 6. 前端操作流程设计（类型卡片首页）

### 6.1 信息架构与路由（v2.4：类型页侧边栏导航）

```
/                          首页：7 个文档类型卡片（替代原知识库列表）
/models  /agent            全局页（现有，不动）
/type/:docType             类型框架页（TypeLayout：侧边栏 + 内容区）→ 默认重定向到 upload
/type/:docType/upload      文档上传与解析（上传 / 列表 / 解析 / 删除）
/type/:docType/view        内容查看（选择文档 → 四视图查看）
/type/:docType/logs        日志管理（解析 / 图谱 / 删除任务日志）
/type/:docType/graph       知识图谱生成（仅 fta/rbd：kg_pending 文档 → 确认三元组 → 生成）
/type/:docType/visualize   知识图谱可视化（仅 fta/rbd：HugeGraph 图渲染）
/type/:docType/doc/:docId  文档详情四视图（侧边栏内渲染，高亮"内容查看"）
/retrieval                 全局检索（doc_type 筛选）
```

**侧边栏菜单（按 data_form 差异化）：**

| 侧边栏项 | 路由 key | 表格类 hmc/fmeca/mfl | 文本类 fault_history/function_locator | 图类 fta/rbd |
|----------|----------|:---:|:---:|:---:|
| 文档上传 | `upload` | ✅ | ✅ | ✅ |
| 内容查看 | `view` | ✅ | ✅ | ✅ |
| 日志管理 | `logs` | ✅ | ✅ | ✅ |
| 知识图谱生成 | `graph` | — | — | ✅ |
| 知识图谱可视化 | `visualize` | — | — | ✅ |

> 图类保留"生成 / 可视化"两个独立入口的原因：图类是两阶段流程（解析 → kg_pending → 人工确认生成 → kg_done → 可视化），生成页承担"人工确认 + 生成"操作，可视化页承担"查看 HugeGraph 图"操作，与旧知识库"知识图谱构建 / 图谱可视化"一致。

### 6.2 页面与组件清单

| 文件 | 说明 |
|------|------|
| `layouts/TypeLayout.tsx`（新，仿 KbLayout） | 类型框架页：左侧边栏（按 data_form 差异化菜单）+ 顶部类型信息 + 内容区 Outlet |
| `pages/TypeHome.tsx`（新，替代 KbList） | 首页：7 张类型卡片（图标 + 名称 + 说明 + 格式 + 文档数/记录数统计），点击进入类型框架页 |
| `pages/TypeDocs.tsx`（改，挂 upload 子路由） | 文档上传与解析页：上传入口 + 文档列表（状态/结果统计/操作） |
| `pages/TypeView.tsx`（新） | 内容查看页：**默认直接展示该类型全部文档的内容**（无需逐份点击），胶囊选择器支持单选/多选文档；每份文档一个四视图卡片（原文 / 向量化 / PG 结构 / 三元组+图谱，按类型组合），复用 `DocContentView` 组件 |
| `pages/TypeLogs.tsx`（新，复用 Logs 逻辑） | 日志管理页：该类型下解析/图谱/删除任务日志（时间线 + 筛选） |
| `pages/TypeGraph.tsx`（新，仅 fta/rbd） | 知识图谱生成页：kg_pending 文档列表 → 三元组确认 → 生成知识图谱（SSE 进度） |
| `pages/TypeVisualize.tsx`（新，仅 fta/rbd） | 知识图谱可视化页：选择 kg_done 文档 → GraphCanvas 渲染 HugeGraph 图 |
| `pages/DocDetail.tsx`（改） | 文档详情四视图页（在 TypeLayout 内渲染，返回/高亮 view） |
| `components/TypeGuide.tsx`（新） | 类型卡片渲染（TypeHome 与上传向导共用） |
| `components/DocTypeTag.tsx`（新） | 类型标签（颜色/名称映射） |
| `components/UploadWizard.tsx`（新） | 两步上传向导（类型已定）：拖拽文件（仅接受该类型格式）→ 确认 |
| `components/views/OriginalView.tsx`（新） | 原文视图：docx 文本树 / 表格渲染 / 图片画廊 |
| `components/views/VectorView.tsx`（新） | 向量化视图：分块/行/节点的 `content` + ES chunk_id + 相似度入口 |
| `components/views/PGView.tsx`（新） | PG 结构视图：真源表行数据（搜索 / 过滤 / 分页） |
| `components/views/TripletView.tsx`（新） | 三元组视图：G6 渲染故障树/框图结构 + 节点/边列表 |
| `api/icp.ts`（改） | 新增 fetchDocumentTypes（含统计）、doc_type 路径的上传/列表/解析/结果/查看/日志 API |
| `types/index.ts`（改） | 新增 DocType、ICPDocument、ParseResult、RowView 类型 |

### 6.3 解析操作流程（按类型差异化）

**三类解析流程差异一览：**

| 维度 | 文本类 | 表格类 | 图类 |
|------|--------|--------|------|
| 解析动作 | 一键自动 | 一键自动 | **两阶段**：自动识别三元组 + 人工确认生成图谱 |
| 人工干预 | 无 | 无 | 生成图谱前需人工确认识别结果 |
| 入库动作 | 分块向量化 → ES | 逐行入 PG + 重要字段向量化 → ES | 三元组入 PG + 整图描述/节点向量化 → ES |
| 解析产物 | N 个分块 | N 行记录 | N 个节点 + M 条边 |

**A. 表格类（hmc / fmeca / mfl）**

```
1. 首页点击「HMC 码查询」卡片 → /type/hmc
2. 点击「上传」→ 两步向导：拖拽 .xlsx 或含表格的 .docx（其他格式灰置）→ 确认
3. 列表出现新文档（类型标签 = HMC，状态 = 已上传）
4. 点击「解析」→ SSE 日志：表头校验 → 逐行入库 PG（行进度）→ 重要字段向量化 → 完成
5. 成功：状态 = 已索引，显示"N 行成功 / M 行失败"
   → 点击「查看」进入文档详情页（见 §6.4 A）
6. 失败处理：修正表格后重新上传（同一 doc_id 覆盖，幂等）
```

**B. 图类（fta / rbd）——两阶段流程**

```
1. 首页点击「FTA 故障树」卡片 → /type/fta
2. 上传向导 → 拖拽 .docx（含故障树图片）
3. 点击「解析」（阶段一·自动）→ SSE 日志：提取图片 → vision 识别 →
   三元组(节点/边)结构化 JSON 存 PG → 整图描述 + 节点向量化 → 完成
4. 完成：状态 = 待生成图谱（kg_pending），列表显示"N 节点 / M 边"
5. 点击「查看」→ 三元组预览（G6 树形结构）→ 人工确认识别无误（阶段二·人工确认）
6. 点击「生成知识图谱」→ convert 入 HugeGraph，状态 → kg_done
7. 图谱可视化页 /graph/fta 查看，检索可用
```

**C. 文本类（fault_history / function_locator）**

```
1. 首页点击「故障历史」卡片 → /type/fault_history
2. 上传向导 → 拖拽 .docx/.txt
3. 点击「解析」→ SSE 日志：分块 → 业务标识抽取 → 向量化 → 完成
4. 完成：状态 = 已索引，显示"N 个分块"
5. 「查看」→ 文档详情页（见 §6.4 C）
```

### 6.4 内容查看操作流程（按类型差异化）

**四视图查看模型**：解析完成后，文档详情页（DocDetail）按 doc_type 自动组合以下视图 Tab，用户无需理解底层存储：

| 视图 | 展示内容 | 文本类 | 表格类 | 图类 |
|------|----------|:------:|:------:|:----:|
| 原文 `OriginalView` | 原文件内容渲染（docx 标题树/表格、Excel 表、图片） | ✅ | ✅ | ✅ |
| 向量化 `VectorView` | 分块/行/节点的 `content` + ES chunk_id + 状态 | ✅ | ✅ | ✅ |
| PG 结构 `PGView` | 真源表行数据（搜索 / 过滤 / 分页） | — | ✅ | ✅（三元组 JSON） |
| 图谱 `TripletView` + Graph | 故障树/框图结构 + HugeGraph 可视化 | — | — | ✅ |

**A. 表格类查看流程**

```
「查看」→ DocDetail（表格类 Tab 组合）：
  ① 原文：渲染 Excel / docx 表格原始数据（与源文件一致）
  ② 向量化：每条入库行对应的 content（向量化文本）+ ES 状态
  ③ PG 结构：fault_hmc / fmea / mfl 行记录表，支持按三码 / severity / api_reportable 搜索过滤、分页
  ④ 失败明细（如有）：行号 + 原因
```

**B. 图类查看流程**

```
「查看」→ DocDetail（图类 Tab 组合）：
  ① 原文：故障树/框图原图（图片画廊）
  ② 三元组：G6 渲染节点/边结构 + 节点/边明细表
  ③ 向量化：整图描述 + 每个节点的向量化 content
  ④ 图谱（kg_done 后）：/graph/fta HugeGraph 可视化，支持图路径查询
  ⑤ 操作：kg_pending 状态展示「生成知识图谱」按钮（人工确认步骤）
```

**C. 文本类查看流程**

```
「查看」→ DocDetail（文本类 Tab 组合）：
  ① 原文：docx/txt 原文结构（标题树 + 正文）
  ② 向量化：分块列表（块标题 / 内容 / fault_code / device_code / ES chunk_id）
```

### 6.5 文档状态机

```
uploaded ──解析──> parsing ──成功──> indexed ──(fta/rbd)──> kg_pending ──生成图谱──> kg_done
                    │                                    │
                    └────失败──> failed <────重解析───────┘
```

### 6.6 后端 API（doc_type 路径）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/document-types` | 类型注册表 + 每类统计（文档数、已索引记录数） |
| POST | `/api/v1/document-types/{docType}/documents/upload` | multipart 上传（类型即路径，无需再传类型） |
| GET | `/api/v1/document-types/{docType}/documents` | 该类型文档列表 |
| POST | `/api/v1/document-types/{docType}/documents/{docId}/parse` | 触发解析（后端按 doc_type 路由解析器） |
| GET | `/api/v1/document-types/{docType}/documents/{docId}/parse-result` | 解析结果统计（行/节点/分块 + 失败明细） |
| GET | `/api/v1/document-types/{docType}/documents/{docId}/original` | 原文内容（docx 标题树/表格、图片元数据） |
| GET | `/api/v1/document-types/{docType}/documents/{docId}/vectors` | 向量化记录（content + ES chunk_id，分页） |
| GET | `/api/v1/document-types/{docType}/documents/{docId}/rows` | PG 真源表行数据（搜索/过滤/分页，表格类） |
| GET | `/api/v1/document-types/{docType}/documents/{docId}/fault-trees` | 三元组 JSON（图类，document_fault_trees） |
| POST | `/api/v1/document-types/{docType}/documents/{docId}/generate_kg` | 图谱生成（仅 fta/rbd，人工确认后触发） |
| DELETE | `/api/v1/document-types/{docType}/documents/{docId}` | 删除文档（8 步清理） |

---

## 7. 检索与问答

- 新增 `structured_retrieval(doc_type, query, filters)`：doc_type 过滤 + ES 混合检索 + 业务过滤（三码 / severity / boolean）→ 按 pg_id/business_id 回 PG 增强。
- 全局检索页 `/retrieval`：可切换 doc_type（或全部）。
- agent 工具：`vector_search` 用 doc_type 替代 kb_id；`sql_query` 白名单扩展 fault_hmc/fmea/mfl；`kg_search` 用于 fta/rbd 图路径。

---

## 8. 实施步骤

| Phase | 内容 | 涉及文件 |
|-------|------|----------|
| 1 数据层 | document_types / icp_documents / 真源表建表 + 类型初始化；ES 模板扩展 | `postgres_client.py` / `es_client.py` |
| 2 解析架构 | ParserRegistry + ExcelRowParser/FaultTreeParser/TextParser + 统一编排 + doc_type 路由 | 新增 `backend/core/parsers/*`，改 `document_service.py` |
| 3 表格类闭环 | hmc/fmeca/mfl 校验+入库+ES+parse-result API | `postgres_client.py` / `documents.py`（或新 router） |
| 4 图类闭环 | FaultTreeParser（复用 vision）+ 节点展平 + 生成图谱联动 | `document_service.py` / `kg_service.py` |
| 5 文本类闭环 | TextParser 专用分块 + 业务标识抽取 | `document_service.py` |
| 6 前端 | TypeHome / TypeDocs / UploadWizard / ParseDetail / DocDetail 四视图 + 路由替换 | `frontend/src/pages/*` / `components/*` / `api/kb.ts` |
| 7 检索治理 | structured_retrieval + agent 工具 doc_type 化 + 删除流程 | `retrieval.py` / `agent/tools/*` / `_deletion_helpers.py` |

---

## 9. 附录

### 9.1 PG 新表 DDL（真源表去 kb_id，doc_id 关联）

```sql
-- HMC（组合唯一：同一 hmc_code 可对应多个 hsc）
CREATE TABLE IF NOT EXISTS fault_hmc (
    id            BIGSERIAL PRIMARY KEY,
    doc_id        VARCHAR(36) NOT NULL REFERENCES icp_documents(doc_id) ON DELETE CASCADE,
    hmc_code      VARCHAR(20) NOT NULL,
    lcn           VARCHAR(10) DEFAULT '**',
    hamc          VARCHAR(10) DEFAULT '**',
    hsc           VARCHAR(10) DEFAULT '**',
    fault_desc    TEXT NOT NULL,
    repair_method TEXT NOT NULL,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (hmc_code, lcn, hamc, hsc)
);
CREATE INDEX IF NOT EXISTS idx_hmc_3code ON fault_hmc (lcn, hamc, hsc);
CREATE INDEX IF NOT EXISTS idx_hmc_doc   ON fault_hmc (doc_id);

-- FMECA
CREATE TABLE IF NOT EXISTS fmea (
    id                BIGSERIAL PRIMARY KEY,
    doc_id            VARCHAR(36) NOT NULL REFERENCES icp_documents(doc_id) ON DELETE CASCADE,
    code              VARCHAR(50) NOT NULL UNIQUE,
    product_flag      VARCHAR(100),
    func_management   VARCHAR(100),
    fault_mode        TEXT NOT NULL,
    fault_cause       TEXT,
    mission_phase     VARCHAR(100),
    local_impact      TEXT,
    final_impact      TEXT,
    severity          VARCHAR(20),
    probability       NUMERIC(10,6),
    detect_method     TEXT,
    design_improve    TEXT,
    compensate_action TEXT,
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_fmea_severity ON fmea (severity);
CREATE INDEX IF NOT EXISTS idx_fmea_doc      ON fmea (doc_id);

-- MFL
CREATE TABLE IF NOT EXISTS mfl (
    id               BIGSERIAL PRIMARY KEY,
    doc_id           VARCHAR(36) NOT NULL REFERENCES icp_documents(doc_id) ON DELETE CASCADE,
    mfl_code         VARCHAR(50) NOT NULL UNIQUE,
    module_fault     TEXT,
    sur_fault        TEXT,
    subcircuit_fault TEXT,
    api_reportable   BOOLEAN DEFAULT FALSE,
    api_name         VARCHAR(200),
    api_method       VARCHAR(20),
    api_usage        TEXT,
    platform_test    BOOLEAN DEFAULT FALSE,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_mfl_api_name   ON mfl (api_name);
CREATE INDEX IF NOT EXISTS idx_mfl_reportable ON mfl (api_reportable);
CREATE INDEX IF NOT EXISTS idx_mfl_doc        ON mfl (doc_id);
```

### 9.2 ES mapping 扩展

```json
{
  "properties": {
    "doc_type_kwd":       { "type": "keyword" },
    "business_id_kwd":    { "type": "keyword" },
    "pg_id_kwd":          { "type": "keyword" },
    "hmc_code_kwd":       { "type": "keyword" },
    "lcn_kwd":            { "type": "keyword" },
    "hamc_kwd":           { "type": "keyword" },
    "hsc_kwd":            { "type": "keyword" },
    "severity_kwd":       { "type": "keyword" },
    "product_kwd":        { "type": "keyword" },
    "api_reportable_kwd": { "type": "keyword" },
    "platform_test_kwd":  { "type": "keyword" }
  }
}
```

### 9.3 涉及代码位置

| 改动点 | 后端 | 前端 |
|--------|------|------|
| 类型注册表 / 文档主表 / 真源表 | `postgres_client.py` | `api/kb.ts`（fetchDocumentTypes） |
| 解析器注册表 | 新增 `backend/core/parsers/{registry,excel_row,fault_tree,text}.py` | — |
| 解析编排 | `services/document_service.py` | `pages/TypeDocs.tsx` / `pages/ParseDetail.tsx` |
| 类型 API（doc_type 路径） | `api/routers/documents.py` 或新 `api/routers/types.py` | `api/kb.ts` |
| 首页类型卡片 | — | `pages/TypeHome.tsx` / `components/TypeGuide.tsx` |
| 上传向导 | — | `components/UploadWizard.tsx` |
| 文档详情查看 | — | `pages/DocDetail.tsx` / `components/views/{Original,Vector,PG,Triplet}View.tsx` |
| 检索 | `core/retrieval.py`（doc_type 化） | `pages/Retrieval.tsx` |
| agent 工具 | `agent/tools/*`（kb_id → doc_type） | — |
| 删除流程 | `routers/_deletion_helpers.py` | — |
