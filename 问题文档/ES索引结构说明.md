# Elasticsearch 索引结构说明

> 更新时间：2026-08-18
> 索引：`knowledge_graph_default`

## 一、总体设计：单索引 + 字段分层

系统**只有一个业务索引** `knowledge_graph_default`，所有知识库的数据都写入该索引，通过 **`kb_id` 字段**进行隔离。每创建一个知识库**不会**新建索引。

```
knowledge_graph_default（唯一索引，当前共 316 条）
│
├─ 知识库层 —— kb_id 字段（如 07718714… 为 ICP故障库）
│
├─ 文档层 —— doc_id / doc_name 字段（如 mod_1.docx）
│
└─ 内容层 —— 两类实体，用不同字段区分：
    ├─ 分块（chunk）：有 content_type + content_vector（1024 维向量）
    │   ├─ content_type=text    文本段落
    │   ├─ content_type=image   图片 + AI 视觉描述
    │   └─ content_type=table   表格结构化内容
    └─ 社区报告：knowledge_graph_kwd=community_report（无向量）
```

### 区分规则

| 实体类型 | 判定字段 | 向量 |
|---|---|---|
| 文本/图片/表格分块 | `content_type` = text / image / table | 有 `content_vector` |
| 社区报告 | `knowledge_graph_kwd` = community_report | 无 |

## 二、字段结构（mapping）

主要字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `kb_id` | keyword | 知识库 ID（隔离维度） |
| `doc_id` / `doc_name` | keyword / text | 文档 ID / 文档名 |
| `chunk_id` | keyword | 分块 ID（如 `{doc_id}_h2_…`） |
| `content_type` | keyword | text / image / table |
| `title` / `section` | text | 分块标题 / 章节位置 |
| `content` | text | 分块正文内容 |
| `content_vector` | dense_vector(1024) | 向量（cosine 相似度） |
| `image_url` | text | 图片存储地址（仅图片分块） |
| `knowledge_graph_kwd` | keyword | 社区报告标记 = community_report |
| `report_id` / `community_id` | text | 社区报告 ID / 社区 ID |
| `entities` | text | 社区包含的实体名列表 |
| `report` | object | `{title, summary, rating, rating_explanation, findings}` |
| `created_at` / `updated_at` | date | 时间戳 |

## 三、实际数据示例（ICP 故障库 mod_1.docx）

### 1. 文本分块（content_type=text）

```json
{
  "kb_id": "07718714-eb7a-…",
  "doc_id": "2fb90130-b23f-…",
  "doc_name": "mod_1.docx",
  "chunk_id": "2fb90130-…_h2_…",
  "content_type": "text",
  "title": "3.9 用于敏感性试验的 EUT 结构",
  "section": "h2_021_3_9_用于敏感性试验的_EUT_结构",
  "content": "在进行本文件中规定那个的敏感性试验期间，所测试的设备应在现场正常运行中可能遇到的最敏感的功能模式下配置……",
  "content_vector": "[1024 维向量]"
}
```

### 2. 图片分块（content_type=image，带 AI 视觉描述）

```json
{
  "content_type": "image",
  "title": "注解：1）从T0到T1的温度变化率未做说明。",
  "section": "h3_034_4_6_3_过压测试",
  "image_url": "http://192.168.14.159:9999/images/2fb90130-…/img_h3_034_….png",
  "content": "图片: 注解：…\n这是一张DO-160标准中的环境测试程序示意图，具体展示了地面存活低温测试与短期操作低温测试的结合流程……",
  "content_vector": "[1024 维向量]"
}
```

### 3. 表格分块（content_type=table）

```json
{
  "content_type": "table",
  "title": "温度和高程标准",
  "content": "表格: 温度和高程标准\n作为航空标准文档分析专家，以下是对该《温度和高程标准》表格的深度结构化解析……",
  "content_vector": "[1024 维向量]"
}
```

### 4. 社区报告（knowledge_graph_kwd=community_report，无向量）

```json
{
  "knowledge_graph_kwd": "community_report",
  "report_id": "community_1_52",
  "community_id": "52",
  "entities": ["测试", "辐射热来源", "温度", "公差", "设备性能标准"],
  "report": {
    "title": "设备性能测试验证与参数体系",
    "rating": 5.0,
    "rating_explanation": "该评分反映了设备测试验证体系在保障产品质量……",
    "summary": "该社区围绕设备测试验证流程构建，核心实体为“测试”活动……",
    "findings": [{ "summary": "测试活动作为社区的核心枢纽……" }]
  }
}
```

## 四、各层数据在智能体中的使用方式

| 数据层 | 存储位置 | 查询方式 | 智能体工具 |
|---|---|---|---|
| 文本/图片/表格分块 | ES | 向量检索（content_vector 余弦）+ 关键词 | `vector_search` |
| 实体/关系/故障树 | HugeGraph | 图遍历（按 kb_id 过滤） | `kg_search` |
| 社区报告 | ES | BM25 文本检索（entities 精确匹配 + 关键词） | `kg_search` |
| 表格/图片元数据 | PostgreSQL | SQL 查询（document_tables / document_images） | `sql_query` |

> 三者通过 `doc_id` / `kb_id` 字段关联：ES 管内容分块与社区报告，HugeGraph 管实体关系，PostgreSQL 管结构化元数据。
