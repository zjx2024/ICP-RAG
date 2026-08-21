// ===== 全局 TS 类型定义 =====

// ---------- ICP 类型驱动文档（七类） ----------
export interface DocTypeInfo {
  doc_type: string;
  name: string;
  description?: string;
  data_form: 'table' | 'image' | 'text';
  input_formats?: string;
  parser_key?: string;
  pg_source?: string;
  es_doc_type?: string;
  needs_kg?: boolean;
  is_active?: boolean;
  sort_order?: number;
  doc_count?: number;
  record_count?: number;
}

export interface ICPDocument {
  doc_id: string;
  doc_type: string;
  file_name: string;
  minio_path?: string;
  status: string; // uploaded/parsing/indexed/kg_pending/kg_done/failed
  parse_count?: number;
  last_result?: Record<string, unknown> | null;
  created_at?: string;
  updated_at?: string;
}

export interface ICPParseResult {
  doc_id: string;
  doc_type: string;
  status: string;
  parse_count: number;
  last_result?: Record<string, unknown> | null;
}

export interface ICPOriginalBlock {
  title: string;
  level: number;
  text?: string;
  table?: string[][];
}

export interface ICPOriginalResponse {
  doc_type: string;
  file_name: string;
  total: number;
  page: number;
  size: number;
  blocks: ICPOriginalBlock[];
}

export interface ICPVectorResponse {
  total: number;
  page: number;
  size: number;
  vectors: Array<Record<string, unknown>>;
}

export interface ICPRowsResponse {
  total: number;
  page: number;
  size: number;
  rows: Array<Record<string, unknown>>;
}

export interface ICPFaultTree {
  fault_tree_id: string;
  doc_id: string;
  kb_id: string;
  chunk_id: string;
  image_id: string;
  image_title: string;
  graph: {
    root_fault_id?: string;
    metadata?: Record<string, unknown>;
    vertices?: Array<{
      id: string;
      label?: string;
      properties?: Record<string, unknown>;
    }>;
    edges?: Array<{
      id: string;
      label?: string;
      source: string;
      target: string;
      properties?: Record<string, unknown>;
    }>;
  };
}

// ---------- 知识库 ----------
export interface KnowledgeBase {
  kb_id: string;
  name: string;
  description?: string;
  created_at?: string;
  updated_at?: string;
}

// ---------- 文档 ----------
export type DocStatus =
  | 'uploaded'
  | 'parsing'
  | 'indexed'
  | 'generating_kg'
  | 'completed'
  | 'failed'
  | 'deleting'
  | 'delete_failed'
  | 'cancelled';

export interface DocumentStatus {
  doc_id: string;
  file_name: string;
  kb_id: string;
  status: DocStatus;
  total_chunks?: number;
  created_at?: string;
  kb_name?: string;
}

export interface UploadResult {
  doc_id: string;
  file_name: string;
  status: string;
}

export interface ParseResult {
  task_id: string;
  doc_id: string;
  file_name: string;
  status: string;
}

export interface Chunk {
  chunk_id: string;
  doc_id: string;
  section: string;
  title: string;
  content: string;
  content_type: string;
  chunk_index: number;
  selected_for_kg: boolean;
  images?: unknown[];
  tables?: unknown[];
}

// ---------- 任务状态 ----------
export interface TaskState {
  task_id: string;
  doc_id: string;
  file_name: string;
  task_type: string;
  status: string;
  progress: number;
  logs: string[];
  result?: unknown;
  error?: string;
}

// ---------- 检索 ----------
export interface RetrievalStats {
  kb_id: string;
  total_documents: number;
  total_chunks: number;
  total_entities: number;
  total_relations: number;
}

export interface SearchResultItem {
  title: string;
  score: number;
  content: string;
  source?: string;
  metadata?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface SearchResponse {
  kb_id: string;
  query: string;
  mode: string;
  results: SearchResultItem[];
  chunks?: SearchResultItem[];
  entities?: SearchResultItem[];
  community_reports?: SearchResultItem[];
  kg_context?: SearchResultItem[] | string;
  total_results: number;
  total_chunks?: number;
  total_entities?: number;
  total_reports?: number;
}

// ---------- 图谱数据 ----------
export interface GraphVertex {
  id: string;
  label?: string;
  type?: string;
  properties?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface GraphEdge {
  source: string;
  target: string;
  label?: string;
  properties?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface GraphData {
  kb_id?: string;
  kb_name?: string;
  doc_id?: string;
  vertices: GraphVertex[];
  edges: GraphEdge[];
  total_vertices: number;
  total_edges: number;
  error?: string;
}

export interface CommunityReport {
  report_id: string;
  community_id: string;
  entities: string[];
  title: string;
  summary: string;
  rating: number;
  rating_explanation: string;
  findings: unknown[];
  doc_id: string;
  doc_ids?: string[];
  kb_id: string;
  created_at?: string;
}

export interface CommunityReportsResponse {
  kb_id: string;
  doc_id?: string;
  total_reports: number;
  reports: CommunityReport[];
  error?: string;
}

// ---------- 模型 ----------
export type ModelType = 'llm' | 'embedding' | 'rerank' | 'vision';

export interface LLMModel {
  model_id: string;
  name: string;
  model_type: ModelType | string;
  provider?: string;
  api_key?: string;
  api_url?: string;
  model_name?: string;
  base_url?: string;
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  timeout?: number;
  is_default: boolean;
  is_active: boolean;
  description?: string;
  created_at?: string;
}

export interface CurrentModelConfig {
  chat_model: LLMModel | null;
  embedding_model: LLMModel | null;
  rerank_model: LLMModel | null;
  config_status: {
    chat_configured: boolean;
    embedding_configured: boolean;
    ready_for_ai: boolean;
  };
}

// ---------- 日志 ----------
export interface LogTask {
  task_type?: string;
  task_id: string;
  kb_id: string;
  doc_id: string;
  kb_name?: string;
  doc_name?: string;
  kb_exists?: boolean;
  doc_exists?: boolean;
  status?: string;
  start_time?: string;
  end_time?: string;
  progress?: number;
  message?: string;
}

export interface LogEntry {
  log_id: string;
  task_id: string;
  doc_id: string;
  kb_id: string;
  log_level: string;
  log_type: string;
  message: string;
  progress: number;
  status?: string;
  created_at?: string;
  [key: string]: unknown;
}

// ---------- 图片/表格（画廊） ----------
export interface ImageDocSummary {
  doc_id: string;
  doc_name?: string;
  image_count: number;
  table_count?: number;
}

export interface ImageItem {
  image_id: string;
  title: string;
  description: string;
  image_type: string;
  image_url: string;
  chunk_id: string;
  minio_path: string;
  created_at?: string;
}

export interface TableItem {
  table_id: string;
  title: string;
  description: string;
  meta_data: Record<string, unknown>;
  chunk_id: string;
  created_at?: string;
}

export interface DocMediaDetail {
  doc_id: string;
  images: ImageItem[];
  tables: TableItem[];
  image_count: number;
  table_count: number;
}

export interface ImageDocsResponse {
  documents: ImageDocSummary[];
  total_documents: number;
}

// ---------- Agent ----------
export interface AgentConfig {
  agent_id: string;
  name: string;
  description?: string;
  system_prompt?: string;
  kb_id?: string;
  kb_ids?: string[];  // 关联的多个知识库（优先于 kb_id）
  enabled_tools?: string[];
  chat_model_id?: string;
  embedding_model_id?: string;
  rerank_model_id?: string;
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  max_reflections?: number;
  retrieval_top_k?: number;
  kg_top_k?: number;
  sql_limit?: number;
  summary_threshold?: number;
  keep_recent_turns?: number;
  streaming?: boolean;
  memory_enabled?: boolean;  // 长期记忆开关
  model_roles?: Array<{
    role: string;
    model_id: string;
    is_default: boolean;
    model_name?: string;
    model_type?: string;
  }>;
}

export interface AgentSession {
  id: string;
  agent_id: string;
  user_id: string;
  title: string;
  pinned?: boolean;
  archived?: boolean;
  message_count?: number;
  created_at?: string;
  updated_at?: string;
}

export interface AgentMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  thinking_events?: unknown[];
  citations?: Array<{ doc_name?: string; title?: string; content?: string; [k: string]: unknown }>;
  feedback?: 'like' | 'dislike';
  created_at?: string;
}

export interface AgentToolDef {
  name: string;
  description?: string;
  [k: string]: unknown;
}
