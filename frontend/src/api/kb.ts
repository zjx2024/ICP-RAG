import { request, buildQuery } from './request';
import type {
  KnowledgeBase,
  DocumentStatus,
  ParseResult,
  Chunk,
  TaskState,
  GraphData,
  CommunityReportsResponse,
  LogEntry,
  LogTask,
} from '@/types';

// ---------- 知识库 ----------
export const fetchKnowledgeBases = () => request<KnowledgeBase[]>('/knowledge-bases/');
export const getKnowledgeBase = (kbId: string) => request<KnowledgeBase>(`/knowledge-bases/${kbId}`);
export const createKnowledgeBase = (data: { name: string; description?: string }) =>
  request<KnowledgeBase>('/knowledge-bases/', { method: 'POST', body: JSON.stringify(data) });
export const updateKnowledgeBase = (kbId: string, data: { name: string; description?: string }) =>
  request<KnowledgeBase>(`/knowledge-bases/${kbId}/`, { method: 'PUT', body: JSON.stringify(data) });
export const deleteKnowledgeBase = (kbId: string) =>
  request<{ status: string; message: string }>(`/knowledge-bases/${kbId}/`, { method: 'DELETE' });

// ---------- 文档 ----------
export const fetchDocuments = (kbId: string) =>
  request<DocumentStatus[]>(`/knowledge-bases/${kbId}/documents`);
export const getDocument = (kbId: string, docId: string) =>
  request<DocumentStatus>(`/knowledge-bases/${kbId}/documents/${docId}`);

export const parseDocument = (kbId: string, docId: string, modelId?: string | null) =>
  request<ParseResult>(`/knowledge-bases/${kbId}/documents/${docId}/parse`, {
    method: 'POST',
    body: JSON.stringify({ model_id: modelId ?? null }),
  });

export const deleteDocument = (kbId: string, docId: string) =>
  request<Record<string, unknown>>(`/knowledge-bases/${kbId}/documents/${docId}`, { method: 'DELETE' });

export const deleteDocGraphData = (kbId: string, docId: string) =>
  request<Record<string, unknown>>(`/knowledge-bases/${kbId}/documents/${docId}/graph-data`, {
    method: 'DELETE',
  });

export const fetchChunks = (kbId: string, docId: string) =>
  request<Chunk[]>(`/knowledge-bases/${kbId}/documents/${docId}/chunks`);

export const updateChunkSelection = (kbId: string, docId: string, chunkIds: string[], selected: boolean) =>
  request<{ status: string; message: string }>(`/knowledge-bases/${kbId}/documents/${docId}/chunks/selection`, {
    method: 'POST',
    body: JSON.stringify({ chunk_ids: chunkIds, selected }),
  });

export const generateKG = (
  kbId: string,
  docId: string,
  data: {
    selected_chunk_ids: string[];
    model_id?: string | null;
    mode?: string;
    entity_resolution?: boolean;
    with_community?: boolean;
  },
) =>
  request<{ task_id: string; doc_id: string; status: string }>(
    `/knowledge-bases/${kbId}/documents/${docId}/generate_kg`,
    { method: 'POST', body: JSON.stringify({ model_id: null, mode: 'enhanced', entity_resolution: true, with_community: true, ...data }) },
  );

// ---------- 文档内容查看 ----------
export const fetchOriginalChunks = (kbId: string, docId: string) =>
  request<{ doc_id: string; doc_name: string; chunks: Chunk[] }>(
    `/knowledge-bases/${kbId}/documents/${docId}/original-chunks`,
  );

export const fetchVectorStoreContent = (kbId: string, docId: string) =>
  request<{ doc_id: string; doc_name: string; contents: Array<{ chunk_id: string; content_type: string; content: string; doc_name: string }>; error?: string }>(
    `/knowledge-bases/${kbId}/documents/${docId}/vector-store-content`,
  );

export const fetchPostgresContent = (kbId: string, docId: string) =>
  request<{ doc_id: string; doc_name: string; images: unknown[]; tables: unknown[] }>(
    `/knowledge-bases/${kbId}/documents/${docId}/postgres-content`,
  );

// 清除向量库中重复解析产生的冗余索引块（以 PG 文档为有效依据）
export const dedupIndex = (kbId: string) =>
  request<{ total_deleted: number; details: Array<{ doc_id: string; deleted: number }>; message: string }>(
    `/knowledge-bases/${kbId}/documents/dedup-index`,
    { method: 'POST', body: JSON.stringify({}) },
  );

// ---------- 图谱数据 ----------
export const fetchKBGraphData = (kbId: string) => request<GraphData>(`/knowledge-bases/${kbId}/graph-data`);
export const fetchDocGraphData = (kbId: string, docId: string) =>
  request<GraphData>(`/knowledge-bases/${kbId}/documents/${docId}/graph-data`);
export const fetchCommunityReports = (kbId: string, docId?: string) =>
  request<CommunityReportsResponse>(
    `/knowledge-bases/${kbId}/community-reports${docId ? buildQuery({ doc_id: docId }) : ''}`,
  );
export const fetchKGStats = (kbId: string) =>
  request<{ kb_id: string; stats: { vertices: number; edges: number } }>(`/knowledge-bases/${kbId}/hugegraph-stats`);

// ---------- 任务 ----------
export const fetchTaskStatus = (taskId: string) => request<TaskState>(`/knowledge-bases/tasks/${taskId}/status`);
export const cancelTask = (taskId: string) =>
  request<{ success: boolean; message: string; task_id: string }>(
    `/knowledge-bases/tasks/${taskId}/cancel`,
    { method: 'POST' },
  );

export function openTaskStream(taskId: string): EventSource {
  return new EventSource(`/api/v1/knowledge-bases/tasks/${taskId}/stream`);
}

// ---------- 日志 ----------
export const fetchLogTasks = (kbId?: string) =>
  request<{ kb_id?: string; kb_name?: string; kb_exists?: boolean; total_tasks: number; tasks: LogTask[] }>(
    kbId ? `/knowledge-bases/${kbId}/logs/tasks` : `/knowledge-bases/logs/tasks`,
  );

export const fetchDeletionLogTasks = (kbId?: string) =>
  request<{ kb_id?: string; total_tasks: number; tasks: LogTask[] }>(
    kbId ? `/knowledge-bases/${kbId}/deletion-logs/tasks` : `/knowledge-bases/deletion-logs/tasks`,
  );

export const fetchLogEntries = (kbId: string, docId: string, taskId: string, type: 'parse' | 'kg') =>
  request<{ task_id: string; doc_name: string; doc_exists?: boolean; kb_name?: string; kb_exists?: boolean; summary?: string | Record<string, unknown>; total_logs: number; logs: LogEntry[] }>(
    `/knowledge-bases/${kbId}/documents/${docId}/${type}-logs/${taskId}`,
  );

export const fetchDeletionLogEntries = (kbId: string, taskId: string) =>
  request<{ task_id: string; kb_id: string; doc_id: string; kb_name?: string; doc_name?: string; kb_exists?: boolean; doc_exists?: boolean; logs: LogEntry[]; total: number }>(
    `/knowledge-bases/${kbId}/deletion-logs/${taskId}`,
  );

export const deleteDeletionLogs = (kbId: string, taskId: string) =>
  request<{ task_id: string; logs_deleted: number }>(`/knowledge-bases/${kbId}/deletion-logs/${taskId}`, {
    method: 'DELETE',
  });

// ---------- Dify ----------
export const fetchDifyServerInfo = () =>
  request<{ lan_ip: string; source: string }>('/dify/server-info');
