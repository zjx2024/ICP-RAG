import { request, buildQuery } from './request';
import type {
  DocTypeInfo,
  ICPDocument,
  ICPParseResult,
  ICPOriginalResponse,
  ICPVectorResponse,
  ICPRowsResponse,
  ICPFaultTree,
} from '@/types';

// ---------- 类型注册表 ----------
export const fetchDocumentTypes = () => request<DocTypeInfo[]>('/document-types');
export const fetchDocumentType = (docType: string) =>
  request<DocTypeInfo>(`/document-types/${docType}`);

// ---------- 类型下文档管理 ----------
export const fetchIcpDocuments = (docType: string, status?: string) =>
  request<ICPDocument[]>(`/document-types/${docType}/documents${buildQuery({ status })}`);

export const uploadIcpDocuments = (docType: string, files: File[]) => {
  const fd = new FormData();
  for (const f of files) fd.append('files', f);
  return request<Array<{ doc_id: string; file_name: string; status: string }>>(
    `/document-types/${docType}/documents/upload`,
    { method: 'POST', body: fd },
  );
};

export const parseIcpDocument = (docType: string, docId: string, modelId?: string | null) =>
  request<{ task_id: string; doc_id: string; doc_type: string; status: string }>(
    `/document-types/${docType}/documents/${docId}/parse`,
    { method: 'POST', body: JSON.stringify({ model_id: modelId ?? null }) },
  );

export const fetchParseResult = (docType: string, docId: string) =>
  request<ICPParseResult>(`/document-types/${docType}/documents/${docId}/parse-result`);

export const deleteIcpDocument = (docType: string, docId: string) =>
  request<{ status: string; doc_id: string; message: string }>(
    `/document-types/${docType}/documents/${docId}`,
    { method: 'DELETE' },
  );

// ---------- 四视图内容 ----------
export const fetchOriginal = (
  docType: string,
  docId: string,
  params?: { page?: number; size?: number },
) =>
  request<ICPOriginalResponse>(
    `/document-types/${docType}/documents/${docId}/original${buildQuery({ ...params })}`,
  );

export const fetchVectors = (
  docType: string,
  docId: string,
  params?: { page?: number; size?: number },
) =>
  request<ICPVectorResponse>(
    `/document-types/${docType}/documents/${docId}/vectors${buildQuery({ ...params })}`,
  );

export const fetchRows = (
  docType: string,
  docId: string,
  params?: {
    page?: number;
    size?: number;
    keyword?: string;
    severity?: string;
    api_reportable?: string;
    platform_test?: string;
  },
) =>
  request<ICPRowsResponse>(
    `/document-types/${docType}/documents/${docId}/rows${buildQuery({ ...(params || {}) })}`,
  );

export const fetchFaultTrees = (docType: string, docId: string) =>
  request<ICPFaultTree[]>(`/document-types/${docType}/documents/${docId}/fault-trees`);

export const fetchIcpGraphData = (docType: string, docId: string) =>
  request<{ doc_id: string; doc_type: string; vertices: Array<Record<string, unknown>>; edges: Array<Record<string, unknown>>; total_vertices: number; total_edges: number; error?: string }>(
    `/document-types/${docType}/documents/${docId}/graph-data`,
  );

export const generateIcpKg = (docType: string, docId: string, modelId?: string | null) =>
  request<{ task_id: string; doc_id: string; doc_type: string; status: string }>(
    `/document-types/${docType}/documents/${docId}/generate_kg`,
    { method: 'POST', body: JSON.stringify({ model_id: modelId ?? null }) },
  );

// ---------- 混合检索 ----------
export const icpStructuredSearch = (data: {
  doc_type: string;
  query: string;
  k?: number;
  filters?: Record<string, string>;
}) => request<{ doc_type: string; query: string; total: number; hits: unknown[] }>(
  '/document-types/search',
  { method: 'POST', body: JSON.stringify(data) },
);

// 任务 SSE 复用全局任务流端点（types.py 与 documents.py 共用 task_manager）
export const openTaskStream = (taskId: string): EventSource =>
  new EventSource(`/api/v1/knowledge-bases/tasks/${taskId}/stream`);
