import { request, buildQuery } from './request';
import type {
  DocTypeInfo,
  ICPDocument,
  ICPParseResult,
  ICPOriginalResponse,
  ICPEvidencePreview,
  ICPVectorResponse,
  ICPRowsResponse,
  ICPFaultTree,
} from '@/types';

export const fetchDocumentTypes = () => request<DocTypeInfo[]>('/document-types');
export const fetchDocumentType = (docType: string) =>
  request<DocTypeInfo>(`/document-types/${docType}`);

export const fetchIcpDocuments = (docType: string, status?: string) =>
  request<ICPDocument[]>(`/document-types/${docType}/documents${buildQuery({ status })}`);

export const uploadIcpDocuments = (docType: string, files: File[]) => {
  const fd = new FormData();
  for (const file of files) fd.append('files', file);
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

export const fetchOriginal = (
  docType: string,
  docId: string,
  params?: { page?: number; size?: number },
) => request<ICPOriginalResponse>(
  `/document-types/${docType}/documents/${docId}/original${buildQuery({ ...params })}`,
);

export const fetchEvidencePreview = (
  docType: string,
  docId: string,
  params?: { source_ref?: string; record_id?: string; title?: string },
) => request<ICPEvidencePreview>(
  `/document-types/${docType}/documents/${docId}/evidence-preview${buildQuery({ ...(params || {}) })}`,
);

export const fetchVectors = (
  docType: string,
  docId: string,
  params?: { page?: number; size?: number },
) => request<ICPVectorResponse>(
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
) => request<ICPRowsResponse>(
  `/document-types/${docType}/documents/${docId}/rows${buildQuery({ ...(params || {}) })}`,
);

/** Function Locator 与故障历史使用专用结构表，其余表格类沿用 /rows。 */
export const fetchDomainRows = (
  docType: string,
  docId: string,
  params?: { page?: number; size?: number; keyword?: string },
) => {
  if (docType === 'function_locator') {
    return request<ICPRowsResponse>(`/icp-domain/functions/${docId}${buildQuery({ ...(params || {}) })}`);
  }
  if (docType === 'fault_history') {
    return request<ICPRowsResponse>(`/icp-domain/fault-history/${docId}${buildQuery({ ...(params || {}) })}`);
  }
  return fetchRows(docType, docId, params);
};

export const fetchFaultTrees = (docType: string, docId: string) =>
  request<ICPFaultTree[]>(`/document-types/${docType}/documents/${docId}/fault-trees`);

export const updateDiagramRevision = (
  docType: string,
  docId: string,
  revisionId: string,
  canonicalGraph: ICPFaultTree['graph'],
  lockVersion: number,
) => request<Record<string, unknown>>(
  `/document-types/${docType}/documents/${docId}/diagram-revisions/${revisionId}`,
  {
    method: 'PATCH',
    body: JSON.stringify({ canonical_graph: canonicalGraph, lock_version: lockVersion }),
  },
);

export const confirmDiagramRevision = (
  docType: string,
  docId: string,
  revisionId: string,
  data: { reviewed_by: string; review_note?: string; lock_version?: number },
) => request<Record<string, unknown>>(
  `/document-types/${docType}/documents/${docId}/diagram-revisions/${revisionId}/confirm`,
  { method: 'POST', body: JSON.stringify(data) },
);

export const rejectDiagramRevision = (
  docType: string,
  docId: string,
  revisionId: string,
  data: { reviewed_by: string; review_note: string; lock_version?: number },
) => request<Record<string, unknown>>(
  `/document-types/${docType}/documents/${docId}/diagram-revisions/${revisionId}/reject`,
  { method: 'POST', body: JSON.stringify(data) },
);

export const fetchIcpGraphData = (docType: string, docId: string) =>
  request<{
    doc_id: string;
    doc_type: string;
    vertices: Array<Record<string, unknown>>;
    edges: Array<Record<string, unknown>>;
    total_vertices: number;
    total_edges: number;
    structured_vertices?: number;
    structured_edges?: number;
    diagram_validation?: Array<{
      revision_id?: string;
      validation_status?: string;
      validation_tier?: string;
      validation_method?: string;
    }>;
    error?: string;
  }>(`/document-types/${docType}/documents/${docId}/graph-data`);

export const generateIcpKg = (docType: string, docId: string, revisionIds?: string[]) =>
  request<{ task_id: string; doc_id: string; doc_type: string; revision_ids?: string[]; status: string }>(
    `/document-types/${docType}/documents/${docId}/generate_kg`,
    {
      method: 'POST',
      // 省略 revision_ids 表示由后端重试全部未完成项；显式空数组会被 API 视为错误输入。
      body: JSON.stringify(revisionIds?.length ? { revision_ids: revisionIds } : {}),
    },
  );

export const fetchDiagnosisReadiness = () => request<{
  types: Array<{
    doc_type: string;
    name: string;
    doc_count: number;
    ready_count: number;
    failed_count: number;
    record_count: number;
  }>;
  all_types_present: boolean;
  ready_type_count: number;
  total_records: number;
}>('/icp-domain/readiness');

export const calculateFta = (diagramId: string, missionTime?: number) =>
  request<Record<string, unknown>>(`/icp-domain/fta/${diagramId}/calculate`, {
    method: 'POST',
    body: JSON.stringify({ mission_time: missionTime ?? null }),
  });

export const calculateRbd = (diagramId: string, missionTime?: number, maxPaths = 12) =>
  request<Record<string, unknown>>(`/icp-domain/rbd/${diagramId}/calculate`, {
    method: 'POST',
    body: JSON.stringify({ mission_time: missionTime ?? null, max_paths: maxPaths }),
  });

export const icpStructuredSearch = (data: {
  doc_type: string;
  query: string;
  k?: number;
  filters?: Record<string, string>;
}) => request<{ doc_type: string; query: string; total: number; hits: unknown[] }>(
  '/document-types/search',
  { method: 'POST', body: JSON.stringify(data) },
);
