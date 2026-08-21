import { request, buildQuery } from './request';
import type { RetrievalStats, SearchResponse } from '@/types';

// ---------- 检索 ----------
export const fetchRetrievalModes = (kbId: string) =>
  request<{ modes: Array<{ mode: string; name: string; description: string }>; description: string }>(
    `/knowledge-bases/${kbId}/retrieval-modes`,
  );

export const fetchRetrievalStats = (kbId: string) => request<RetrievalStats>(`/knowledge-bases/${kbId}/stats`);

// 统一检索入口
export function search(kbId: string, mode: string, params: Record<string, string | number | boolean>) {
  // 下划线转连字符
  const pathMode = mode.replaceAll('_', '-');
  const qs = buildQuery(params);
  return request<SearchResponse>(`/knowledge-bases/${kbId}/search/${pathMode}${qs}`);
}

// 高级检索：实体关系等
export const fetchTopEntities = (kbId: string, limit = 20) =>
  request<{ kb_id: string; entities: unknown[]; total: number }>(
    `/knowledge-bases/${kbId}/entities/top${buildQuery({ limit })}`,
  );

export const fetchEntitiesByType = (kbId: string, entityType: string, limit = 100) =>
  request<{ kb_id: string; entity_type: string; entities: unknown[]; total: number }>(
    `/knowledge-bases/${kbId}/entities/type${buildQuery({ entity_type: entityType, limit })}`,
  );

export const fetchEntityRelations = (kbId: string, entityName: string, limit = 50) =>
  request<{ kb_id: string; entity_name: string; relations: unknown[]; total: number }>(
    `/knowledge-bases/${kbId}/entities/${encodeURIComponent(entityName)}/relations${buildQuery({ limit })}`,
  );

export const fetchEntityPath = (kbId: string, source: string, target: string) =>
  request<{ kb_id: string; source: string; target: string; paths: unknown[]; total_paths: number }>(
    `/knowledge-bases/${kbId}/entities/path${buildQuery({ source, target })}`,
  );

export const fetchDifyRetrieval = (data: {
  knowledge_id: string;
  query: string;
  retrieval_setting?: { top_k?: number; score_threshold?: number };
  metadata_condition?: unknown;
}) =>
  request<{ records: unknown[] }>('/dify/retrieval', {
    method: 'POST',
    body: JSON.stringify(data),
  });
