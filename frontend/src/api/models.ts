import { request, buildQuery } from './request';
import type { LLMModel, ModelType, CurrentModelConfig } from '@/types';

// ---------- 模型管理 ----------
export const fetchModels = (modelType?: string) =>
  request<{ models: LLMModel[] }>(`/models/${modelType ? buildQuery({ model_type: modelType }) : ''}`);

export const fetchModelTypes = () => request<{ types: Array<{ value: string; label: string; description: string }> }>('/models/types');

export const fetchDefaultModels = () => request<Record<string, unknown>>('/models/default');

export const fetchModel = (modelId: string) => request<LLMModel>(`/models/${modelId}`);

export const createModel = (data: Partial<LLMModel>) =>
  request<{ model_id: string; name: string; model_type: string; message: string }>('/models/', {
    method: 'POST',
    body: JSON.stringify(data),
  });

export const updateModel = (modelId: string, data: Partial<LLMModel>) =>
  request<{ message: string }>(`/models/${modelId}`, { method: 'PUT', body: JSON.stringify(data) });

export const deleteModel = (modelId: string) =>
  request<{ message: string }>(`/models/${modelId}`, { method: 'DELETE' });

export const setDefaultModel = (modelType: string, modelId: string) =>
  request<{ message: string }>('/models/set-default', {
    method: 'POST',
    body: JSON.stringify({ model_type: modelType, model_id: modelId }),
  });

export const fetchCurrentModelConfig = () => request<CurrentModelConfig>('/models/current-config');

export const testModelConnection = (modelId: string) =>
  request<{ ok: boolean; message: string; latency_ms?: number }>(`/models/${modelId}/test`, {
    method: 'POST',
    body: JSON.stringify({}),
  });

export const refreshModels = () => request<{ message: string }>('/models/refresh', { method: 'POST' });

export const MODEL_TYPES: Array<{ value: ModelType; label: string }> = [
  { value: 'llm', label: 'Chat LLM' },
  { value: 'embedding', label: 'Embedding' },
  { value: 'rerank', label: 'Rerank' },
  { value: 'vision', label: 'Vision' },
];

export const PROVIDER_LABELS: Record<string, string> = {
  ollama: 'Ollama',
  qwen: '通义千问(Qwen)',
  openai: 'OpenAI',
  huggingface: 'HuggingFace',
};
