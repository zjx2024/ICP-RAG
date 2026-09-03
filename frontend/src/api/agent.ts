import { request, buildQuery } from './request';
import type {
  AgentConfig,
  AgentSession,
  AgentMessage,
  AgentToolDef,
} from '@/types';

const API = '/api';

// ---------- 智能体 ----------
export const fetchAgents = (activeOnly = false) =>
  request<AgentConfig[]>(`${API}/agent${activeOnly ? '?active_only=true' : ''}`);

export const fetchAgentTools = () => request<AgentToolDef[]>(`${API}/agent/tools`);

export const fetchAgent = (agentId: string) => request<AgentConfig>(`${API}/agent/${agentId}`);

export const createAgent = (data: Record<string, unknown>) =>
  request<AgentConfig>(`${API}/agent`, { method: 'POST', body: JSON.stringify(data) });

export const updateAgent = (agentId: string, data: Record<string, unknown>) =>
  request<AgentConfig>(`${API}/agent/${agentId}`, { method: 'PUT', body: JSON.stringify(data) });

export const deleteAgent = (agentId: string) =>
  request<{ deleted: boolean }>(`${API}/agent/${agentId}`, { method: 'DELETE' });

export const bindAgentModel = (agentId: string, role: string, modelId: string, isDefault = true) =>
  request<Record<string, unknown>>(`${API}/agent/${agentId}/models`, {
    method: 'POST',
    body: JSON.stringify({ role, model_id: modelId, is_default: isDefault }),
  });

export const fetchAgentModels = (agentId: string) =>
  request<Array<{ role: string; model_id: string; is_default: boolean; model_name?: string; model_type?: string }>>(
    `${API}/agent/${agentId}/models`,
  );

export const unbindAgentModel = (agentId: string, role: string) =>
  request<{ unbound: boolean }>(`${API}/agent/${agentId}/models/${role}`, { method: 'DELETE' });

// ---------- 会话 ----------
export const fetchSessions = (agentId: string) =>
  request<AgentSession[]>(`${API}/session${buildQuery({ agent_id: agentId })}`);

export const createSession = (agentId: string, title = '新诊断会话') =>
  request<AgentSession>(`${API}/session`, {
    method: 'POST',
    body: JSON.stringify({ agent_id: agentId, user_id: 'default', title }),
  });

export const fetchSession = (sessionId: string) =>
  request<AgentSession & { messages: AgentMessage[] }>(`${API}/session/${sessionId}`);

export const updateSession = (sessionId: string, data: Record<string, unknown>) =>
  request<AgentSession>(`${API}/session/${sessionId}`, { method: 'PUT', body: JSON.stringify(data) });

export const deleteSession = (sessionId: string) =>
  request<{ deleted: boolean }>(`${API}/session/${sessionId}`, { method: 'DELETE' });

export const generateSessionTitle = (sessionId: string) =>
  request<AgentSession>(`${API}/session/${sessionId}/title`, { method: 'POST' });

export const compressSession = (sessionId: string) =>
  request<{ session_id: string; before_tokens: number; after_tokens: number; deleted_count: number; summary: string }>(
    `${API}/session/${sessionId}/compress`,
    { method: 'POST' },
  );

// ---------- 对话（SSE 流式） ----------
export function chatStream(
  body: { agent_id: string; session_id: string; query: string; user_id: string; chat_model_id: string; embedding_model_id: string },
  signal?: AbortSignal,
): Promise<Response> {
  return fetch(`${API}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });
}

export const stopChat = (sessionId: string) =>
  request<{ stopped: boolean; session_id: string; reason?: string }>(
    `${API}/chat/stop${buildQuery({ session_id: sessionId })}`,
    { method: 'POST' },
  );

export const sendFeedback = (messageId: string, feedback: 'like' | 'dislike') =>
  request<{ message_id: string; feedback: string }>(
    `${API}/message/${messageId}/feedback${buildQuery({ feedback })}`,
    { method: 'POST' },
  );

// ---------- 长期记忆 ----------
export const fetchMemories = () => request<unknown[]>(`${API}/memory`);
export const deleteMemory = (key: string) =>
  request<{ deleted: boolean; key: string }>(`${API}/memory/${encodeURIComponent(key)}`, { method: 'DELETE' });
export const clearMemories = () => request<{ cleared: boolean }>(`${API}/memory`, { method: 'DELETE' });

// ---------- 模型（复用 /api/v1） ----------
export const fetchLLMModels = () =>
  request<{ models: Array<{ model_id: string; name: string; model_type?: string; is_default: boolean }> }>(
    `/api/v1/models/${buildQuery({ model_type: 'llm' })}`,
  );
