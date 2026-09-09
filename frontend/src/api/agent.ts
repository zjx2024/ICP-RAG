import { request, buildQuery } from './request';
import type {
  AgentConfig,
  AgentSession,
  AgentMessage,
} from '@/types';

const API = '/api';

// 默认 ICP 智能体由后端 icp_defaults 幂等创建，前端只读取与选择；
// agent CRUD、模型绑定、长期记忆与压缩等通用 Agent 平台入口已不在诊断工作台暴露。

// ---------- 智能体 ----------
export const fetchAgents = (activeOnly = false) =>
  request<AgentConfig[]>(`${API}/agent${activeOnly ? '?active_only=true' : ''}`);

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
