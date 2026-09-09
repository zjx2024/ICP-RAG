import { request } from './request';
import type { LogEntry, LogTask } from '@/types';

// 通用知识库 CRUD、上传、解析、KG 生成、社区报告与 Dify 前端入口已随旧链路退役。
// 这里只保留 ICP 文档管理页仍在复用的任务监控与解析/图谱日志查询。

// ---------- 任务 ----------
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

export const fetchLogEntries = (kbId: string, docId: string, taskId: string, type: 'parse' | 'kg') =>
  request<{ task_id: string; doc_name: string; doc_exists?: boolean; kb_name?: string; kb_exists?: boolean; summary?: string | Record<string, unknown>; total_logs: number; logs: LogEntry[] }>(
    `/knowledge-bases/${kbId}/documents/${docId}/${type}-logs/${taskId}`,
  );
