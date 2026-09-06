import type { AgentEvidence, AgentExecutionEvent } from '../types';

export type DiagnosisEventType =
  | 'plan' | 'stage' | 'tool_call' | 'reflect' | 'thinking_delta'
  | 'answer_delta' | 'answer' | 'citations' | 'error' | 'done' | 'stopped';

export interface DiagnosisStreamEvent extends AgentExecutionEvent {
  type: DiagnosisEventType;
  content?: string;
  message?: string;
  citations?: AgentEvidence[];
  evidence_catalog?: AgentEvidence[];
  items?: AgentEvidence[];
  completion_status?: 'full' | 'partial';
  unconfirmed_fields?: string[];
  used_evidence_ids?: string[];
}

const EVENT_TYPES = new Set<string>([
  'plan', 'stage', 'tool_call', 'reflect', 'thinking_delta',
  'answer_delta', 'answer', 'citations', 'error', 'done', 'stopped',
]);
const STRING_FIELDS = [
  'stage', 'status', 'route', 'summary', 'tool', 'error', 'intent', 'answer_mode',
  'rewritten_query', 'next_action', 'result_summary', 'content', 'message', 'failure_kind',
];
const NUMBER_FIELDS = ['count', 'evidence_count', 'duration', 'postgres_authoritative_count'];
const ARRAY_FIELDS = [
  'entity_keywords', 'type_keywords', 'tried_tools', 'evidence_ids',
  'approved_evidence_ids', 'blocked_evidence_ids', 'excluded_evidence_ids',
  'reviewed_evidence_ids', 'approved', 'blocked', 'used',
];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function requireShape(condition: boolean): asserts condition {
  if (!condition) throw new Error('诊断事件格式异常，请重试或检查服务日志');
}

function isEvidence(value: unknown): value is AgentEvidence {
  if (!isRecord(value)) return false;
  const strings = [
    'evidence_id', 'status', 'doc_id', 'doc_name', 'title', 'display_summary',
    'content', 'snippet', 'doc_type', 'authority', 'evidence_level',
    'validation_tier', 'source_ref', 'record_id', 'chunk_id', 'image_url',
  ];
  return strings.every((key) => value[key] === undefined || typeof value[key] === 'string')
    && (value.answer_citation_id === undefined || value.answer_citation_id === null
      || (Number.isInteger(value.answer_citation_id) && Number(value.answer_citation_id) > 0))
    && (value.document_metadata === undefined || isRecord(value.document_metadata));
}

export function parseDiagnosisEvent(value: unknown): DiagnosisStreamEvent | null {
  requireShape(isRecord(value) && typeof value.type === 'string');
  // Unknown extension events are forward compatible; known malformed payloads
  // are errors, never silently accepted as a successful answer.
  if (!EVENT_TYPES.has(value.type)) return null;
  for (const key of STRING_FIELDS) requireShape(value[key] === undefined || typeof value[key] === 'string');
  for (const key of NUMBER_FIELDS) requireShape(value[key] === undefined || (typeof value[key] === 'number' && Number.isFinite(value[key])));
  for (const key of ARRAY_FIELDS) requireShape(value[key] === undefined || Array.isArray(value[key]));
  for (const key of ['metrics', 'params', 'retrieval_info']) requireShape(value[key] === undefined || isRecord(value[key]));
  for (const key of ['tools_plan', 'results_preview']) requireShape(value[key] === undefined || (Array.isArray(value[key]) && value[key].every(isRecord)));
  for (const key of ['citations', 'evidence_catalog', 'items']) requireShape(value[key] === undefined || (Array.isArray(value[key]) && value[key].every(isEvidence)));
  for (const key of ['used_evidence_ids', 'unconfirmed_fields']) requireShape(value[key] === undefined || (Array.isArray(value[key]) && value[key].every((item) => typeof item === 'string')));
  for (const key of ['need_more', 'evidence_ids_truncated']) requireShape(value[key] === undefined || typeof value[key] === 'boolean');
  requireShape(value.completion_status === undefined || value.completion_status === 'full' || value.completion_status === 'partial');
  if (value.type === 'answer' || value.type === 'answer_delta') requireShape(typeof value.content === 'string');
  return value as unknown as DiagnosisStreamEvent;
}

export function parseSseChunk(buffer: string): { events: DiagnosisStreamEvent[]; rest: string } {
  const parts = buffer.split(/\r\n\r\n|\n\n|\r\r|\r\n\n|\n\r\n/);
  const rest = parts.pop() || '';
  const events: DiagnosisStreamEvent[] = [];
  for (const part of parts) {
    const lines = part.replace(/^\uFEFF/, '').split(/\r\n|\r|\n/);
    const data = lines.filter((line) => line.startsWith('data:'))
      .map((line) => line.slice(5).replace(/^ /, '')).join('\n');
    if (!data) continue;
    let payload: unknown;
    try { payload = JSON.parse(data); } catch { throw new Error('诊断事件 JSON 不完整或格式错误'); }
    const event = parseDiagnosisEvent(payload);
    if (event) events.push(event);
  }
  return { events, rest };
}

/** One reader owns decoding, resource cleanup and the terminal-event contract. */
export async function* readDiagnosisStream(body: ReadableStream<Uint8Array>): AsyncGenerator<DiagnosisStreamEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) throw new Error('诊断连接提前结束，未收到完成确认；请重新打开会话核对结果');
      buffer += decoder.decode(value, { stream: true });
      const parsed = parseSseChunk(buffer);
      buffer = parsed.rest;
      for (const event of parsed.events) {
        yield event;
        if (event.type === 'done' || event.type === 'stopped') return;
      }
    }
  } finally {
    // Cancellation must not mask a protocol/read error already on its way out.
    try { await reader.cancel(); } catch { /* preserve the original failure */ }
    reader.releaseLock();
  }
}
