import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react';
import type { CSSProperties, KeyboardEvent as ReactKeyboardEvent, PointerEvent as ReactPointerEvent } from 'react';
import {
  Button, Card, Empty, Input, Layout, List, Modal, Popconfirm,
  Space, Spin, Tag, Tooltip, Typography, message,
} from 'antd';
import {
  DeleteOutlined, EditOutlined, FileSearchOutlined, PlusOutlined, RobotOutlined,
  SendOutlined, StopOutlined,
} from '@ant-design/icons';
import {
  chatStream, createSession, deleteSession, fetchAgents, fetchSession,
  fetchSessions, generateSessionTitle, stopChat, updateSession,
} from '@/api/agent';
import { GlassButton } from '@/components/GlassUI';
import EvidenceSourcePreview from '@/components/EvidenceSourcePreview';
import { extractCitationNumbers, formatMarkdown, formatTimeShort } from '@/utils';
import type { Citation } from '@/utils';
import type { AgentConfig, AgentEvidence, AgentMessage, AgentSession } from '@/types';

const { Sider, Content } = Layout;
const { TextArea } = Input;
const DEFAULT_SESSION_TITLE = '新诊断会话';
const EVIDENCE_WIDTH_STORAGE_KEY = 'icp-diagnosis-evidence-width';
const DEFAULT_EVIDENCE_WIDTH = 340;
const MIN_EVIDENCE_WIDTH = 280;
const MAX_EVIDENCE_WIDTH = 520;
const MIN_CHAT_WIDTH = 460;
const RESIZER_WIDTH = 14;

function clampEvidenceWidth(value: number, containerWidth?: number) {
  const availableMax = containerWidth
    ? Math.max(MIN_EVIDENCE_WIDTH, containerWidth - MIN_CHAT_WIDTH - RESIZER_WIDTH)
    : MAX_EVIDENCE_WIDTH;
  return Math.min(Math.max(value, MIN_EVIDENCE_WIDTH), Math.min(MAX_EVIDENCE_WIDTH, availableMax));
}

function storedEvidenceWidth() {
  if (typeof window === 'undefined') return DEFAULT_EVIDENCE_WIDTH;
  try {
    const value = Number.parseInt(window.localStorage.getItem(EVIDENCE_WIDTH_STORAGE_KEY) || '', 10);
    return Number.isFinite(value) ? clampEvidenceWidth(value) : DEFAULT_EVIDENCE_WIDTH;
  } catch {
    return DEFAULT_EVIDENCE_WIDTH;
  }
}

interface ExecutionEvent {
  type?: string;
  stage?: string;
  status?: string;
  route?: string;
  summary?: string;
  metrics?: Record<string, unknown>;
  tool?: string;
  count?: number;
  evidence_count?: number;
  duration?: number;
  params?: Record<string, unknown>;
  results_preview?: Array<Record<string, unknown>>;
  retrieval_info?: Record<string, unknown>;
  postgres_authoritative_count?: number;
  error?: string;
  intent?: string;
  answer_mode?: string;
  rewritten_query?: string;
  entity_keywords?: unknown[];
  type_keywords?: unknown[];
  tools_plan?: Array<Record<string, unknown>>;
  need_more?: boolean;
  next_action?: string;
  result_summary?: string;
  tried_tools?: unknown[];
  evidence_ids?: unknown[];
  approved_evidence_ids?: unknown[];
  blocked_evidence_ids?: unknown[];
  excluded_evidence_ids?: unknown[];
  reviewed_evidence_ids?: unknown[];
  evidence_ids_truncated?: boolean;
  used_evidence_ids?: unknown[];
  approved?: unknown[];
  blocked?: unknown[];
  used?: unknown[];
}

const TOOL_LABEL: Record<string, string> = {
  function_locator: '功能定位',
  hybrid_search: '七类文档混合召回',
  hmc_query: 'HMC 查询',
  fmeca_query: 'FMECA 查询',
  fta_query: 'FTA 原因链',
  fta_analyzer: 'FTA 确定性分析',
  rbd_query: 'RBD 可靠性结构',
  rbd_calculator: 'RBD 可靠度计算',
  mfl_query: 'MFL 维修定位',
  fault_history_query: '故障历史',
  graph_traversal: '跨文档关系查询',
  fault_reasoning: '候选融合与置信度判断',
  vector_search: '文档检索',
  kg_search: '关系检索',
  sql_query: '结构化数据查询',
};

const INTENT_LABEL: Record<string, string> = {
  pending: '理解用户问题',
  exact_code: '故障代码查询',
  symptom_diagnosis: '故障现象诊断',
  architecture_query: '系统架构查询',
  entity_profile: '对象基础信息',
  function_location: '功能与模块定位',
  cause_analysis: '故障原因分析',
  impact_analysis: '故障影响分析',
  maintenance_query: '维修方法查询',
  reliability_query: '可靠性分析',
  history_query: '历史案例查询',
  hybrid: '综合知识查询',
};

const DOC_TYPE_LABEL: Record<string, string> = {
  function_locator: '系统架构与功能说明',
  hmc: 'HMC 故障代码手册',
  fta: 'FTA 故障树',
  rbd: 'RBD 可靠性框图',
  fmeca: 'FMECA 分析',
  mfl: 'MFL 维修清单',
  fault_history: '故障历史',
};

const ANSWER_MODE_LABEL: Record<string, string> = {
  architecture_knowledge: '架构知识回答',
  knowledge_explanation: '知识解释',
  fault_diagnosis: '故障诊断',
  maintenance_guidance: '维修指导',
  exact_code_lookup: '代码精确查询',
};

type ExecutionStepKind = 'plan' | 'tool' | 'review' | 'answer';
type ExecutionStepStatus = 'complete' | 'active' | 'empty' | 'error';

interface ExecutionDetailItem {
  title: string;
  description?: string;
  evidenceId?: string;
  citationNumber?: number;
  disposition?: EvidenceDisposition;
  sourceRef?: string;
}

interface ExecutionDetailSection {
  label: string;
  text?: string;
  tags?: string[];
  items?: ExecutionDetailItem[];
  note?: string;
}

interface ExecutionStep {
  key: string;
  kind: ExecutionStepKind;
  title: string;
  facts: string[];
  status: ExecutionStepStatus;
  sections: ExecutionDetailSection[];
}

type EvidenceDisposition = 'used' | 'approved' | 'blocked' | 'candidate';

interface EvidenceViewItem {
  key: string;
  evidenceId?: string;
  citationNumber?: number;
  evidence: Citation;
  disposition: EvidenceDisposition;
}

interface EvidenceView {
  answer: EvidenceViewItem[];
  retrieval: EvidenceViewItem[];
  all: EvidenceViewItem[];
  candidateCount: number;
  approvedCount: number;
  blockedCount: number;
}

function asFiniteNumber(value: unknown): number | undefined {
  if (value === null || value === undefined || value === '') return undefined;
  const number = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(number) ? number : undefined;
}

function scalarText(value: unknown): string {
  if (typeof value === 'string') return value.trim();
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  return '';
}

function compactText(value: unknown, maxLength = 220): string {
  const text = scalarText(value).replace(/\s+/g, ' ').trim();
  if (!text) return '';
  return text.length > maxLength ? `${text.slice(0, maxLength).trim()}…` : text;
}

type EvidenceSummaryKind = 'source' | 'structure' | 'calculation';

interface EvidencePresentation {
  label: '原文摘要' | '结构摘要' | '计算摘要';
  text: string;
}

function objectRecord(value: unknown): Record<string, unknown> | undefined {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return undefined;
  return value as Record<string, unknown>;
}

function evidenceSummaryKind(evidence: Record<string, unknown>): EvidenceSummaryKind {
  const claimType = scalarText(evidence.claim_type || evidence.type).toLowerCase();
  if (claimType.includes('calculation') || scalarText(evidence.source_kind) === 'calculation') {
    return 'calculation';
  }
  if (claimType.includes('structure') || claimType.includes('relation') || claimType.includes('diagram')) {
    return 'structure';
  }
  return 'source';
}

function evidenceSummaryLabel(
  evidence: Record<string, unknown>,
  detectedKind?: EvidenceSummaryKind,
): EvidencePresentation['label'] {
  const kind = detectedKind || evidenceSummaryKind(evidence);
  if (kind === 'structure') return '结构摘要';
  if (kind === 'calculation') return '计算摘要';
  return '原文摘要';
}

function structuredValue(value: unknown): { value?: unknown; lookedStructured: boolean } {
  if (value && typeof value === 'object') return { value, lookedStructured: true };
  const text = scalarText(value);
  const objectLike = /^\{\s*(?:"|})/.test(text);
  const arrayLike = /^\[\s*(?:[\[{"\]\d-]|true\b|false\b|null\b)/.test(text);
  if (!text || (!objectLike && !arrayLike)) {
    return { lookedStructured: false };
  }
  try {
    return { value: JSON.parse(text), lookedStructured: true };
  } catch {
    // Older messages may contain JSON truncated by the server-side evidence cap.
    // Treat it as structured data even though it can no longer be parsed safely.
    const truncatedArrayLike = /^\[\s*[\[{"\]]/.test(text);
    return { lookedStructured: objectLike || truncatedArrayLike };
  }
}

function recordArray(record: Record<string, unknown>, ...keys: string[]): Array<Record<string, unknown>> {
  for (const key of keys) {
    const value = record[key];
    if (!Array.isArray(value)) continue;
    return value.map(objectRecord).filter((item): item is Record<string, unknown> => Boolean(item));
  }
  return [];
}

function readableList(value: unknown, limit = 4): string[] {
  if (!Array.isArray(value)) return [];
  return Array.from(new Set(value.flatMap((item) => {
    if (Array.isArray(item)) {
      const path = item.map((part) => compactText(part, 45)).filter(Boolean).join(' → ');
      return path ? [path] : [];
    }
    const text = compactText(item, 80);
    return text ? [text] : [];
  }))).slice(0, limit);
}

function isTechnicalIdentifier(value: string): boolean {
  return /^(?:node|edge|vertex|revision|diagram|ev)_[a-z0-9_-]+$/i.test(value)
    || /^[0-9a-f]{8}-[0-9a-f-]{27,}$/i.test(value)
    || /^[0-9a-f]{24,}$/i.test(value);
}

function graphNodeName(node: Record<string, unknown>): string {
  const properties = objectRecord(node.properties) || {};
  const candidates = [
    properties.name, properties.title, properties.text, properties.event_name, properties.block_name,
    node.name, node.title, node.text,
  ];
  for (const candidate of candidates) {
    const text = compactText(candidate, 70);
    if (text && !isTechnicalIdentifier(text)) return text;
  }
  return '';
}

function graphStructureSummary(
  graph: Record<string, unknown>,
  evidence: Record<string, unknown>,
): string {
  const docType = scalarText(evidence.doc_type).toLowerCase();
  const graphType = scalarText(graph.graph_type || graph.diagram_type).toLowerCase();
  const isRbd = docType === 'rbd' || graphType === 'rbd';
  const nodes = recordArray(graph, 'vertices', 'nodes');
  const edges = recordArray(graph, 'edges', 'relations', 'links');
  const nodeNames = Array.from(new Set(nodes.map(graphNodeName).filter(Boolean)));
  const gateTypes = Array.from(new Set(edges.flatMap((edge) => {
    const properties = objectRecord(edge.properties) || {};
    const gate = compactText(properties.gate_type ?? edge.gate_type, 12).toUpperCase();
    return ['AND', 'OR', 'XOR'].includes(gate) ? [gate] : [];
  })));
  const parts: string[] = [];
  const subject = compactText(graph.top_event ?? graph.title ?? evidence.title, 120);
  if (subject) parts.push(`${isRbd ? '框图主题' : '故障树主题'}：${subject}`);
  if (nodes.length || edges.length) {
    parts.push(`${nodes.length} 个${isRbd ? '功能节点' : '事件节点'}、${edges.length} 条${isRbd ? '连接关系' : '因果关系'}`);
  }
  if (gateTypes.length) parts.push(`逻辑门：${gateTypes.join('、')}`);
  if (nodeNames.length) parts.push(`关键节点：${nodeNames.slice(0, 6).join('、')}`);
  return parts.join('；');
}

function calculationSummary(
  result: Record<string, unknown>,
  evidence: Record<string, unknown>,
): string {
  const model = scalarText(result.model || evidence.doc_type).toUpperCase();
  const parts: string[] = [];
  const title = compactText(result.title ?? evidence.title, 120);
  if (title) parts.push(`${model === 'RBD' ? '可靠性模型' : '故障树'}：${title}`);
  if (model === 'RBD') {
    const pathCount = asFiniteNumber(result.success_path_count);
    if (pathCount !== undefined) parts.push(`成功路径 ${pathCount} 条`);
    const paths = readableList(result.success_paths, 2);
    if (paths.length) parts.push(`代表路径：${paths.join('；')}`);
    const reliability = asFiniteNumber(result.system_reliability);
    if (reliability !== undefined) parts.push(`系统可靠度：${reliability}`);
    const missing = readableList(result.missing_reliability_blocks, 4);
    if (missing.length) parts.push(`缺少可靠度参数：${missing.join('、')}`);
    const limitReason = compactText(result.calculation_limit_reason, 160);
    if (limitReason) parts.push(limitReason);
  } else {
    const topEvent = compactText(result.top_event, 120);
    if (topEvent) parts.push(`顶事件：${topEvent}`);
    const cutCount = asFiniteNumber(result.minimal_cut_set_count);
    if (cutCount !== undefined) parts.push(`最小割集 ${cutCount} 个`);
    const cutSets = readableList(result.minimal_cut_sets, 2);
    if (cutSets.length) parts.push(`代表割集：${cutSets.join('；')}`);
    const probability = asFiniteNumber(result.top_event_probability);
    if (probability !== undefined) parts.push(`顶事件概率：${probability}`);
    const missing = readableList(result.missing_probability_nodes, 4);
    if (missing.length) parts.push(`缺少概率参数：${missing.join('、')}`);
    const blockReason = compactText(result.probability_block_reason, 160);
    if (blockReason) parts.push(blockReason);
  }
  return parts.join('；');
}

function detectedStructuredKind(
  evidence: Record<string, unknown>,
  record?: Record<string, unknown>,
  rawText = '',
): EvidenceSummaryKind | undefined {
  const declaredKind = evidenceSummaryKind(evidence);
  if (declaredKind !== 'source') return declaredKind;
  if (!record) {
    if (/"(?:minimal_cut_sets|top_event_probability|success_paths|system_reliability)"\s*:/.test(rawText)) {
      return 'calculation';
    }
    if (/"edges"\s*:\s*\[/.test(rawText) && /"(?:source|target)"\s*:/.test(rawText)) {
      return 'structure';
    }
    return undefined;
  }

  const model = scalarText(record.model).toUpperCase();
  const hasCalculationFields = (
    model === 'FTA' && (
      'minimal_cut_sets' in record || 'top_event_probability' in record
      || 'missing_probability_nodes' in record
    )
  ) || (
    model === 'RBD' && (
      'success_paths' in record || 'system_reliability' in record
      || 'missing_reliability_blocks' in record
    )
  );
  if (hasCalculationFields) return 'calculation';

  const nestedGraph = objectRecord(record.canonical_graph) || objectRecord(record.graph);
  const graph = nestedGraph || record;
  const nodes = graph.vertices ?? graph.nodes;
  const edges = graph.edges ?? graph.relations ?? graph.links;
  const graphType = scalarText(graph.graph_type || graph.diagram_type).toLowerCase();
  const hasCanonicalGraphFields = Array.isArray(nodes) && Array.isArray(edges);
  if (hasCanonicalGraphFields || ['fta', 'rbd'].includes(graphType)) return 'structure';
  return undefined;
}

function structuredEvidenceFallback(
  evidence: Record<string, unknown>,
  detectedKind?: EvidenceSummaryKind,
): string {
  const kind = detectedKind || evidenceSummaryKind(evidence);
  const docType = scalarText(evidence.doc_type).toUpperCase();
  if (kind === 'calculation') {
    return `${docType || '该'}确定性计算结果为结构化数据，请结合证据标题和来源定位查看。`;
  }
  if (kind === 'structure' || ['FTA', 'RBD'].includes(docType)) {
    return `${docType || '工程图'}结构化关系证据，请结合证据标题和来源定位查看。`;
  }
  return '该证据为结构化数据，已隐藏内部字段，请结合证据标题和来源定位查看。';
}

function evidenceWithToolContext(
  evidence: Record<string, unknown>,
  tool: unknown,
): Record<string, unknown> {
  const normalizedTool = scalarText(tool);
  const defaults: Record<string, Record<string, string>> = {
    fta_query: { doc_type: 'fta', type: 'fta_structure' },
    rbd_query: { doc_type: 'rbd', type: 'rbd_structure' },
    fta_analyzer: { doc_type: 'fta', type: 'fta_calculation', source_kind: 'calculation' },
    rbd_calculator: { doc_type: 'rbd', type: 'rbd_calculation', source_kind: 'calculation' },
  };
  const contextual = { ...evidence };
  Object.entries(defaults[normalizedTool] || {}).forEach(([key, value]) => {
    if (!scalarText(contextual[key])) contextual[key] = value;
  });
  return contextual;
}

/**
 * Turn every evidence payload into bounded user-facing copy.
 * Raw graph/calculation JSON remains available in the evidence contract, but is
 * never exposed directly in the diagnosis UI (including legacy truncated JSON).
 */
function evidencePresentation(
  evidence: Record<string, unknown>,
  maxLength = 1200,
): EvidencePresentation {
  const label = evidenceSummaryLabel(evidence);
  const displaySummary = scalarText(evidence.display_summary);
  if (displaySummary) return { label, text: compactText(displaySummary, maxLength) };

  const candidates = [evidence.content, evidence.snippet];
  for (const candidate of candidates) {
    const candidateIsObject = Boolean(candidate && typeof candidate === 'object');
    const plain = scalarText(candidate);
    const parsed = structuredValue(candidate);
    if (!parsed.lookedStructured && plain) return { label, text: compactText(plain, maxLength) };
    if (!parsed.lookedStructured) continue;

    const record = objectRecord(parsed.value);
    const kind = detectedStructuredKind(evidence, record, plain);
    if (record && kind) {
      const nestedGraph = objectRecord(record.canonical_graph) || objectRecord(record.graph);
      const summary = kind === 'calculation'
        ? calculationSummary(record, evidence)
        : graphStructureSummary(nestedGraph || record, evidence);
      if (summary) {
        return {
          label: evidenceSummaryLabel(evidence, kind),
          text: compactText(summary, maxLength),
        };
      }
    }
    if (kind) {
      return {
        label: evidenceSummaryLabel(evidence, kind),
        text: compactText(structuredEvidenceFallback(evidence, kind), maxLength),
      };
    }
    // A JSON string can itself be quoted source material.  Keep it intact for
    // ordinary retrieval evidence; only actual object values need a safe UI
    // fallback because String(object) would expose implementation noise.
    if (!candidateIsObject && plain) return { label, text: compactText(plain, maxLength) };
    return { label, text: compactText(structuredEvidenceFallback(evidence), maxLength) };
  }
  return { label, text: '未提供可展示的证据摘要。' };
}

function stringList(value: unknown, limit = 10): string[] {
  if (!Array.isArray(value)) return [];
  return Array.from(new Set(
    value.map((item) => compactText(item, 48)).filter(Boolean),
  )).slice(0, limit);
}

function evidenceIds(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return Array.from(new Set(value.flatMap((raw) => {
    if (typeof raw === 'string' || typeof raw === 'number') {
      const id = String(raw).trim();
      return id ? [id] : [];
    }
    if (!raw || typeof raw !== 'object') return [];
    const item = raw as Record<string, unknown>;
    const id = String(item.evidence_id || item.id || '').trim();
    return id ? [id] : [];
  })));
}

function evidenceStableId(evidence?: Citation): string {
  if (!evidence) return '';
  return String(evidence.evidence_id || '').trim();
}

function legacyEvidenceKey(evidence: Citation, index: number, origin: 'catalog' | 'citation' | 'preview') {
  const source = String(
    evidence.record_id || evidence.chunk_id || evidence.source_ref || '',
  ).trim();
  const doc = String(evidence.doc_name || '').trim();
  const title = String(evidence.title || '').trim();
  const claim = String(evidence.claim_type || evidence.type || '').trim();
  return `legacy:${origin}:${index}:${source}|${claim}|${doc}|${title}`;
}

function citationDisplayNumber(evidence: Citation, fallbackIndex: number): number {
  const explicit = asFiniteNumber(evidence.answer_citation_id ?? evidence.id);
  return explicit && explicit >= 1 ? explicit : fallbackIndex + 1;
}

function normalizeDisposition(value: unknown): EvidenceDisposition | undefined {
  const status = String(value || '').trim().toLowerCase();
  if (['answer_used', 'used', 'adopted', 'cited'].includes(status)) return 'used';
  if (['blocked', 'excluded', 'rejected', 'quarantined'].includes(status)) return 'blocked';
  if (['approved', 'reviewed', 'accepted'].includes(status)) return 'approved';
  if (['candidate', 'retrieved', 'recalled'].includes(status)) return 'candidate';
  return undefined;
}

function eventEvidenceSets(events?: unknown[]) {
  const tool = new Set<string>();
  const approved = new Set<string>();
  const blocked = new Set<string>();
  const used = new Set<string>();
  if (!Array.isArray(events)) return { tool, approved, blocked, used };
  events.forEach((raw) => {
    if (!raw || typeof raw !== 'object') return;
    const event = raw as Record<string, unknown>;
    evidenceIds(event.evidence_ids).forEach((id) => tool.add(id));
    evidenceIds(event.approved_evidence_ids ?? event.approved).forEach((id) => approved.add(id));
    [
      ...evidenceIds(event.blocked_evidence_ids ?? event.blocked),
      ...evidenceIds(event.excluded_evidence_ids),
    ].forEach((id) => blocked.add(id));
    evidenceIds(event.used_evidence_ids ?? event.used).forEach((id) => used.add(id));
  });
  return { tool, approved, blocked, used };
}

function previewEvidenceCatalog(events?: unknown[]): Citation[] {
  if (!Array.isArray(events)) return [];
  const catalog: Citation[] = [];
  const seen = new Set<string>();
  events.forEach((raw) => {
    if (!raw || typeof raw !== 'object') return;
    const event = raw as Record<string, unknown>;
    if (String(event.type || '') !== 'tool_call' || !Array.isArray(event.results_preview)) return;
    event.results_preview.forEach((previewRaw) => {
      if (!previewRaw || typeof previewRaw !== 'object') return;
      const preview = previewRaw as Record<string, unknown>;
      const id = String(preview.evidence_id || '').trim();
      if (!id || seen.has(id)) return;
      seen.add(id);
      const contextualPreview = evidenceWithToolContext(preview, event.tool);
      const presentation = evidencePresentation(contextualPreview, 600);
      catalog.push({
        evidence_id: id,
        status: 'candidate',
        title: compactText(preview.title, 120) || '检索候选',
        doc_name: compactText(preview.doc_name, 100) || toolLabel(event.tool),
        content: presentation.text,
        source_ref: compactText(preview.source_ref, 240),
        doc_type: compactText(contextualPreview.doc_type, 40),
        type: compactText(contextualPreview.claim_type ?? contextualPreview.type, 80),
      });
    });
  });
  return catalog;
}

function buildEvidenceView(
  content: string | undefined,
  citations?: Citation[],
  catalog?: Citation[],
  events?: unknown[],
  messageUsedIds?: unknown[],
): EvidenceView {
  const citationList = Array.isArray(citations) ? citations : [];
  const catalogList = Array.isArray(catalog) ? catalog : [];
  const previewCatalog = previewEvidenceCatalog(events);
  const citedNumbers = extractCitationNumbers(content);
  const sets = eventEvidenceSets(events);
  evidenceIds(messageUsedIds).forEach((id) => sets.used.add(id));

  const byKey = new Map<string, EvidenceViewItem>();
  const citationNumberById = new Map<string, number>();

  citationList.forEach((evidence, index) => {
    const citationNumber = citationDisplayNumber(evidence, index);
    const id = evidenceStableId(evidence);
    if (id) citationNumberById.set(id, citationNumber);
    if (citedNumbers.includes(citationNumber) && id) {
      if (String(evidence.evidence_id || '').trim()) sets.used.add(String(evidence.evidence_id));
    }
  });

  const addEvidence = (
    evidence: Citation,
    fallbackIndex: number,
    origin: 'catalog' | 'citation' | 'preview',
  ) => {
    const stableId = evidenceStableId(evidence);
    const key = stableId || legacyEvidenceKey(evidence, fallbackIndex, origin);
    const existing = byKey.get(key);
    const explicitNumber = asFiniteNumber(evidence.answer_citation_id ?? evidence.id);
    const citationNumber = explicitNumber && explicitNumber >= 1
      ? explicitNumber
      : citationNumberById.get(stableId)
        ?? (origin === 'citation' ? fallbackIndex + 1 : undefined);
    byKey.set(key, {
      key,
      evidenceId: String(evidence.evidence_id || '').trim() || existing?.evidenceId,
      citationNumber: existing?.citationNumber ?? citationNumber,
      evidence: { ...(existing?.evidence || {}), ...evidence },
      disposition: existing?.disposition || 'candidate',
    });
  };
  previewCatalog.forEach((evidence, index) => addEvidence(evidence, index, 'preview'));
  catalogList.forEach((evidence, index) => addEvidence(evidence, index, 'catalog'));
  citationList.forEach((evidence, index) => addEvidence(evidence, index, 'citation'));

  const all = Array.from(byKey.values()).map((item) => {
    const explicit = normalizeDisposition(item.evidence.status ?? item.evidence.usage_status);
    const id = item.evidenceId || '';
    const markerUsed = item.citationNumber !== undefined && citedNumbers.includes(item.citationNumber);
    let disposition: EvidenceDisposition = explicit || 'candidate';
    if (markerUsed || sets.used.has(id) || explicit === 'used') disposition = 'used';
    else if (sets.blocked.has(id) || explicit === 'blocked') disposition = 'blocked';
    else if (sets.approved.has(id) || explicit === 'approved') disposition = 'approved';
    else if (sets.tool.has(id)) disposition = 'candidate';
    return { ...item, disposition };
  });

  const answer = all.filter((item) => item.disposition === 'used').sort((a, b) => {
    const aIndex = a.citationNumber === undefined ? -1 : citedNumbers.indexOf(a.citationNumber);
    const bIndex = b.citationNumber === undefined ? -1 : citedNumbers.indexOf(b.citationNumber);
    const aOrder = aIndex >= 0 ? aIndex : Number.MAX_SAFE_INTEGER;
    const bOrder = bIndex >= 0 ? bIndex : Number.MAX_SAFE_INTEGER;
    if (aOrder !== bOrder) return aOrder - bOrder;
    return (a.citationNumber ?? Number.MAX_SAFE_INTEGER) - (b.citationNumber ?? Number.MAX_SAFE_INTEGER);
  });
  const retrieval = all.filter((item) => item.disposition !== 'used').sort((a, b) => {
    const weight: Record<EvidenceDisposition, number> = { approved: 0, candidate: 1, blocked: 2, used: -1 };
    return weight[a.disposition] - weight[b.disposition];
  });
  return {
    answer,
    retrieval,
    all,
    candidateCount: all.filter((item) => item.disposition === 'candidate').length,
    approvedCount: all.filter((item) => item.disposition === 'approved').length,
    blockedCount: all.filter((item) => item.disposition === 'blocked').length,
  };
}

const EVIDENCE_DISPOSITION_LABEL: Record<EvidenceDisposition, string> = {
  used: '回答采用',
  approved: '已审阅未采用',
  blocked: '已排除',
  candidate: '检索候选',
};

function toolLabel(tool: unknown): string {
  return TOOL_LABEL[String(tool || '')] || '资料查询';
}

function toolCountLabel(tool: string, count: number): string {
  if (tool === 'fta_analyzer') return `分析 ${count} 个故障树`;
  if (tool === 'rbd_calculator') return `计算 ${count} 个可靠性模型`;
  if (tool === 'fault_reasoning') return `形成 ${count} 个候选`;
  if (tool === 'function_locator') return `定位 ${count} 项功能`;
  if (tool === 'graph_traversal') return `找到 ${count} 条关系路径`;
  if (tool === 'hybrid_search' || tool === 'vector_search') return `召回 ${count} 条`;
  return `命中 ${count} 条`;
}

function durationLabel(duration: number): string {
  if (duration < 0.01) return '耗时 <0.01 秒';
  if (duration < 10) return `耗时 ${duration.toFixed(2)} 秒`;
  return `耗时 ${Math.round(duration)} 秒`;
}

function toolScopeTags(params?: Record<string, unknown>): string[] {
  if (!params) return [];
  const tags: string[] = [];
  stringList(params.doc_types, 7).forEach((docType) => {
    tags.push(DOC_TYPE_LABEL[docType] || docType);
  });
  const codeFields: Array<[string, string]> = [
    ['lcn', 'LCN'], ['hamc', 'HAMC'], ['hsc', 'HSC'], ['hmc_code', 'HMC'],
  ];
  codeFields.forEach(([field, label]) => {
    const value = compactText(params[field], 40);
    if (value) tags.push(`${label} ${value}`);
  });
  const missionTime = asFiniteNumber(params.mission_time);
  if (missionTime !== undefined) tags.push(`任务时间 ${missionTime}`);
  const limit = asFiniteNumber(params.top_k ?? params.limit);
  if (limit !== undefined) tags.push(`最多返回 ${limit} 条`);
  return Array.from(new Set(tags)).slice(0, 10);
}

function retrievalSummary(info?: Record<string, unknown>): string {
  if (!info) return '';
  const parts: string[] = [];
  const recalled = asFiniteNumber(info.recall_count);
  const deduplicated = asFiniteNumber(info.deduplicated_count);
  if (recalled !== undefined) parts.push(`初步召回 ${recalled} 条`);
  if (deduplicated !== undefined && deduplicated !== recalled) parts.push(`去重后 ${deduplicated} 条`);
  if (info.reranked === true) parts.push('已进行相关性重排');
  if (info.fallback) parts.push('部分检索能力已自动降级');
  return parts.join('，');
}

function previewItems(value: unknown, tool?: unknown): ExecutionDetailItem[] {
  if (!Array.isArray(value)) return [];
  return value.slice(0, 3).map((raw, index) => {
    const item = raw && typeof raw === 'object' ? raw as Record<string, unknown> : {};
    const presentation = evidencePresentation(evidenceWithToolContext(item, tool), 180);
    return {
      title: compactText(item.title, 80) || `候选结果 ${index + 1}`,
      description: presentation.text || undefined,
      evidenceId: compactText(item.evidence_id, 160) || undefined,
      sourceRef: compactText(item.source_ref, 220) || undefined,
      disposition: item.evidence_id ? 'candidate' : undefined,
    };
  });
}

function evidenceDetailItems(items: EvidenceViewItem[], limit = 8): ExecutionDetailItem[] {
  return items.slice(0, limit).map((item) => {
    const citation = item.evidence;
    const docName = compactText(citation.doc_name, 90) || '未命名文档';
    const title = compactText(citation.title, 90);
    const presentation = evidencePresentation(citation, 180);
    return {
      title: `${item.citationNumber ? `[${item.citationNumber}] ` : ''}${title || docName}`,
      description: [title && title !== docName ? docName : '', presentation.text]
        .filter(Boolean)
        .join(' · ') || undefined,
      evidenceId: item.evidenceId,
      citationNumber: item.citationNumber,
      disposition: item.disposition,
    };
  });
}

function evidenceForIds(view: EvidenceView, ids: string[]): EvidenceViewItem[] {
  if (!ids.length) return [];
  const requested = new Set(ids);
  return view.all.filter((item) => item.evidenceId && requested.has(item.evidenceId));
}

function scopeRouteFromEvents(events: unknown[] | undefined): string {
  if (!Array.isArray(events)) return '';
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const raw = events[index];
    if (!raw || typeof raw !== 'object') continue;
    const event = raw as Record<string, unknown>;
    if (event.type === 'stage' && event.stage === 'scope_routing') {
      return String(event.route || '');
    }
  }
  return '';
}

/** 将持久化事件收敛为可审计摘要；明确忽略 thinking_text / thinking_delta 和原始 JSON。 */
function executionSteps(events: unknown[] | undefined, evidenceView: EvidenceView, live = false): ExecutionStep[] {
  if (!Array.isArray(events)) return [];
  const steps: ExecutionStep[] = [];
  const scopeRoute = scopeRouteFromEvents(events);
  const directRoute = ['agent_capability', 'out_of_scope', 'clarification'].includes(scopeRoute);
  const planEvent = events.find((raw) => (
    raw && typeof raw === 'object' && String((raw as Record<string, unknown>).type || '') === 'plan'
  )) as Record<string, unknown> | undefined;
  const planItems = Array.isArray(planEvent?.tools_plan) ? planEvent.tools_plan : [];
  const toolPurposes = new Map<string, string>();
  planItems.forEach((raw) => {
    if (!raw || typeof raw !== 'object') return;
    const item = raw as Record<string, unknown>;
    const tool = String(item.tool || '');
    const reason = compactText(item.reason, 180);
    if (tool && reason) toolPurposes.set(tool, reason);
  });

  events.forEach((raw, index) => {
    if (!raw || typeof raw !== 'object') return;
    const event = raw as Record<string, unknown>;
    const type = String(event.type || '');
    if (type === 'stage' && String(event.stage || '') === 'scope_routing') {
      const route = String(event.route || '');
      const routeFact: Record<string, string> = {
        knowledge_base: '进入工程知识处理',
        agent_capability: '无需查询工程资料',
        out_of_scope: '当前知识范围外',
        clarification: '需要先补充对象信息',
      };
      steps.push({
        key: `scope-${index}`,
        kind: 'plan',
        title: '判断问题处理方式',
        facts: [routeFact[route] || '已完成问题范围判断'],
        status: 'complete',
        sections: compactText(event.summary, 220)
          ? [{ label: '处理方式', text: compactText(event.summary, 220) }]
          : [],
      });
      return;
    }
    if (type === 'plan') {
      const intent = String(event.intent || '');
      const pending = intent === 'pending';
      if (pending && scopeRoute) return;
      const plannedTools = Array.isArray(event.tools_plan) ? event.tools_plan : [];
      const keywords = [
        ...stringList(event.entity_keywords, 8),
        ...stringList(event.type_keywords, 5),
      ];
      const sections: ExecutionDetailSection[] = [];
      const rewrittenQuery = compactText(event.rewritten_query, 240);
      if (rewrittenQuery) sections.push({ label: '用于检索的问题', text: rewrittenQuery });
      if (keywords.length) sections.push({ label: '识别要点', tags: Array.from(new Set(keywords)) });
      const plannedItems = plannedTools.slice(0, 8).flatMap((item) => {
        if (!item || typeof item !== 'object') return [];
        const toolPlan = item as Record<string, unknown>;
        return [{
          title: toolLabel(toolPlan.tool),
          description: compactText(toolPlan.reason, 180) || undefined,
        }];
      });
      if (plannedItems.length) sections.push({ label: '检索计划', items: plannedItems });
      steps.push({
        key: `plan-${index}`,
        kind: 'plan',
        title: pending ? '理解问题并准备检索' : '理解问题并制定检索计划',
        facts: [
          INTENT_LABEL[intent] || '综合知识查询',
          ...(plannedTools.length ? [`计划 ${plannedTools.length} 项查询`] : []),
          ...(event.answer_mode && ANSWER_MODE_LABEL[String(event.answer_mode)]
            ? [ANSWER_MODE_LABEL[String(event.answer_mode)]] : []),
        ],
        status: pending && live ? 'active' : 'complete',
        sections,
      });
      return;
    }
    if (type === 'tool_call') {
      const count = asFiniteNumber(event.count);
      const duration = asFiniteNumber(event.duration);
      const tool = String(event.tool || '');
      const toolFailed = Boolean(event.error)
        || ['error', 'failed'].includes(String(event.status || '').toLowerCase());
      const exposedEvidenceIds = evidenceIds(event.evidence_ids);
      const relatedEvidence = evidenceForIds(evidenceView, exposedEvidenceIds);
      const relatedApproved = relatedEvidence.filter(
        (item) => item.disposition === 'approved' || item.disposition === 'used',
      ).length;
      const relatedUsed = relatedEvidence.filter((item) => item.disposition === 'used').length;
      const reportedEvidenceCount = asFiniteNumber(event.evidence_count);
      const totalMatched = Math.max(
        count ?? 0,
        reportedEvidenceCount ?? 0,
        exposedEvidenceIds.length,
        relatedEvidence.length,
      );
      const traceableCount = relatedEvidence.length;
      const partialTrace = event.evidence_ids_truncated === true || traceableCount < totalMatched;
      const genuinelyEmpty = count === 0
        && (reportedEvidenceCount === undefined || reportedEvidenceCount === 0)
        && exposedEvidenceIds.length === 0
        && relatedEvidence.length === 0;
      const facts: string[] = [];
      if (relatedEvidence.length) {
        facts.push(partialTrace
          ? `命中 ${totalMatched} 条 · 本轮可追溯 ${traceableCount}/${totalMatched} 条 · 可追溯资料中通过 ${relatedApproved} 条、回答采用 ${relatedUsed} 条`
          : `命中 ${totalMatched} 条 → 通过 ${relatedApproved} 条 → 回答采用 ${relatedUsed} 条`);
      } else if (count !== undefined || reportedEvidenceCount !== undefined) {
        facts.push(toolCountLabel(tool, totalMatched));
        if (totalMatched > 0) {
          facts.push(live ? '证据目录整理中' : '该历史记录未保存稳定证据标识');
        }
      }
      if (duration !== undefined) facts.push(durationLabel(duration));
      if (event.retrieval_info && typeof event.retrieval_info === 'object'
        && (event.retrieval_info as Record<string, unknown>).reranked === true) facts.push('已重排');
      const authoritativeCount = asFiniteNumber(event.postgres_authoritative_count);
      if (authoritativeCount !== undefined) facts.push(`权威记录 ${authoritativeCount} 条`);

      const sections: ExecutionDetailSection[] = [];
      const purpose = toolPurposes.get(tool);
      if (purpose) sections.push({ label: '执行目的', text: purpose });
      const scopes = toolScopeTags(
        event.params && typeof event.params === 'object'
          ? event.params as Record<string, unknown> : undefined,
      );
      if (scopes.length) sections.push({ label: '查询范围', tags: scopes });
      const retrieval = retrievalSummary(
        event.retrieval_info && typeof event.retrieval_info === 'object'
          ? event.retrieval_info as Record<string, unknown> : undefined,
      );
      if (retrieval) sections.push({ label: '检索处理', text: retrieval });
      const previews = previewItems(event.results_preview, event.tool);
      const evidenceItems = evidenceDetailItems(relatedEvidence);
      if (evidenceItems.length) {
        sections.push({
          label: '证据去向',
          items: evidenceItems,
          note: '正文实际采用的证据可点击查看；其余项仅在此记录检索去向。',
        });
      }
      if (!evidenceItems.length) {
        if (previews.length) {
          sections.push({
            label: '初步命中',
            items: previews,
            note: '这里展示的是候选结果预览；最终采用的依据以正文编号和右侧证据栏为准。',
          });
        } else if (genuinelyEmpty) {
          sections.push({ label: '查询结果', text: '此次查询没有找到直接匹配的记录。' });
        }
      }
      steps.push({
        key: `tool-${index}`,
        kind: 'tool',
        title: toolLabel(tool),
        facts,
        status: toolFailed ? 'error' : genuinelyEmpty ? 'empty' : 'complete',
        sections,
      });
      return;
    }
    if (type === 'reflect') {
      const needMore = Boolean(event.need_more);
      const triedTools = stringList(event.tried_tools, 8).map(toolLabel);
      const sections: ExecutionDetailSection[] = [];
      const resultSummary = compactText(event.result_summary, 220);
      if (resultSummary) sections.push({ label: '检查结果', text: resultSummary });
      if (triedTools.length) sections.push({ label: '已检查来源', tags: triedTools });
      steps.push({
        key: `review-${index}`,
        kind: 'review',
        title: needMore ? '检查证据并补充检索' : '检查证据是否足以回答',
        facts: [needMore ? '需要补充资料' : '证据检查完成'],
        status: 'complete',
        sections,
      });
      return;
    }
    if (type === 'stage' && String(event.stage || '') === 'evidence_review') {
      const metrics = event.metrics && typeof event.metrics === 'object'
        ? event.metrics as Record<string, unknown> : {};
      const reviewed = asFiniteNumber(metrics.reviewed_count) || 0;
      const approved = asFiniteNumber(metrics.approved_count) || 0;
      const excluded = asFiniteNumber(metrics.excluded_count) || 0;
      const requiresClarification = Boolean(metrics.requires_clarification);
      const stageStatus = String(event.status || 'completed');
      const approvedIds = evidenceIds(event.approved_evidence_ids ?? event.approved);
      const blockedIds = [
        ...evidenceIds(event.blocked_evidence_ids ?? event.blocked),
        ...evidenceIds(event.excluded_evidence_ids),
      ];
      const reviewedIds = evidenceIds(event.reviewed_evidence_ids);
      const reviewedEvidence = evidenceForIds(
        evidenceView,
        [...reviewedIds, ...approvedIds, ...blockedIds],
      );
      const facts = [
        `检查 ${reviewed} 条候选`,
        `通过 ${approved} 条`,
        ...(excluded ? [`排除 ${excluded} 条`] : []),
        ...(requiresClarification ? ['需要补充信息'] : []),
      ];
      steps.push({
        key: `evidence-review-${index}`,
        kind: 'review',
        title: '核对候选资料与证据边界',
        facts,
        status: stageStatus === 'blocked' ? 'error' : 'complete',
        sections: [
          {
            label: '核对结果',
            text: stageStatus === 'blocked'
              ? '当前权威资料不可用或证据未通过安全边界，本轮不会形成确定性结论。'
              : requiresClarification
                ? '已保留可引用资料，但现有信息仍不足以唯一收敛，回答中会明确需要补充的内容。'
                : `已从候选资料中保留 ${approved} 条可用于组织回答的证据。`,
          },
          ...(reviewedEvidence.length ? [{
            label: '审阅明细',
            items: evidenceDetailItems(reviewedEvidence),
            note: '“通过”表示可供回答参考，不代表最终正文一定采用。',
          }] : []),
        ],
      });
    }
  });

  const hasResolvedPlan = steps.some((step) => step.kind === 'plan' && step.status !== 'active');
  const hasToolResult = steps.some((step) => step.kind === 'tool');
  if (hasResolvedPlan && (!live || directRoute || hasToolResult || evidenceView.answer.length > 0)) {
    const citedItems = evidenceDetailItems(evidenceView.answer);
    const sections: ExecutionDetailSection[] = [{
      label: live ? '当前阶段' : '整理方式',
      text: live
        ? directRoute
          ? '正在组织不需要工程资料的直接答复。'
          : '正在核对候选资料、保留可引用证据，并由大模型组织面向用户的回答。'
        : directRoute
          ? '本轮根据问题范围直接答复，没有调用工程知识工具。'
          : '系统只把通过筛选的资料写入最终回答，正文中的编号与右侧证据栏一一对应。',
    }];
    if (citedItems.length) sections.push({ label: '最终采用的证据', items: citedItems });
    steps.push({
      key: 'answer-composition',
      kind: 'answer',
      title: directRoute
        ? (live ? '整理直接答复' : '已完成直接答复')
        : (live ? '筛选证据并整理回答' : '完成证据筛选与答案整理'),
      facts: citedItems.length
        ? [`采用 ${citedItems.length} 条证据`]
        : live ? ['正在处理'] : [directRoute ? '本轮无需引用工程证据' : '未引用直接证据'],
      status: live ? 'active' : 'complete',
      sections,
    });
  }
  return steps;
}

function ExecutionTrace({
  events,
  evidenceView,
  onEvidenceSelect,
  live = false,
}: {
  events?: unknown[];
  evidenceView: EvidenceView;
  onEvidenceSelect?: (item: ExecutionDetailItem) => void;
  live?: boolean;
}) {
  const steps = useMemo(() => executionSteps(events, evidenceView, live), [events, evidenceView, live]);
  const [open, setOpen] = useState(live);
  const [expandedStepKeys, setExpandedStepKeys] = useState<Set<string>>(new Set());
  const autoFollowLatestStepRef = useRef(true);
  const traceSummaryRef = useRef<HTMLElement>(null);
  const traceId = useId().replace(/:/g, '');
  const latestStepKey = steps[steps.length - 1]?.key;
  const toolCount = steps.filter((step) => step.kind === 'tool').length;
  const citationCount = evidenceView.answer.length;
  const scopeRoute = scopeRouteFromEvents(events);
  const directRoute = ['agent_capability', 'out_of_scope', 'clarification'].includes(scopeRoute);

  useEffect(() => {
    if (live && latestStepKey && autoFollowLatestStepRef.current) {
      setExpandedStepKeys(new Set([latestStepKey]));
    }
  }, [latestStepKey, live]);

  if (steps.length === 0) return null;
  const allExpanded = steps.every((step) => expandedStepKeys.has(step.key));
  const expandedCount = expandedStepKeys.size;
  const completedCount = steps.filter((step) => step.status !== 'active').length;
  const summary = live
    ? `处理中 · 已完成 ${completedCount}/${steps.length} 步`
    : directRoute
      ? '本轮无需查询工程资料'
      : `${toolCount} 次查询 · ${citationCount} 条引用证据`;

  return (
    <details
      className="diagnosis-execution-trace"
      open={open}
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary ref={traceSummaryRef}>
        <span>执行过程</span>
        <small role={live ? 'status' : undefined} aria-live={live ? 'polite' : undefined}>
          {summary}
        </small>
      </summary>
      <div className="diagnosis-execution-toolbar">
        <span>点击步骤查看查询范围和结果摘要</span>
        <button
          type="button"
          aria-expanded={allExpanded}
          onClick={() => {
            autoFollowLatestStepRef.current = false;
            setExpandedStepKeys(
              allExpanded ? new Set() : new Set(steps.map((step) => step.key)),
            );
          }}
        >
          {allExpanded ? '收起全部' : '展开全部'}
        </button>
      </div>
      <ol>
        {steps.map((step) => (
          <li key={step.key} className={`diagnosis-execution-step ${step.kind} ${step.status}`}>
            <span className="diagnosis-execution-dot" aria-hidden="true" />
            <button
              type="button"
              className="diagnosis-execution-step-toggle"
              aria-expanded={expandedStepKeys.has(step.key)}
              aria-controls={`${traceId}-${step.key}`}
              onClick={() => {
                autoFollowLatestStepRef.current = false;
                setExpandedStepKeys((current) => (
                  current.has(step.key) ? new Set() : new Set([step.key])
                ));
              }}
            >
              <span className="diagnosis-execution-copy">
                <strong>{step.status === 'error' ? `${step.title}失败` : step.title}</strong>
                {step.facts.length ? <small>{step.facts.join(' · ')}</small> : null}
              </span>
              <span className="diagnosis-execution-step-state">
                {step.status === 'active'
                  ? '进行中'
                  : step.status === 'empty' ? '无结果' : step.status === 'error' ? '失败' : '已完成'}
              </span>
              <span className="diagnosis-execution-step-chevron" aria-hidden="true" />
            </button>
            <div
              id={`${traceId}-${step.key}`}
              className="diagnosis-execution-detail"
              hidden={!expandedStepKeys.has(step.key)}
            >
              {step.sections.length ? step.sections.map((section, sectionIndex) => (
                <section key={`${step.key}-section-${sectionIndex}`}>
                  <h5>{section.label}</h5>
                  {section.text ? <p>{section.text}</p> : null}
                  {section.tags?.length ? (
                    <div className="diagnosis-execution-tags">
                      {section.tags.map((tag) => <span key={tag}>{tag}</span>)}
                    </div>
                  ) : null}
                  {section.items?.length ? (
                    <ul className="diagnosis-execution-detail-list">
                      {section.items.map((item, itemIndex) => {
                        const content = (
                          <>
                            <span className="diagnosis-execution-item-copy">
                              <strong>{item.title}</strong>
                              {item.description ? <span>{item.description}</span> : null}
                            </span>
                            {item.disposition ? (
                              <span className={`diagnosis-evidence-status ${item.disposition}`}>
                                {EVIDENCE_DISPOSITION_LABEL[item.disposition]}
                              </span>
                            ) : null}
                          </>
                        );
                        return (
                          <li key={`${item.evidenceId || item.title}-${itemIndex}`}>
                            {item.disposition === 'used' && (item.evidenceId || item.citationNumber) ? (
                              <button
                                type="button"
                                className="diagnosis-execution-evidence-link"
                                onClick={() => onEvidenceSelect?.(item)}
                                aria-label={`查看证据：${item.title}`}
                              >
                                {content}
                              </button>
                            ) : content}
                          </li>
                        );
                      })}
                    </ul>
                  ) : null}
                  {section.note ? <small className="diagnosis-execution-note">{section.note}</small> : null}
                </section>
              )) : (
                <p className="diagnosis-execution-empty-detail">该步骤已完成，没有更多需要展示的信息。</p>
              )}
            </div>
          </li>
        ))}
      </ol>
      {expandedCount > 0 ? (
        <div className="diagnosis-execution-footer">
          <span>已展开 {expandedCount} 个步骤</span>
          <button
            type="button"
            onClick={() => {
              autoFollowLatestStepRef.current = false;
              setExpandedStepKeys(new Set());
              requestAnimationFrame(() => {
                traceSummaryRef.current?.focus({ preventScroll: true });
                traceSummaryRef.current?.scrollIntoView({ block: 'nearest' });
              });
            }}
          >
            收起已展开步骤
          </button>
        </div>
      ) : null}
    </details>
  );
}

const CITATION_AUTHORITY_LABEL: Record<string, string> = {
  authoritative_record: '文档结构化记录',
  source_artifact: '原始文档',
  confirmed_relation: '结构已确认',
  validated_relation: '结构已校验',
  deterministic_derivation: '确定性计算',
  search_projection: '文档检索',
  unconfirmed_extraction: '结构提取结果',
};

const SUGGESTIONS = [
  'ICP 上电后模板一直加载失败，重新上电仍不能恢复。',
  '设备运行几分钟后反复重启，偶尔伴随显示闪烁。',
  '查询 HMC 31810001-**-0001 的故障和维修方法。',
  'ICP 功能丧失可能由哪些原因引起？',
];

function parseSseChunk(buffer: string): { events: any[]; rest: string } {
  const parts = buffer.split('\n\n');
  const rest = parts.pop() || '';
  const events: any[] = [];
  for (const part of parts) {
    const line = part.split('\n').find((value) => value.startsWith('data:'));
    if (!line) continue;
    try {
      events.push(JSON.parse(line.slice(5).trim()));
    } catch {
      /* ignore malformed event */
    }
  }
  return { events, rest };
}

export default function Diagnosis() {
  const [agent, setAgent] = useState<AgentConfig | null>(null);
  const [sessions, setSessions] = useState<AgentSession[]>([]);
  const [sessionId, setSessionId] = useState<string>();
  const [messages, setMessages] = useState<AgentMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [streamingAnswer, setStreamingAnswer] = useState('');
  const [liveExecutionEvents, setLiveExecutionEvents] = useState<ExecutionEvent[]>([]);
  const [liveCitations, setLiveCitations] = useState<AgentEvidence[]>([]);
  const [liveEvidenceCatalog, setLiveEvidenceCatalog] = useState<AgentEvidence[]>([]);
  const [liveUsedEvidenceIds, setLiveUsedEvidenceIds] = useState<string[]>([]);
  // 部分字段未确认状态：与系统错误区分，展示“部分确认”提示而不是失败。
  const [streamPartialFields, setStreamPartialFields] = useState<string[]>([]);
  const abortRef = useRef<AbortController | null>(null);
  const chatBottomRef = useRef<HTMLDivElement | null>(null);
  const evidenceRailRef = useRef<HTMLElement | null>(null);
  const citationCardRefs = useRef<Map<string, HTMLElement>>(new Map());
  const [selectedEvidenceKey, setSelectedEvidenceKey] = useState<string>();
  const [highlightedCitationNumber, setHighlightedCitationNumber] = useState<number>();
  const [highlightedEvidenceId, setHighlightedEvidenceId] = useState<string>();
  const [renameTarget, setRenameTarget] = useState<AgentSession | null>(null);
  const [renameValue, setRenameValue] = useState('');
  const [renaming, setRenaming] = useState(false);
  const [evidenceWidth, setEvidenceWidth] = useState(storedEvidenceWidth);
  const [resizingEvidence, setResizingEvidence] = useState(false);
  const diagnosisGridRef = useRef<HTMLElement | null>(null);
  const resizeStartRef = useRef({ x: 0, width: DEFAULT_EVIDENCE_WIDTH });

  const loadSession = useCallback(async (id: string) => {
    const detail = await fetchSession(id);
    const nextMessages = detail.messages || [];
    setMessages(nextMessages);
    const latestAnswer = [...nextMessages].reverse().find((item) => item.role === 'assistant');
    setSelectedEvidenceKey(latestAnswer?.id);
    setHighlightedCitationNumber(undefined);
    setHighlightedEvidenceId(undefined);
    setLiveExecutionEvents([]);
    setLiveCitations([]);
    setLiveEvidenceCatalog([]);
    setLiveUsedEvidenceIds([]);
    setSessionId(id);
  }, []);

  const bootstrap = useCallback(async () => {
    setLoading(true);
    try {
      const agents = await fetchAgents(true);
      const selected = agents.find((item) => item.name === 'ICP故障诊断智能体')
        || agents.find((item) => item.name.includes('ICP'))
        || agents[0];
      if (!selected) {
        setAgent(null);
        setSessions([]);
        setMessages([]);
        return;
      }
      setAgent(selected);
      let list = await fetchSessions(selected.agent_id);
      if (list.length === 0) {
        const created = await createSession(selected.agent_id, DEFAULT_SESSION_TITLE);
        list = [created];
      } else {
        // 为升级前已有首问、但仍保留占位名称的历史会话补齐标题。
        // 接口内部使用 title_origin 条件更新，不会覆盖人工命名。
        const repaired: AgentSession[] = [];
        for (const session of list) {
          if (session.title_origin === 'default') {
            try {
              repaired.push(await generateSessionTitle(session.id));
            } catch {
              repaired.push(session);
            }
          } else {
            repaired.push(session);
          }
        }
        list = repaired;
      }
      setSessions(list);
      await loadSession(list[0].id);
    } catch (error) {
      message.error(`诊断工作台初始化失败：${(error as Error).message}`);
    } finally {
      setLoading(false);
    }
  }, [loadSession]);

  useEffect(() => { bootstrap(); }, [bootstrap]);

  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages, streamingAnswer]);

  useEffect(() => {
    try {
      window.localStorage.setItem(EVIDENCE_WIDTH_STORAGE_KEY, String(Math.round(evidenceWidth)));
    } catch {
      // 浏览器禁用本地存储时仍保留本次会话内的拖拽结果。
    }
  }, [evidenceWidth]);

  useEffect(() => {
    if (!resizingEvidence) return;
    const handlePointerMove = (event: PointerEvent) => {
      const containerWidth = diagnosisGridRef.current?.getBoundingClientRect().width;
      const next = resizeStartRef.current.width + resizeStartRef.current.x - event.clientX;
      setEvidenceWidth(clampEvidenceWidth(next, containerWidth));
    };
    const stopResizing = () => setResizingEvidence(false);
    document.body.classList.add('diagnosis-column-resizing');
    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', stopResizing, { once: true });
    window.addEventListener('pointercancel', stopResizing, { once: true });
    return () => {
      document.body.classList.remove('diagnosis-column-resizing');
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', stopResizing);
      window.removeEventListener('pointercancel', stopResizing);
    };
  }, [resizingEvidence]);

  // 正文引用只负责选择右侧证据；证据内容只在右栏展示，避免弹层与侧栏重复。
  useEffect(() => {
    const clickHandler = (event: MouseEvent) => {
      const target = event.target as HTMLElement | null;
      const cite = target?.closest?.('.inline-cite') as HTMLElement | null;
      if (!cite) return;
      const number = Number.parseInt(cite.dataset.citeNumber || '', 10);
      const messageKey = cite.dataset.citeMessageKey || '';
      if (!messageKey || !Number.isFinite(number) || number < 1) return;
      document.querySelectorAll<HTMLElement>('.inline-cite[aria-expanded="true"]')
        .forEach((item) => item.setAttribute('aria-expanded', 'false'));
      cite.setAttribute('aria-expanded', 'true');
      setSelectedEvidenceKey(messageKey);
      setHighlightedCitationNumber(number);
      setHighlightedEvidenceId(undefined);
    };
    document.addEventListener('click', clickHandler);
    return () => document.removeEventListener('click', clickHandler);
  }, []);

  useEffect(() => {
    if ((!highlightedCitationNumber && !highlightedEvidenceId) || !selectedEvidenceKey) {
      evidenceRailRef.current?.scrollTo({ top: 0, behavior: 'smooth' });
      return;
    }
    const frame = window.requestAnimationFrame(() => {
      const refKey = highlightedEvidenceId
        ? `${selectedEvidenceKey}:id:${highlightedEvidenceId}`
        : `${selectedEvidenceKey}:citation:${highlightedCitationNumber}`;
      citationCardRefs.current
        .get(refKey)
        ?.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'nearest' });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [highlightedCitationNumber, highlightedEvidenceId, selectedEvidenceKey]);

  const currentSession = sessions.find((item) => item.id === sessionId);

  const createNewSession = async () => {
    if (!agent) return;
    const created = await createSession(agent.agent_id, DEFAULT_SESSION_TITLE);
    setSessions((prev) => [created, ...prev]);
    setMessages([]);
    setSessionId(created.id);
    setLiveExecutionEvents([]);
    setLiveCitations([]);
    setLiveEvidenceCatalog([]);
    setLiveUsedEvidenceIds([]);
    setSelectedEvidenceKey(undefined);
    setHighlightedCitationNumber(undefined);
    setHighlightedEvidenceId(undefined);
  };

  const removeSession = async (id: string) => {
    await deleteSession(id);
    const next = sessions.filter((item) => item.id !== id);
    setSessions(next);
    if (sessionId === id) {
      if (next[0]) await loadSession(next[0].id);
      else if (agent) {
        const created = await createSession(agent.agent_id, DEFAULT_SESSION_TITLE);
        setSessions([created]);
        setSessionId(created.id);
        setMessages([]);
        setLiveExecutionEvents([]);
        setLiveCitations([]);
        setLiveEvidenceCatalog([]);
        setLiveUsedEvidenceIds([]);
        setSelectedEvidenceKey(undefined);
        setHighlightedCitationNumber(undefined);
        setHighlightedEvidenceId(undefined);
      }
    }
  };

  const openRenameSession = (session: AgentSession) => {
    setRenameTarget(session);
    setRenameValue(session.title || '');
  };

  const saveSessionTitle = async () => {
    if (!renameTarget || renaming) return;
    const title = renameValue.replace(/\s+/g, ' ').trim();
    if (!title) {
      message.warning('会话名称不能为空');
      return;
    }
    setRenaming(true);
    try {
      const updated = await updateSession(renameTarget.id, { title });
      setSessions((prev) => prev.map((item) => (item.id === updated.id ? updated : item)));
      setRenameTarget(null);
      setRenameValue('');
      message.success('会话名称已更新');
    } catch (error) {
      message.error(`重命名失败：${(error as Error).message}`);
    } finally {
      setRenaming(false);
    }
  };

  const stopGeneration = async () => {
    const controller = abortRef.current;
    try {
      if (sessionId) await stopChat(sessionId);
    } catch {
      // 即使服务端停止请求失败，也立即断开本地 SSE，避免界面继续等待。
    } finally {
      controller?.abort();
      abortRef.current = null;
      setGenerating(false);
    }
  };

  const send = async (preset?: string) => {
    const query = (preset ?? input).trim();
    if (!query || !agent || !sessionId || generating) return;
    const sessionAtSend = sessions.find((item) => item.id === sessionId);
    const shouldGenerateTitle = sessionAtSend?.title_origin === 'default'
      || (!sessionAtSend?.title_origin && sessionAtSend?.title === DEFAULT_SESSION_TITLE);

    const tempUser: AgentMessage = {
      id: `temp-user-${Date.now()}`,
      role: 'user',
      content: query,
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, tempUser]);
    setInput('');
    setGenerating(true);
    setStreamingAnswer('');
    setLiveCitations([]);
    setLiveEvidenceCatalog([]);
    setLiveUsedEvidenceIds([]);
    setStreamPartialFields([]);
    setLiveExecutionEvents([{ type: 'plan', intent: 'pending' }]);
    setSelectedEvidenceKey('stream');
    setHighlightedCitationNumber(undefined);
    setHighlightedEvidenceId(undefined);

    const controller = new AbortController();
    abortRef.current = controller;
    let answer = '';
    let citations: AgentEvidence[] = [];
    let evidenceCatalog: AgentEvidence[] = [];
    let usedEvidenceIds: string[] = [];

    try {
      const response = await chatStream({
        agent_id: agent.agent_id,
        session_id: sessionId,
        query,
        user_id: 'default',
        chat_model_id: '',
        embedding_model_id: '',
      }, controller.signal);
      if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parsed = parseSseChunk(buffer);
        buffer = parsed.rest;
        for (const event of parsed.events) {
          if (event.type === 'plan') {
            setLiveExecutionEvents((previous) => [
              event as ExecutionEvent,
              ...previous.filter((item) => item.type !== 'plan'),
            ]);
          } else if (event.type === 'thinking_delta') {
            // 规划模型的原始推理仅用于内部执行，不直接展示给最终用户。
          } else if (event.type === 'tool_call') {
            setLiveExecutionEvents((previous) => [...previous, event as ExecutionEvent]);
          } else if (event.type === 'reflect') {
            setLiveExecutionEvents((previous) => [...previous, event as ExecutionEvent]);
          } else if (event.type === 'stage') {
            setLiveExecutionEvents((previous) => {
              const withoutPendingPlan = event.stage === 'scope_routing'
                ? previous.filter((item) => !(item.type === 'plan' && item.intent === 'pending'))
                : previous;
              return [...withoutPendingPlan, event as ExecutionEvent];
            });
          } else if (event.type === 'answer_delta') {
            answer += event.content || '';
            setStreamingAnswer(answer);
          } else if (event.type === 'answer') {
            answer = event.content || answer;
            setStreamingAnswer(answer);
            if (Array.isArray(event.citations)) {
              citations = event.citations;
              setLiveCitations(citations);
            }
            if (Array.isArray(event.evidence_catalog)) {
              evidenceCatalog = event.evidence_catalog;
              setLiveEvidenceCatalog(evidenceCatalog);
            }
            usedEvidenceIds = evidenceIds(event.used_evidence_ids);
            if (usedEvidenceIds.length) setLiveUsedEvidenceIds(usedEvidenceIds);
            // “部分字段未确认”是正常答案的完成度状态，不是系统错误。
            if (event.completion_status === 'partial') {
              setStreamPartialFields(
                Array.isArray(event.unconfirmed_fields) ? event.unconfirmed_fields : [],
              );
            } else if (event.completion_status === 'full') {
              setStreamPartialFields([]);
            }
          } else if (event.type === 'citations') {
            citations = event.items || [];
            setLiveCitations(citations);
          } else if (event.type === 'error') {
            throw new Error(event.message || '诊断流程执行失败');
          } else if (event.type === 'done' || event.type === 'stopped') {
            if (Array.isArray(event.evidence_catalog)) {
              evidenceCatalog = event.evidence_catalog;
              setLiveEvidenceCatalog(evidenceCatalog);
            }
            const doneUsedIds = evidenceIds(event.used_evidence_ids);
            if (doneUsedIds.length) {
              usedEvidenceIds = doneUsedIds;
              setLiveUsedEvidenceIds(doneUsedIds);
            }
            // 服务端已完成持久化，稍后重新拉取会话。
          }
        }
      }

      if (answer) {
        setMessages((prev) => [...prev, {
          id: `temp-assistant-${Date.now()}`,
          role: 'assistant',
          content: answer,
          citations,
          evidence_catalog: evidenceCatalog,
          used_evidence_ids: usedEvidenceIds,
          created_at: new Date().toISOString(),
        }]);
      }
      // 后端 done 事件在消息保存后发送，重新拉取以取得真实消息 ID 与引用。
      await loadSession(sessionId);
      if (shouldGenerateTitle) {
        try {
          const titled = await generateSessionTitle(sessionId);
          setSessions((prev) => prev.map((item) => (item.id === titled.id ? titled : item)));
        } catch {
          // 标题生成不影响诊断结果；下次首问仍可安全重试。
        }
      }
      const refreshed = await fetchSessions(agent.agent_id);
      setSessions((previous) => refreshed.map((remote) => {
        const local = previous.find((item) => item.id === remote.id);
        // 极窄竞态下，较早发出的列表请求可能在手动重命名后
        // 才返回旧快照。数据库已有条件保护，这里同步保护本地 UI。
        return local?.title_origin === 'manual' ? local : remote;
      }));
    } catch (error) {
      if ((error as Error).name !== 'AbortError') {
        message.error(`诊断失败：${(error as Error).message}`);
      }
    } finally {
      abortRef.current = null;
      setGenerating(false);
      setStreamingAnswer('');
    }
  };

  const selectedEvidenceMessage = useMemo(
    () => (selectedEvidenceKey && selectedEvidenceKey !== 'stream'
      ? messages.find((item) => item.id === selectedEvidenceKey)
      : undefined),
    [messages, selectedEvidenceKey],
  );
  const liveEvidenceView = useMemo(() => buildEvidenceView(
    streamingAnswer,
    liveCitations,
    liveEvidenceCatalog,
    liveExecutionEvents,
    liveUsedEvidenceIds,
  ), [
    liveCitations,
    liveEvidenceCatalog,
    liveExecutionEvents,
    liveUsedEvidenceIds,
    streamingAnswer,
  ]);
  const selectedEvidenceView = useMemo(() => {
    if (selectedEvidenceKey === 'stream') return liveEvidenceView;
    return buildEvidenceView(
      selectedEvidenceMessage?.content,
      selectedEvidenceMessage?.citations as Citation[] | undefined,
      selectedEvidenceMessage?.evidence_catalog as Citation[] | undefined,
      selectedEvidenceMessage?.thinking_events,
      selectedEvidenceMessage?.used_evidence_ids,
    );
  }, [liveEvidenceView, selectedEvidenceKey, selectedEvidenceMessage]);
  const selectedEvidenceItems = selectedEvidenceView.answer;
  const selectedExecutionEvents = selectedEvidenceKey === 'stream'
    ? liveExecutionEvents
    : selectedEvidenceMessage?.thinking_events;
  const selectedScopeRoute = scopeRouteFromEvents(selectedExecutionEvents);
  const emptyEvidenceDescription = selectedEvidenceKey === 'stream' && generating
    ? '正在核对回答依据'
    : selectedScopeRoute === 'agent_capability'
      ? '本轮是能力或使用说明，无需引用工程文档'
      : selectedScopeRoute === 'out_of_scope'
        ? '本轮未调用工程知识工具'
        : selectedScopeRoute === 'clarification'
          ? '补充对象信息后再查询相关工程资料'
          : '本轮未找到可直接核验的回答依据';
  const selectedEvidenceTime = selectedEvidenceKey === 'stream'
    ? '当前回答'
    : formatTimeShort(selectedEvidenceMessage?.created_at);

  const showExecutionEvidence = useCallback((
    messageKey: string,
    item: ExecutionDetailItem,
  ) => {
    // 未用于正文的候选只在折叠执行过程里保留摘要。右栏是面向用户的
    // 回答依据，不应因为点击检索候选而切换成几十条调试/审计记录。
    if (item.disposition !== 'used') return;
    setSelectedEvidenceKey(messageKey);
    setHighlightedEvidenceId(item.evidenceId);
    setHighlightedCitationNumber(item.citationNumber);
  }, []);

  useEffect(() => {
    if (loading || !diagnosisGridRef.current) return;
    const container = diagnosisGridRef.current;
    const clampToContainer = () => {
      const containerWidth = container.getBoundingClientRect().width;
      setEvidenceWidth((current) => clampEvidenceWidth(current, containerWidth));
    };
    clampToContainer();
    if (typeof ResizeObserver === 'undefined') {
      window.addEventListener('resize', clampToContainer);
      return () => window.removeEventListener('resize', clampToContainer);
    }
    const observer = new ResizeObserver(clampToContainer);
    observer.observe(container);
    return () => observer.disconnect();
  }, [loading]);

  const beginEvidenceResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;
    event.preventDefault();
    resizeStartRef.current = { x: event.clientX, width: evidenceWidth };
    setResizingEvidence(true);
  };

  const resizeEvidenceByKeyboard = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    const containerWidth = diagnosisGridRef.current?.getBoundingClientRect().width;
    const step = event.shiftKey ? 40 : 16;
    let next: number | undefined;
    if (event.key === 'ArrowLeft') next = evidenceWidth + step;
    if (event.key === 'ArrowRight') next = evidenceWidth - step;
    if (event.key === 'Home') next = MIN_EVIDENCE_WIDTH;
    if (event.key === 'End') next = MAX_EVIDENCE_WIDTH;
    if (next === undefined) return;
    event.preventDefault();
    setEvidenceWidth(clampEvidenceWidth(next, containerWidth));
  };

  if (loading) {
    return <div style={{ display: 'grid', placeItems: 'center', minHeight: '70vh' }}><Spin size="large" /></div>;
  }

  if (!agent) {
    return (
      <Card className="glass">
        <Empty
          description="诊断服务暂时不可用，请稍后重试。"
        />
      </Card>
    );
  }

  return (
    <>
      <Layout hasSider className="glass-fade-in diagnosis-workbench">
      <Sider
        width={232}
        theme="light"
        className="diagnosis-session-sider"
      >
        <div className="diagnosis-session-header">
          <Space style={{ width: '100%', justifyContent: 'space-between' }}>
            <Typography.Text strong>诊断会话</Typography.Text>
            <Tooltip title="新建诊断会话">
              <Button type="text" size="small" icon={<PlusOutlined />} onClick={createNewSession} />
            </Tooltip>
          </Space>
          <Typography.Text type="secondary" className="diagnosis-timezone-label">
            会话时间 · 北京时间
          </Typography.Text>
        </div>
        <List
          className="diagnosis-session-list"
          dataSource={sessions}
          locale={{ emptyText: '暂无会话' }}
          renderItem={(item) => (
            <List.Item
              className={`agent-session-item${item.id === sessionId ? ' ant-list-item-active' : ''}`}
              onClick={() => loadSession(item.id)}
              style={{
                cursor: 'pointer', padding: '11px 14px',
                background: item.id === sessionId ? 'rgba(22,119,255,.10)' : 'transparent',
              }}
              actions={[
                <Space key="actions" size={0} onClick={(event) => event.stopPropagation()}>
                  <Tooltip title="重命名">
                    <Button
                      type="text"
                      size="small"
                      icon={<EditOutlined />}
                      aria-label={`重命名 ${item.title || '诊断会话'}`}
                      onClick={() => openRenameSession(item)}
                    />
                  </Tooltip>
                  <Popconfirm title="删除此诊断会话？" onConfirm={() => removeSession(item.id)}>
                    <Tooltip title="删除">
                      <Button type="text" size="small" icon={<DeleteOutlined />} aria-label={`删除 ${item.title || '诊断会话'}`} />
                    </Tooltip>
                  </Popconfirm>
                </Space>,
              ]}
            >
              <List.Item.Meta
                avatar={<RobotOutlined style={{ color: item.id === sessionId ? '#1677ff' : '#8c8c8c' }} />}
                title={<Typography.Text ellipsis style={{ maxWidth: 150 }}>{item.title || '诊断会话'}</Typography.Text>}
                description={(
                  <span className="diagnosis-session-time" title="北京时间">
                    {formatTimeShort(item.updated_at || item.created_at)}
                  </span>
                )}
              />
            </List.Item>
          )}
        />
      </Sider>

      <Content
        ref={diagnosisGridRef}
        className="diagnosis-main-grid with-evidence"
        style={{ '--diagnosis-evidence-width': `${evidenceWidth}px` } as CSSProperties}
      >
        <Card
          className="glass diagnosis-chat-card"
          styles={{ body: { padding: 0, display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0, overflow: 'hidden' } }}
          title={
            <Space className="diagnosis-card-title" wrap>
              <RobotOutlined />
              <span>ICP 故障诊断</span>
            </Space>
          }
          extra={<Typography.Text type="secondary" ellipsis className="diagnosis-current-session">{currentSession?.title || '诊断会话'}</Typography.Text>}
        >
          <div className="diagnosis-message-scroll">
            {messages.length === 0 && !streamingAnswer ? (
              <div className="diagnosis-welcome">
                <Typography.Title level={3}>描述你观察到的故障现象</Typography.Title>
                <Typography.Paragraph type="secondary">
                  不需要知道具体模块或故障码。系统会先做功能定位，再联合 HMC、FMECA、FTA、RBD、MFL 和历史案例检索。
                </Typography.Paragraph>
                <Space wrap>
                  {SUGGESTIONS.map((item) => (
                    <Button key={item} onClick={() => send(item)}>{item}</Button>
                  ))}
                </Space>
              </div>
            ) : null}

            {messages.map((item, index) => {
              const messageKey = item.id || String(index);
              if (item.role !== 'assistant') {
                return (
                  <div key={messageKey} className="diagnosis-message-row user">
                    <div className="diagnosis-message-bubble user">{item.content}</div>
                  </div>
                );
              }
              const messageEvidenceView = buildEvidenceView(
                item.content,
                item.citations as Citation[] | undefined,
                item.evidence_catalog as Citation[] | undefined,
                item.thinking_events,
                item.used_evidence_ids,
              );
              const citationCount = messageEvidenceView.answer.length;
              return (
                <div key={messageKey} className="diagnosis-message-row assistant">
                  <div className="diagnosis-assistant-turn">
                    <ExecutionTrace
                      events={item.thinking_events}
                      evidenceView={messageEvidenceView}
                      onEvidenceSelect={(evidenceItem) => showExecutionEvidence(messageKey, evidenceItem)}
                    />
                    <div className="diagnosis-message-bubble assistant">
                      <div className="diagnosis-assistant-label">
                        <span><RobotOutlined /> 诊断助手</span>
                        {citationCount ? (
                          <button
                            type="button"
                            className="diagnosis-answer-evidence-button"
                            aria-pressed={selectedEvidenceKey === messageKey}
                            onClick={() => {
                              document.querySelectorAll<HTMLElement>('.inline-cite[aria-expanded="true"]')
                                .forEach((node) => node.setAttribute('aria-expanded', 'false'));
                              setSelectedEvidenceKey(messageKey);
                              setHighlightedCitationNumber(undefined);
                              setHighlightedEvidenceId(undefined);
                            }}
                          >
                            回答引用 {citationCount}
                          </button>
                        ) : null}
                      </div>
                      <div
                        className="diagnosis-markdown"
                        dangerouslySetInnerHTML={{ __html: formatMarkdown(item.content, item.citations, messageKey) }}
                      />
                    </div>
                  </div>
                </div>
              );
            })}

            {generating && (
              <div className="diagnosis-message-row assistant">
                <div className="diagnosis-assistant-turn">
                  <ExecutionTrace
                    events={liveExecutionEvents}
                    evidenceView={liveEvidenceView}
                    onEvidenceSelect={(evidenceItem) => showExecutionEvidence('stream', evidenceItem)}
                    live
                  />
                  <div className="diagnosis-message-bubble assistant">
                    <div className="diagnosis-assistant-label">
                      <span><RobotOutlined /> 诊断助手</span>
                      {liveEvidenceView.answer.length ? (
                        <button
                          type="button"
                          className="diagnosis-answer-evidence-button"
                          aria-pressed={selectedEvidenceKey === 'stream'}
                          onClick={() => {
                            setSelectedEvidenceKey('stream');
                            setHighlightedCitationNumber(undefined);
                            setHighlightedEvidenceId(undefined);
                          }}
                        >
                          回答引用 {liveEvidenceView.answer.length}
                        </button>
                      ) : null}
                    </div>
                    {streamingAnswer ? (
                      <div
                        className="diagnosis-markdown"
                        dangerouslySetInnerHTML={{ __html: formatMarkdown(streamingAnswer, liveCitations, 'stream') }}
                      />
                    ) : (
                      <Space><Spin size="small" /><Typography.Text type="secondary">正在理解并处理问题…</Typography.Text></Space>
                    )}
                    {streamPartialFields.length > 0 && (
                      <div className="diagnosis-partial-fields" style={{ marginTop: 8 }}>
                        <Tag color="orange">部分字段未确认</Tag>
                        <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                          以下必答字段尚未从已批准证据确认：{streamPartialFields.join('、')}。以上内容不构成唯一诊断结论。
                        </Typography.Text>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
            <div ref={chatBottomRef} />
          </div>

          <div className="diagnosis-composer">
            <div className="diagnosis-composer-row">
              <TextArea
                className="agent-input diagnosis-composer-input"
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onPressEnter={(event) => {
                  if (!event.shiftKey) { event.preventDefault(); send(); }
                }}
                autoSize={{ minRows: 2, maxRows: 5 }}
                placeholder="描述现象、发生阶段、告警文字、是否可复现、已执行操作等…"
                disabled={generating}
                aria-label="故障现象描述"
              />
              {generating ? (
                <GlassButton className="glass-btn-black diagnosis-composer-button" icon={<StopOutlined />} onClick={stopGeneration}>停止</GlassButton>
              ) : (
                <GlassButton className="diagnosis-composer-button" variant="primary" icon={<SendOutlined />} onClick={() => send()} disabled={!input.trim()}>诊断</GlassButton>
              )}
            </div>
          </div>
        </Card>

        <>
          <div
            className={`diagnosis-column-resizer${resizingEvidence ? ' active' : ''}`}
            role="separator"
            aria-label="调整对话区与引用证据栏宽度"
            aria-orientation="vertical"
            aria-valuemin={MIN_EVIDENCE_WIDTH}
            aria-valuemax={MAX_EVIDENCE_WIDTH}
            aria-valuenow={Math.round(evidenceWidth)}
            tabIndex={0}
            title="拖动调整宽度；双击恢复默认宽度"
            onPointerDown={beginEvidenceResize}
            onKeyDown={resizeEvidenceByKeyboard}
            onDoubleClick={() => setEvidenceWidth(clampEvidenceWidth(
              DEFAULT_EVIDENCE_WIDTH,
              diagnosisGridRef.current?.getBoundingClientRect().width,
            ))}
          >
            <span aria-hidden="true" />
          </div>
          <aside
            id="diagnosis-evidence-rail"
            ref={evidenceRailRef}
            className="diagnosis-evidence-rail"
            aria-label="当前回答的引用依据"
          >
            <Card
              className="diagnosis-evidence-card diagnosis-citations-card"
              size="small"
              title={<Space><FileSearchOutlined />回答依据</Space>}
              extra={<Tag color="blue">本轮</Tag>}
            >
              {selectedEvidenceTime ? (
                <Typography.Text type="secondary" className="diagnosis-evidence-answer-time">
                  {selectedEvidenceTime}
                </Typography.Text>
              ) : null}
              <div className="diagnosis-evidence-summary">
                {['agent_capability', 'out_of_scope', 'clarification'].includes(selectedScopeRoute)
                  ? '这个回答分支不需要工程资料引用。'
                  : '仅展示最终回答实际采用、可由正文编号定位的来源。'}
              </div>
              {selectedEvidenceItems.length === 0 ? (
                <Empty
                  image={Empty.PRESENTED_IMAGE_SIMPLE}
                  description={emptyEvidenceDescription}
                />
              ) : (
                <div className="diagnosis-citation-list">
                  {selectedEvidenceItems.map((viewItem) => {
                    const citation = viewItem.evidence;
                    const number = viewItem.citationNumber;
                    const cardKey = `${selectedEvidenceKey}:${viewItem.key}`;
                    const idRefKey = viewItem.evidenceId
                      ? `${selectedEvidenceKey}:id:${viewItem.evidenceId}`
                      : '';
                    const citationRefKey = number
                      ? `${selectedEvidenceKey}:citation:${number}`
                      : '';
                    const docName = compactText(citation.doc_name, 180) || '未命名文档';
                    const title = compactText(citation.title, 220);
                    const presentation = evidencePresentation(citation);
                    const authorityKey = scalarText(
                      citation.authority || citation.evidence_level || citation.validation_tier || '',
                    );
                    const authority = CITATION_AUTHORITY_LABEL[authorityKey] || authorityKey;
                    const docTypeKey = scalarText(citation.doc_type);
                    const docType = DOC_TYPE_LABEL[docTypeKey] || docTypeKey.toUpperCase();
                    const highlighted = (
                      Boolean(highlightedEvidenceId && viewItem.evidenceId === highlightedEvidenceId)
                      || Boolean(highlightedCitationNumber && number === highlightedCitationNumber)
                    );
                    return (
                      <article
                        key={cardKey}
                        ref={(node) => {
                          [idRefKey, citationRefKey].filter(Boolean).forEach((refKey) => {
                            if (node) citationCardRefs.current.set(refKey, node);
                            else citationCardRefs.current.delete(refKey);
                          });
                        }}
                        className={`diagnosis-citation-card${highlighted ? ' highlighted' : ''}`}
                        tabIndex={-1}
                        aria-label={`${number ? `证据 ${number}` : '回答依据'}：${docName}`}
                      >
                        <header>
                          {number ? <span className="diagnosis-citation-number">[{number}]</span> : null}
                          <strong title={docName}>{docName}</strong>
                          <span className={`diagnosis-evidence-status ${viewItem.disposition}`}>
                            {EVIDENCE_DISPOSITION_LABEL[viewItem.disposition]}
                          </span>
                        </header>
                        {title && title !== docName ? <div className="diagnosis-citation-title">{title}</div> : null}
                        {docType || authority ? (
                          <div className="diagnosis-citation-tags">
                            {docType ? <Tag>{docType}</Tag> : null}
                            {authority ? <Tag color="blue">{authority}</Tag> : null}
                          </div>
                        ) : null}
                        <EvidenceSourcePreview
                          evidence={citation as AgentEvidence}
                          summaryLabel={presentation.label}
                          summaryText={presentation.text}
                          autoLoad
                        />
                      </article>
                    );
                  })}
                </div>
              )}
            </Card>
        </aside>
        </>
      </Content>
      </Layout>
      <Modal
        title="重命名诊断会话"
        open={Boolean(renameTarget)}
        okText="保存"
        cancelText="取消"
        confirmLoading={renaming}
        onOk={saveSessionTitle}
        onCancel={() => {
          if (renaming) return;
          setRenameTarget(null);
          setRenameValue('');
        }}
      >
        <Input
          autoFocus
          value={renameValue}
          maxLength={64}
          showCount
          placeholder="输入会话名称"
          aria-label="会话名称"
          onChange={(event) => setRenameValue(event.target.value)}
          onPressEnter={saveSessionTitle}
        />
      </Modal>
    </>
  );
}
