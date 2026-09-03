import { useEffect, useRef } from 'react';
import G6 from '@antv/g6';
import type { GraphVertex, GraphEdge } from '@/types';

// 用克制的分类色帮助用户区分对象；同一业务族保持相近色相。
export const ENTITY_TYPE_MAP: Record<string, { color: string; icon: string; name: string }> = {
  DOCUMENT: { color: '#334155', icon: '📄', name: '来源文档' },
  SYSTEM: { color: '#1d4ed8', icon: '🖥️', name: '系统' },
  SUBSYSTEM: { color: '#2563eb', icon: '🧩', name: '子系统' },
  MODULE: { color: '#3b82f6', icon: '📦', name: '模块' },
  COMPONENT: { color: '#0f766e', icon: '⚙️', name: '组件' },
  FUNCTION: { color: '#0d9488', icon: 'ƒ', name: '功能' },
  INTERFACE: { color: '#0891b2', icon: '🔌', name: '接口' },
  HMCCODE: { color: '#7c3aed', icon: '🔢', name: 'HMC 代码' },
  FMECARECORD: { color: '#9333ea', icon: '🧭', name: 'FMECA 记录' },
  MFLRECORD: { color: '#a855f7', icon: '🛠️', name: 'MFL 记录' },
  API: { color: '#0284c7', icon: '🔗', name: 'API' },
  FAULTHISTORYCASE: { color: '#6366f1', icon: '🗂️', name: '历史案例' },
  FAULTHISTORYSTEP: { color: '#4f46e5', icon: '↳', name: '排故步骤' },
  FTADIAGRAMREF: { color: '#be123c', icon: '△', name: 'FTA 引用' },
  RBDDIAGRAMREF: { color: '#15803d', icon: '▰', name: 'RBD 引用' },
  TEXTVALUE: { color: '#64748b', icon: '📝', name: '文本信息' },
  TOPEVENT: { color: '#b91c1c', icon: '🔺', name: '顶事件' },
  INTERMEDIATEEVENT: { color: '#dc2626', icon: '◆', name: '中间事件' },
  BASICEVENT: { color: '#ef4444', icon: '●', name: '基本事件' },
  GATE: { color: '#f97316', icon: '◇', name: '逻辑门' },
  START: { color: '#15803d', icon: '▶', name: 'RBD 起点' },
  END: { color: '#166534', icon: '■', name: 'RBD 终点' },
  RELIABILITYBLOCK: { color: '#16a34a', icon: '▣', name: '可靠性模块' },
  FAULT: { color: '#be123c', icon: '⚡', name: '故障' },
  CAUSE: { color: '#ea580c', icon: '🔍', name: '原因' },
  ACTION: { color: '#059669', icon: '✓', name: '措施' },
  CHECK: { color: '#0f766e', icon: '🔧', name: '检查' },
  CATEGORY: { color: '#475569', icon: '📁', name: '分类' },
  SYMPTOM: { color: '#d97706', icon: '!', name: '症状' },
  EQUIPMENT: { color: '#0369a1', icon: '⚙️', name: '设备' },
  PART: { color: '#1e40af', icon: '🔩', name: '部件' },
  default: { color: '#64748b', icon: '•', name: '其他' },
};

const BASE_NODE_SIZE = 40;
const KEY_NODE_SIZE = 58;
const FORCE_REHEAT_ALPHA = 0.34;

export type DiagramMode = 'architecture' | 'fta' | 'rbd' | 'record';

interface PreparedVertex extends GraphVertex {
  synthetic?: boolean;
  gateOperator?: string;
  gateCaption?: string;
}

interface PreparedEdge extends GraphEdge {
  synthetic?: boolean;
}

const LIGHT_NODE_FILL: Record<string, string> = {
  SYSTEM: '#eaf2ff',
  SUBSYSTEM: '#f1f6ff',
  MODULE: '#f7faff',
  COMPONENT: '#f0fdfa',
  FUNCTION: '#ecfeff',
  INTERFACE: '#ecfeff',
  TOPEVENT: '#fff1f2',
  INTERMEDIATEEVENT: '#fff7ed',
  BASICEVENT: '#fff8f8',
  GATE: '#fff7ed',
  START: '#15803d',
  END: '#166534',
  RELIABILITYBLOCK: '#f0fdf4',
  RBDCONNECTOR: '#166534',
  default: '#f8fafc',
};

// FTA 门必须位于事件与原因之间。这里仅把边上已经存在的 AND/OR/XOR
// 画成工程图符号，不猜测缺失的逻辑，也不写回任何业务数据。
G6.registerNode('icp-fta-gate', {
  options: {
    size: [74, 56],
    style: { fill: '#fff7ed', stroke: '#ea580c', lineWidth: 2 },
    labelCfg: {
      style: { fill: '#9a3412', fontSize: 14, fontWeight: 700 },
    },
  },
  shapeType: 'icp-fta-gate',
  labelPosition: 'center',
  drawShape(cfg: any, group: any) {
    const rawSize = Array.isArray(cfg?.size) ? cfg.size : [74, 56];
    const width = Number(rawSize[0]) || 74;
    const height = Number(rawSize[1]) || 56;
    const operator = String(cfg?.gateOperator || '').toUpperCase();
    let path: any[];
    if (operator === 'OR' || operator === 'XOR') {
      path = [
        ['M', -width / 2, height / 2],
        ['Q', 0, height / 2 - 13, width / 2, height / 2],
        ['Q', width / 2 - 5, -height / 5, 0, -height / 2],
        ['Q', -width / 2 + 5, -height / 5, -width / 2, height / 2],
        ['Z'],
      ];
    } else if (operator === 'AND') {
      path = [
        ['M', -width / 2, height / 2],
        ['L', width / 2, height / 2],
        ['L', width / 2, 0],
        ['C', width / 2, -height / 2, -width / 2, -height / 2, -width / 2, 0],
        ['Z'],
      ];
    } else {
      path = [
        ['M', -width / 2 + 12, -height / 2],
        ['L', width / 2 - 12, -height / 2],
        ['L', width / 2, 0],
        ['L', width / 2 - 12, height / 2],
        ['L', -width / 2 + 12, height / 2],
        ['L', -width / 2, 0],
        ['Z'],
      ];
    }
    const keyShape = group.addShape('path', {
      attrs: { ...cfg.style, path },
      className: 'icp-fta-gate-keyShape',
      name: 'icp-fta-gate-keyShape',
      draggable: true,
    });
    group.shapeMap['icp-fta-gate-keyShape'] = keyShape;
    if (cfg?.gateCaption) {
      group.addShape('text', {
        attrs: {
          x: 0,
          y: height / 2 + 15,
          text: cfg.gateCaption,
          textAlign: 'center',
          textBaseline: 'middle',
          fill: '#7c2d12',
          fontSize: 10,
          fontWeight: 500,
        },
        name: 'gate-caption',
        draggable: true,
      });
    }
    return keyShape;
  },
  getAnchorPoints() {
    return [[0.5, 0], [0.5, 1]];
  },
}, 'single-node');
const CARD_NODE_TYPES = new Set([
  'SYSTEM', 'SUBSYSTEM', 'MODULE', 'COMPONENT', 'FUNCTION', 'INTERFACE',
  'HMCCODE', 'FMECARECORD', 'MFLRECORD', 'FAULTHISTORYCASE', 'FAULTHISTORYSTEP',
  'TEXTVALUE', 'TOPEVENT', 'RELIABILITYBLOCK', 'API', 'FTADIAGRAMREF', 'RBDDIAGRAMREF',
]);

const RELATION_LABELS: Record<string, string> = {
  extracted_from: '来源于', contains: '包含', contains_component: '包含组件',
  contains_function: '包含功能', provides_function: '提供功能', describes_fault: '描述故障',
  has_repair_method: '维修方法', causes: '导致', success_path: '成功路径',
  has_fault_mode: '故障模式', has_fault_cause: '故障原因', has_local_impact: '局部影响',
  has_final_impact: '最终影响', has_detection_method: '检测方法', has_design_improvement: '设计改进',
  has_compensation_action: '补偿措施', applies_in_mission_phase: '适用任务阶段',
  has_function_management: '功能管理', has_product_flag: '产品标识',
  has_module_fault: '模块故障', has_sur_fault: 'SUR 板故障',
  has_subcircuit_fault: '子电路故障', reports_via_api: '通过 API 上报',
  has_api_method: 'API 调用方法', has_api_usage: 'API 使用说明',
  has_maintenance_action: '维修措施', has_diagnosis_step: '包含排故步骤',
  has_step_content: '步骤内容', next_step: '下一步骤', has_fault_symptom: '故障现象',
  has_diagnosis_method: '排故方法', has_recorded_root_cause: '记录根因',
  has_final_solution: '最终方案', has_preventive_action: '预防措施',
  has_verification_result: '验证结果', occurred_in_mission_phase: '发生于任务阶段',
  mentions_function: '涉及功能', mentions_module: '涉及模块', references_hmc: '关联 HMC',
  references_fmeca: '关联 FMECA', references_fta: '关联 FTA', references_rbd: '关联 RBD',
  references_mfl: '关联 MFL',
};

function relationLabel(value: unknown): string {
  const raw = String(value || '').trim();
  return RELATION_LABELS[raw.toLowerCase()] || raw;
}

// 工程图中的天然入口、边界或结论类型；其他关键节点由显式标记、根节点和度数补充。
const SEMANTIC_KEY_TYPES = new Set(['DOCUMENT', 'SYSTEM', 'TOPEVENT', 'START', 'END']);

const FORCE_LAYOUT: Record<string, unknown> = {
  type: 'force',
  preventOverlap: true,
  nodeSize: (node: any) => nodeSizeForLayout(node?.size),
  nodeSpacing: (node: any) => (node?.isKey ? 34 : 24),
  linkDistance: (edge: any) => (edge?.source?.isKey || edge?.target?.isKey ? 190 : 155),
  nodeStrength: (node: any) => (node?.isKey ? -300 : -210),
  edgeStrength: 0.28,
  collideStrength: 0.85,
  alpha: 0.3,
  alphaDecay: 0.035,
  alphaMin: 0.002,
};

const LAYOUTS: Record<string, any> = {
  force: FORCE_LAYOUT,
  circular: { type: 'circular', radius: 280, divisions: 5, spacing: 80 },
  dagre: { type: 'dagre', rankdir: 'TB', nodesep: 90, ranksep: 160 },
  radial: {
    type: 'radial', radius: 300, linkDistance: 220, preventOverlap: true,
    nodeSpacing: 36, nodeSize: (node: any) => nodeSizeForLayout(node?.size),
  },
};

function nodeSizeForLayout(size: unknown): number {
  if (Array.isArray(size)) return Math.max(...size.map(Number).filter(Number.isFinite), BASE_NODE_SIZE);
  return Number(size) || BASE_NODE_SIZE;
}

interface NodePresentation {
  shape: string;
  size: number | [number, number];
  card?: boolean;
}

function nodePresentation(type: string, isKey: boolean, diagramMode: DiagramMode): NodePresentation {
  if (type === 'RBDCONNECTOR') return { shape: 'circle', size: 13 };

  if (diagramMode === 'fta') {
    if (type === 'GATE') return { shape: 'icp-fta-gate', size: [86, 66] };
    if (type === 'TOPEVENT') return { shape: 'modelRect', size: [360, 104], card: true };
    if (type === 'INTERMEDIATEEVENT') return { shape: 'modelRect', size: [272, 88], card: true };
    if (type === 'BASICEVENT') return { shape: 'ellipse', size: [190, 70] };
  }

  if (diagramMode === 'rbd') {
    if (type === 'START' || type === 'END') return { shape: 'circle', size: 66 };
    // RBD 名称通常是短代码，节点无需做成宽卡片；把空间留给完整路径。
    if (type === 'RELIABILITYBLOCK') return { shape: 'rect', size: [128, 66] };
  }

  if (diagramMode === 'architecture') {
    if (type === 'SYSTEM') return { shape: 'modelRect', size: [310, 100], card: true };
    if (type === 'SUBSYSTEM') return { shape: 'modelRect', size: [250, 84], card: true };
    if (type === 'MODULE' || type === 'COMPONENT') return { shape: 'modelRect', size: [228, 78], card: true };
    if (type === 'FUNCTION' || type === 'INTERFACE') return { shape: 'modelRect', size: [218, 74], card: true };
  }

  if (type === 'FAULTHISTORYSTEP') {
    return { shape: 'modelRect', size: [300, 120], card: true };
  }

  if (CARD_NODE_TYPES.has(type)) {
    return { shape: 'modelRect', size: isKey ? [286, 98] : [240, 104], card: true };
  }
  return { shape: 'circle', size: isKey ? KEY_NODE_SIZE : BASE_NODE_SIZE };
}

function endpointId(endpoint: unknown): string {
  if (endpoint && typeof endpoint === 'object' && 'id' in endpoint) {
    return String((endpoint as { id: unknown }).id || '');
  }
  return String(endpoint || '');
}

function componentEntityType(properties: Record<string, unknown>): string | undefined {
  let summary: unknown = properties.summary;
  if (typeof summary === 'string' && summary.trim().startsWith('{')) {
    try {
      summary = JSON.parse(summary);
    } catch {
      summary = undefined;
    }
  }
  if (!summary || typeof summary !== 'object') return undefined;
  const componentType = String((summary as Record<string, unknown>).component_type || '').toLowerCase();
  if (componentType === 'system') return 'SYSTEM';
  if (componentType === 'subsystem') return 'SUBSYSTEM';
  if (componentType === 'module') return 'MODULE';
  return undefined;
}

function entityTypeOf(vertex: GraphVertex): string {
  const properties = vertex.properties || {};
  const rawType = String(properties.entity_type || vertex.label || vertex.type || '').trim().toUpperCase();
  if (rawType === 'COMPONENT') return componentEntityType(properties) || rawType;
  return rawType;
}

function nodeNameOf(vertex: GraphVertex): string {
  const properties = vertex.properties || {};
  return String(
    properties.display_name
    || properties.name
    || properties.title
    || properties.business_id
    || properties.code
    || vertex.label
    || vertex.id,
  ).trim();
}

function isExplicitKey(properties: Record<string, unknown> = {}): boolean {
  const truthy = (value: unknown) => value === true || value === 1 || String(value).toLowerCase() === 'true';
  if (truthy(properties.is_key) || truthy(properties.is_root) || truthy(properties.key_node)) return true;
  const role = String(properties.role || properties.node_role || '').trim().toUpperCase();
  return ['ROOT', 'TOP', 'START', 'END', 'KEY'].includes(role);
}

function normalizeGateOperator(value: unknown): string | undefined {
  const operator = String(value || '').trim().toUpperCase();
  return ['AND', 'OR', 'XOR'].includes(operator) ? operator : undefined;
}

function edgeGateOperator(edge: GraphEdge): string | undefined {
  return normalizeGateOperator(
    edge.properties?.gate_type
    || edge.properties?.operator
    || edge.properties?.logic,
  );
}

function inferDiagramMode(vertices: GraphVertex[], requested?: DiagramMode): DiagramMode {
  if (requested) return requested;
  const types = new Set(vertices.map(entityTypeOf));
  if (types.has('TOPEVENT') || types.has('BASICEVENT') || types.has('GATE')) return 'fta';
  if (types.has('START') || types.has('END') || types.has('RELIABILITYBLOCK')) return 'rbd';
  if (types.has('SYSTEM') || types.has('SUBSYSTEM') || types.has('MODULE')) return 'architecture';
  return 'record';
}

function normalizedEdge(edge: GraphEdge): PreparedEdge {
  return {
    ...edge,
    source: endpointId(edge.source || edge.outV),
    target: endpointId(edge.target || edge.inV),
  };
}

/**
 * 把 FTA 边上已经明确存在的门类型提升为可见的门节点。显式 Gate 节点沿用其
 * 出边的 gate_type；顶事件等非门节点的同类多出边增加一个仅用于绘图的门。
 */
function prepareFtaGraph(vertices: GraphVertex[], edges: GraphEdge[]) {
  const preparedVertices: PreparedVertex[] = vertices.map((vertex) => ({ ...vertex }));
  const vertexById = new Map(preparedVertices.map((vertex) => [vertex.id, vertex]));
  const outgoing = new Map<string, PreparedEdge[]>();
  edges.map(normalizedEdge).forEach((edge) => {
    const list = outgoing.get(edge.source) || [];
    list.push(edge);
    outgoing.set(edge.source, list);
  });
  const preparedEdges: PreparedEdge[] = [];

  preparedVertices.forEach((vertex) => {
    const sourceEdges = outgoing.get(vertex.id) || [];
    if (!sourceEdges.length) return;
    const operators = new Set(sourceEdges.map(edgeGateOperator).filter(Boolean) as string[]);
    const operator = operators.size === 1 ? [...operators][0] : undefined;

    if (entityTypeOf(vertex) === 'GATE') {
      vertex.gateOperator = operator;
      vertex.gateCaption = nodeNameOf(vertex);
      preparedEdges.push(...sourceEdges);
      return;
    }

    if (operator && sourceEdges.length >= 2) {
      const gateId = `__display_gate__${vertex.id}__${operator}`;
      preparedVertices.push({
        id: gateId,
        label: 'Entity',
        type: 'vertex',
        synthetic: true,
        gateOperator: operator,
        properties: {
          entity_type: 'Gate',
          name: operator,
          description: '依据原关系中的 gate_type 生成的显示门',
          display_only: true,
        },
      });
      const first = sourceEdges[0];
      preparedEdges.push({
        source: vertex.id,
        target: gateId,
        label: first.label,
        properties: { ...(first.properties || {}), relation_type: first.label || 'causes' },
        synthetic: true,
      });
      sourceEdges.forEach((edge) => preparedEdges.push({ ...edge, source: gateId }));
      return;
    }

    preparedEdges.push(...sourceEdges);
  });

  // 防御性保留没有对应源节点的边，正常业务图不会进入此分支。
  edges.map(normalizedEdge).forEach((edge) => {
    if (!vertexById.has(edge.source)) preparedEdges.push(edge);
  });
  return { vertices: preparedVertices, edges: preparedEdges };
}

/** RBD 分支/汇合点是显示连接器，不是新的可靠性实体。 */
function prepareRbdGraph(vertices: GraphVertex[], edges: GraphEdge[]) {
  const preparedVertices: PreparedVertex[] = vertices.map((vertex) => ({ ...vertex }));
  let preparedEdges: PreparedEdge[] = edges.map(normalizedEdge);
  const outDegree = new Map<string, number>();
  const inDegree = new Map<string, number>();
  preparedEdges.forEach((edge) => {
    outDegree.set(edge.source, (outDegree.get(edge.source) || 0) + 1);
    inDegree.set(edge.target, (inDegree.get(edge.target) || 0) + 1);
  });

  vertices.forEach((vertex) => {
    if ((outDegree.get(vertex.id) || 0) <= 1) return;
    const connectorId = `__rbd_branch__${vertex.id}`;
    preparedVertices.push({
      id: connectorId,
      synthetic: true,
      properties: { entity_type: 'RbdConnector', name: '分支点', display_only: true },
    });
    const outgoing = preparedEdges.filter((edge) => edge.source === vertex.id);
    preparedEdges = preparedEdges.map((edge) => (
      edge.source === vertex.id ? { ...edge, source: connectorId } : edge
    ));
    preparedEdges.push({
      source: vertex.id,
      target: connectorId,
      label: outgoing[0]?.label,
      properties: { ...(outgoing[0]?.properties || {}), description: 'branch' },
      synthetic: true,
    });
  });

  vertices.forEach((vertex) => {
    if ((inDegree.get(vertex.id) || 0) <= 1) return;
    const connectorId = `__rbd_join__${vertex.id}`;
    preparedVertices.push({
      id: connectorId,
      synthetic: true,
      properties: { entity_type: 'RbdConnector', name: '汇合点', display_only: true },
    });
    const incoming = preparedEdges.filter((edge) => edge.target === vertex.id);
    preparedEdges = preparedEdges.map((edge) => (
      edge.target === vertex.id ? { ...edge, target: connectorId } : edge
    ));
    preparedEdges.push({
      source: connectorId,
      target: vertex.id,
      label: incoming[0]?.label,
      properties: { ...(incoming[0]?.properties || {}), description: 'join' },
      synthetic: true,
    });
  });
  return { vertices: preparedVertices, edges: preparedEdges };
}

function prepareDiagramGraph(vertices: GraphVertex[], edges: GraphEdge[], diagramMode: DiagramMode) {
  if (diagramMode === 'fta') return prepareFtaGraph(vertices, edges);
  if (diagramMode === 'rbd') return prepareRbdGraph(vertices, edges);
  return { vertices: vertices.map((vertex) => ({ ...vertex })), edges: edges.map(normalizedEdge) };
}

/**
 * 关键节点由领域入口/边界类型、显式 key/root 标记、分支根与少量高度连接枢纽确定。
 * 枢纽最多取约 6%，避免把整张图都放大。
 */
function findKeyNodeIds(vertices: GraphVertex[], edges: GraphEdge[]): Set<string> {
  const keyIds = new Set<string>();
  const degree = new Map<string, number>();
  const inDegree = new Map<string, number>();
  const outDegree = new Map<string, number>();

  vertices.forEach((vertex) => {
    degree.set(vertex.id, 0);
    inDegree.set(vertex.id, 0);
    outDegree.set(vertex.id, 0);
    if (SEMANTIC_KEY_TYPES.has(entityTypeOf(vertex)) || isExplicitKey(vertex.properties)) keyIds.add(vertex.id);
  });

  edges.forEach((edge) => {
    const source = endpointId(edge.source || edge.outV);
    const target = endpointId(edge.target || edge.inV);
    if (!degree.has(source) || !degree.has(target)) return;
    degree.set(source, (degree.get(source) || 0) + 1);
    degree.set(target, (degree.get(target) || 0) + 1);
    outDegree.set(source, (outDegree.get(source) || 0) + 1);
    inDegree.set(target, (inDegree.get(target) || 0) + 1);
  });

  const structuralRoots = vertices.filter((vertex) => (
    (inDegree.get(vertex.id) || 0) === 0 && (outDegree.get(vertex.id) || 0) >= 2
  ));
  // 同层表格记录也可能都没有上游；数量过多时不能把整层误判为关键节点。
  if (structuralRoots.length <= 12) {
    structuralRoots.forEach((vertex) => keyIds.add(vertex.id));
  }

  const hubLimit = Math.min(6, Math.max(1, Math.ceil(vertices.length * 0.06)));
  [...vertices]
    .sort((left, right) => (degree.get(right.id) || 0) - (degree.get(left.id) || 0))
    .filter((vertex) => (degree.get(vertex.id) || 0) >= 4)
    .slice(0, hubLimit)
    .forEach((vertex) => keyIds.add(vertex.id));

  return keyIds;
}

function ellipsize(value: string, maxCharacters: number): string {
  const characters = Array.from(value);
  if (characters.length <= maxCharacters) return value;
  return `${characters.slice(0, Math.max(1, maxCharacters - 1)).join('')}…`;
}

function characterWidth(value: string): number {
  // 英文代码和数字比汉字窄；按视觉宽度换行比固定字符数更不容易溢出。
  if (/\s/.test(value)) return 0.35;
  if (/^[\x00-\xff]$/.test(value)) return 0.58;
  return 1;
}

function wrapLabel(value: string, maxWidth: number, maxLines: number): string {
  const characters = Array.from(String(value || '').replace(/\s+/g, ' ').trim());
  if (!characters.length) return '';

  const lines: string[] = [];
  let current = '';
  let width = 0;
  let index = 0;
  for (; index < characters.length; index += 1) {
    const character = characters[index];
    const nextWidth = width + characterWidth(character);
    // 中文标点不能孤零零地出现在下一行开头；允许前一行略微超宽来保留自然阅读节奏。
    const cannotStartLine = /^[，。；：、？！,.!?;:）】》]$/.test(character);
    if (current && nextWidth > maxWidth && !cannotStartLine) {
      lines.push(current.trimEnd());
      if (lines.length >= maxLines) break;
      current = '';
      width = 0;
    }
    current += character;
    width += characterWidth(character);
  }
  if (lines.length < maxLines && current) lines.push(current.trimEnd());

  if (index < characters.length) {
    let last = lines[Math.max(0, lines.length - 1)] || '';
    let lastWidth = Array.from(last).reduce((total, character) => total + characterWidth(character), 0);
    while (last && lastWidth + 1 > maxWidth) {
      const removed = Array.from(last).pop();
      last = Array.from(last).slice(0, -1).join('');
      lastWidth -= removed ? characterWidth(removed) : 0;
    }
    lines[Math.max(0, lines.length - 1)] = `${last.trimEnd()}…`;
  }
  return lines.slice(0, maxLines).join('\n');
}

function nodeDisplayName(vertex: PreparedVertex, entityType: string): string {
  if (entityType === 'GATE') return vertex.gateOperator || '组合门';
  if (entityType === 'START') return '起点';
  if (entityType === 'END') return '终点';
  return nodeNameOf(vertex);
}

function wrappedNodeLabel(value: string, entityType: string, isKey: boolean, nodeCount: number): string {
  if (entityType === 'TOPEVENT') return wrapLabel(value, 11, 2);
  if (entityType === 'SYSTEM') return wrapLabel(value, 11, 2);
  if (entityType === 'SUBSYSTEM') return wrapLabel(value, 10, 2);
  if (entityType === 'MODULE' || entityType === 'COMPONENT') return wrapLabel(value, 10.5, 2);
  if (entityType === 'INTERMEDIATEEVENT') return wrapLabel(value, 8, 2);
  if (entityType === 'BASICEVENT') return wrapLabel(value, 6.5, 2);
  if (entityType === 'RELIABILITYBLOCK') return wrapLabel(value, 5, 2);
  if (entityType === 'GATE' || entityType === 'START' || entityType === 'END') return wrapLabel(value, 7, 1);
  if (entityType === 'TEXTVALUE') return wrapLabel(value, nodeCount > 20 ? 8 : 9, 2);
  if (entityType === 'FAULTHISTORYSTEP') return wrapLabel(value, 14, 3);
  if (isKey) return wrapLabel(value, 14, 2);
  return wrapLabel(value, nodeCount > 20 ? 8 : 9, 2);
}

function nodeFill(entityType: string): string {
  return LIGHT_NODE_FILL[entityType] || LIGHT_NODE_FILL.default;
}

function edgeTheme(diagramMode: DiagramMode) {
  if (diagramMode === 'architecture') {
    return { stroke: '#64748b', lineWidth: 1.9, opacity: 0.94, arrow: G6.Arrow.triangle(8, 10, 1) };
  }
  if (diagramMode === 'fta') {
    return { stroke: '#9f4960', lineWidth: 2, opacity: 0.88, arrow: undefined };
  }
  if (diagramMode === 'rbd') {
    return { stroke: '#15803d', lineWidth: 2.2, opacity: 0.92, arrow: G6.Arrow.triangle(8, 10, 1) };
  }
  return { stroke: '#8793a4', lineWidth: 1.6, opacity: 0.86, arrow: G6.Arrow.triangle(7, 9, 1) };
}

function dagreLayout(diagramMode: DiagramMode, rankDirection: 'TB' | 'LR', historyStepChain = false) {
  const settings = historyStepChain
    ? { nodesep: 18, ranksep: 42 }
    : diagramMode === 'architecture'
    ? { nodesep: 38, ranksep: 82 }
    : diagramMode === 'fta'
      ? { nodesep: 32, ranksep: 44 }
      : diagramMode === 'rbd'
        ? { nodesep: 44, ranksep: 42 }
        : { nodesep: 18, ranksep: 86 };
  return {
    ...LAYOUTS.dagre,
    rankdir: rankDirection,
    ...settings,
    controlPoints: true,
    edgeLabelSpace: false,
  };
}

function relationTypeOf(edge: GraphEdge): string {
  return String(
    edge.properties?.relation_type
    || edge.properties?.description
    || edge.label
    || '',
  ).trim().toLowerCase();
}

function recordFanoutRootId(
  vertices: PreparedVertex[],
  edges: PreparedEdge[],
  diagramMode: DiagramMode,
): string | undefined {
  if (diagramMode !== 'record' || vertices.length < 8) return undefined;
  const vertexIds = new Set(vertices.map((vertex) => String(vertex.id)));
  for (const vertex of vertices) {
    const rootId = String(vertex.id);
    const directTargets = new Set(
      edges
        .filter((edge) => endpointId(edge.source || edge.outV) === rootId)
        .map((edge) => endpointId(edge.target || edge.inV))
        .filter((target) => target && vertexIds.has(target)),
    );
    // FMECA、故障历史概览等记录是“主记录 + 多个明确字段”。继续用单排树会
    // 在常见屏幕上把内容压成细条；只有纯直接扇出才使用环形字段布局，避免
    // 改变带步骤或二级关系的业务结构。
    if (directTargets.size < 7 || directTargets.size !== vertices.length - 1) continue;
    const allDirect = edges.every((edge) => (
      endpointId(edge.source || edge.outV) === rootId
      && directTargets.has(endpointId(edge.target || edge.inV))
    ));
    if (allDirect) return rootId;
  }
  return undefined;
}

function recordFanoutPositions(
  vertices: PreparedVertex[],
  rootId: string,
): Map<string, { x: number; y: number }> {
  const positions = new Map<string, { x: number; y: number }>();
  positions.set(rootId, { x: 0, y: 0 });
  const children = vertices.filter((vertex) => String(vertex.id) !== rootId);
  // 卡片是横向矩形，圆环过扁会让斜对角的相邻字段相互覆盖。九到十个字段
  // 使用更宽的椭圆，在适配视图后仍保持约 14px 的实际屏幕字号。
  const radiusX = children.length >= 9 ? 650 : Math.min(620, Math.max(500, children.length * 62));
  const radiusY = children.length >= 9 ? 220 : 190;
  children.forEach((vertex, index) => {
    const angle = -Math.PI / 2 + (index * Math.PI * 2) / children.length;
    positions.set(String(vertex.id), {
      x: Math.cos(angle) * radiusX,
      y: Math.sin(angle) * radiusY,
    });
  });
  return positions;
}

function historyStepPositions(
  vertices: PreparedVertex[],
  edges: PreparedEdge[],
): Map<string, { x: number; y: number }> {
  const positions = new Map<string, { x: number; y: number }>();
  const caseVertex = vertices.find((vertex) => entityTypeOf(vertex) === 'FAULTHISTORYCASE');
  const steps = vertices.filter((vertex) => entityTypeOf(vertex) === 'FAULTHISTORYSTEP');
  if (!caseVertex || !steps.length) return positions;

  const stepIds = new Set(steps.map((step) => String(step.id)));
  const nextBySource = new Map<string, string>();
  const nextTargets = new Set<string>();
  let firstStepId = '';
  edges.forEach((edge) => {
    const source = endpointId(edge.source || edge.outV);
    const target = endpointId(edge.target || edge.inV);
    const relation = relationTypeOf(edge);
    if (relation === 'has_diagnosis_step' && target && stepIds.has(target)) firstStepId = target;
    if (relation === 'next_step' && stepIds.has(source) && stepIds.has(target)) {
      nextBySource.set(source, target);
      nextTargets.add(target);
    }
  });
  if (!firstStepId) {
    firstStepId = steps
      .map((step) => String(step.id))
      .find((stepId) => !nextTargets.has(stepId)) || String(steps[0].id);
  }

  const orderedIds: string[] = [];
  const visited = new Set<string>();
  let cursor = firstStepId;
  while (cursor && stepIds.has(cursor) && !visited.has(cursor)) {
    orderedIds.push(cursor);
    visited.add(cursor);
    cursor = nextBySource.get(cursor) || '';
  }
  steps
    .map((step) => String(step.id))
    .filter((stepId) => !visited.has(stepId))
    .sort((left, right) => left.localeCompare(right, 'zh-Hans-CN', { numeric: true }))
    .forEach((stepId) => orderedIds.push(stepId));

  positions.set(String(caseVertex.id), { x: 0, y: -190 });
  const columnCount = Math.min(3, Math.max(1, orderedIds.length));
  const xGap = 350;
  const yGap = 190;
  orderedIds.forEach((stepId, index) => {
    const row = Math.floor(index / columnCount);
    const offset = index % columnCount;
    const column = row % 2 === 0 ? offset : columnCount - 1 - offset;
    positions.set(stepId, {
      x: (column - (columnCount - 1) / 2) * xGap,
      y: -20 + row * yGap,
    });
  });
  return positions;
}

function closestCardinalAnchors(
  source: { x: number; y: number } | undefined,
  target: { x: number; y: number } | undefined,
): { sourceAnchor?: number; targetAnchor?: number } {
  if (!source || !target) return {};
  const dx = target.x - source.x;
  const dy = target.y - source.y;
  // 四个锚点依次为上、右、下、左。
  if (Math.abs(dx) >= Math.abs(dy)) {
    return dx >= 0 ? { sourceAnchor: 1, targetAnchor: 3 } : { sourceAnchor: 3, targetAnchor: 1 };
  }
  return dy >= 0 ? { sourceAnchor: 2, targetAnchor: 0 } : { sourceAnchor: 0, targetAnchor: 2 };
}

function ordinaryLabelZoomThreshold(nodeCount: number): number {
  // 普通标签在远景隐藏；节点较少时通过反向字号缩放仍能保持 14px 左右的屏幕字号。
  if (nodeCount <= 15) return 0.68;
  if (nodeCount <= 35) return 0.78;
  if (nodeCount <= 80) return 0.92;
  return 1.08;
}

function readableFitZoomFor(nodeCount: number, diagramMode: DiagramMode): number {
  if (nodeCount <= 20) {
    if (diagramMode === 'fta') return 0.52;
    // RBD 节点以短设备代码为主，可以在不牺牲实际 14px 字号的前提下展示
    // 完整串并联系统路径；较高下限会把终点和末端模块裁出常见 1440px 屏幕。
    if (diagramMode === 'rbd') return 0.47;
    if (diagramMode === 'record') return 0.6;
    // 六到七个子系统横向展开时，0.76 会在 1440px 屏幕裁掉两端；标签字号
    // 会反向补偿，因此降低几何缩放不会使文字一并变小。
    return 0.6;
  }
  if (nodeCount <= 50) {
    if (diagramMode === 'fta') return 0.52;
    if (diagramMode === 'record') return 0.5;
    return 0.7;
  }
  return 0.55;
}

function medianNodeSpacing(graph: any): number | undefined {
  const allItems = graph?.getNodes?.() || [];
  if (allItems.length < 2) return allItems.length === 1 ? Number.POSITIVE_INFINITY : undefined;
  const step = Math.max(1, Math.ceil(allItems.length / 64));
  const points = allItems
    .filter((_: any, index: number) => index % step === 0)
    .slice(0, 64)
    .map((item: any) => {
      const model = item.getModel?.() || {};
      if (!Number.isFinite(model.x) || !Number.isFinite(model.y)) return undefined;
      return graph.getCanvasByPoint?.(model.x, model.y) || { x: model.x, y: model.y };
    })
    .filter(Boolean) as Array<{ x: number; y: number }>;
  if (points.length < 2) return undefined;
  const nearest = points.map((point, index) => {
    let distance = Number.POSITIVE_INFINITY;
    points.forEach((candidate, candidateIndex) => {
      if (candidateIndex === index) return;
      distance = Math.min(distance, Math.hypot(point.x - candidate.x, point.y - candidate.y));
    });
    return distance;
  }).sort((left, right) => left - right);
  return nearest[Math.floor(nearest.length / 2)];
}

function labelShapes(item: any, itemType: 'node' | 'edge'): { label?: any; background?: any } {
  const group = item?.getContainer?.() || item?.get?.('group');
  if (!group) return {};
  const shapeMap = group.shapeMap || group['shapeMap'] || {};
  const children = group.getChildren?.() || group.get?.('children') || [];
  const label = shapeMap[`${itemType}-label`] || shapeMap['text-shape'] || children.find?.((shape: any) => {
    const name = String(shape.get?.('name') || shape.get?.('className') || '');
    return name === `${itemType}-label` || name === 'text-shape';
  });
  const background = shapeMap[`${itemType}-label-bg`] || shapeMap['text-bg-shape'] || children.find?.((shape: any) => {
    const name = String(shape.get?.('name') || shape.get?.('className') || '');
    return name === `${itemType}-label-bg` || name === 'text-bg-shape';
  });
  return { label, background };
}

function keepImportantLabelReadable(item: any, zoom: number) {
  const { label, background } = labelShapes(item, 'node');
  if (!label || label.destroyed) return;
  const safeZoom = Math.max(0.15, zoom);
  const fontSize = Math.min(26, Math.max(15, 15 / safeZoom));
  label.attr({
    fontSize,
    lineHeight: fontSize * 1.24,
    lineWidth: Math.min(4, Math.max(2.5, 2.5 / safeZoom)),
  });
  const group = item.getContainer?.();
  const description = group?.shapeMap?.['rect-description'];
  if (description && !description.destroyed) {
    const descriptionSize = Math.min(18, Math.max(11, 11 / safeZoom));
    description.attr({ fontSize: descriptionSize, lineHeight: descriptionSize * 1.2 });
  }
  // 远景只保留带白色描边的文字，避免放大的标签底板遮挡周围节点。
  if (background && !background.destroyed) background.attr('opacity', zoom >= 0.68 ? 1 : 0);
}

function keepOrdinaryLabelReadable(item: any, itemType: 'node' | 'edge', zoom: number) {
  const { label } = labelShapes(item, itemType);
  if (!label || label.destroyed) return;
  const safeZoom = Math.max(0.15, zoom);
  const baseSize = itemType === 'node' ? 14 : 13;
  const fontSize = Math.min(itemType === 'node' ? 24 : 22, Math.max(baseSize, baseSize / safeZoom));
  label.attr({
    fontSize,
    lineHeight: fontSize * 1.24,
    lineWidth: Math.min(4, Math.max(1.5, 2.3 / safeZoom)),
  });
  if (itemType === 'node') {
    const group = item.getContainer?.();
    const description = group?.shapeMap?.['rect-description'];
    const gateCaption = group?.find?.((shape: any) => shape.get?.('name') === 'gate-caption');
    const secondarySize = Math.min(18, Math.max(11, 11 / safeZoom));
    description?.attr?.({ fontSize: secondarySize, lineHeight: secondarySize * 1.2 });
    gateCaption?.attr?.({ fontSize: secondarySize, lineHeight: secondarySize * 1.2 });
  }
}

function setItemLabel(item: any, visible: boolean) {
  const model = item?.getModel?.();
  if (!model) return;
  const group = item.getContainer?.();
  const shapeMap = group?.shapeMap || group?.['shapeMap'] || {};
  const description = shapeMap['rect-description'] || group?.find?.((shape: any) => (
    shape.get?.('name') === 'rect-description'
  ));
  const gateCaption = group?.find?.((shape: any) => shape.get?.('name') === 'gate-caption');
  if (visible) {
    description?.show?.();
    gateCaption?.show?.();
  } else {
    description?.hide?.();
    gateCaption?.hide?.();
  }
  if (Boolean(model.label) === visible && Boolean(model.labelVisible) === visible) return;
  if (visible) {
    model.label = model.shortLabel;
  } else {
    delete model.label;
    const itemType = item.getType?.() === 'edge' ? 'edge' : 'node';
    const label = shapeMap[`${itemType}-label`] || shapeMap['text-shape'];
    const background = shapeMap[`${itemType}-label-bg`] || shapeMap['text-bg-shape'];
    if (label) group.removeChild?.(label);
    if (background) group.removeChild?.(background);
    delete shapeMap[`${itemType}-label`];
    delete shapeMap[`${itemType}-label-bg`];
    delete shapeMap['text-shape'];
    delete shapeMap['text-bg-shape'];
  }
  model.labelVisible = visible;
  if (visible) item.refresh?.('bbox|label');
  item.clearCache?.();
}

function forceLayoutMethod(graph: any): any | undefined {
  const controller = graph?.get?.('layoutController');
  return controller?.layoutMethods?.find?.((method: any) => method?.getType?.() === 'force')
    || controller?.layoutMethods?.[0];
}

function pauseForceLayout(graph: any) {
  const method = forceLayoutMethod(graph);
  method?.forceSimulation?.stop?.();
  if (method) method.ticking = false;
}

function fitGraphWithinCanvas(graph: any) {
  if (!graph || graph.get?.('destroyed')) return;
  try {
    const width = Number(graph.get?.('width')) || 800;
    const height = Number(graph.get?.('height')) || 600;
    const bounds = (graph.getNodes?.() || []).map((item: any) => {
      // G6 的 getBBox 会把当前字号纳入包围盒；而字号会在缩放后为可读性动态
      // 调整，若再拿该包围盒 fit，会形成“越放大文字、整图越缩小”的反馈环。
      // 视图范围只由节点几何尺寸决定，标签始终留在节点内部。
      const model = item.getModel?.() || {};
      const x = Number(model.x);
      const y = Number(model.y);
      if (!Number.isFinite(x) || !Number.isFinite(y)) return undefined;
      const size = Array.isArray(model.size) ? model.size : [model.size, model.size];
      const nodeWidth = Number(size[0]) || BASE_NODE_SIZE;
      const nodeHeight = Number(size[1]) || BASE_NODE_SIZE;
      return {
        minX: x - nodeWidth / 2,
        maxX: x + nodeWidth / 2,
        minY: y - nodeHeight / 2,
        maxY: y + nodeHeight / 2,
      };
    }).filter(Boolean);
    if (!bounds.length) return;
    const minX = Math.min(...bounds.map((box: any) => box.minX));
    const maxX = Math.max(...bounds.map((box: any) => box.maxX));
    const minY = Math.min(...bounds.map((box: any) => box.minY));
    const maxY = Math.max(...bounds.map((box: any) => box.maxY));
    const contentWidth = Math.max(1, maxX - minX);
    const contentHeight = Math.max(1, maxY - minY);
    const usableWidth = Math.max(120, width - 104);
    const usableHeight = Math.max(120, height - 104);
    // 不再为了“一屏塞下整图”把工程节点压到不可读。超宽图保持可读比例，
    // 用户通过拖动画布浏览其余部分；重置视图也遵循同一规则。
    const readableFloor = Number(graph.get?.('readableFitZoom')) || 0.6;
    const zoom = Math.max(readableFloor, Math.min(1.08, usableWidth / contentWidth, usableHeight / contentHeight));
    graph.zoomTo(zoom, { x: width / 2, y: height / 2 });

    // 超出一屏时不能居中裁掉结构的入口。纵向树从顶节点开始浏览，RBD 横向
    // 路径从起点开始浏览；只有自由关系图才以几何中心定位。
    const alignment = String(graph.get?.('fitAlignment') || 'center');
    const dataX = alignment === 'start-x' ? minX : (minX + maxX) / 2;
    const dataY = alignment === 'start-y' ? minY : (minY + maxY) / 2;
    const current = graph.getCanvasByPoint?.(dataX, dataY);
    if (!current || !Number.isFinite(current.x) || !Number.isFinite(current.y)) return;
    const startPadding = Number(graph.get?.('fitStartPadding')) || 48;
    const targetX = alignment === 'start-x' ? startPadding : width / 2;
    const targetY = alignment === 'start-y' ? startPadding : height / 2;
    graph.translate(targetX - current.x, targetY - current.y);
  } catch {
    try {
      graph.fitCenter();
    } catch {
      /* ignore */
    }
  }
}

function reheatForceLayout(graph: any, draggedItem?: any) {
  const model = draggedItem?.getModel?.();
  if (model) {
    // drag-node 已写入 x/y；清掉可能残留的固定坐标，让节点重新参与整体收敛。
    model.fx = null;
    model.fy = null;
    model.vx = 0;
    model.vy = 0;
  }

  const method = forceLayoutMethod(graph);
  const simulation = method?.forceSimulation;
  if (simulation?.alpha && simulation?.restart) {
    const currentAlpha = Number(simulation.alpha()) || 0;
    if (method) method.ticking = true;
    simulation.alpha(Math.max(currentAlpha, FORCE_REHEAT_ALPHA));
    simulation.alphaTarget?.(0);
    simulation.restart();
    return;
  }

  // 初始布局尚未创建 simulation 时使用公开 API 兜底。
  graph?.updateLayout?.({ ...FORCE_LAYOUT, alpha: FORCE_REHEAT_ALPHA }, undefined, undefined, false);
}

export interface GraphCanvasProps {
  vertices: GraphVertex[];
  edges: GraphEdge[];
  layout: string;
  /** 领域图类型决定节点、连线和布局语法，而不仅是颜色。 */
  diagramMode?: DiagramMode;
  onNodeClick?: (node: any) => void;
  onEdgeClick?: (edge: any) => void;
  onCanvasClick?: () => void;
  highlightStates?: { highlightIds: Set<string>; dimIds: Set<string>; color?: string };
  /** 递增该值可从外部触发适应视图，不影响已有调用方。 */
  fitViewSignal?: number;
  /** 层级图方向；FTA/架构默认自上而下，RBD 可从左向右。 */
  rankDirection?: 'TB' | 'LR';
  /** 外部查找结果需要定位的节点。 */
  focusNodeId?: string;
  focusSignal?: number;
}

export default function GraphCanvas({
  vertices,
  edges,
  layout,
  diagramMode: requestedDiagramMode,
  onNodeClick,
  onEdgeClick,
  onCanvasClick,
  highlightStates,
  fitViewSignal,
  rankDirection = 'TB',
  focusNodeId,
  focusSignal,
}: GraphCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const graphRef = useRef<any>(null);
  const refreshLabelsRef = useRef<() => void>(() => undefined);
  const onNodeClickRef = useRef(onNodeClick);
  const onEdgeClickRef = useRef(onEdgeClick);
  const onCanvasClickRef = useRef(onCanvasClick);
  const highlightStatesRef = useRef(highlightStates);

  // 图实例不应因父组件每次渲染产生的新回调而重建。
  onNodeClickRef.current = onNodeClick;
  onEdgeClickRef.current = onEdgeClick;
  onCanvasClickRef.current = onCanvasClick;
  highlightStatesRef.current = highlightStates;

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    if (graphRef.current) {
      graphRef.current.destroy();
      graphRef.current = null;
    }
    if (vertices.length === 0) return;

    const diagramMode = inferDiagramMode(vertices, requestedDiagramMode);
    const prepared = prepareDiagramGraph(vertices, edges, diagramMode);
    const preparedVertices = prepared.vertices;
    const preparedEdges = prepared.edges;
    const preparedTypes = new Set(preparedVertices.map(entityTypeOf));
    const historyStepChain = diagramMode === 'record'
      && preparedTypes.has('FAULTHISTORYSTEP')
      && [...preparedTypes].every((type) => type === 'FAULTHISTORYCASE' || type === 'FAULTHISTORYSTEP');
    const fanoutRootId = historyStepChain
      ? undefined
      : recordFanoutRootId(preparedVertices, preparedEdges, diagramMode);
    const presetPositions = historyStepChain
      ? historyStepPositions(preparedVertices, preparedEdges)
      : (fanoutRootId ? recordFanoutPositions(preparedVertices, fanoutRootId) : new Map());
    const usesPresetLayout = presetPositions.size === preparedVertices.length;
    const usesDagreLayout = layout === 'dagre' && !usesPresetLayout;
    const keyNodeIds = findKeyNodeIds(preparedVertices, preparedEdges);
    const nodeLabelThreshold = ordinaryLabelZoomThreshold(preparedVertices.length);
    const incomingRelationByNode = new Map<string, string>();
    preparedEdges.forEach((edge) => {
      const target = endpointId(edge.target || edge.inV);
      if (!target || incomingRelationByNode.has(target)) return;
      incomingRelationByNode.set(target, relationLabel(
        edge.properties?.relation_type
        || edge.properties?.description
        || edge.label
        || '',
      ));
    });

    const nodes = preparedVertices.map((vertex) => {
      const name = nodeNameOf(vertex);
      const entityType = entityTypeOf(vertex);
      const typeMeta = ENTITY_TYPE_MAP[entityType] || ENTITY_TYPE_MAP.default;
      const isKey = keyNodeIds.has(vertex.id);
      const presentation = nodePresentation(entityType, isKey, diagramMode);
      const displayName = nodeDisplayName(vertex, entityType);
      const shortLabel = wrappedNodeLabel(displayName, entityType, isKey, preparedVertices.length);
      const labelLineCount = Math.max(1, shortLabel.split('\n').length);
      const isConnector = entityType === 'RBDCONNECTOR';
      const baseStroke = typeMeta.color;
      const baseLineWidth = isKey ? 2.8 : (isConnector ? 1.5 : 1.9);
      const presetPosition = presetPositions.get(String(vertex.id));
      return {
        id: vertex.id,
        x: presetPosition?.x,
        y: presetPosition?.y,
        label: isKey && !isConnector ? shortLabel : undefined,
        shortLabel,
        labelVisible: isKey && !isConnector,
        fullLabel: name,
        type: presentation.shape,
        vertexType: vertex.label || vertex.type,
        entityType,
        isKey,
        labelLineCount,
        synthetic: Boolean(vertex.synthetic),
        size: presentation.size,
        properties: vertex.properties,
        gateOperator: vertex.gateOperator,
        gateCaption: vertex.gateCaption,
        // 扇出记录图把关系名放在目标节点副标题中，既交代字段语义，又避免
        // 多条边在同一横线上堆叠标签。边本身仍可悬停查看完整关系。
        description: presentation.card
          ? (diagramMode === 'record'
            ? (incomingRelationByNode.get(vertex.id) || typeMeta.name)
            : typeMeta.name)
          : undefined,
        anchorPoints: usesPresetLayout
          ? [[0.5, 0], [1, 0.5], [0.5, 1], [0, 0.5]]
          : (rankDirection === 'LR' ? [[0, 0.5], [1, 0.5]] : [[0.5, 0], [0.5, 1]]),
        baseStroke,
        baseLineWidth,
        logoIcon: { show: false },
        stateIcon: { show: false },
        preRect: presentation.card ? {
          show: true,
          width: 6,
          fill: typeMeta.color,
          radius: 3,
        } : undefined,
        style: {
          fill: nodeFill(entityType),
          stroke: baseStroke,
          lineWidth: baseLineWidth,
          shadowColor: isKey ? 'rgba(15, 23, 42, 0.20)' : 'rgba(15, 23, 42, 0.10)',
          shadowBlur: isKey ? 12 : 5,
          shadowOffsetX: 0,
          shadowOffsetY: isKey ? 3 : 2,
          radius: presentation.shape === 'rect' || presentation.shape === 'modelRect' ? 8 : undefined,
        },
        labelCfg: {
          position: presentation.card ? undefined : 'center',
          offset: presentation.card ? 16 : 0,
          style: {
            fill: isConnector || entityType === 'START' || entityType === 'END' ? '#ffffff' : '#172033',
            stroke: undefined,
            lineWidth: 0,
            fontSize: isKey ? 15 : 14,
            lineHeight: isKey ? 19 : 18,
            fontWeight: isKey || entityType === 'GATE' ? 650 : 550,
            textAlign: presentation.card ? 'left' : 'center',
            textBaseline: 'middle',
          },
        },
        descriptionCfg: presentation.card ? {
          // modelRect 的标题基线固定为 y=-5；两行标题需要把类型说明下移，
          // 否则第二行与说明文字会重叠。
          paddingTop: (labelLineCount - 1) * 18,
          style: {
            fill: typeMeta.color,
            fontSize: 11,
            lineHeight: 13,
            fontWeight: 650,
            textAlign: 'left',
          },
        } : undefined,
      };
    });

    const currentEdgeTheme = edgeTheme(diagramMode);
    const edgeList = preparedEdges.map((edge) => {
      const fullLabel = relationLabel(
        edge.properties?.relation_type
        || edge.properties?.description
        || edge.label
        || '',
      );
      const source = endpointId(edge.source || edge.outV);
      const target = endpointId(edge.target || edge.inV);
      const presetAnchors = usesPresetLayout
        ? closestCardinalAnchors(presetPositions.get(source), presetPositions.get(target))
        : {};
      return {
        source,
        target,
        label: undefined,
        shortLabel: ellipsize(fullLabel, 24),
        labelVisible: false,
        fullLabel,
        properties: edge.properties,
        synthetic: Boolean(edge.synthetic),
        sourceAnchor: usesPresetLayout ? presetAnchors.sourceAnchor : (usesDagreLayout ? 1 : undefined),
        targetAnchor: usesPresetLayout ? presetAnchors.targetAnchor : (usesDagreLayout ? 0 : undefined),
        type: usesDagreLayout ? 'polyline' : 'line',
        style: {
          stroke: currentEdgeTheme.stroke,
          lineWidth: currentEdgeTheme.lineWidth,
          opacity: currentEdgeTheme.opacity,
          radius: 10,
          endArrow: currentEdgeTheme.arrow,
          lineAppendWidth: 14,
        },
        labelCfg: {
          autoRotate: layout !== 'dagre',
          style: {
            fill: '#3f4752',
            stroke: 'rgba(255,255,255,0.98)',
            lineWidth: 2.5,
            fontSize: 13,
            lineHeight: 16,
            fontWeight: 500,
          },
        },
      };
    });

    const graph = new G6.Graph({
      container,
      width: container.offsetWidth || 800,
      height: container.offsetHeight || 600,
      fitView: false,
      fitViewPadding: 48,
      minZoom: 0.35,
      maxZoom: 4,
      // 结构图应加载后立即稳定；只有自由关联布局保留运动过程。
      animate: layout !== 'dagre' && !usesPresetLayout,
      modes: {
        default: [
          'drag-canvas',
          { type: 'zoom-canvas', sensitivity: 1.4 },
          'drag-node',
          'click-select',
          'brush-select',
        ],
      },
      defaultNode: { size: BASE_NODE_SIZE, type: 'circle' },
      defaultEdge: { type: 'polyline' },
      layout: usesPresetLayout
        ? { type: 'preset' }
        : layout === 'dagre'
        ? dagreLayout(diagramMode, rankDirection, historyStepChain)
        : (LAYOUTS[layout] || FORCE_LAYOUT),
      nodeStateStyles: {
        highlight: { stroke: '#0f172a', lineWidth: 3.5, shadowBlur: 16 },
        dim: { opacity: 0.14, shadowBlur: 0 },
        selected: { stroke: '#1a1a1a', lineWidth: 3.5 },
        related: { stroke: '#2563eb', lineWidth: 2.5, shadowBlur: 10 },
      },
      edgeStateStyles: {
        highlight: { stroke: '#0f172a', lineWidth: 3 },
        dim: { opacity: 0.06 },
        related: { stroke: '#2563eb', lineWidth: 2.5, opacity: 1 },
        selected: { stroke: '#1d4ed8', lineWidth: 3, opacity: 1 },
      },
    });
    const readableFitZoom = readableFitZoomFor(preparedVertices.length, diagramMode);
    graph.set('readableFitZoom', readableFitZoom);
    graph.set('fitStartPadding', diagramMode === 'fta' ? 24 : 48);
    graph.set('fitAlignment', usesDagreLayout
      ? (rankDirection === 'LR' ? 'start-x' : 'start-y')
      : 'center');

    let labelFrame = 0;
    const applyLabelVisibility = () => {
      labelFrame = 0;
      if (graph.get('destroyed')) return;
      const zoom = graph.getZoom();
      const highlighted = highlightStatesRef.current?.highlightIds || new Set<string>();
      const spacing = medianNodeSpacing(graph);
      const ordinaryMinZoom = preparedVertices.length <= 35
        ? Math.min(nodeLabelThreshold, readableFitZoom)
        : nodeLabelThreshold;
      const ordinaryReadable = zoom >= ordinaryMinZoom && (
        preparedVertices.length <= 35 || spacing === undefined || spacing >= 82
      );
      // 关系名已进入节点副标题和上方图例。默认不在线条上重复堆字，悬停某条边
      // 时再显示完整关系，这对 FMECA/MFL 等宽扇出图尤其重要。
      const edgeReadable = false;
      graph.getNodes().forEach((item: any) => {
        const model = item.getModel();
        const persistent = Boolean(model.isKey || highlighted.has(String(model.id)));
        const visible = model.entityType !== 'RBDCONNECTOR' && (persistent || ordinaryReadable);
        setItemLabel(item, visible);
        if (persistent) keepImportantLabelReadable(item, zoom);
        else if (visible) keepOrdinaryLabelReadable(item, 'node', zoom);
      });
      graph.getEdges().forEach((item: any) => {
        setItemLabel(item, edgeReadable);
        if (edgeReadable) keepOrdinaryLabelReadable(item, 'edge', zoom);
      });
      graph.paint();
    };
    const scheduleLabelVisibility = () => {
      if (labelFrame) window.cancelAnimationFrame(labelFrame);
      labelFrame = window.requestAnimationFrame(applyLabelVisibility);
    };
    refreshLabelsRef.current = scheduleLabelVisibility;

    // G6 的 dagre `afterlayout` 可能早于节点最终包围盒稳定。立即 fit 会偶发只
    // 居中根节点，而把下一层留在画布之外；双帧后再补一次短延迟校正，确保切换
    // 小分支时用户始终能看到完整关系，而不是一个看似孤立的根节点。
    let fitFrame = 0;
    let settleFitTimer = 0;
    const applySettledFit = () => {
      if (graph.get('destroyed')) return;
      fitGraphWithinCanvas(graph);
      scheduleLabelVisibility();
    };
    const scheduleSettledFit = () => {
      if (fitFrame) window.cancelAnimationFrame(fitFrame);
      if (settleFitTimer) window.clearTimeout(settleFitTimer);
      fitFrame = window.requestAnimationFrame(() => {
        fitFrame = window.requestAnimationFrame(() => {
          fitFrame = 0;
          applySettledFit();
        });
      });
      settleFitTimer = window.setTimeout(() => {
        settleFitTimer = 0;
        applySettledFit();
      }, 160);
    };

    let initialFitDone = false;
    graph.on('afterlayout', () => {
      if (!initialFitDone) {
        initialFitDone = true;
        scheduleSettledFit();
      }
      scheduleLabelVisibility();
    });
    graph.on('viewportchange', (event: any) => {
      if (event?.action === 'zoom') scheduleLabelVisibility();
    });
    graph.on('node:dragstart', () => {
      if (layout === 'force') pauseForceLayout(graph);
    });
    graph.on('node:dragend', (event: any) => {
      if (layout === 'force') reheatForceLayout(graph, event?.item);
    });
    graph.on('node:mouseenter', (event: any) => {
      if (!event?.item) return;
      const model = event.item.getModel?.();
      if (model && !model.labelVisible) setItemLabel(event.item, true);
      graph.paint();
    });
    graph.on('node:mouseleave', scheduleLabelVisibility);
    graph.on('edge:mouseenter', (event: any) => {
      if (!event?.item) return;
      setItemLabel(event.item, true);
      keepOrdinaryLabelReadable(event.item, 'edge', graph.getZoom());
      graph.paint();
    });
    graph.on('edge:mouseleave', scheduleLabelVisibility);
    graph.on('node:click', (event: any) => {
      graph.getNodes().forEach((item: any) => graph.setItemState(item, 'related', false));
      graph.getEdges().forEach((item: any) => graph.setItemState(item, 'related', false));
      if (event?.item) {
        event.item.getEdges?.().forEach((edgeItem: any) => {
          graph.setItemState(edgeItem, 'related', true);
          const source = edgeItem.getSource?.();
          const target = edgeItem.getTarget?.();
          if (source && source !== event.item) graph.setItemState(source, 'related', true);
          if (target && target !== event.item) graph.setItemState(target, 'related', true);
        });
      }
      const model = event?.item?.getModel?.();
      if (model && !model.synthetic && onNodeClickRef.current) {
        onNodeClickRef.current({ ...model, label: model.fullLabel || model.label });
      }
    });
    graph.on('edge:click', (event: any) => {
      graph.getNodes().forEach((item: any) => {
        graph.setItemState(item, 'related', false);
        graph.setItemState(item, 'selected', false);
      });
      graph.getEdges().forEach((item: any) => {
        graph.setItemState(item, 'related', false);
        graph.setItemState(item, 'selected', false);
      });
      if (event?.item) {
        graph.setItemState(event.item, 'selected', true);
        const source = event.item.getSource?.();
        const target = event.item.getTarget?.();
        if (source) graph.setItemState(source, 'related', true);
        if (target) graph.setItemState(target, 'related', true);
      }
      const model = event?.item?.getModel?.();
      if (model && !model.synthetic && onEdgeClickRef.current) {
        onEdgeClickRef.current({ ...model, label: model.fullLabel || model.label });
      }
    });
    graph.on('canvas:click', () => {
      graph.getNodes().forEach((item: any) => {
        graph.setItemState(item, 'related', false);
        graph.setItemState(item, 'selected', false);
      });
      graph.getEdges().forEach((item: any) => {
        graph.setItemState(item, 'related', false);
        graph.setItemState(item, 'selected', false);
      });
      onCanvasClickRef.current?.();
    });

    graph.data({ nodes, edges: edgeList } as any);
    graph.render();
    graphRef.current = graph;
    scheduleSettledFit();

    return () => {
      if (labelFrame) window.cancelAnimationFrame(labelFrame);
      if (fitFrame) window.cancelAnimationFrame(fitFrame);
      if (settleFitTimer) window.clearTimeout(settleFitTimer);
      refreshLabelsRef.current = () => undefined;
      if (graphRef.current === graph) graphRef.current = null;
      graph.destroy();
    };
  }, [vertices, edges, layout, rankDirection, requestedDiagramMode]);

  // 应用高亮/dim 状态；高亮节点即使在缩小时也显示标签。
  useEffect(() => {
    const graph = graphRef.current;
    if (!graph) return;
    const highlightIds = highlightStates?.highlightIds || new Set<string>();
    const dimIds = highlightStates?.dimIds || new Set<string>();
    const color = highlightStates?.color;
    const data = graph.get('data') as any;
    const nodes = data.nodes || [];
    nodes.forEach((node: any) => {
      const item = graph.findById(node.id);
      if (!item) return;
      if (highlightIds.has(node.id)) {
        graph.setItemState(node.id, 'highlight', true);
        graph.setItemState(node.id, 'dim', false);
        if (color) item.update({ style: { stroke: color, lineWidth: 4 } });
      } else if (dimIds.has(node.id)) {
        graph.setItemState(node.id, 'highlight', false);
        graph.setItemState(node.id, 'dim', true);
      } else {
        graph.setItemState(node.id, 'highlight', false);
        graph.setItemState(node.id, 'dim', false);
        item.update({
          style: {
            stroke: node.baseStroke || '#64748b',
            lineWidth: node.baseLineWidth || (node.isKey ? 2.8 : 1.9),
          },
        });
      }
    });
    graph.paint();
    refreshLabelsRef.current();
  }, [highlightStates]);

  useEffect(() => {
    if (fitViewSignal === undefined) return;
    const graph = graphRef.current;
    if (!graph || graph.get('destroyed')) return;
    fitGraphWithinCanvas(graph);
    refreshLabelsRef.current();
  }, [fitViewSignal]);

  useEffect(() => {
    if (!focusNodeId) return;
    // 跨范围搜索会先重建画布并执行延迟 fit；聚焦必须晚于该过程，否则 fit 会
    // 把刚居中的目标再次推到角落。力导向图需要多留一点收敛时间。
    let frame = 0;
    const timer = window.setTimeout(() => {
      const graph = graphRef.current;
      if (!graph || graph.get('destroyed')) return;
      frame = window.requestAnimationFrame(() => {
        const item = graph.findById(focusNodeId);
        if (!item) return;
        try {
          const currentZoom = graph.getZoom();
          if (currentZoom < 0.55) {
            graph.zoomTo(0.72, {
              x: (containerRef.current?.offsetWidth || 800) / 2,
              y: (containerRef.current?.offsetHeight || 600) / 2,
            });
          }
          graph.focusItem(item, true, { duration: 320, easing: 'easeCubic' });
          setItemLabel(item, true);
          graph.paint();
          refreshLabelsRef.current();
        } catch {
          /* ignore */
        }
      });
    }, layout === 'force' ? 480 : 220);
    return () => {
      window.clearTimeout(timer);
      if (frame) window.cancelAnimationFrame(frame);
    };
  }, [focusNodeId, focusSignal, layout]);

  // 只调整画布尺寸，不销毁并重跑图布局，从而保留用户已经整理的视图位置。
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    let resizeFrame = 0;
    const resize = () => {
      if (resizeFrame) window.cancelAnimationFrame(resizeFrame);
      resizeFrame = window.requestAnimationFrame(() => {
        const graph = graphRef.current;
        if (!graph || graph.get('destroyed')) return;
        const width = container.offsetWidth;
        const height = container.offsetHeight;
        if (width > 0 && height > 0) {
          graph.changeSize(width, height);
          fitGraphWithinCanvas(graph);
          refreshLabelsRef.current();
        }
      });
    };

    if (typeof ResizeObserver !== 'undefined') {
      const observer = new ResizeObserver(resize);
      observer.observe(container);
      return () => {
        observer.disconnect();
        if (resizeFrame) window.cancelAnimationFrame(resizeFrame);
      };
    }

    window.addEventListener('resize', resize);
    return () => {
      window.removeEventListener('resize', resize);
      if (resizeFrame) window.cancelAnimationFrame(resizeFrame);
    };
  }, []);

  return (
    <div
      ref={containerRef}
      style={{ width: '100%', height: '100%', minHeight: 0, background: 'transparent', position: 'relative' }}
    />
  );
}
