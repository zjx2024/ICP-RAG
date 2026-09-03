import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Card, Descriptions, Drawer, Empty, Select, Space, Tag, message,
} from 'antd';
import {
  AppstoreOutlined, DownOutlined, ReloadOutlined, SearchOutlined, UndoOutlined,
} from '@ant-design/icons';
import { useParams } from 'react-router-dom';
import { fetchIcpDocuments, fetchIcpGraphData } from '@/api/icp';
import GraphCanvas, { ENTITY_TYPE_MAP, type DiagramMode } from '@/components/GraphCanvas';
import type { GraphEdge, GraphVertex, ICPDocument } from '@/types';
import { GlassButton } from '@/components/GlassUI';

const PROVENANCE_RELATIONS = new Set(['extracted_from']);

const RECORD_ANCHOR_TYPES: Record<string, string> = {
  hmc: 'HMCCODE',
  fmeca: 'FMECARECORD',
  mfl: 'MFLRECORD',
  fault_history: 'FAULTHISTORYCASE',
};

const RANGE_LABELS: Record<string, string> = {
  function_locator: '系统或功能分支',
  fta: '故障树',
  rbd: '可靠性路径',
  hmc: 'HMC 条目',
  fmeca: '故障模式',
  mfl: '维修条目',
  fault_history: '历史案例',
};

type HistoryView = 'overview' | 'steps' | 'references';

const HISTORY_VIEW_OPTIONS: Array<{ value: HistoryView; label: string }> = [
  { value: 'overview', label: '案例概览' },
  { value: 'steps', label: '排故步骤' },
  { value: 'references', label: '关联资料' },
];

const HISTORY_OVERVIEW_RELATIONS = new Set([
  'has_fault_symptom', 'occurred_in_mission_phase', 'mentions_module', 'mentions_function',
  'has_diagnosis_method', 'has_recorded_root_cause', 'has_final_solution',
  'has_verification_result', 'has_preventive_action',
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

interface GraphScope {
  id: string;
  label: string;
  description: string;
  nodeIds: Set<string>;
  anchorId?: string;
}

function endpointId(endpoint: unknown): string {
  if (endpoint && typeof endpoint === 'object' && 'id' in endpoint) {
    return String((endpoint as { id: unknown }).id || '');
  }
  return String(endpoint || '');
}

function edgeSource(edge: GraphEdge): string {
  return endpointId(edge.source || edge.outV);
}

function edgeTarget(edge: GraphEdge): string {
  return endpointId(edge.target || edge.inV);
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

function entityType(vertex: GraphVertex): string {
  const properties = vertex.properties || {};
  const rawType = String(properties.entity_type || vertex.type || vertex.label || '').trim().toUpperCase();
  if (rawType === 'COMPONENT') return componentEntityType(properties) || rawType;
  return rawType;
}

function nodeName(vertex: GraphVertex): string {
  const properties = vertex.properties || {};
  return String(
    properties.display_name || properties.name || properties.title
    || properties.business_id || properties.code || vertex.label || vertex.id,
  ).trim();
}

function rawRelation(edge: GraphEdge): string {
  return String(edge.properties?.relation_type || edge.label || '').trim().toLowerCase();
}

function relationName(edge: GraphEdge | any): string {
  const raw = String(edge?.properties?.relation_type || edge?.label || '').trim();
  return RELATION_LABELS[raw.toLowerCase()] || raw || '关联关系';
}

function defaultLayout(_docType: string): string {
  // 默认范围已经收敛到单棵树、单条路径或单条业务记录，层级排列比随机散开更容易读懂方向。
  return 'dagre';
}

function diagramModeFor(docType: string): DiagramMode {
  if (docType === 'function_locator') return 'architecture';
  if (docType === 'fta') return 'fta';
  if (docType === 'rbd') return 'rbd';
  return 'record';
}

function semanticGraph(vertices: GraphVertex[], edges: GraphEdge[]) {
  const businessVertices = vertices.filter((vertex) => entityType(vertex) !== 'DOCUMENT');
  const businessIds = new Set(businessVertices.map((vertex) => vertex.id));
  const businessEdges = edges.filter((edge) => (
    !PROVENANCE_RELATIONS.has(rawRelation(edge))
    && businessIds.has(edgeSource(edge))
    && businessIds.has(edgeTarget(edge))
  ));
  return { businessVertices, businessEdges };
}

function scopeDescription(docType: string, title: string): string {
  switch (docType) {
    case 'function_locator': return `查看“${title}”中的组成、功能和明确连接关系`;
    case 'fta': return `以“${title}”为顶事件，向下查看逻辑门与基本事件`;
    case 'rbd': return '从起点到终点查看可靠性模块、串联路径与并行分支';
    case 'hmc': return `查看 HMC ${title} 对应的故障现象和维修方法`;
    case 'fmeca': return `查看故障模式 ${title} 的原因、影响、检测与改进信息`;
    case 'mfl': return `查看维修条目 ${title} 的模块故障、API 与维修措施`;
    case 'fault_history': return `查看历史案例 ${title} 的现象、原因、步骤和最终方案`;
    default: return `查看“${title}”及其直接关联内容`;
  }
}

function buildFunctionScopes(vertices: GraphVertex[], edges: GraphEdge[]): GraphScope[] {
  const vertexById = new Map(vertices.map((vertex) => [vertex.id, vertex]));
  const outgoing = new Map(vertices.map((vertex) => [vertex.id, [] as string[]]));
  const inDegree = new Map(vertices.map((vertex) => [vertex.id, 0]));

  edges.forEach((edge) => {
    const source = edgeSource(edge);
    const target = edgeTarget(edge);
    if (!vertexById.has(source) || !vertexById.has(target)) return;
    outgoing.get(source)?.push(target);
    inDegree.set(target, (inDegree.get(target) || 0) + 1);
  });

  const descendants = (startId: string) => {
    const found = new Set<string>([startId]);
    const queue = [startId];
    while (queue.length) {
      const current = queue.shift()!;
      (outgoing.get(current) || []).forEach((next) => {
        if (found.has(next)) return;
        found.add(next);
        queue.push(next);
      });
    }
    return found;
  };

  const roots = vertices
    .filter((vertex) => (inDegree.get(vertex.id) || 0) === 0 && (outgoing.get(vertex.id)?.length || 0) > 0)
    .sort((left, right) => {
      const priority = (vertex: GraphVertex) => {
        const name = nodeName(vertex);
        if (name.includes('综合信息处理与显示系统')) return 10000;
        if (entityType(vertex) === 'SYSTEM') return 9000;
        if (name === '系统任务') return 8000;
        return descendants(vertex.id).size;
      };
      return priority(right) - priority(left) || nodeName(left).localeCompare(nodeName(right), 'zh-CN');
    });

  const scopes: GraphScope[] = [];
  const seenSignatures = new Set<string>();
  const appendScope = (scope: GraphScope) => {
    const signature = [...scope.nodeIds].sort().join('|');
    if (seenSignatures.has(signature)) return;
    seenSignatures.add(signature);
    scopes.push(scope);
  };

  roots.forEach((root) => {
    const children = [...new Set(outgoing.get(root.id) || [])];
    const title = nodeName(root);
    if (children.length >= 2) {
      appendScope({
        id: `function:${root.id}:overview`, label: title,
        description: scopeDescription('function_locator', title),
        nodeIds: new Set([root.id, ...children]), anchorId: root.id,
      });

      children.forEach((childId) => {
        const branchIds = descendants(childId);
        if (branchIds.size <= 1) return;
        branchIds.add(root.id);
        const child = vertexById.get(childId);
        if (!child) return;
        const childTitle = nodeName(child);
        appendScope({
          id: `function:${root.id}:${childId}`, label: childTitle,
          description: scopeDescription('function_locator', childTitle),
          nodeIds: branchIds, anchorId: childId,
        });
      });
      return;
    }

    appendScope({
      id: `function:${root.id}`, label: title,
      description: scopeDescription('function_locator', title),
      nodeIds: descendants(root.id), anchorId: root.id,
    });
  });

  // 孤立事实仍可查找和查看，但不为了丰富画面补造关系。
  const covered = new Set(scopes.flatMap((scope) => [...scope.nodeIds]));
  vertices.filter((vertex) => !covered.has(vertex.id)).forEach((vertex) => {
    const title = nodeName(vertex);
    appendScope({
      id: `function:${vertex.id}`, label: title,
      description: scopeDescription('function_locator', title),
      nodeIds: new Set([vertex.id]), anchorId: vertex.id,
    });
  });
  return scopes;
}

function buildConnectedScopes(docType: string, vertices: GraphVertex[], edges: GraphEdge[]): GraphScope[] {
  const vertexById = new Map(vertices.map((vertex) => [vertex.id, vertex]));
  const adjacency = new Map(vertices.map((vertex) => [vertex.id, new Set<string>()]));
  const inDegree = new Map(vertices.map((vertex) => [vertex.id, 0]));
  const outDegree = new Map(vertices.map((vertex) => [vertex.id, 0]));
  edges.forEach((edge) => {
    const source = edgeSource(edge);
    const target = edgeTarget(edge);
    if (!vertexById.has(source) || !vertexById.has(target)) return;
    adjacency.get(source)?.add(target);
    adjacency.get(target)?.add(source);
    inDegree.set(target, (inDegree.get(target) || 0) + 1);
    outDegree.set(source, (outDegree.get(source) || 0) + 1);
  });

  const visited = new Set<string>();
  const components: Set<string>[] = [];
  vertices.forEach((vertex) => {
    if (visited.has(vertex.id)) return;
    const ids = new Set<string>([vertex.id]);
    const queue = [vertex.id];
    visited.add(vertex.id);
    while (queue.length) {
      const current = queue.shift()!;
      (adjacency.get(current) || []).forEach((next) => {
        if (visited.has(next)) return;
        visited.add(next);
        ids.add(next);
        queue.push(next);
      });
    }
    components.push(ids);
  });

  const priorityTypes = docType === 'fta' ? ['TOPEVENT'] : docType === 'rbd' ? ['START'] : [];
  const scopes = components.map((nodeIds, index) => {
    const nodes = [...nodeIds].map((id) => vertexById.get(id)!).filter(Boolean);
    const rootCandidates = nodes
      .filter((node) => (inDegree.get(node.id) || 0) === 0)
      .sort((left, right) => (outDegree.get(right.id) || 0) - (outDegree.get(left.id) || 0));
    const anchor = priorityTypes
      .map((type) => nodes.find((node) => entityType(node) === type))
      .find(Boolean) || rootCandidates[0] || nodes[0];
    let label = nodeName(anchor);
    if (docType === 'rbd') {
      const blocks = nodes.filter((node) => entityType(node) === 'RELIABILITYBLOCK').map(nodeName).slice(0, 3);
      label = `可靠性路径 ${index + 1}${blocks.length ? ` · ${blocks.join(' / ')}` : ''}`;
    }
    return {
      id: `${docType}:component:${anchor.id}`, label,
      description: scopeDescription(docType, label), nodeIds, anchorId: anchor.id,
    };
  });
  return scopes.sort((left, right) => (
    right.nodeIds.size - left.nodeIds.size || left.label.localeCompare(right.label, 'zh-CN', { numeric: true })
  ));
}

function buildRecordScopes(docType: string, vertices: GraphVertex[], edges: GraphEdge[]): GraphScope[] {
  const anchorType = RECORD_ANCHOR_TYPES[docType];
  const vertexById = new Map(vertices.map((vertex) => [vertex.id, vertex]));
  const connectedEdges = new Map(vertices.map((vertex) => [vertex.id, [] as GraphEdge[]]));
  edges.forEach((edge) => {
    connectedEdges.get(edgeSource(edge))?.push(edge);
    connectedEdges.get(edgeTarget(edge))?.push(edge);
  });
  const anchors = vertices
    .filter((vertex) => entityType(vertex) === anchorType)
    .sort((left, right) => nodeName(left).localeCompare(nodeName(right), 'zh-CN', { numeric: true }));

  if (!anchors.length) return buildConnectedScopes(docType, vertices, edges);
  return anchors.map((anchor) => {
    const nodeIds = new Set<string>([anchor.id]);
    (connectedEdges.get(anchor.id) || []).forEach((edge) => {
      nodeIds.add(edgeSource(edge));
      nodeIds.add(edgeTarget(edge));
    });

    // 历史案例的步骤正文位于步骤节点的下一层，只沿明确步骤关系多取一层。
    if (docType === 'fault_history') {
      [...nodeIds].forEach((id) => {
        const vertex = vertexById.get(id);
        if (!vertex || entityType(vertex) !== 'FAULTHISTORYSTEP') return;
        (connectedEdges.get(id) || []).forEach((edge) => {
          nodeIds.add(edgeSource(edge));
          nodeIds.add(edgeTarget(edge));
        });
      });
    }

    const title = nodeName(anchor);
    return {
      id: `${docType}:record:${anchor.id}`, label: title,
      description: scopeDescription(docType, title), nodeIds, anchorId: anchor.id,
    };
  });
}

function buildScopes(docType: string, vertices: GraphVertex[], edges: GraphEdge[]): GraphScope[] {
  const { businessVertices, businessEdges } = semanticGraph(vertices, edges);
  if (!businessVertices.length) return [];
  if (docType === 'function_locator') return buildFunctionScopes(businessVertices, businessEdges);
  if (docType === 'fta' || docType === 'rbd') return buildConnectedScopes(docType, businessVertices, businessEdges);
  return buildRecordScopes(docType, businessVertices, businessEdges);
}

function historyViewGraph(
  scope: GraphScope,
  vertices: GraphVertex[],
  edges: GraphEdge[],
  view: HistoryView,
): { vertices: GraphVertex[]; edges: GraphEdge[] } {
  const scopedVertices = vertices.filter((vertex) => scope.nodeIds.has(vertex.id));
  const vertexById = new Map(scopedVertices.map((vertex) => [vertex.id, vertex]));
  const scopedEdges = edges.filter((edge) => (
    scope.nodeIds.has(edgeSource(edge)) && scope.nodeIds.has(edgeTarget(edge))
  ));
  const anchorId = scope.anchorId;
  if (!anchorId) return { vertices: scopedVertices, edges: scopedEdges };

  if (view === 'overview' || view === 'references') {
    const selectedEdges = scopedEdges.filter((edge) => {
      if (edgeSource(edge) !== anchorId) return false;
      const relation = rawRelation(edge);
      return view === 'overview'
        ? HISTORY_OVERVIEW_RELATIONS.has(relation)
        : relation.startsWith('references_');
    });
    const selectedIds = new Set<string>([anchorId]);
    selectedEdges.forEach((edge) => selectedIds.add(edgeTarget(edge)));
    return {
      vertices: scopedVertices.filter((vertex) => selectedIds.has(vertex.id)),
      edges: selectedEdges,
    };
  }

  const diagnosisEdges = scopedEdges.filter((edge) => (
    edgeSource(edge) === anchorId && rawRelation(edge) === 'has_diagnosis_step'
  ));
  const stepIds = new Set(diagnosisEdges.map(edgeTarget));
  const nextEdges = scopedEdges.filter((edge) => (
    rawRelation(edge) === 'next_step'
    && stepIds.has(edgeSource(edge))
    && stepIds.has(edgeTarget(edge))
  ));
  const contentByStep = new Map<string, GraphVertex>();
  scopedEdges.forEach((edge) => {
    if (rawRelation(edge) !== 'has_step_content' || !stepIds.has(edgeSource(edge))) return;
    const content = vertexById.get(edgeTarget(edge));
    if (content) contentByStep.set(edgeSource(edge), content);
  });

  // 步骤正文仍来自原 has_step_content 节点；这里只把它折叠进步骤卡，避免
  // “步骤节点 + 正文节点 + 案例到每一步”三套线同时铺开。原节点和边未改写。
  const stepVertices = scopedVertices
    .filter((vertex) => stepIds.has(vertex.id))
    .map((vertex) => {
      const content = contentByStep.get(vertex.id);
      return {
        ...vertex,
        properties: {
          ...(vertex.properties || {}),
          display_name: content ? `${nodeName(vertex)}｜${nodeName(content)}` : nodeName(vertex),
          description: content ? nodeName(content) : vertex.properties?.description,
        },
      };
    });
  const nextTargets = new Set(nextEdges.map(edgeTarget));
  const firstStepId = [...stepIds].find((id) => !nextTargets.has(id));
  const firstEdge = diagnosisEdges.find((edge) => edgeTarget(edge) === firstStepId) || diagnosisEdges[0];
  const anchor = vertexById.get(anchorId);
  return {
    vertices: [...(anchor ? [anchor] : []), ...stepVertices],
    edges: [...(firstEdge ? [firstEdge] : []), ...nextEdges],
  };
}

export default function TypeVisualize() {
  const { docType = '' } = useParams<{ docType: string }>();
  const [docs, setDocs] = useState<ICPDocument[]>([]);
  const [docId, setDocId] = useState<string>();
  const [vertices, setVertices] = useState<GraphVertex[]>([]);
  const [edges, setEdges] = useState<GraphEdge[]>([]);
  const [scopeId, setScopeId] = useState<string>();
  const [historyView, setHistoryView] = useState<HistoryView>('overview');
  const [layout, setLayout] = useState(() => defaultLayout(docType));
  const [searchText, setSearchText] = useState('');
  const [selectedNodeId, setSelectedNodeId] = useState<string>();
  const [detailNode, setDetailNode] = useState<any>(null);
  const [detailEdge, setDetailEdge] = useState<any>(null);
  const [legendOpen, setLegendOpen] = useState(false);
  const [fitViewSignal, setFitViewSignal] = useState(0);
  const [focusSignal, setFocusSignal] = useState(0);
  const [loading, setLoading] = useState(false);
  const diagramMode = diagramModeFor(docType);
  const layoutOptions = diagramMode === 'record'
    ? [
      { value: 'dagre', label: '层级排列' },
      { value: 'force', label: '关联排列' },
      { value: 'circular', label: '环形排列' },
      { value: 'radial', label: '中心排列' },
    ]
    : [
      { value: 'dagre', label: docType === 'rbd' ? '可靠性路径' : (docType === 'fta' ? '故障树' : '架构树') },
      { value: 'force', label: '自由查看' },
    ];

  useEffect(() => {
    setLayout(defaultLayout(docType));
    setSearchText('');
    setSelectedNodeId(undefined);
    setScopeId(undefined);
    setHistoryView('overview');
    setLegendOpen(false);
    if (!docType) return;
    fetchIcpDocuments(docType).then((list) => {
      setDocs(list);
      setDocId((current) => (
        current && list.some((doc) => doc.doc_id === current) ? current : list[0]?.doc_id
      ));
    }).catch(() => {
      setDocs([]);
      setDocId(undefined);
    });
  }, [docType]);

  const loadGraph = useCallback(async (did?: string) => {
    if (!docType || !did) {
      setVertices([]);
      setEdges([]);
      setLoading(false);
      return;
    }
    setLoading(true);
    try {
      const response = await fetchIcpGraphData(docType, did);
      if (response.error) {
        message.error('关系图暂时无法加载，请稍后重试');
        setVertices([]);
        setEdges([]);
        return;
      }
      setVertices((response.vertices || []) as GraphVertex[]);
      setEdges((response.edges || []) as GraphEdge[]);
    } catch {
      message.error('关系图加载失败，请稍后重试');
      setVertices([]);
      setEdges([]);
    } finally {
      setLoading(false);
    }
  }, [docType]);

  useEffect(() => {
    setSearchText('');
    setSelectedNodeId(undefined);
    setScopeId(undefined);
    setHistoryView('overview');
    setDetailNode(null);
    setDetailEdge(null);
    loadGraph(docId);
  }, [docId, loadGraph]);

  const scopes = useMemo(() => buildScopes(docType, vertices, edges), [docType, vertices, edges]);
  const activeScope = useMemo(
    () => scopes.find((scope) => scope.id === scopeId) || scopes[0],
    [scopes, scopeId],
  );

  useEffect(() => {
    if (activeScope && activeScope.id !== scopeId) setScopeId(activeScope.id);
  }, [activeScope, scopeId]);

  const { businessVertices, businessEdges } = useMemo(() => semanticGraph(vertices, edges), [vertices, edges]);
  const vertexById = useMemo(
    () => new Map(businessVertices.map((vertex) => [vertex.id, vertex])),
    [businessVertices],
  );
  const displayedGraph = useMemo(() => {
    if (!activeScope) return { vertices: [] as GraphVertex[], edges: [] as GraphEdge[] };
    if (docType === 'fault_history') {
      return historyViewGraph(activeScope, businessVertices, businessEdges, historyView);
    }
    return {
      vertices: businessVertices.filter((vertex) => activeScope.nodeIds.has(vertex.id)),
      edges: businessEdges.filter((edge) => (
        activeScope.nodeIds.has(edgeSource(edge)) && activeScope.nodeIds.has(edgeTarget(edge))
      )),
    };
  }, [activeScope, businessEdges, businessVertices, docType, historyView]);
  const displayedVertices = displayedGraph.vertices;
  const displayedEdges = displayedGraph.edges;

  const selectedDocument = docs.find((doc) => doc.doc_id === docId);
  const vertexNameById = useMemo(
    () => new Map(businessVertices.map((vertex) => [vertex.id, nodeName(vertex)])),
    [businessVertices],
  );
  const presentEntityTypes = useMemo(() => new Set(displayedVertices.map(entityType).filter(Boolean)), [displayedVertices]);
  const presentRelations = useMemo(() => [...new Set(displayedEdges.map(relationName))], [displayedEdges]);

  const searchOptions = useMemo(() => {
    const query = searchText.trim().toLocaleLowerCase('zh-CN');
    const pool = query ? businessVertices : displayedVertices;
    return pool
      .filter((vertex) => !query || nodeName(vertex).toLocaleLowerCase('zh-CN').includes(query))
      .sort((left, right) => nodeName(left).localeCompare(nodeName(right), 'zh-CN', { numeric: true }))
      .slice(0, 80)
      .map((vertex) => ({
        value: vertex.id,
        label: `${nodeName(vertex)} · ${(ENTITY_TYPE_MAP[entityType(vertex)] || ENTITY_TYPE_MAP.default).name}`,
      }));
  }, [businessVertices, displayedVertices, searchText]);

  const searchHighlight = useMemo(() => {
    if (selectedNodeId && activeScope?.nodeIds.has(selectedNodeId)) {
      return { highlightIds: new Set([selectedNodeId]), dimIds: new Set<string>() };
    }
    const query = searchText.trim().toLocaleLowerCase('zh-CN');
    if (!query) return undefined;
    const highlightIds = new Set(displayedVertices
      .filter((vertex) => nodeName(vertex).toLocaleLowerCase('zh-CN').includes(query))
      .map((vertex) => vertex.id));
    if (!highlightIds.size) return undefined;
    return {
      highlightIds,
      dimIds: new Set(displayedVertices.filter((vertex) => !highlightIds.has(vertex.id)).map((vertex) => vertex.id)),
    };
  }, [activeScope, displayedVertices, searchText, selectedNodeId]);

  const focusNode = useCallback((nodeId: string) => {
    const containingScopes = scopes
      .filter((scope) => scope.nodeIds.has(nodeId))
      .sort((left, right) => left.nodeIds.size - right.nodeIds.size);
    const targetScope = activeScope?.nodeIds.has(nodeId) ? activeScope : containingScopes[0];
    if (targetScope && targetScope.id !== activeScope?.id) setScopeId(targetScope.id);
    let focusId = nodeId;
    if (docType === 'fault_history') {
      const targetType = entityType(vertexById.get(nodeId) || { id: nodeId });
      if (targetType === 'FAULTHISTORYSTEP') {
        setHistoryView('steps');
      } else if (['HMCCODE', 'FMECARECORD', 'MFLRECORD', 'FTADIAGRAMREF', 'RBDDIAGRAMREF'].includes(targetType)) {
        setHistoryView('references');
      } else {
        const contentEdge = businessEdges.find((edge) => (
          edgeTarget(edge) === nodeId && rawRelation(edge) === 'has_step_content'
        ));
        if (contentEdge) {
          setHistoryView('steps');
          focusId = edgeSource(contentEdge);
        } else {
          setHistoryView('overview');
        }
      }
    }
    setSelectedNodeId(focusId);
    setSearchText('');
    setFocusSignal((value) => value + 1);
  }, [activeScope, businessEdges, docType, scopes, vertexById]);

  const resetView = useCallback(() => {
    setSearchText('');
    setSelectedNodeId(undefined);
    setDetailNode(null);
    setDetailEdge(null);
    setFitViewSignal((value) => value + 1);
  }, []);

  const nodeDetailRows = useMemo(() => {
    if (!detailNode) return [];
    const properties = detailNode.properties || {};
    const name = String(properties.name || detailNode.label || detailNode.id);
    const rows = [
      { label: '名称', value: name },
      { label: '类型', value: (ENTITY_TYPE_MAP[detailNode.entityType] || ENTITY_TYPE_MAP.default).name },
    ];
    if (properties.description && String(properties.description) !== name) {
      rows.push({ label: '说明', value: String(properties.description) });
    }
    if (properties.probability != null) rows.push({ label: '概率', value: String(properties.probability) });
    if (selectedDocument?.file_name) rows.push({ label: '来源文档', value: selectedDocument.file_name });
    return rows;
  }, [detailNode, selectedDocument]);

  const edgeDetailRows = useMemo(() => {
    if (!detailEdge) return [];
    const properties = detailEdge.properties || {};
    const source = endpointId(detailEdge.source);
    const target = endpointId(detailEdge.target);
    const rows = [
      { label: '关系', value: relationName(detailEdge) },
      { label: '起点', value: vertexNameById.get(source) || source },
      { label: '终点', value: vertexNameById.get(target) || target },
    ];
    if (properties.description && String(properties.description) !== String(properties.relation_type || detailEdge.label || '')) {
      rows.push({ label: '说明', value: String(properties.description) });
    }
    if (properties.gate_type != null) rows.push({ label: '连接条件', value: String(properties.gate_type) });
    return rows;
  }, [detailEdge, vertexNameById]);

  const relatedItems = useMemo(() => {
    if (!detailNode) return [];
    const currentId = String(detailNode.id || '');
    return displayedEdges.flatMap((edge) => {
      const source = edgeSource(edge);
      const target = edgeTarget(edge);
      if (source !== currentId && target !== currentId) return [];
      const otherId = source === currentId ? target : source;
      const other = vertexById.get(otherId);
      if (!other) return [];
      return [{
        id: otherId, name: nodeName(other), relation: relationName(edge),
        direction: source === currentId ? '向下关联' : '上游关联',
      }];
    }).slice(0, 24);
  }, [detailNode, displayedEdges, vertexById]);

  const historyViewMeta = HISTORY_VIEW_OPTIONS.find((item) => item.value === historyView);
  const workspaceTitle = docType === 'fault_history' && activeScope
    ? `${activeScope.label} · ${historyViewMeta?.label || '案例概览'}`
    : (activeScope?.label || '关系图');
  const workspaceDescription = docType !== 'fault_history'
    ? (activeScope?.description || '选择一个文档后，可查看其中明确提取出的对象和关系。')
    : historyView === 'steps'
      ? '按先后顺序查看排故步骤；步骤正文已收进对应卡片，点击可查看完整内容。'
      : historyView === 'references'
        ? '查看该案例明确引用的 HMC、FMECA、FTA、RBD 和 MFL 资料。'
        : '查看故障现象、发生阶段、涉及位置、根因、方案、验证与预防措施。';

  return (
    <div className="glass-fade-in type-page type-visualize-page">
      <Card
        className="glass type-section-card graph-workspace-card"
        styles={{ body: { padding: 0, height: '100%', minHeight: 0, display: 'flex', flexDirection: 'column' } }}
      >
        <div className="graph-workspace-context">
          <div className="graph-workspace-copy">
            <span className="graph-workspace-document">{selectedDocument?.file_name || '请选择文档'}</span>
            <h2>{workspaceTitle}</h2>
            <p>{workspaceDescription}</p>
          </div>
          <div className="graph-workspace-counts" aria-label="当前显示数量">
            <span><strong>{displayedVertices.length}</strong> 个对象</span>
            <span><strong>{displayedEdges.length}</strong> 条关系</span>
          </div>
        </div>

        <div className={`graph-workspace-toolbar${scopes.length > 1 ? '' : ' graph-workspace-toolbar--no-scope'}${docType === 'fault_history' ? ' graph-workspace-toolbar--history' : ''}`}>
          <Select
            className="graph-document-select"
            placeholder="选择文档"
            value={docId}
            suffixIcon={<DownOutlined />}
            onChange={(value) => setDocId(value || undefined)}
            options={docs.map((doc) => ({ value: doc.doc_id, label: doc.file_name }))}
          />
          {scopes.length > 1 && (
            <Select
              className="graph-scope-select"
              aria-label={RANGE_LABELS[docType] || '查看范围'}
              placeholder={`选择${RANGE_LABELS[docType] || '查看范围'}`}
              value={activeScope?.id}
              showSearch
              optionFilterProp="label"
              onChange={(value) => {
                setScopeId(value);
                setHistoryView('overview');
                setSearchText('');
                setSelectedNodeId(undefined);
                setDetailNode(null);
                setDetailEdge(null);
              }}
              options={scopes.map((scope) => ({ value: scope.id, label: scope.label }))}
            />
          )}
          {docType === 'fault_history' && (
            <Select
              className="graph-history-view-select"
              aria-label="故障历史查看内容"
              value={historyView}
              onChange={(value: HistoryView) => {
                setHistoryView(value);
                setLayout('dagre');
                setSearchText('');
                setSelectedNodeId(undefined);
                setDetailNode(null);
                setDetailEdge(null);
              }}
              options={HISTORY_VIEW_OPTIONS}
            />
          )}
          <Select
            className="graph-node-search"
            aria-label="查找并定位对象"
            placeholder="查找并定位对象"
            showSearch
            allowClear
            filterOption={false}
            value={selectedNodeId}
            searchValue={searchText}
            suffixIcon={<SearchOutlined />}
            notFoundContent={searchText ? '没有找到匹配对象' : '输入名称或代码'}
            onSearch={(value) => {
              setSearchText(value);
              if (value) setSelectedNodeId(undefined);
            }}
            onChange={(value) => {
              if (value) focusNode(value);
              else {
                setSelectedNodeId(undefined);
                setSearchText('');
              }
            }}
            options={searchOptions}
          />
          <Select
            className="graph-layout-select"
            aria-label="关系图排列方式"
            value={layout}
            onChange={setLayout}
            options={layoutOptions}
          />
          <div className="graph-toolbar-actions">
            <GlassButton variant="text" size="small" icon={<ReloadOutlined />} loading={loading} onClick={() => loadGraph(docId)}>
              刷新
            </GlassButton>
            <GlassButton variant="text" size="small" icon={<UndoOutlined />} disabled={!displayedVertices.length} onClick={resetView}>
              重置视图
            </GlassButton>
            <GlassButton variant="text" size="small" icon={<AppstoreOutlined />} onClick={() => setLegendOpen((open) => !open)}>
              {legendOpen ? '收起图例' : '查看图例'}
            </GlassButton>
          </div>
        </div>

        {legendOpen && (
          <div className="graph-workspace-legend">
            <div>
              <span className="graph-legend-title">对象</span>
              <Space size={[6, 6]} wrap>
                {Object.entries(ENTITY_TYPE_MAP)
                  .filter(([key]) => key !== 'default' && presentEntityTypes.has(key))
                  .map(([key, item]) => (
                    <Tag key={key} color={item.color}>{item.icon} {item.name}</Tag>
                  ))}
              </Space>
            </div>
            {!!presentRelations.length && (
              <div>
                <span className="graph-legend-title">连线</span>
                <Space size={[12, 6]} wrap>
                  {presentRelations.map((relation) => (
                    <span className="graph-relation-legend" key={relation}><i />{relation}</span>
                  ))}
                </Space>
              </div>
            )}
          </div>
        )}

        <div className="graph-workspace-canvas">
          {!displayedVertices.length ? (
            <Empty description={docId ? '该文档还没有可显示的明确关系，请先完成文档解析' : '请选择文档'} />
          ) : (
            <>
              <GraphCanvas
                vertices={displayedVertices}
                edges={displayedEdges}
                layout={layout}
                diagramMode={diagramMode}
                rankDirection={docType === 'rbd' || (docType === 'fault_history' && historyView === 'steps') ? 'LR' : 'TB'}
                fitViewSignal={fitViewSignal}
                focusNodeId={selectedNodeId}
                focusSignal={focusSignal}
                highlightStates={searchHighlight}
                onNodeClick={(node) => {
                  setSelectedNodeId(node.id);
                  setSearchText('');
                  setDetailNode(node);
                  setDetailEdge(null);
                }}
                onEdgeClick={(edge) => { setDetailEdge(edge); setDetailNode(null); }}
                onCanvasClick={() => { setDetailNode(null); setDetailEdge(null); }}
              />
              <div className="graph-canvas-help">滚轮缩放 · 拖动画布 · 点击对象查看详情</div>
            </>
          )}
        </div>
      </Card>

      <Drawer
        title={detailEdge ? '关系详情' : (detailNode ? String(detailNode.properties?.name || detailNode.label || '对象详情') : '对象详情')}
        open={!!detailNode || !!detailEdge}
        onClose={() => { setDetailNode(null); setDetailEdge(null); }}
        width={440}
      >
        {detailNode && (
          <>
            <Descriptions column={1} size="small" bordered>
              {nodeDetailRows.map((row) => (
                <Descriptions.Item key={row.label} label={row.label}>{row.value}</Descriptions.Item>
              ))}
            </Descriptions>
            {!!relatedItems.length && (
              <div className="graph-related-section">
                <h3>与它直接相关</h3>
                <div className="graph-related-list">
                  {relatedItems.map((item) => (
                    <button key={`${item.id}:${item.relation}`} type="button" onClick={() => focusNode(item.id)}>
                      <span>{item.name}</span>
                      <small>{item.relation} · {item.direction}</small>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
        {detailEdge && (
          <Descriptions column={1} size="small" bordered>
            {edgeDetailRows.map((row) => (
              <Descriptions.Item key={row.label} label={row.label}>{row.value}</Descriptions.Item>
            ))}
          </Descriptions>
        )}
      </Drawer>
    </div>
  );
}
