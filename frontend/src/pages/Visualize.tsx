import { useEffect, useState, useCallback, useMemo } from 'react';
import {
  Card, Select, Space, Tag, Typography, Input, Empty,
  Popconfirm, message, Drawer, Descriptions, Checkbox, Divider,
} from 'antd';
import {
  DeleteOutlined, ReloadOutlined, DownOutlined,
  SearchOutlined, ClearOutlined,
} from '@ant-design/icons';
import { fetchDocuments, fetchKBGraphData, fetchCommunityReports, deleteDocGraphData } from '@/api/kb';
import { useParams } from 'react-router-dom';
import GraphCanvas, { ENTITY_TYPE_MAP } from '@/components/GraphCanvas';
import type { GraphVertex, GraphEdge, CommunityReport } from '@/types';
import { formatDateTime } from '@/utils';
import { GlassButton } from '@/components/GlassUI';

// 社区报告色板（黑白主题：统一灰阶明度梯度）
const COMMUNITY_COLORS = [
  '#1a1a1a', '#3d3d3d', '#595959', '#737373', '#8f8f8f', '#a6a6a6', '#bfbfbf', '#4d4d4d',
  '#666666', '#808080', '#999999', '#b3b3b3', '#333333', '#555555', '#777777',
];
const COMMUNITY_BGS = [
  'rgba(0,0,0,0.10)', 'rgba(0,0,0,0.08)', 'rgba(0,0,0,0.06)', 'rgba(0,0,0,0.05)', 'rgba(0,0,0,0.04)',
  'rgba(0,0,0,0.03)', 'rgba(0,0,0,0.02)', 'rgba(0,0,0,0.07)', 'rgba(0,0,0,0.06)', 'rgba(0,0,0,0.05)',
  'rgba(0,0,0,0.04)', 'rgba(0,0,0,0.03)', 'rgba(0,0,0,0.09)', 'rgba(0,0,0,0.08)', 'rgba(0,0,0,0.07)',
];

function containsDocId(docId: unknown, target: string): boolean {
  if (!docId) return false;
  if (Array.isArray(docId)) return docId.includes(target);
  if (typeof docId === 'string') {
    if (docId.includes(',')) return docId.split(',').map((s) => s.trim()).includes(target);
    return docId === target;
  }
  return false;
}

// 把顶点/边属性里的 doc_id（逗号分隔串或数组）转成数组
function splitDocIds(docId: unknown): string[] {
  if (!docId) return [];
  if (Array.isArray(docId)) return docId.filter((x): x is string => typeof x === 'string' && !!x);
  if (typeof docId === 'string') {
    return docId.split(',').map((s) => s.trim()).filter(Boolean);
  }
  return [];
}

export default function Visualize() {
  // kbId 由知识库详情路由提供（/kb/:kbId/visualize）
  const { kbId } = useParams<{ kbId: string }>();
  const [docId, setDocId] = useState<string>();
  const [docs, setDocs] = useState<any[]>([]);
  const [vertices, setVertices] = useState<GraphVertex[]>([]);
  const [edges, setEdges] = useState<GraphEdge[]>([]);
  const [reports, setReports] = useState<CommunityReport[]>([]);
  const [stats, setStats] = useState({ vertices: 0, edges: 0 });
  const [layout, setLayout] = useState('force');
  const [searchText, setSearchText] = useState('');
  const [reportSearch, setReportSearch] = useState('');
  const [detailNode, setDetailNode] = useState<any>(null);
  const [detailEdge, setDetailEdge] = useState<any>(null);
  const [activeCommunities, setActiveCommunities] = useState<Set<number>>(new Set());
  const [expandedReports, setExpandedReports] = useState<Set<number>>(new Set());

  useEffect(() => {
    if (kbId) {
      fetchDocuments(kbId).then(setDocs).catch(() => setDocs([]));
    }
  }, [kbId]);

  // 文档名映射：doc_id -> file_name（详情/社区报告展示用，不暴露裸 ID）
  const docNameMap = useMemo(() => {
    const m = new Map<string, string>();
    docs.forEach((d) => { if (d?.doc_id && d?.file_name) m.set(d.doc_id, d.file_name); });
    return m;
  }, [docs]);

  const docIdsToNames = useCallback((docId: unknown): string[] => {
    return splitDocIds(docId).map((id) => docNameMap.get(id) || id);
  }, [docNameMap]);

  // 实体名映射：id -> name（关系详情展示用）
  const vertexNameById = useMemo(() => {
    const m = new Map<string, string>();
    vertices.forEach((v) => m.set(v.id, String(v.properties?.name || v.label || v.id)));
    return m;
  }, [vertices]);

  const loadGraph = useCallback(async (kid: string, did?: string) => {
    try {
      const [graph, reportRes] = await Promise.all([
        fetchKBGraphData(kid),
        fetchCommunityReports(kid, did),
      ]);
      let v = graph.vertices || [];
      let e = graph.edges || [];
      if (did) {
        v = v.filter((item) => containsDocId(item.properties?.doc_id, did));
        e = e.filter(
          (item) =>
            containsDocId(item.properties?.doc_id, did) ||
            (v.some((x) => x.id === (item.source || item.outV)) && v.some((y) => y.id === (item.target || item.inV))),
        );
      }
      setVertices(v);
      setEdges(e);
      setReports(reportRes.reports || []);
      setStats({ vertices: v.length, edges: e.length });
    } catch (err) {
      message.error(`加载图谱失败: ${(err as Error).message}`);
    }
  }, []);

  useEffect(() => {
    if (kbId) loadGraph(kbId);
  }, [kbId, loadGraph]);

  useEffect(() => {
    if (docId && kbId) loadGraph(kbId, docId);
  }, [docId, kbId, loadGraph]);

  // 社区高亮
  const highlightStates = useMemo(() => {
    if (activeCommunities.size === 0) return undefined;
    const highlightIds = new Set<string>();
    for (const idx of activeCommunities) {
      const report = reports[idx];
      if (!report) continue;
      const nameSet = new Set((report.entities || []).map((e) => e.toLowerCase()));
      for (const v of vertices) {
        const name = String(v.properties?.name || v.label || v.id || '');
        const labelNoIcon = name.replace(/^[^\s]+\s/, '');
        if (nameSet.has(labelNoIcon.toLowerCase()) || nameSet.has(name.toLowerCase()) || nameSet.has(String(v.id).toLowerCase())) {
          highlightIds.add(v.id);
        }
      }
    }
    return { highlightIds, dimIds: new Set(vertices.map((v) => v.id).filter((id) => !highlightIds.has(id))) };
  }, [activeCommunities, reports, vertices]);

  // 社区报告搜索过滤（按标题/摘要/实体关键词）
  const filteredReports = useMemo(() => {
    const q = reportSearch.trim().toLowerCase();
    if (!q) return reports;
    return reports.filter((r) =>
      (r.title || '').toLowerCase().includes(q) ||
      (r.summary || '').toLowerCase().includes(q) ||
      (r.entities || []).some((e) => e.toLowerCase().includes(q)),
    );
  }, [reports, reportSearch]);

  const handleDeleteDocGraph = async () => {
    if (!kbId || !docId) return;
    try {
      const res = await deleteDocGraphData(kbId, docId);
      message.success(`图谱数据已删除（移除实体 ${(res as any).removed_entities?.length || 0} 个）`);
      loadGraph(kbId, docId);
    } catch (err) {
      message.error(`删除失败: ${(err as Error).message}`);
    }
  };

  // 节点详情字段
  const nodeDetailRows = useMemo(() => {
    if (!detailNode) return [];
    const p = detailNode.properties || {};
    const docNames = docIdsToNames(p.doc_id);
    return [
      { label: '名称', value: String(p.name || detailNode.label || detailNode.id) },
      { label: '类型', value: ENTITY_TYPE_MAP[detailNode.entityType]?.name || detailNode.entityType || '-' },
      { label: '描述', value: String(p.description || '-') },
      { label: '来源文档', value: docNames.length ? docNames.join('、') : '-' },
      { label: '创建时间', value: formatDateTime(String(p.created_at || '')) || '-' },
    ];
  }, [detailNode, docIdsToNames]);

  // 关系详情字段（两端实体显示名称，来源文档显示文档名）
  const edgeDetailRows = useMemo(() => {
    if (!detailEdge) return [];
    const p = detailEdge.properties || {};
    const sName = vertexNameById.get(detailEdge.source) || detailEdge.source;
    const tName = vertexNameById.get(detailEdge.target) || detailEdge.target;
    const docNames = docIdsToNames(p.doc_id);
    return [
      { label: '关系类型', value: String(p.relation_type || detailEdge.label || p.description || '-') },
      { label: '描述', value: String(p.description || '-') },
      { label: '起点实体', value: sName },
      { label: '终点实体', value: tName },
      { label: '权重', value: p.weight != null ? String(p.weight) : '-' },
      { label: '来源文档', value: docNames.length ? docNames.join('、') : '-' },
    ];
  }, [detailEdge, vertexNameById, docIdsToNames]);

  return (
    <div className="glass-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 12, height: 'calc(100vh - 88px)', overflow: 'hidden' }}>
      <Card className="glass glass-card-bordered" style={{ flexShrink: 0 }} bodyStyle={{ padding: 12 }}>
        <Space wrap>
          <Select
            style={{ width: 260 }}
            placeholder="全部文档（整库图谱）"
            value={docId}
            allowClear
            suffixIcon={<DownOutlined />}
            onChange={(v) => setDocId(v || undefined)}
            options={docs.map((d) => ({ value: d.doc_id, label: d.file_name }))}
          />
          <Tag>顶点 {stats.vertices}</Tag>
          <Tag>边 {stats.edges}</Tag>
          <Tag>社区报告 {reports.length}</Tag>
          {docId && (
            <Popconfirm
              title="删除该文档的图谱数据？"
              onConfirm={handleDeleteDocGraph}
            >
              <GlassButton variant="danger" size="small" icon={<DeleteOutlined />}>删除图谱数据</GlassButton>
            </Popconfirm>
          )}
          <GlassButton variant="text" size="small" icon={<ReloadOutlined />} onClick={() => kbId && loadGraph(kbId, docId)}>刷新</GlassButton>
        </Space>
        <Divider style={{ margin: '10px 0' }} />
        <Space wrap>
          <Input.Search
            placeholder="搜索实体"
            allowClear
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
            style={{ width: 220 }}
          />
          <Select
            style={{ width: 140 }}
            value={layout}
            onChange={setLayout}
            options={[
              { value: 'force', label: '力导向' },
              { value: 'circular', label: '环形' },
              { value: 'dagre', label: '层次' },
              { value: 'radial', label: '径向' },
            ]}
          />
          <Tag>图例:</Tag>
          {Object.entries(ENTITY_TYPE_MAP).filter(([k]) => k !== 'default').map(([k, v]) => (
            <Tag key={k} color={v.color}>{v.icon} {v.name}</Tag>
          ))}
        </Space>
      </Card>

      <div style={{ display: 'flex', gap: 12, flex: 1, minHeight: 0 }}>
        {/* 图谱画布：占满剩余空间 */}
        <Card
          className="glass glass-card-bordered"
          style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}
          bodyStyle={{ padding: 8, flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}
        >
          {vertices.length === 0 ? (
            <Empty description="暂无图谱数据，请先在知识图谱构建页生成" style={{ marginTop: 80 }} />
          ) : (
            <div style={{ flex: 1, minHeight: 0, width: '100%', position: 'relative' }}>
              <GraphCanvas
                vertices={vertices}
                edges={edges}
                layout={layout}
                highlightStates={highlightStates}
                onNodeClick={(node) => { setDetailNode(node); setDetailEdge(null); }}
                onEdgeClick={(edge) => { setDetailEdge(edge); setDetailNode(null); }}
                onCanvasClick={() => { setDetailNode(null); setDetailEdge(null); }}
              />
            </div>
          )}
        </Card>

        {/* 社区报告：右侧独立滚动 + 可搜索 */}
        <Card
          className="glass glass-card-bordered"
          title={`社区报告 (${filteredReports.length})`}
          style={{ width: 400, flexShrink: 0, minHeight: 0, display: 'flex', flexDirection: 'column' }}
          bodyStyle={{ padding: 12, flex: 1, minHeight: 0, overflow: 'auto' }}
        >
          <Space style={{ marginBottom: 10 }} wrap>
            <Input
              prefix={<SearchOutlined style={{ color: '#8c8c8c' }} />}
              placeholder="搜索社区报告"
              allowClear
              value={reportSearch}
              onChange={(e) => setReportSearch(e.target.value)}
              style={{ width: 230 }}
            />
            <GlassButton variant="text" size="small" icon={<ClearOutlined />} onClick={() => setActiveCommunities(new Set())}>清除高亮</GlassButton>
          </Space>
          {filteredReports.length === 0 ? (
            <Empty description={reports.length === 0 ? '暂无社区报告' : '没有匹配的社区报告'} />
          ) : (
            filteredReports.map((r) => {
              const idx = reports.findIndex((x) => x.report_id === r.report_id);
              const color = COMMUNITY_COLORS[idx % COMMUNITY_COLORS.length];
              const bg = COMMUNITY_BGS[idx % COMMUNITY_BGS.length];
              const checked = activeCommunities.has(idx);
              const expanded = expandedReports.has(idx);
              const docNames = docIdsToNames(r.doc_ids || r.doc_id);
              const summaryText = r.summary || '';
              const clipped = summaryText.length > 160;
              return (
                <div
                  key={r.report_id}
                  className="glass"
                  style={{
                    marginBottom: 10,
                    padding: 12,
                    borderLeft: `3px solid ${color}`,
                    background: checked ? bg : undefined,
                    cursor: 'pointer',
                    border: '1px solid rgba(0,0,0,0.25)',
                    boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
                  }}
                  onClick={() => {
                    setActiveCommunities((prev) => {
                      const next = new Set(prev);
                      if (next.has(idx)) next.delete(idx);
                      else next.add(idx);
                      return next;
                    });
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography.Text strong style={{ fontSize: 13 }}>
                      {r.title || `社区 ${idx + 1}`}
                    </Typography.Text>
                    <Checkbox checked={checked} onClick={(e) => e.stopPropagation()} />
                  </div>
                  <Space size={4} style={{ marginTop: 4 }}>
                    <Tag color="gold">⭐ {r.rating}</Tag>
                    <Tag>{r.entities?.length || 0} 实体</Tag>
                  </Space>
                  <div style={{ fontSize: 12, color: '#595959', marginTop: 4, lineHeight: 1.6 }}>
                    {clipped && !expanded ? `${summaryText.slice(0, 160)}...` : summaryText}
                  </div>
                  {clipped && (
                    <Typography.Link
                      style={{ fontSize: 12 }}
                      onClick={(e) => {
                        e.stopPropagation();
                        setExpandedReports((prev) => {
                          const next = new Set(prev);
                          if (next.has(idx)) next.delete(idx);
                          else next.add(idx);
                          return next;
                        });
                      }}
                    >
                      {expanded ? '收起' : '展开全文'}
                    </Typography.Link>
                  )}
                  <div style={{ marginTop: 6, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                    {(r.entities || []).slice(0, 8).map((e, i) => (
                      <Tag key={i} style={{ fontSize: 11 }}>{e}</Tag>
                    ))}
                    {(r.entities || []).length > 8 && <Tag>+{(r.entities || []).length - 8}</Tag>}
                  </div>
                  <div style={{ marginTop: 6 }}>
                    {docNames.length ? (
                      docNames.map((n, i) => <Tag key={i} color="blue" style={{ fontSize: 10 }}>{n}</Tag>)
                    ) : (
                      <Tag style={{ fontSize: 10 }}>未知来源</Tag>
                    )}
                  </div>
                </div>
              );
            })
          )}
        </Card>
      </div>

      <Drawer
        title={detailEdge ? '关系详情' : '节点详情'}
        open={!!detailNode || !!detailEdge}
        onClose={() => { setDetailNode(null); setDetailEdge(null); }}
        width={440}
      >
        {detailNode && (
          <Descriptions column={1} size="small" bordered>
            {nodeDetailRows.map((row) => (
              <Descriptions.Item key={row.label} label={row.label}>{row.value}</Descriptions.Item>
            ))}
          </Descriptions>
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
