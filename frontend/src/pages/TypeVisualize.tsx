import { useEffect, useState, useCallback, useMemo } from 'react';
import {
  Card, Select, Space, Tag, Typography, Input, Empty,
  Drawer, Descriptions, Divider, message,
} from 'antd';
import {
  ReloadOutlined, DownOutlined, SearchOutlined,
} from '@ant-design/icons';
import { useParams } from 'react-router-dom';
import { fetchIcpDocuments, fetchIcpGraphData } from '@/api/icp';
import GraphCanvas, { ENTITY_TYPE_MAP } from '@/components/GraphCanvas';
import type { GraphVertex, GraphEdge } from '@/types';
import { GlassButton } from '@/components/GlassUI';
import { DOC_STATUS_LABEL, DOC_STATUS_COLOR } from '@/utils';

// 节点实体类型映射（GraphCanvas 共用色板）
export default function TypeVisualize() {
  // docType 由类型框架路由提供（/type/:docType/visualize，仅 fta/rbd 图类）
  const { docType = '' } = useParams<{ docType: string }>();
  const [docs, setDocs] = useState<any[]>([]);
  const [docId, setDocId] = useState<string>();
  const [vertices, setVertices] = useState<GraphVertex[]>([]);
  const [edges, setEdges] = useState<GraphEdge[]>([]);
  const [stats, setStats] = useState({ vertices: 0, edges: 0 });
  const [layout, setLayout] = useState('force');
  const [searchText, setSearchText] = useState('');
  const [detailNode, setDetailNode] = useState<any>(null);
  const [detailEdge, setDetailEdge] = useState<any>(null);

  useEffect(() => {
    if (docType) {
      fetchIcpDocuments(docType).then(setDocs).catch(() => setDocs([]));
    }
  }, [docType]);

  // 文档名映射：doc_id -> file_name（详情展示用，不暴露裸 ID）
  const docNameMap = useMemo(() => {
    const m = new Map<string, string>();
    docs.forEach((d) => { if (d?.doc_id && d?.file_name) m.set(d.doc_id, d.file_name); });
    return m;
  }, [docs]);

  // 实体名映射：id -> name（关系详情展示用）
  const vertexNameById = useMemo(() => {
    const m = new Map<string, string>();
    vertices.forEach((v) => m.set(v.id, String(v.properties?.name || v.label || v.id)));
    return m;
  }, [vertices]);

  const loadGraph = useCallback(async (did?: string) => {
    if (!docType || !did) { setVertices([]); setEdges([]); setStats({ vertices: 0, edges: 0 }); return; }
    try {
      const res = await fetchIcpGraphData(docType, did);
      if (res.error) {
        message.error(res.error);
        setVertices([]);
        setEdges([]);
        setStats({ vertices: 0, edges: 0 });
        return;
      }
      const v = (res.vertices || []) as GraphVertex[];
      const e = (res.edges || []) as GraphEdge[];
      setVertices(v);
      setEdges(e);
      setStats({ vertices: v.length, edges: e.length });
    } catch (err) {
      message.error(`加载图谱失败: ${(err as Error).message}`);
    }
  }, [docType]);

  useEffect(() => {
    if (docId) loadGraph(docId);
  }, [docId, loadGraph]);

  // 节点搜索过滤
  const filteredVertices = useMemo(() => {
    const q = searchText.trim().toLowerCase();
    if (!q) return vertices;
    return vertices.filter((v) => {
      const name = String(v.properties?.name || v.label || v.id || '');
      const labelNoIcon = name.replace(/^[^\s]+\s/, '');
      return name.toLowerCase().includes(q) || labelNoIcon.toLowerCase().includes(q) || String(v.id).toLowerCase().includes(q);
    });
  }, [vertices, searchText]);

  // 节点详情字段
  const nodeDetailRows = useMemo(() => {
    if (!detailNode) return [];
    const p = detailNode.properties || {};
    return [
      { label: '名称', value: String(p.name || detailNode.label || detailNode.id) },
      { label: '类型', value: ENTITY_TYPE_MAP[detailNode.entityType]?.name || detailNode.entityType || '-' },
      { label: '描述', value: String(p.description || '-') },
      { label: '概率', value: p.probability != null ? String(p.probability) : '-' },
    ];
  }, [detailNode]);

  // 关系详情字段（两端实体显示名称）
  const edgeDetailRows = useMemo(() => {
    if (!detailEdge) return [];
    const p = detailEdge.properties || {};
    const sName = vertexNameById.get(detailEdge.source) || detailEdge.source;
    const tName = vertexNameById.get(detailEdge.target) || detailEdge.target;
    return [
      { label: '关系类型', value: String(p.relation_type || detailEdge.label || p.description || '-') },
      { label: '描述', value: String(p.description || '-') },
      { label: '起点实体', value: sName },
      { label: '终点实体', value: tName },
      { label: '逻辑门', value: p.gate_type != null ? String(p.gate_type) : '-' },
    ];
  }, [detailEdge, vertexNameById]);

  const selectedDoc = docs.find((d) => d.doc_id === docId);

  return (
    <div className="glass-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 12, height: 'calc(100vh - 88px)', overflow: 'hidden' }}>
      <Card className="glass glass-card-bordered" style={{ flexShrink: 0 }} styles={{ body: { padding: 12 } }}>
        <Space wrap>
          <Select
            style={{ width: 260 }}
            placeholder="选择已生成图谱的文档"
            value={docId}
            allowClear
            suffixIcon={<DownOutlined />}
            onChange={(v) => setDocId(v || undefined)}
            options={docs.map((d) => ({
              value: d.doc_id,
              label: `${d.file_name}（${DOC_STATUS_LABEL[d.status] || d.status}）`,
            }))}
          />
          <Tag>顶点 {stats.vertices}</Tag>
          <Tag>边 {stats.edges}</Tag>
          {selectedDoc && <Tag color={DOC_STATUS_COLOR[selectedDoc.status]}>{DOC_STATUS_LABEL[selectedDoc.status] || selectedDoc.status}</Tag>}
          <GlassButton variant="text" size="small" icon={<ReloadOutlined />} onClick={() => loadGraph(docId)}>刷新</GlassButton>
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

      <Card
        className="glass glass-card-bordered"
        style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}
        styles={{ body: { padding: 8, flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' } }}
      >
        {vertices.length === 0 ? (
          <Empty
            description={docId ? '该文档暂无图谱数据，请先在「知识图谱生成」中确认生成' : '请选择已生成知识图谱（kg_done）的文档'}
            style={{ marginTop: 80 }}
          />
        ) : (
          <div style={{ flex: 1, minHeight: 0, width: '100%', position: 'relative' }}>
            <GraphCanvas
              vertices={filteredVertices}
              edges={edges}
              layout={layout}
              onNodeClick={(node) => { setDetailNode(node); setDetailEdge(null); }}
              onEdgeClick={(edge) => { setDetailEdge(edge); setDetailNode(null); }}
              onCanvasClick={() => { setDetailNode(null); setDetailEdge(null); }}
            />
          </div>
        )}
      </Card>

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
