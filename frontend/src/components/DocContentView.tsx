import { useEffect, useState, useCallback } from 'react';
import {
  Card, Tabs, Tag, Spin, Typography, Space, Empty, Alert,
  Button, Table, Input, Select,
} from 'antd';
import {
  ReloadOutlined, FileTextOutlined, DatabaseOutlined,
  ThunderboltOutlined, BranchesOutlined, ApartmentOutlined,
} from '@ant-design/icons';
import {
  fetchOriginal, fetchVectors, fetchDomainRows, fetchFaultTrees, fetchIcpGraphData,
} from '@/api/icp';
import { GlassButton } from '@/components/GlassUI';
import GraphCanvas from '@/components/GraphCanvas';
import type {
  ICPOriginalBlock, ICPVectorResponse, ICPRowsResponse, ICPFaultTree,
  GraphVertex, GraphEdge,
} from '@/types';

const STRUCTURED_TYPES = new Set(['hmc', 'fmeca', 'mfl', 'function_locator', 'fault_history']);
const IMAGE_TYPES = new Set(['fta', 'rbd']);

interface Props {
  docType: string;
  docId: string;
  file_name?: string;
}

/** 七类文档内容视图：原文 / 向量 / 结构化数据 / 图结构。 */
export default function DocContentView({ docType, docId, file_name }: Props) {
  const [origPage, setOrigPage] = useState(1);
  const [origData, setOrigData] = useState<{ file_name: string; total: number; blocks: ICPOriginalBlock[] } | null>(null);
  const [vecPage, setVecPage] = useState(1);
  const [vecData, setVecData] = useState<ICPVectorResponse | null>(null);
  const [rowsPage, setRowsPage] = useState(1);
  const [rowsData, setRowsData] = useState<ICPRowsResponse | null>(null);
  const [rowFilters, setRowFilters] = useState<{
    keyword?: string;
    severity?: string;
    api_reportable?: string;
    platform_test?: string;
  }>({});
  const [faultTrees, setFaultTrees] = useState<ICPFaultTree[] | null>(null);
  const [graphData, setGraphData] = useState<{ vertices: GraphVertex[]; edges: GraphEdge[]; error?: string } | null>(null);

  const [activeTab, setActiveTab] = useState('original');
  const [vecLoading, setVecLoading] = useState(false);
  const [rowsLoading, setRowsLoading] = useState(false);

  const loadOriginal = useCallback(async (page = 1) => {
    try {
      const res = await fetchOriginal(docType, docId, { page, size: 50 });
      setOrigPage(res.page);
      setOrigData(res);
    } catch {
      setOrigData(null);
    }
  }, [docType, docId]);

  const loadVectors = useCallback(async (page = 1) => {
    setVecLoading(true);
    try {
      setVecData(await fetchVectors(docType, docId, { page, size: 20 }));
    } catch {
      setVecData(null);
    } finally {
      setVecLoading(false);
    }
  }, [docType, docId]);

  const loadRows = useCallback(async (page = 1, filters = rowFilters) => {
    setRowsLoading(true);
    try {
      setRowsData(await fetchDomainRows(docType, docId, { page, size: 20, ...filters }));
      setRowsPage(page);
    } catch {
      setRowsData(null);
    } finally {
      setRowsLoading(false);
    }
  }, [docType, docId, rowFilters]);

  const loadFaultTrees = useCallback(async () => {
    try {
      setFaultTrees(await fetchFaultTrees(docType, docId));
    } catch {
      setFaultTrees(null);
    }
  }, [docType, docId]);

  const loadGraph = useCallback(async () => {
    try {
      const res = await fetchIcpGraphData(docType, docId);
      setGraphData({ vertices: res.vertices as GraphVertex[], edges: res.edges as GraphEdge[], error: res.error });
    } catch {
      setGraphData({ vertices: [], edges: [] });
    }
  }, [docType, docId]);

  useEffect(() => { loadOriginal(1); }, [docType, docId, loadOriginal]);

  useEffect(() => {
    if (activeTab === 'vectors') loadVectors(1);
    if (activeTab === 'rows') loadRows(1);
    if (activeTab === 'triplets') loadFaultTrees();
    if (activeTab === 'graph') loadGraph();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab]);

  const isImage = IMAGE_TYPES.has(docType);
  const isStructured = STRUCTURED_TYPES.has(docType);

  const renderOriginal = () => {
    if (!origData) return <Spin style={{ display: 'block', margin: '40px auto' }} />;
    if (origData.blocks.length === 0) return <Empty description="无原文内容" />;
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {origData.blocks.map((block, index) => (
          <div key={index} className="glass-dense" style={{ padding: '10px 14px', borderRadius: 10, border: '1px solid rgba(0,0,0,0.1)' }}>
            {block.title && (
              <div style={{
                fontWeight: 600,
                fontSize: block.level === 1 ? 16 : 14,
                color: '#262626',
                paddingLeft: block.level > 1 ? (block.level - 1) * 16 : 0,
                marginBottom: block.text ? 6 : 0,
              }}>
                {block.title}
              </div>
            )}
            {block.text && <div style={{ whiteSpace: 'pre-wrap', fontSize: 13, color: '#595959' }}>{block.text}</div>}
            {block.table && block.table.length > 0 && (
              <div style={{ overflowX: 'auto', marginTop: 8 }}>
                <table style={{ borderCollapse: 'collapse', fontSize: 12, width: '100%' }}>
                  <tbody>
                    {block.table.map((row, rowIndex) => (
                      <tr key={rowIndex}>
                        {row.map((cell, cellIndex) => (
                          <td key={cellIndex} style={{
                            border: '1px solid #e0e0e0', padding: '4px 8px',
                            background: rowIndex === 0 ? 'rgba(0,0,0,0.04)' : 'transparent',
                            whiteSpace: 'nowrap',
                          }}>
                            {cell}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        ))}
        {origData.total > 50 && (
          <Space style={{ justifyContent: 'center' }}>
            <Button size="small" disabled={origPage <= 1} onClick={() => loadOriginal(origPage - 1)}>上一页</Button>
            <Typography.Text type="secondary">第 {origPage} / {Math.ceil(origData.total / 50)} 页</Typography.Text>
            <Button size="small" disabled={origPage * 50 >= origData.total} onClick={() => loadOriginal(origPage + 1)}>下一页</Button>
          </Space>
        )}
      </div>
    );
  };

  const renderVectors = () => {
    if (vecLoading && !vecData) return <Spin style={{ display: 'block', margin: '40px auto' }} />;
    if (!vecData || vecData.vectors.length === 0) return <Empty description="暂无向量化内容" />;
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {vecData.vectors.map((vector: any, index: number) => (
          <div key={index} className="glass-dense" style={{ padding: '10px 14px', borderRadius: 10, border: '1px solid rgba(0,0,0,0.1)' }}>
            <Space size={6} wrap style={{ marginBottom: 6 }}>
              <Tag color="blue">{vector.content_type || 'text'}</Tag>
              {vector.title ? <Typography.Text strong style={{ fontSize: 13 }}>{vector.title}</Typography.Text> : null}
              {vector.business_id_kwd && <Tag>{vector.business_id_kwd}</Tag>}
              {vector.section && <Tag color="geekblue">{vector.section}</Tag>}
            </Space>
            <div style={{ fontSize: 13, color: '#595959', whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
              {(vector.content || '').slice(0, 600)}
              {(vector.content || '').length > 600 ? '…' : ''}
            </div>
          </div>
        ))}
        <Space style={{ justifyContent: 'center' }}>
          <Button size="small" disabled={vecPage <= 1} onClick={() => { const page = vecPage - 1; setVecPage(page); loadVectors(page); }}>上一页</Button>
          <Typography.Text type="secondary">第 {vecPage} / {Math.max(1, Math.ceil((vecData.total || 0) / 20))} 页（共 {vecData.total} 条）</Typography.Text>
          <Button size="small" disabled={vecPage * 20 >= (vecData.total || 0)} onClick={() => { const page = vecPage + 1; setVecPage(page); loadVectors(page); }}>下一页</Button>
        </Space>
      </div>
    );
  };

  const renderRows = () => {
    if (rowsLoading && !rowsData) return <Spin style={{ display: 'block', margin: '40px auto' }} />;
    if (!rowsData || rowsData.rows.length === 0) return <Empty description="暂无结构化记录，请先完成解析" />;
    const columns = Object.keys(rowsData.rows[0]);
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        <Space wrap>
          <Input.Search
            allowClear
            placeholder="关键词检索全部字段"
            style={{ width: 260 }}
            onSearch={(keyword) => {
              const filters = { ...rowFilters, keyword: keyword || undefined };
              setRowFilters(filters);
              loadRows(1, filters);
            }}
          />
          {docType === 'fmeca' && (
            <Select
              allowClear
              placeholder="严酷度"
              style={{ width: 130 }}
              options={['I', 'II', 'III', 'IV', 'A', 'B', 'C', 'D'].map((value) => ({ value, label: value }))}
              onChange={(value) => {
                const filters = { ...rowFilters, severity: value || undefined };
                setRowFilters(filters);
                loadRows(1, filters);
              }}
            />
          )}
          {docType === 'mfl' && (
            <>
              <Select
                allowClear placeholder="API可上报" style={{ width: 130 }}
                options={[{ value: 'true', label: '是' }, { value: 'false', label: '否' }]}
                onChange={(value) => {
                  const filters = { ...rowFilters, api_reportable: value || undefined };
                  setRowFilters(filters);
                  loadRows(1, filters);
                }}
              />
              <Select
                allowClear placeholder="平台实验" style={{ width: 130 }}
                options={[{ value: 'true', label: '是' }, { value: 'false', label: '否' }]}
                onChange={(value) => {
                  const filters = { ...rowFilters, platform_test: value || undefined };
                  setRowFilters(filters);
                  loadRows(1, filters);
                }}
              />
            </>
          )}
        </Space>
        <Table<Record<string, unknown>>
          size="small"
          rowKey={(record) => String(record.id || record.function_id || record.case_id)}
          loading={rowsLoading}
          dataSource={rowsData.rows}
          columns={columns.map((column) => ({
            title: column,
            dataIndex: column,
            ellipsis: true,
            render: (value: unknown) => {
              if (value === null || value === undefined || value === '') return '-';
              if (typeof value === 'object') return JSON.stringify(value, null, 0);
              return String(value);
            },
          }))}
          pagination={false}
          scroll={{ x: 'max-content' }}
        />
        <Space style={{ justifyContent: 'center' }}>
          <Button size="small" disabled={rowsPage <= 1} onClick={() => loadRows(rowsPage - 1)}>上一页</Button>
          <Typography.Text type="secondary">
            第 {rowsPage} / {Math.max(1, Math.ceil((rowsData.total || 0) / 20))} 页（共 {rowsData.total} 条）
          </Typography.Text>
          <Button size="small" disabled={rowsPage * 20 >= (rowsData.total || 0)} onClick={() => loadRows(rowsPage + 1)}>下一页</Button>
        </Space>
      </div>
    );
  };

  const renderTriplets = () => {
    if (!faultTrees) return <Spin style={{ display: 'block', margin: '40px auto' }} />;
    if (faultTrees.length === 0) return <Empty description="暂无结构化图数据，请先解析" />;
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        {faultTrees.map((tree, treeIndex) => {
          const graph = tree.graph || {};
          const vertices = graph.vertices || [];
          const edges = graph.edges || [];
          return (
            <Card key={tree.fault_tree_id || treeIndex} size="small" title={tree.image_title || `图 ${treeIndex + 1}`} className="glass">
              <Space style={{ marginBottom: 8 }}>
                <Tag>根/起点: {graph.root_fault_id || '-'}</Tag>
                <Tag color="purple">节点 {vertices.length}</Tag>
                <Tag color="orange">边 {edges.length}</Tag>
              </Space>
              {vertices.length > 0 && (
                <>
                  <Typography.Text strong style={{ fontSize: 13 }}>节点</Typography.Text>
                  <Table<Record<string, unknown>>
                    size="small"
                    rowKey="id"
                    style={{ margin: '8px 0 16px' }}
                    dataSource={vertices as unknown as Record<string, unknown>[]}
                    pagination={false}
                    columns={[
                      { title: 'ID', dataIndex: 'id', width: 200, ellipsis: true },
                      { title: '类型', dataIndex: 'label', width: 130 },
                      { title: '名称', render: (_, record) => (record as any).properties?.name || '-' },
                      { title: '描述', render: (_, record) => (record as any).properties?.description || '-' },
                    ]}
                  />
                </>
              )}
              {edges.length > 0 && (
                <>
                  <Typography.Text strong style={{ fontSize: 13 }}>关系</Typography.Text>
                  <Table<Record<string, unknown>>
                    size="small"
                    rowKey="id"
                    style={{ margin: '8px 0' }}
                    dataSource={edges as unknown as Record<string, unknown>[]}
                    pagination={false}
                    columns={[
                      { title: '源', dataIndex: 'source', width: 200, ellipsis: true },
                      { title: '目标', dataIndex: 'target', width: 200, ellipsis: true },
                      { title: '关系', dataIndex: 'label', width: 130 },
                      {
                        title: docType === 'fta' ? '逻辑门' : '连接属性',
                        render: (_, record) => {
                          const props = (record as any).properties || {};
                          return props.gate_type || props.connection_type || props.branch_id || '-';
                        },
                      },
                    ]}
                  />
                </>
              )}
            </Card>
          );
        })}
      </div>
    );
  };

  const renderGraph = () => {
    if (!graphData) return <Spin style={{ display: 'block', margin: '40px auto' }} />;
    if (graphData.error) return <Alert type="warning" showIcon message="图谱数据加载失败" description={graphData.error} />;
    if (graphData.vertices.length === 0) return <Empty description="暂无 HugeGraph 投影；请确认三元组后生成图谱" />;
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        <Space>
          <Typography.Text type="secondary">节点 {graphData.vertices.length} · 边 {graphData.edges.length}</Typography.Text>
          <GlassButton variant="text" size="small" icon={<ReloadOutlined />} onClick={loadGraph}>刷新</GlassButton>
        </Space>
        <Card className="glass" style={{ height: 560 }}>
          <GraphCanvas vertices={graphData.vertices} edges={graphData.edges} layout="force" />
        </Card>
      </div>
    );
  };

  const tabs = [
    { key: 'original', label: '原文证据', icon: <FileTextOutlined />, children: renderOriginal() },
    { key: 'vectors', label: '语义索引', icon: <ThunderboltOutlined />, children: renderVectors() },
  ];
  if (isStructured) {
    tabs.push({ key: 'rows', label: '结构化记录', icon: <DatabaseOutlined />, children: renderRows() });
  }
  if (isImage) {
    tabs.push({ key: 'triplets', label: docType === 'fta' ? 'FTA 节点/边' : 'RBD 节点/边', icon: <BranchesOutlined />, children: renderTriplets() });
    tabs.push({ key: 'graph', label: '图谱投影', icon: <ApartmentOutlined />, children: renderGraph() });
  }

  return (
    <Tabs
      activeKey={activeTab}
      onChange={setActiveTab}
      items={tabs as any}
      tabBarExtraContent={
        file_name ? <Typography.Text type="secondary" style={{ fontSize: 12 }}>{file_name}</Typography.Text> : null
      }
    />
  );
}
