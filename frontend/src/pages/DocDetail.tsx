import { useEffect, useState, useCallback, useRef } from 'react';
import {
  Card, Tabs, Tag, Spin, Typography, Space, Empty, Alert,
  Modal, Button, Popconfirm, message, Progress, Table, Input, Select,
} from 'antd';
import {
  ArrowLeftOutlined, PlayCircleOutlined, DeleteOutlined,
  ReloadOutlined, ApartmentOutlined, FileTextOutlined,
  DatabaseOutlined, ThunderboltOutlined, BranchesOutlined,
} from '@ant-design/icons';
import { useNavigate, useParams } from 'react-router-dom';
import {
  fetchDocumentType, fetchParseResult, fetchOriginal, fetchVectors,
  fetchRows, fetchFaultTrees, fetchIcpGraphData, generateIcpKg,
  parseIcpDocument, deleteIcpDocument, openTaskStream,
} from '@/api/icp';
import { cancelTask } from '@/api/kb';
import { GlassButton } from '@/components/GlassUI';
import GraphCanvas from '@/components/GraphCanvas';
import type {
  DocTypeInfo, ICPOriginalBlock, ICPVectorResponse, ICPRowsResponse,
  ICPFaultTree, GraphVertex, GraphEdge,
} from '@/types';
import { DOC_STATUS_LABEL, DOC_STATUS_COLOR, formatDateTime } from '@/utils';

const TABLE_TYPES = new Set(['hmc', 'fmeca', 'mfl']);
const IMAGE_TYPES = new Set(['fta', 'rbd']);
const RUNNING_STATUS = new Set(['parsing', 'generating_kg']);

export default function DocDetail() {
  const { docType = '', docId = '' } = useParams<{ docType: string; docId: string }>();
  const navigate = useNavigate();
  const [typeInfo, setTypeInfo] = useState<DocTypeInfo | null>(null);
  const [parseInfo, setParseInfo] = useState<{ status: string; parse_count: number; last_result?: Record<string, unknown> | null } | null>(null);
  const [progress, setProgress] = useState(0);
  const [running, setRunning] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [logModal, setLogModal] = useState(false);
  const sourceRef = useRef<EventSource | null>(null);
  const taskIdRef = useRef('');

  // 各 Tab 数据
  const [origPage, setOrigPage] = useState(1);
  const [origData, setOrigData] = useState<{ file_name: string; total: number; blocks: ICPOriginalBlock[] } | null>(null);
  const [vecPage, setVecPage] = useState(1);
  const [vecData, setVecData] = useState<ICPVectorResponse | null>(null);
  const [rowsPage, setRowsPage] = useState(1);
  const [rowsData, setRowsData] = useState<ICPRowsResponse | null>(null);
  const [rowFilters, setRowFilters] = useState<{ keyword?: string; severity?: string; api_reportable?: string; platform_test?: string }>({});
  const [faultTrees, setFaultTrees] = useState<ICPFaultTree[] | null>(null);
  const [graphData, setGraphData] = useState<{ vertices: GraphVertex[]; edges: GraphEdge[]; error?: string } | null>(null);

  const [activeTab, setActiveTab] = useState('original');
  const [vecLoading, setVecLoading] = useState(false);
  const [rowsLoading, setRowsLoading] = useState(false);

  const loadParseInfo = useCallback(async () => {
    try {
      setParseInfo(await fetchParseResult(docType, docId));
    } catch { /* ignore */ }
  }, [docType, docId]);

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
      setRowsData(await fetchRows(docType, docId, { page, size: 20, ...filters }));
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

  useEffect(() => {
    fetchDocumentType(docType).then(setTypeInfo).catch(() => setTypeInfo(null));
    loadParseInfo();
    loadOriginal(1);
  }, [docType, docId, loadParseInfo, loadOriginal]);

  useEffect(() => {
    if (!activeTab) return;
    if (activeTab === 'vectors') loadVectors(1);
    if (activeTab === 'rows') loadRows(1);
    if (activeTab === 'triplets') loadFaultTrees();
    if (activeTab === 'graph') loadGraph();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab]);

  useEffect(() => () => {
    sourceRef.current?.close();
  }, []);

  const stopRun = () => {
    sourceRef.current?.close();
    sourceRef.current = null;
    setRunning(false);
  };

  const startTask = (taskId: string) => {
    stopRun();
    taskIdRef.current = taskId;
    const source = openTaskStream(taskId);
    sourceRef.current = source;
    const ls: string[] = [];
    setLogs(ls);
    setProgress(0);
    setRunning(true);
    source.onmessage = (ev) => {
      let data: any;
      try { data = JSON.parse(ev.data); } catch { return; }
      if (data.type === 'progress') {
        setProgress(data.progress);
        ls.push(data.log ? `[${formatDateTime()}] ${data.log}` : '');
      } else if (data.type === 'log') {
        ls.push(`[${formatDateTime()}] ${data.log}`);
      } else if (data.type === 'status' && data.status === 'cancelled') {
        ls.push(`[${formatDateTime()}] 任务已取消`);
        stopRun();
        loadParseInfo();
      } else if (data.type === 'final') {
        stopRun();
        loadParseInfo();
        if (data.status === 'failed') message.error('任务失败');
        else message.success('任务完成');
      }
    };
    source.onerror = () => {
      stopRun();
      loadParseInfo();
    };
  };

  const handleParse = async () => {
    try {
      const res = await parseIcpDocument(docType, docId);
      startTask(res.task_id);
    } catch (e) {
      message.error(`解析启动失败: ${(e as Error).message}`);
    }
  };

  const handleGenerateKg = async () => {
    try {
      const res = await generateIcpKg(docType, docId);
      startTask(res.task_id);
    } catch (e) {
      message.error(`图谱生成启动失败: ${(e as Error).message}`);
    }
  };

  const handleDelete = async () => {
    stopRun();
    try {
      await deleteIcpDocument(docType, docId);
      message.success('文档已删除');
      navigate(`/type/${docType}/view`);
    } catch (e) {
      message.error(`删除失败: ${(e as Error).message}`);
    }
  };

  const isImage = IMAGE_TYPES.has(docType);
  const isTable = TABLE_TYPES.has(docType);
  const status = parseInfo?.status || '';
  const canReparse = !RUNNING_STATUS.has(status) && status !== '';
  const canGenerateKg = isImage && status === 'kg_pending';
  const stats = parseInfo?.last_result as Record<string, unknown> | undefined;

  // ===== Tab 内容 =====
  const renderOriginal = () => {
    if (!origData) return <Spin style={{ display: 'block', margin: '40px auto' }} />;
    if (origData.blocks.length === 0) return <Empty description="无原文内容" />;
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {origData.blocks.map((b, i) => (
          <div key={i} className="glass-dense" style={{ padding: '10px 14px', borderRadius: 10, border: '1px solid rgba(0,0,0,0.1)' }}>
            {b.title && (
              <div style={{ fontWeight: 600, fontSize: b.level === 1 ? 16 : 14, color: '#262626', paddingLeft: b.level > 1 ? (b.level - 1) * 16 : 0, marginBottom: b.text ? 6 : 0 }}>
                {b.title}
              </div>
            )}
            {b.text && <div style={{ whiteSpace: 'pre-wrap', fontSize: 13, color: '#595959' }}>{b.text}</div>}
            {b.table && b.table.length > 0 && (
              <div style={{ overflowX: 'auto', marginTop: 8 }}>
                <table style={{ borderCollapse: 'collapse', fontSize: 12, width: '100%' }}>
                  <tbody>
                    {b.table.map((row, ri) => (
                      <tr key={ri}>
                        {row.map((cell, ci) => (
                          <td key={ci} style={{ border: '1px solid #e0e0e0', padding: '4px 8px', background: ri === 0 ? 'rgba(0,0,0,0.04)' : 'transparent', whiteSpace: 'nowrap' }}>{cell}</td>
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
        {vecData.vectors.map((v: any, i: number) => (
          <div key={i} className="glass-dense" style={{ padding: '10px 14px', borderRadius: 10, border: '1px solid rgba(0,0,0,0.1)' }}>
            <Space size={6} wrap style={{ marginBottom: 6 }}>
              <Tag color="blue">{v.content_type || 'text'}</Tag>
              {v.title ? <Typography.Text strong style={{ fontSize: 13 }}>{v.title}</Typography.Text> : null}
              {v.business_id_kwd && <Tag>{v.business_id_kwd}</Tag>}
              {v.section && <Tag color="geekblue">{v.section}</Tag>}
            </Space>
            <div style={{ fontSize: 13, color: '#595959', whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
              {(v.content || '').slice(0, 600)}
              {(v.content || '').length > 600 ? '…' : ''}
            </div>
          </div>
        ))}
        <Space style={{ justifyContent: 'center' }}>
          <Button size="small" disabled={vecPage <= 1} onClick={() => { const p = vecPage - 1; setVecPage(p); loadVectors(p); }}>上一页</Button>
          <Typography.Text type="secondary">第 {vecPage} / {Math.max(1, Math.ceil((vecData.total || 0) / 20))} 页（共 {vecData.total} 条）</Typography.Text>
          <Button size="small" disabled={vecPage * 20 >= (vecData.total || 0)} onClick={() => { const p = vecPage + 1; setVecPage(p); loadVectors(p); }}>下一页</Button>
        </Space>
      </div>
    );
  };

  const renderRows = () => {
    if (rowsLoading && !rowsData) return <Spin style={{ display: 'block', margin: '40px auto' }} />;
    if (!rowsData || rowsData.rows.length === 0) return <Empty description="暂无 PG 行数据" />;
    const cols = Object.keys(rowsData.rows[0]);
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        <Space wrap>
          <Input.Search
            allowClear
            placeholder="关键词检索全列"
            style={{ width: 240 }}
            onSearch={(kw) => { const f = { ...rowFilters, keyword: kw || undefined }; setRowFilters(f); loadRows(1, f); }}
          />
          {docType === 'fmeca' && (
            <>
              <Select
                allowClear placeholder="严重度" style={{ width: 110 }}
                options={[{ value: 'A', label: 'A' }, { value: 'B', label: 'B' }, { value: 'C', label: 'C' }, { value: 'D', label: 'D' }]}
                onChange={(v) => { const f = { ...rowFilters, severity: v || undefined }; setRowFilters(f); loadRows(1, f); }}
              />
              <Select
                allowClear placeholder="API上报" style={{ width: 120 }}
                options={[{ value: 'true', label: '是' }, { value: 'false', label: '否' }]}
                onChange={(v) => { const f = { ...rowFilters, api_reportable: v || undefined }; setRowFilters(f); loadRows(1, f); }}
              />
              <Select
                allowClear placeholder="平台测试" style={{ width: 120 }}
                options={[{ value: 'true', label: '是' }, { value: 'false', label: '否' }]}
                onChange={(v) => { const f = { ...rowFilters, platform_test: v || undefined }; setRowFilters(f); loadRows(1, f); }}
              />
            </>
          )}
        </Space>
        <Table<Record<string, unknown>>
          size="small"
          rowKey="id"
          loading={rowsLoading}
          dataSource={rowsData.rows}
          columns={cols.map((c) => ({
            title: c,
            dataIndex: c,
            ellipsis: true,
            render: (val: unknown) => (val === null || val === undefined || val === '' ? '-' : String(val)),
          }))}
          pagination={false}
          scroll={{ x: 'max-content' }}
        />
        <Space style={{ justifyContent: 'center' }}>
          <Button size="small" disabled={rowsPage <= 1} onClick={() => { const p = rowsPage - 1; setRowsPage(p); loadRows(p); }}>上一页</Button>
          <Typography.Text type="secondary">第 {rowsPage} / {Math.max(1, Math.ceil((rowsData.total || 0) / 20))} 页（共 {rowsData.total} 条）</Typography.Text>
          <Button size="small" disabled={rowsPage * 20 >= (rowsData.total || 0)} onClick={() => { const p = rowsPage + 1; setRowsPage(p); loadRows(p); }}>下一页</Button>
        </Space>
      </div>
    );
  };

  const renderTriplets = () => {
    if (!faultTrees) return <Spin style={{ display: 'block', margin: '40px auto' }} />;
    if (faultTrees.length === 0) return <Empty description="暂无三元组数据，请先解析" />;
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        {faultTrees.map((t, ti) => {
          const g = t.graph || {};
          const vertices = g.vertices || [];
          const edges = g.edges || [];
          return (
            <Card key={t.fault_tree_id || ti} size="small" title={t.image_title || `图 ${ti + 1}`} className="glass">
              <Space style={{ marginBottom: 8 }}>
                <Tag>根节点: {g.root_fault_id || '-'}</Tag>
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
                      { title: '类型', dataIndex: 'label', width: 100 },
                      { title: '名称', render: (_, r) => (r as any).properties?.name || '-' },
                      { title: '描述', render: (_, r) => (r as any).properties?.description || '-' },
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
                      { title: '关系', dataIndex: 'label', width: 100 },
                      { title: '逻辑门', render: (_, r) => (r as any).properties?.gate_type || '-' },
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
    if (graphData.vertices.length === 0) return <Empty description="暂无知识图谱数据" />;
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

  // ===== Tab 组合（按 data_form 差异化） =====
  const tabs = [
    { key: 'original', label: '原文', icon: <FileTextOutlined />, children: renderOriginal() },
    { key: 'vectors', label: '向量化', icon: <ThunderboltOutlined />, children: renderVectors() },
  ];
  if (isTable) {
    tabs.push({ key: 'rows', label: 'PG 结构', icon: <DatabaseOutlined />, children: renderRows() });
  }
  if (isImage) {
    tabs.push({ key: 'triplets', label: '三元组', icon: <BranchesOutlined />, children: renderTriplets() });
    tabs.push({ key: 'graph', label: '图谱', icon: <ApartmentOutlined />, children: renderGraph() });
  }

  return (
    <div className="glass-fade-in" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16, flexWrap: 'wrap', flexShrink: 0 }}>
        <GlassButton variant="text" icon={<ArrowLeftOutlined />} onClick={() => navigate(`/type/${docType}/view`)}>返回内容查看</GlassButton>
        <div style={{ minWidth: 0 }}>
          <Space size={8} wrap>
            <Typography.Title level={5} style={{ margin: 0, color: '#262626' }} ellipsis>
              {origData?.file_name || docId}
            </Typography.Title>
            <Tag color={DOC_STATUS_COLOR[status]}>{DOC_STATUS_LABEL[status] || status}</Tag>
            {parseInfo?.parse_count ? <Tag>解析 {parseInfo.parse_count} 次</Tag> : null}
          </Space>
          <Typography.Text type="secondary" style={{ fontSize: 12 }}>{typeInfo?.name || docType}</Typography.Text>
        </div>
        <div style={{ flex: 1 }} />

        {running && (
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}>
            <Progress percent={progress} size="small" style={{ width: 160 }} />
            <GlassButton variant="text" size="small" onClick={() => setLogModal(true)}>日志</GlassButton>
            <GlassButton variant="text" size="small" onClick={() => { cancelTask(taskIdRef.current).catch(() => { /* ignore */ }); }}>取消</GlassButton>
          </span>
        )}

        {stats && (
          <Space size={4} wrap style={{ maxWidth: 400 }}>
            {Object.entries(stats).filter(([k]) => !['kg_pending', 'message'].includes(k)).map(([k, v]) => (
              <Tag key={k}>{k}: {String(v ?? '')}</Tag>
            ))}
          </Space>
        )}

        {canReparse && (
          <GlassButton variant="primary" size="small" icon={<PlayCircleOutlined />} onClick={handleParse}>重新解析</GlassButton>
        )}
        {canGenerateKg && (
          <GlassButton size="small" icon={<ApartmentOutlined />} onClick={handleGenerateKg}>生成知识图谱</GlassButton>
        )}
        <Popconfirm title="确定删除该文档？" description="将删除全部解析数据、向量与图谱，不可恢复" okText="删除" okButtonProps={{ danger: true }} cancelText="取消" onConfirm={handleDelete}>
          <GlassButton variant="danger" size="small" icon={<DeleteOutlined />}>删除</GlassButton>
        </Popconfirm>
      </div>

      <Card className="glass" style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }} styles={{ body: { flex: 1, minHeight: 0, overflow: 'auto', paddingTop: 8 } }}>
        <Tabs
          activeKey={activeTab}
          onChange={setActiveTab}
          items={tabs as any}
          style={{ height: '100%' }}
        />
      </Card>

      <Modal title="任务日志" open={logModal} onCancel={() => setLogModal(false)} footer={<Button onClick={() => setLogModal(false)}>关闭</Button>} width={640}>
        <Progress percent={progress} size="small" style={{ marginBottom: 12 }} />
        <div style={{ background: 'rgba(0,0,0,0.85)', color: '#d9d9d9', borderRadius: 8, padding: 12, maxHeight: 360, overflow: 'auto', fontFamily: 'monospace', fontSize: 12, whiteSpace: 'pre-wrap' }}>
          {logs.length === 0 ? '暂无日志输出' : logs.join('\n')}
        </div>
      </Modal>
    </div>
  );
}
