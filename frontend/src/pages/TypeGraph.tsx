import { useEffect, useState, useCallback, useRef } from 'react';
import {
  Card, Button, Space, Progress, Typography, message,
  Tag, Alert, Modal, Empty, Table, Popconfirm,
} from 'antd';
import {
  DownOutlined, PlayCircleOutlined, StopOutlined, SyncOutlined, ApartmentOutlined,
} from '@ant-design/icons';
import { useParams } from 'react-router-dom';
import { fetchIcpDocuments, fetchFaultTrees, generateIcpKg, openTaskStream } from '@/api/icp';
import { cancelTask } from '@/api/kb';
import { fetchModels } from '@/api/models';
import { useGlobalStore } from '@/store';
import type { ICPDocument, ICPFaultTree, LLMModel } from '@/types';
import { formatDateTime, DOC_STATUS_LABEL, DOC_STATUS_COLOR } from '@/utils';
import { GlassButton } from '@/components/GlassUI';

export default function TypeGraph() {
  // docType 由类型框架路由提供（/type/:docType/graph，仅 fta/rbd 图类）
  const { docType = '' } = useParams<{ docType: string }>();
  const [docs, setDocs] = useState<ICPDocument[]>([]);
  const [docId, setDocId] = useState<string>();
  const [docOpen, setDocOpen] = useState(false);
  const [models, setModels] = useState<LLMModel[]>([]);
  const [modelId, setModelId] = useState<string>();
  const [modelOpen, setModelOpen] = useState(false);
  const [trees, setTrees] = useState<ICPFaultTree[]>([]);
  const [running, setRunning] = useState(false);
  const [taskId, setTaskId] = useState<string>();
  const [progress, setProgress] = useState(0);
  const [logs, setLogs] = useState<string[]>([]);
  const esRef = useRef<EventSource | null>(null);
  const [armed, setArmed] = useState(false);
  const [modelConfigStatus, setModelConfigStatus] = useState<any>(useGlobalStore.getState().modelConfigStatus);

  useEffect(() => {
    const unsub = useGlobalStore.subscribe((s) => setModelConfigStatus(s.modelConfigStatus));
    return unsub;
  }, []);

  const loadAll = useCallback(async () => {
    const modelRes = await fetchModels().catch(() => ({ models: [] as LLMModel[] }));
    const list = (modelRes.models || []).filter((m) => m.model_type === 'llm' || !m.model_type);
    setModels(list);
    const def = list.find((m) => m.is_default) || list[0];
    if (def) setModelId(def.model_id);
    if (docType) {
      const docList = await fetchIcpDocuments(docType).catch(() => []);
      setDocs(docList);
    }
  }, [docType]);

  useEffect(() => {
    loadAll();
    return () => { if (esRef.current) esRef.current.close(); };
  }, [loadAll]);

  const loadTrees = async (did: string) => {
    try {
      const list = await fetchFaultTrees(docType, did);
      setTrees(list);
    } catch (e) {
      message.error(`加载三元组失败: ${(e as Error).message}`);
      setTrees([]);
    }
  };

  const pickDoc = (did: string) => {
    setDocId(did);
    setDocOpen(false);
    setArmed(false);
    loadTrees(did);
  };

  const handlePrepare = () => setArmed(true);

  const handleGenerate = async () => {
    if (!docType || !docId) { message.warning('请选择文档'); return; }
    const chatOk = modelId || models.some((m) => m.model_type === 'llm' || !m.model_type);
    if (!chatOk) { message.error('未配置 Chat LLM 模型'); return; }
    try {
      const res = await generateIcpKg(docType, docId);
      setRunning(true);
      setTaskId(res.task_id);
      setProgress(0);
      setLogs([]);
      startEventSource(res.task_id);
    } catch (e) {
      message.error(`启动图谱生成失败: ${(e as Error).message}`);
    }
  };

  const startEventSource = (tid: string) => {
    if (esRef.current) esRef.current.close();
    const source = openTaskStream(tid);
    esRef.current = source;
    const cleanup = () => {
      source.close();
      esRef.current = null;
      setRunning(false);
      setArmed(false);
      if (docType) {
        fetchIcpDocuments(docType).then(setDocs).catch(() => setDocs([]));
      }
    };
    source.onmessage = (ev) => {
      let data: any;
      try { data = JSON.parse(ev.data); } catch { return; }
      if (data.type === 'progress') {
        setProgress(data.progress || 0);
        if (data.log) setLogs((prev) => [...prev, `[${formatDateTime()}] ${data.log}`]);
      } else if (data.type === 'log') {
        setLogs((prev) => [...prev, `[${formatDateTime()}] ${data.log}`]);
      } else if (data.type === 'status') {
        if (data.status === 'cancelled') setLogs((prev) => [...prev, `[${formatDateTime()}] ⚠️ 任务已取消`]);
      } else if (data.type === 'final') {
        setLogs((prev) => [...prev, `[${formatDateTime()}] ✅ 任务结束 (${data.status})`]);
        cleanup();
      }
    };
    source.onerror = () => cleanup();
  };

  const handleCancel = async () => {
    if (taskId) await cancelTask(taskId).catch(() => {});
  };

  const selectedDoc = docs.find((d) => d.doc_id === docId);
  const selectedModel = models.find((m) => m.model_id === modelId);

  // 三元组确认预览：每张图一个卡片（节点/边表格）
  const renderTripletPreview = () => {
    if (!docId) return <Typography.Text type="secondary">请选择文档后加载三元组</Typography.Text>;
    if (trees.length === 0) return <Typography.Text type="secondary">该文档暂无三元组数据，请先解析（kg_pending）</Typography.Text>;
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        {trees.map((t, ti) => {
          const g = t.graph || {};
          const vertices = g.vertices || [];
          const edges = g.edges || [];
          return (
            <div
              key={t.fault_tree_id || ti}
              style={{
                marginBottom: 8,
                padding: 12,
                background: 'rgba(255,255,255,0.5)',
                borderRadius: 8,
                border: '1px solid rgba(0,0,0,0.35)',
                boxShadow: '0 2px 6px rgba(0,0,0,0.06)',
              }}
            >
              <Space wrap style={{ marginBottom: 8 }}>
                <Tag color="purple">{t.image_title || `图 ${ti + 1}`}</Tag>
                <Tag>根节点: {g.root_fault_id || '-'}</Tag>
                <Tag color="geekblue">节点 {vertices.length}</Tag>
                <Tag color="orange">边 {edges.length}</Tag>
              </Space>
              {vertices.length > 0 && (
                <Table<Record<string, unknown>>
                  size="small"
                  rowKey="id"
                  style={{ marginBottom: 8 }}
                  dataSource={vertices as unknown as Record<string, unknown>[]}
                  pagination={false}
                  columns={[
                    { title: 'ID', dataIndex: 'id', width: 220, ellipsis: true },
                    { title: '类型', dataIndex: 'label', width: 90 },
                    { title: '名称', render: (_, r) => (r as any).properties?.name || '-' },
                    { title: '描述', render: (_, r) => (r as any).properties?.description || '-' },
                  ]}
                />
              )}
              {edges.length > 0 && (
                <Table<Record<string, unknown>>
                  size="small"
                  rowKey="id"
                  dataSource={edges as unknown as Record<string, unknown>[]}
                  pagination={false}
                  columns={[
                    { title: '源', dataIndex: 'source', width: 220, ellipsis: true },
                    { title: '目标', dataIndex: 'target', width: 220, ellipsis: true },
                    { title: '关系', dataIndex: 'label', width: 90 },
                    { title: '逻辑门', render: (_, r) => (r as any).properties?.gate_type || '-' },
                  ]}
                />
              )}
            </div>
          );
        })}
      </div>
    );
  };

  const canGenerate = selectedDoc?.status === 'kg_pending';

  return (
    <div className="glass-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 12, height: 'calc(100vh - 64px)', overflow: 'hidden' }}>
      <div style={{ display: 'flex', gap: 12, flexWrap: 'nowrap', flexShrink: 0 }}>
        {/* 文档选择：胶囊 + 弹窗 */}
        <Card className="glass glass-card-bordered" size="small" style={{ flex: 1, minWidth: 0 }} title="选择文档">
          <div className="agent-picker" style={{ cursor: 'pointer' }} onClick={() => setDocOpen(true)}>
            <span className="agent-picker-label">选择文档</span>
            <span className="agent-picker-value">
              {selectedDoc ? `${selectedDoc.file_name}（${DOC_STATUS_LABEL[selectedDoc.status] || selectedDoc.status}）` : '点击选择要生成图谱的文档'}
              <DownOutlined style={{ fontSize: 10 }} />
            </span>
          </div>
        </Card>

        {/* 图谱生成配置：紧凑排布 */}
        <Card className="glass glass-card-bordered" size="small" style={{ flex: 1.5, minWidth: 0 }} title="图谱生成配置">
          <Space direction="vertical" style={{ width: '100%' }} size={6}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <Typography.Text style={{ fontSize: 12, color: '#595959', width: 56, flexShrink: 0 }}>生成模型</Typography.Text>
              <div className="agent-picker agent-picker-sm" style={{ flex: 1, cursor: 'pointer' }} onClick={() => setModelOpen(true)}>
                <span className="agent-picker-value" style={{ fontSize: 12 }}>
                  {selectedModel ? `${selectedModel.name}${selectedModel.is_default ? ' (默认)' : ''}` : '点击选择 Chat 模型'}
                </span>
                <DownOutlined style={{ fontSize: 9 }} />
              </div>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <Typography.Text style={{ fontSize: 12, color: '#595959', width: 56, flexShrink: 0 }}>生成方式</Typography.Text>
              <Tag color="purple" style={{ margin: 0 }}>故障树三元组 → 知识图谱</Tag>
            </div>
            {!modelConfigStatus?.chat_configured && (
              <Alert type="warning" showIcon message="未配置 Chat LLM 模型" style={{ padding: '4px 12px', fontSize: 12 }} />
            )}
            {!modelConfigStatus?.embedding_configured && (
              <Alert type="warning" showIcon message="未配置 Embedding 模型（图谱生成可能需要向量化）" style={{ padding: '4px 12px', fontSize: 12 }} />
            )}
            {!docId && <Alert type="info" showIcon message="请先选择待生成图谱（kg_pending）的文档" style={{ padding: '4px 12px', fontSize: 12 }} />}
            {selectedDoc && !canGenerate && selectedDoc.status !== 'running' && (
              <Alert type="info" showIcon message={`当前文档状态「${DOC_STATUS_LABEL[selectedDoc.status] || selectedDoc.status}」，仅 kg_pending 可生成`} style={{ padding: '4px 12px', fontSize: 12 }} />
            )}
          </Space>
        </Card>
      </div>

      <Card
        className="glass glass-card-bordered"
        size="small"
        style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}
        styles={{ body: { flex: 1, overflow: 'auto', minHeight: 0 } }}
        title={`三元组确认预览（${trees.length} 张图）`}
      >
        {renderTripletPreview()}
      </Card>

      <Card
        className="glass glass-card-bordered"
        size="small"
        style={{ flexShrink: 0, boxShadow: '0 -6px 18px rgba(0,0,0,0.15)', backgroundColor: 'rgba(255,255,255,0.92)' }}
        title="生成控制"
      >
        <Space wrap>
          {!running && (
            armed ? (
              <GlassButton variant="primary" icon={<PlayCircleOutlined />} onClick={handleGenerate}>
                AI生成
              </GlassButton>
            ) : (
              <GlassButton
                variant="primary"
                icon={<ApartmentOutlined />}
                disabled={!canGenerate}
                onClick={handlePrepare}
              >
                生成知识图谱
              </GlassButton>
            )
          )}
          {armed && !running && (
            <>
              <GlassButton variant="text" size="small" onClick={() => setArmed(false)}>取消生成</GlassButton>
            </>
          )}
          {running && (
            <GlassButton variant="danger" icon={<StopOutlined />} onClick={handleCancel}>取消任务</GlassButton>
          )}
          <GlassButton variant="text" icon={<SyncOutlined />} onClick={() => loadAll()}>刷新文档</GlassButton>
          {running && (
            <div style={{ width: 200 }}>
              <Progress percent={progress} status="active" />
            </div>
          )}
        </Space>
        {logs.length > 0 && (
          <div
            style={{
              marginTop: 12,
              maxHeight: 300,
              overflow: 'auto',
              background: 'rgba(31,45,78,0.04)',
              borderRadius: 8,
              padding: 12,
              fontFamily: 'monospace',
              fontSize: 12,
              border: '1px solid rgba(0,0,0,0.35)',
            }}
          >
            {logs.map((l, i) => <div key={i}>{l}</div>)}
          </div>
        )}
      </Card>

      {/* 文档选择弹窗 */}
      <Modal
        title="选择文档"
        open={docOpen}
        onCancel={() => setDocOpen(false)}
        width={520}
        footer={[<Button key="c" onClick={() => setDocOpen(false)}>取消</Button>]}
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxHeight: 460, overflow: 'auto' }}>
          {docs.length ? (
            docs.map((d) => (
              <div
                key={d.doc_id}
                className={['agent-picker-option', docId === d.doc_id ? 'selected' : ''].filter(Boolean).join(' ')}
                onClick={() => pickDoc(d.doc_id)}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0 }}>
                  <span className="agent-picker-option-name" style={{ flex: 1 }}>{d.file_name}</span>
                  <Tag color={DOC_STATUS_COLOR[d.status]}>{DOC_STATUS_LABEL[d.status] || d.status}</Tag>
                  {docId === d.doc_id && <span style={{ color: '#000' }}>✓</span>}
                </div>
              </div>
            ))
          ) : (
            <Empty description="暂无文档" />
          )}
        </div>
      </Modal>

      {/* 生成模型弹窗 */}
      <Modal
        title="选择生成模型"
        open={modelOpen}
        onCancel={() => setModelOpen(false)}
        width={520}
        footer={[<Button key="c" onClick={() => setModelOpen(false)}>取消</Button>]}
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxHeight: 460, overflow: 'auto' }}>
          {models.length ? (
            models.map((m) => (
              <div
                key={m.model_id}
                className={['agent-picker-option', modelId === m.model_id ? 'selected' : ''].filter(Boolean).join(' ')}
                onClick={() => { setModelId(m.model_id); setModelOpen(false); }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0 }}>
                  <span className="agent-picker-option-name" style={{ flex: 1 }}>{m.name}</span>
                  <Tag color="blue">Chat LLM</Tag>
                  {m.is_default && <Tag color="gold">默认</Tag>}
                  {modelId === m.model_id && <span style={{ color: '#000' }}>✓</span>}
                </div>
              </div>
            ))
          ) : (
            <Empty description="暂无 Chat 模型，请先在模型管理页配置" />
          )}
        </div>
      </Modal>
    </div>
  );
}
