import { useEffect, useState, useCallback, useRef } from 'react';
import {
  Card, Button, Switch, Space, Progress, Typography, message,
  Tag, Checkbox, Alert, Segmented, Modal, Empty,
} from 'antd';
import { DownOutlined, PlayCircleOutlined, StopOutlined, SyncOutlined } from '@ant-design/icons';
import { useParams } from 'react-router-dom';
import { fetchDocuments, fetchChunks, generateKG, cancelTask } from '@/api/kb';
import { fetchModels } from '@/api/models';
import { useGlobalStore } from '@/store';
import type { DocumentStatus, Chunk, LLMModel } from '@/types';
import { formatDateTime } from '@/utils';
import { GlassButton } from '@/components/GlassUI';

export default function GraphBuilder() {
  // kbId 由知识库详情路由提供（/kb/:kbId/graph）
  const { kbId } = useParams<{ kbId: string }>();
  const [docs, setDocs] = useState<DocumentStatus[]>([]);
  const [docId, setDocId] = useState<string>();
  const [docOpen, setDocOpen] = useState(false);
  const [models, setModels] = useState<LLMModel[]>([]);
  const [modelId, setModelId] = useState<string>();
  const [modelOpen, setModelOpen] = useState(false);
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [selectedChunkIds, setSelectedChunkIds] = useState<string[]>([]);
  const [mode, setMode] = useState('enhanced');
  const [entityResolution, setEntityResolution] = useState(true);
  const [withCommunity, setWithCommunity] = useState(true);
  const [running, setRunning] = useState(false);
  const [taskId, setTaskId] = useState<string>();
  const [progress, setProgress] = useState(0);
  const [logs, setLogs] = useState<string[]>([]);
  const esRef = useRef<EventSource | null>(null);
  const [previewImg, setPreviewImg] = useState<any>(null);
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
    if (kbId) {
      const docList = await fetchDocuments(kbId).catch(() => []);
      const ready = docList.filter((d) => d.status === 'indexed' || d.status === 'completed');
      setDocs(ready);
    }
  }, [kbId]);

  useEffect(() => {
    loadAll();
    return () => { if (esRef.current) esRef.current.close(); };
  }, [loadAll]);

  const loadChunks = async (did: string) => {
    if (!kbId) return;
    try {
      const list = await fetchChunks(kbId, did);
      setChunks(list);
      setSelectedChunkIds(list.filter((c) => c.selected_for_kg).map((c) => c.chunk_id));
    } catch (e) {
      message.error(`加载分块失败: ${(e as Error).message}`);
    }
  };

  const pickDoc = (did: string) => {
    setDocId(did);
    setDocOpen(false);
    setArmed(false);
    loadChunks(did);
  };

  const toggleAll = (checked: boolean) => {
    setSelectedChunkIds(checked ? chunks.map((c) => c.chunk_id) : []);
  };

  const handleCancelArm = () => {
    setArmed(false);
    setSelectedChunkIds([]);
  };

  const handlePrepare = () => {
    setArmed(true);
  };

  const handleGenerate = async () => {
    if (!kbId || !docId) { message.warning('请选择文档'); return; }
    if (selectedChunkIds.length === 0) { message.warning('请至少选中 1 个分块'); return; }
    const chatOk = modelId || models.some((m) => m.model_type === 'llm' || !m.model_type);
    if (!chatOk) { message.error('未配置 Chat LLM 模型'); return; }
    try {
      const res = await generateKG(kbId, docId, {
        selected_chunk_ids: selectedChunkIds,
        model_id: modelId,
        mode,
        entity_resolution: entityResolution,
        with_community: withCommunity,
      });
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
    const source = new EventSource(`/api/v1/knowledge-bases/tasks/${tid}/stream`);
    esRef.current = source;
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
    const cleanup = () => {
      source.close();
      esRef.current = null;
      setRunning(false);
      setArmed(false);
      if (kbId) {
        fetchDocuments(kbId).then((l) => setDocs(l.filter((d) => d.status === 'indexed' || d.status === 'completed')));
      }
    };
  };

  const handleCancel = async () => {
    if (taskId) {
      await cancelTask(taskId).catch(() => {});
    }
  };

  const selectedDoc = docs.find((d) => d.doc_id === docId);
  const selectedModel = models.find((m) => m.model_id === modelId);

  // 分块内图片缩略图：点击放大预览
  const renderChunkImages = (images: any[]) => {
    if (!images || images.length === 0) return null;
    return (
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 8 }}>
        {images.map((img) => (
          <div
            key={img.image_id}
            onClick={(e) => {
              e.stopPropagation();
              setPreviewImg(img);
            }}
            title={img.title || img.description || '点击查看大图'}
            style={{
              width: 84,
              cursor: 'zoom-in',
              border: '1px solid rgba(0,0,0,0.4)',
              borderRadius: 6,
              padding: 3,
              background: 'rgba(255,255,255,0.7)',
              boxShadow: '0 1px 4px rgba(0,0,0,0.10)',
            }}
          >
            <img
              src={img.image_url}
              alt={img.title || '图片'}
              style={{ width: '100%', height: 60, objectFit: 'cover', borderRadius: 4, display: 'block', border: '1px solid rgba(0,0,0,0.1)' }}
            />
          </div>
        ))}
      </div>
    );
  };

  // 分块内表格预览：黑底表头小表格，限高滚动
  const renderChunkTables = (tables: any[]) => {
    if (!tables || tables.length === 0) return null;
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 8 }}>
        {tables.map((t) => {
          const md = t.meta_data;
          const data = md?.data || md?.rows;
          return (
            <div key={t.table_id} style={{ border: '1px solid rgba(0,0,0,0.4)', borderRadius: 6, padding: 6, background: 'rgba(255,255,255,0.65)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
                <Tag color="purple" style={{ marginRight: 0 }}>表格</Tag>
                <Typography.Text strong style={{ fontSize: 12 }}>{t.title || '未命名表格'}</Typography.Text>
              </div>
              {t.description && <div style={{ fontSize: 11, color: '#8c8c8c', marginBottom: 4 }}>{t.description}</div>}
              {Array.isArray(data) && Array.isArray(data[0]) ? (
                <div style={{ maxHeight: 160, overflow: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                    <thead>
                      <tr>
                        {(data[0] as string[]).map((h, i) => (
                          <th key={i} style={{ border: '1px solid #1f1f1f', padding: '4px 8px', background: '#1f1f1f', color: '#fff', fontWeight: 600, textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {(data.slice(1) as string[][]).map((r, ri) => (
                        <tr key={ri} style={{ background: ri % 2 ? 'rgba(0,0,0,0.04)' : '#fff' }}>
                          {r.map((c, ci) => (
                            <td key={ci} style={{ border: '1px solid rgba(0,0,0,0.45)', padding: '4px 8px', color: '#262626', whiteSpace: 'nowrap' }}>{c}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div style={{ fontSize: 11, color: '#8c8c8c' }}>表格内容为空</div>
              )}
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div className="glass-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 12, height: 'calc(100vh - 64px)', overflow: 'hidden' }}>
      <div style={{ display: 'flex', gap: 12, flexWrap: 'nowrap', flexShrink: 0 }}>
        {/* 文档选择：胶囊 + 弹窗 */}
        <Card className="glass glass-card-bordered" size="small" style={{ flex: 1, minWidth: 0 }} title="选择文档">
          <div className="agent-picker" style={{ cursor: 'pointer' }} onClick={() => setDocOpen(true)}>
            <span className="agent-picker-label">选择文档</span>
            <span className="agent-picker-value">
              {selectedDoc?.file_name || '点击选择要生成图谱的文档'}
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
              <Typography.Text style={{ fontSize: 12, color: '#595959', width: 56, flexShrink: 0 }}>生成模式</Typography.Text>
              <Segmented
                size="small"
                style={{ flex: 1, border: '1px solid rgba(0,0,0,0.3)' }}
                value={mode}
                onChange={(v) => setMode(v as string)}
                options={[
                  { value: 'enhanced', label: '增强模式' },
                  { value: 'classic', label: '经典模式' },
                ]}
              />
            </div>
            <div style={{ display: 'flex', gap: 18, padding: '4px 0' }}>
              <span style={{ fontSize: 12 }}>实体归一化 <Switch size="small" checked={entityResolution} onChange={setEntityResolution} /></span>
              <span style={{ fontSize: 12 }}>社区报告 <Switch size="small" checked={withCommunity} onChange={setWithCommunity} /></span>
            </div>
            {!modelConfigStatus?.chat_configured && (
              <Alert type="warning" showIcon message="未配置 Chat LLM 模型" style={{ padding: '4px 12px', fontSize: 12 }} />
            )}
            {!modelConfigStatus?.embedding_configured && (
              <Alert type="warning" showIcon message="未配置 Embedding 模型（图谱生成可能需要向量化）" style={{ padding: '4px 12px', fontSize: 12 }} />
            )}
            {!docId && <Alert type="info" showIcon message="请先选择已索引的文档" style={{ padding: '4px 12px', fontSize: 12 }} />}
          </Space>
        </Card>
      </div>

      <Card
        className="glass glass-card-bordered"
        size="small"
        style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}
        styles={{ body: { flex: 1, overflow: 'auto', minHeight: 0 } }}
        title={`分块列表（${selectedChunkIds.length}/${chunks.length} 已选）`}
      >
        {!docId ? (
          <Typography.Text type="secondary">请选择文档后加载分块</Typography.Text>
        ) : chunks.length === 0 ? (
          <Typography.Text type="secondary">该文档暂无分块，请先解析</Typography.Text>
        ) : (
          <div style={{ paddingRight: 4 }}>
            {chunks.map((c) => (
              <div
                key={c.chunk_id}
                style={{
                  marginBottom: 8,
                  padding: 8,
                  background: 'rgba(255,255,255,0.5)',
                  borderRadius: 8,
                  border: '1px solid rgba(0,0,0,0.35)',
                  boxShadow: '0 2px 6px rgba(0,0,0,0.06)',
                }}
              >
                <Space>
                  {armed && (
                    <Checkbox
                      className="chunk-checkbox"
                      checked={selectedChunkIds.includes(c.chunk_id)}
                      onChange={(e) => {
                        setSelectedChunkIds((prev) =>
                          e.target.checked ? [...prev, c.chunk_id] : prev.filter((x) => x !== c.chunk_id),
                        );
                      }}
                    />
                  )}
                  <Tag color="blue">{c.chunk_index}</Tag>
                  <Typography.Text strong>{c.title || `分块 ${c.chunk_index}`}</Typography.Text>
                  <Tag>{c.section}</Tag>
                </Space>
                <div style={{ fontSize: 12, color: '#595959', marginTop: 4, maxHeight: 60, overflow: 'hidden' }}>
                  {c.content.slice(0, 300)}...
                </div>
                {renderChunkImages((c as any).images)}
                {renderChunkTables((c as any).tables)}
              </div>
            ))}
          </div>
        )}
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
              <GlassButton
                variant="primary"
                icon={<PlayCircleOutlined />}
                onClick={handleGenerate}
              >
                AI生成
              </GlassButton>
            ) : (
              <GlassButton
                variant="primary"
                icon={<PlayCircleOutlined />}
                onClick={handlePrepare}
              >
                生成知识图谱
              </GlassButton>
            )
          )}
          {armed && !running && (
            <>
              <GlassButton variant="text" size="small" onClick={() => toggleAll(true)}>全选</GlassButton>
              <GlassButton variant="text" size="small" onClick={() => toggleAll(false)}>取消全选</GlassButton>
              <GlassButton variant="text" size="small" onClick={handleCancelArm}>取消生成</GlassButton>
            </>
          )}
          {running && (
            <GlassButton variant="danger" icon={<StopOutlined />} onClick={handleCancel}>取消任务</GlassButton>
          )}
          <GlassButton variant="text" icon={<SyncOutlined />} onClick={() => docId && loadChunks(docId)}>刷新分块</GlassButton>
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
                  <Tag color={d.status === 'indexed' ? 'blue' : 'green'}>{d.status === 'indexed' ? '已索引' : '已完成'}</Tag>
                  {docId === d.doc_id && <span style={{ color: '#000' }}>✓</span>}
                </div>
              </div>
            ))
          ) : (
            <Empty description="暂无已索引/已完成的文档" />
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

      {/* 分块图片放大预览 */}
      <Modal
        title={previewImg?.title || '图片预览'}
        open={!!previewImg}
        onCancel={() => setPreviewImg(null)}
        footer={null}
        width={680}
      >
        {previewImg && (
          <div style={{ textAlign: 'center' }}>
            <img
              src={previewImg.image_url}
              alt={previewImg.title || '图片'}
              style={{ maxWidth: '100%', maxHeight: 560, borderRadius: 8, border: '1px solid rgba(0,0,0,0.4)', display: 'block', margin: '0 auto' }}
            />
            {previewImg.description && (
              <div style={{ marginTop: 10, fontSize: 13, color: '#595959', textAlign: 'left', padding: '0 8px' }}>
                {previewImg.description}
              </div>
            )}
          </div>
        )}
      </Modal>
    </div>
  );
}
