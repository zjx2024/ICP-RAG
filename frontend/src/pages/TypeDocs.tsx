import { useEffect, useState, useCallback, useRef } from 'react';
import {
  Card, Button, Tag, Table, Space, Modal, Empty,
  Progress, Typography, message, Popconfirm, Alert,
} from 'antd';
import { Upload as AntdUpload, type UploadFile } from 'antd';
import {
  CloudUploadOutlined, DeleteOutlined, PlayCircleOutlined, EyeOutlined,
  SyncOutlined, InboxOutlined, FileOutlined, DownOutlined, CheckOutlined,
} from '@ant-design/icons';
import { useNavigate, useParams } from 'react-router-dom';
import {
  fetchDocumentType, fetchIcpDocuments, uploadIcpDocuments,
  parseIcpDocument, deleteIcpDocument,
} from '@/api/icp';
import { openTaskStream, cancelTask } from '@/api/kb';
import { fetchModels, fetchCurrentModelConfig } from '@/api/models';
import { useGlobalStore } from '@/store';
import { GlassButton } from '@/components/GlassUI';
import type { ICPDocument, DocTypeInfo } from '@/types';
import { DOC_STATUS_LABEL, DOC_STATUS_COLOR, formatDateTime } from '@/utils';

interface RunningTask {
  source: EventSource;
  taskId: string;
  logs: string[];
  progress: number;
}

const RUNNING_STATUS = new Set(['parsing', 'generating_kg']);

export default function TypeDocs() {
  const { docType = '' } = useParams<{ docType: string }>();
  const navigate = useNavigate();
  const [typeInfo, setTypeInfo] = useState<DocTypeInfo | null>(null);
  const [docs, setDocs] = useState<ICPDocument[]>([]);
  const [models, setModels] = useState<any[]>([]);
  const [modelConfigStatus, setModelConfigStatus] = useState<any>(null);
  const [parseModel, setParseModel] = useState<string | undefined>();
  const [uploadModalOpen, setUploadModalOpen] = useState(false);
  const [uploadFileList, setUploadFileList] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [modelPickerOpen, setModelPickerOpen] = useState(false);
  const [logModal, setLogModal] = useState<{ open: boolean; docId: string; title: string } | null>(null);
  const [logContent, setLogContent] = useState<{ logs: string[]; progress: number }>({ logs: [], progress: 0 });
  const runningTasks = useRef<Map<string, RunningTask>>(new Map());
  const [, forceUpdate] = useState(0);

  const setModelsStore = useGlobalStore((s) => s.setModels);
  const setModelConfigStatusStore = useGlobalStore((s) => s.setModelConfigStatus);

  const loadType = useCallback(async (dt: string) => {
    try {
      setTypeInfo(await fetchDocumentType(dt));
    } catch {
      setTypeInfo(null);
    }
  }, []);

  const loadDocs = useCallback(async (dt: string) => {
    try {
      setDocs(await fetchIcpDocuments(dt));
    } catch {
      setDocs([]);
    }
  }, []);

  const loadModelsData = useCallback(async () => {
    try {
      const res = await fetchModels();
      const list = (res.models || []).filter((m) => m.model_type === 'llm' || !m.model_type);
      setModels(list);
      setModelsStore(list);
      const conf = await fetchCurrentModelConfig();
      setModelConfigStatus(conf.config_status);
      setModelConfigStatusStore(conf.config_status);
    } catch {
      /* ignore */
    }
  }, [setModelsStore, setModelConfigStatusStore]);

  useEffect(() => {
    if (!docType) return;
    loadType(docType);
    loadDocs(docType);
    loadModelsData();
  }, [docType, loadType, loadDocs, loadModelsData]);

  useEffect(() => () => {
    runningTasks.current.forEach((t) => t.source.close());
    runningTasks.current.clear();
  }, []);

  const checkModels = () => {
    if (!modelConfigStatus?.chat_configured) return '未配置 Chat LLM 模型，无法解析';
    if (!modelConfigStatus?.embedding_configured) return '未配置 Embedding 模型，无法向量化';
    return '';
  };

  const openUploadModal = () => { setUploadFileList([]); setUploadModalOpen(true); };

  const handleUploadModalAdd = (files: File[]) => {
    setUploadFileList((prev) => {
      const seen = new Set(prev.map((f) => `${f.name}-${f.size}-${f.lastModified}`));
      const add = files.filter((f) => !seen.has(`${f.name}-${f.size}-${f.lastModified}`));
      return [...prev, ...add];
    });
  };

  const handleUploadModalRemove = (index: number) => {
    setUploadFileList((prev) => prev.filter((_, i) => i !== index));
  };

  const handleConfirmUpload = async () => {
    if (uploadFileList.length === 0) { message.warning('请选择要上传的文件'); return; }
    setUploading(true);
    try {
      const res = await uploadIcpDocuments(docType, uploadFileList);
      message.success(`上传成功 ${res.length} 个文件`);
      setUploadFileList([]);
      setUploadModalOpen(false);
      loadDocs(docType);
    } catch (e) {
      message.error(`上传失败: ${(e as Error).message}`);
    } finally {
      setUploading(false);
    }
  };

  const handleParse = async (doc: ICPDocument) => {
    const tip = checkModels();
    if (tip) { message.error(tip); return; }
    setDocs((prev) => prev.map((d) => (d.doc_id === doc.doc_id ? { ...d, status: 'parsing' } : d)));
    try {
      const res = await parseIcpDocument(docType, doc.doc_id, parseModel);
      startEventSource(doc.doc_id, res.task_id);
    } catch (e) {
      setDocs((prev) => prev.map((d) => (d.doc_id === doc.doc_id ? { ...d, status: 'failed' } : d)));
      message.error(`解析启动失败: ${(e as Error).message}`);
    }
  };

  const startEventSource = (docId: string, taskId: string) => {
    const key = `${docType}|${docId}`;
    const old = runningTasks.current.get(key);
    if (old) old.source.close();

    const logs: string[] = [];
    const source = openTaskStream(taskId);
    const entry: RunningTask = { source, taskId, logs, progress: 0 };
    runningTasks.current.set(key, entry);

    const cleanup = () => {
      source.close();
      runningTasks.current.delete(key);
      loadDocs(docType);
      forceUpdate((n) => n + 1);
    };

    source.onmessage = (ev) => {
      let data: any;
      try { data = JSON.parse(ev.data); } catch { return; }
      if (data.type === 'progress') {
        entry.progress = data.progress;
        entry.logs.push(`[${formatDateTime()}] ${data.log || ''}`);
      } else if (data.type === 'log') {
        entry.logs.push(`[${formatDateTime()}] ${data.log}`);
      } else if (data.type === 'status' && data.status === 'cancelled') {
        entry.logs.push(`[${formatDateTime()}] 任务已取消`);
        cleanup();
      } else if (data.type === 'final') {
        cleanup();
      }
      forceUpdate((n) => n + 1);
    };
    source.onerror = () => cleanup();
  };

  const handleCancelTask = async (docId: string) => {
    const task = runningTasks.current.get(`${docType}|${docId}`);
    if (task) await cancelTask(task.taskId);
  };

  const showLogModal = (docId: string) => {
    const task = runningTasks.current.get(`${docType}|${docId}`);
    const doc = docs.find((d) => d.doc_id === docId);
    setLogContent({ logs: task?.logs || [], progress: task?.progress || 0 });
    setLogModal({ open: true, docId, title: doc?.file_name || docId });
  };

  const handleDeleteDoc = async (docId: string) => {
    const task = runningTasks.current.get(`${docType}|${docId}`);
    if (task) { task.source.close(); runningTasks.current.delete(task ? `${docType}|${docId}` : ''); }
    try {
      await deleteIcpDocument(docType, docId);
      message.success('文档已删除');
      loadDocs(docType);
    } catch (e) {
      message.error(`删除失败: ${(e as Error).message}`);
    }
  };

  const getDocProgress = (docId: string) =>
    runningTasks.current.get(`${docType}|${docId}`)?.progress ?? 0;

  const columns = [
    { title: '文件名', dataIndex: 'file_name', ellipsis: true },
    {
      title: '状态',
      dataIndex: 'status',
      width: 160,
      render: (v: string, r: ICPDocument) => (
        <Space direction="vertical" size={0}>
          <Tag color={DOC_STATUS_COLOR[v]}>{DOC_STATUS_LABEL[v] || v}</Tag>
          {RUNNING_STATUS.has(v) && (
            <Progress percent={getDocProgress(r.doc_id)} size="small" style={{ width: 110 }} />
          )}
        </Space>
      ),
    },
    {
      title: '解析次数',
      dataIndex: 'parse_count',
      width: 90,
      render: (v?: number) => v ?? 0,
    },
    { title: '更新时间', dataIndex: 'updated_at', width: 160, render: (v: string) => (v ? formatDateTime(v) : '-') },
    {
      title: '操作',
      width: 280,
      render: (_: unknown, r: ICPDocument) => (
        <Space size={4} className="kb-row-actions">
          {(r.status === 'uploaded' || r.status === 'failed' || r.status === 'indexed' ||
            r.status === 'kg_pending' || r.status === 'kg_done') && (
            <GlassButton variant="primary" size="small" icon={<PlayCircleOutlined />} onClick={() => handleParse(r)}>
              解析
            </GlassButton>
          )}
          {(r.status === 'parsing' || r.status === 'generating_kg') && (
            <>
              <GlassButton variant="text" size="small" onClick={() => showLogModal(r.doc_id)}>日志</GlassButton>
              <GlassButton variant="text" size="small" onClick={() => handleCancelTask(r.doc_id)}>取消</GlassButton>
            </>
          )}
          <GlassButton size="small" icon={<EyeOutlined />} onClick={() => navigate(`/type/${docType}/doc/${r.doc_id}`)}>
            查看
          </GlassButton>
          <Popconfirm title="确定删除该文档？" onConfirm={() => handleDeleteDoc(r.doc_id)}>
            <GlassButton className="glass-btn-black" size="small" icon={<DeleteOutlined />}>删除</GlassButton>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  const sortedDocs = [...docs].sort((a, b) => (b.updated_at || '').localeCompare(a.updated_at || ''));

  return (
    <div className="glass-fade-in" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16, flexWrap: 'wrap', flexShrink: 0 }}>
        <div>
          <Typography.Title level={5} style={{ margin: 0, color: '#262626' }}>
            文档上传与解析
          </Typography.Title>
          <Typography.Text type="secondary" style={{ fontSize: 12 }}>
            {typeInfo?.name || docType}
            {typeInfo?.input_formats ? ` · 支持格式: ${typeInfo.input_formats}` : ''}
          </Typography.Text>
        </div>
        <div style={{ flex: 1 }} />

        {/* 解析模型胶囊 */}
        <div className="agent-picker agent-picker-sm" onClick={() => setModelPickerOpen(true)} style={{ width: 'auto', maxWidth: 260 }} title="选择解析用 Chat 模型">
          <span className="agent-picker-label">解析模型</span>
          <span className="agent-picker-value">
            {parseModel ? models.find((m) => m.model_id === parseModel)?.name || '已选择' : '使用默认'}
            <DownOutlined />
          </span>
        </div>

        {/* 上传胶囊 */}
        <div className="agent-picker agent-picker-sm" onClick={openUploadModal} style={{ width: 'auto', cursor: 'pointer' }} title="上传文件">
          <span className="agent-picker-label" style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <CloudUploadOutlined /> 上传文件
          </span>
          <span className="agent-picker-value"><DownOutlined /></span>
        </div>

        {(!modelConfigStatus?.chat_configured || !modelConfigStatus?.embedding_configured) && (
          <Alert
            type="warning"
            showIcon
            style={{ padding: '2px 10px', fontSize: 12, flexShrink: 0 }}
            message={
              !modelConfigStatus?.chat_configured
                ? !modelConfigStatus?.embedding_configured
                  ? '未配置 Chat LLM 与 Embedding 模型'
                  : '未配置 Chat LLM 模型，解析不可用'
                : '未配置 Embedding 模型，向量化不可用'
            }
          />
        )}
      </div>

      <Card
        className="glass"
        style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}
        styles={{ body: { display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0, paddingTop: 12 } }}
        title={
          <Space>
            <span>{typeInfo?.name || docType} · 文档列表</span>
            <GlassButton variant="text" size="small" icon={<SyncOutlined />} onClick={() => loadDocs(docType)}>刷新</GlassButton>
          </Space>
        }
      >
        <Table<ICPDocument>
          rowKey="doc_id"
          className="kb-doc-table"
          dataSource={sortedDocs}
          columns={columns}
          pagination={{ pageSize: 10, showSizeChanger: false, showTotal: (t) => `共 ${t} 份` }}
          size="small"
          locale={{ emptyText: '暂无文档，请先上传' }}
          scroll={{ y: 'calc(100vh - 320px)' }}
        />
      </Card>

      {/* 上传弹窗 */}
      <Modal
        title={`上传文件（${typeInfo?.name || docType}）`}
        open={uploadModalOpen}
        onCancel={() => { setUploadModalOpen(false); setUploadFileList([]); }}
        footer={[
          <Button key="cancel" onClick={() => { setUploadModalOpen(false); setUploadFileList([]); }}>取消</Button>,
          <GlassButton key="confirm" variant="primary" icon={<CloudUploadOutlined />} loading={uploading} disabled={uploadFileList.length === 0} onClick={handleConfirmUpload}>
            {uploading ? '上传中…' : `保存并上传${uploadFileList.length > 0 ? `（${uploadFileList.length}）` : ''}`}
          </GlassButton>,
        ]}
        width={520}
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <AntdUpload.Dragger
            multiple
            accept={typeInfo?.input_formats || undefined}
            beforeUpload={() => false}
            fileList={[]}
            onChange={({ fileList: fl }: { fileList: UploadFile[] }) => {
              const newFiles = fl.map((f) => f.originFileObj as File).filter(Boolean);
              if (newFiles.length > 0) handleUploadModalAdd(newFiles);
            }}
          >
            <p className="ant-upload-drag-icon">
              <InboxOutlined style={{ fontSize: 42, color: 'rgba(0,0,0,0.4)' }} />
            </p>
            <p className="ant-upload-text">点击或拖拽文件到此处</p>
            <p className="ant-upload-hint" style={{ fontSize: 12 }}>
              {typeInfo?.input_formats ? `该类型支持格式：${typeInfo.input_formats}` : '仅支持该文档类型声明的格式'}
            </p>
          </AntdUpload.Dragger>

          {uploadFileList.length > 0 && (
            <div style={{ marginTop: 8 }}>
              <Typography.Text strong style={{ fontSize: 13 }}>已选文件（{uploadFileList.length} 个）</Typography.Text>
              <div style={{ maxHeight: 200, overflow: 'auto', marginTop: 6, display: 'flex', flexDirection: 'column', gap: 4 }}>
                {uploadFileList.map((f, i) => (
                  <div key={i} className="glass-dense" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '6px 10px', borderRadius: 8, border: '1px solid rgba(0,0,0,0.12)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0, flex: 1 }}>
                      <FileOutlined style={{ color: 'rgba(0,0,0,0.5)', flexShrink: 0 }} />
                      <div style={{ minWidth: 0, flex: 1 }}>
                        <div style={{ fontSize: 13, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{f.name}</div>
                        <div style={{ fontSize: 11, color: '#8c8c8c' }}>{(f.size / 1024).toFixed(1)} KB</div>
                      </div>
                    </div>
                    <GlassButton variant="text" size="small" icon={<DeleteOutlined />} onClick={() => handleUploadModalRemove(i)} />
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </Modal>

      {/* 解析模型弹窗 */}
      <Modal title="选择解析模型" open={modelPickerOpen} onCancel={() => setModelPickerOpen(false)} onOk={() => setModelPickerOpen(false)} okText="确定" cancelText="取消" width={460} okButtonProps={{ disabled: models.length === 0 }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxHeight: 400, overflow: 'auto' }}>
          {models.length === 0 ? (
            <Empty description="暂无 Chat 模型，请先到模型管理配置" style={{ marginTop: 40 }} />
          ) : (
            <>
              {models.map((m) => {
                const selected = parseModel === m.model_id;
                return (
                  <div key={m.model_id} className={`agent-picker-option${selected ? ' selected' : ''}`} onClick={() => setParseModel(m.model_id)}>
                    <div style={{ minWidth: 0 }}>
                      <div className="agent-picker-option-name">
                        {m.name}
                        <Tag style={{ marginLeft: 6 }}>Chat LLM</Tag>
                        {m.is_default && <Tag color="gold" style={{ marginLeft: 4 }}>★ 默认</Tag>}
                      </div>
                      <div className="agent-picker-option-sub">解析文档 / 表格 / 图片用</div>
                    </div>
                    {selected && <CheckOutlined style={{ color: '#1a1a1a', flexShrink: 0 }} />}
                  </div>
                );
              })}
              <div className={`agent-picker-option${!parseModel ? ' selected' : ''}`} onClick={() => setParseModel(undefined)}>
                <div className="agent-picker-option-name" style={{ color: '#8c8c8c' }}>使用默认模型</div>
                {!parseModel && <CheckOutlined style={{ color: '#8c8c8c', flexShrink: 0 }} />}
              </div>
            </>
          )}
        </div>
      </Modal>

      {/* 日志弹窗 */}
      <Modal
        title={`任务日志：${logModal?.title || ''}`}
        open={logModal?.open}
        onCancel={() => setLogModal(null)}
        footer={<Button onClick={() => setLogModal(null)}>关闭</Button>}
        width={640}
      >
        <Progress percent={logContent.progress} size="small" style={{ marginBottom: 12 }} />
        <div
          style={{
            background: 'rgba(0,0,0,0.85)', color: '#d9d9d9', borderRadius: 8, padding: 12,
            maxHeight: 360, overflow: 'auto', fontFamily: 'monospace', fontSize: 12, whiteSpace: 'pre-wrap',
          }}
        >
          {logContent.logs.length === 0 ? '暂无日志输出' : logContent.logs.join('\n')}
        </div>
      </Modal>
    </div>
  );
}
