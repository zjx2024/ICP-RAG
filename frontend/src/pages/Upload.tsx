import { useEffect, useState, useCallback, useRef } from 'react';
import {
  Card, Button, Tag, Table, Space, Modal, Empty,
  Progress, Typography, message, Popconfirm, Alert, Checkbox,
} from 'antd';
import { Upload as AntdUpload, type UploadFile } from 'antd';
import {
  CloudUploadOutlined, DeleteOutlined, PlayCircleOutlined,
  ApiOutlined, SyncOutlined, InboxOutlined, FileOutlined, DownOutlined, CheckOutlined,
} from '@ant-design/icons';
import { useParams } from 'react-router-dom';
import {
  fetchDocuments,
  uploadDocuments, parseDocument, deleteDocument, cancelTask,
} from '@/api/kb';
import { fetchModels, fetchCurrentModelConfig } from '@/api/models';
import { fetchDifyServerInfo } from '@/api/kb';
import { useGlobalStore } from '@/store';
import { GlassButton, GlassToolbar } from '@/components/GlassUI';
import type { DocumentStatus, LLMModel } from '@/types';
import { DOC_STATUS_LABEL, DOC_STATUS_COLOR, formatDateTime } from '@/utils';

interface RunningTask {
  source: EventSource;
  taskId: string;
  logs: string[];
  progress: number;
}

export default function Upload() {
  const { kbId } = useParams<{ kbId: string }>();
  const [docs, setDocs] = useState<DocumentStatus[]>([]);
  const [models, setModels] = useState<LLMModel[]>([]);
  const [modelConfigStatus, setModelConfigStatus] = useState<any>(null);
  const [parseModel, setParseModel] = useState<string | undefined>();
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);
  const [batchArmed, setBatchArmed] = useState(false);
  const [logModal, setLogModal] = useState<{ open: boolean; kbId: string; docId: string; title: string } | null>(null);
  const [logContent, setLogContent] = useState<{ logs: string[]; progress: number }>({ logs: [], progress: 0 });
  const runningTasks = useRef<Map<string, RunningTask>>(new Map());
  const [, forceUpdate] = useState(0);

  // 上传弹窗状态
  const [uploadModalOpen, setUploadModalOpen] = useState(false);
  const [uploadFileList, setUploadFileList] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);

  // Dify 对接弹窗
  const [difyModalOpen, setDifyModalOpen] = useState(false);
  const [difyInfo, setDifyInfo] = useState<any>(null);
  const [difyLoading, setDifyLoading] = useState(false);

  // 解析模型选择弹窗
  const [modelPickerOpen, setModelPickerOpen] = useState(false);

  // 文档列表表格：动态高度占满剩余可视区域
  const tableWrapRef = useRef<HTMLDivElement>(null);
  const [tableScrollY, setTableScrollY] = useState(420);
  useEffect(() => {
    const calc = () => {
      const el = tableWrapRef.current;
      if (!el) return;
      setTableScrollY(Math.max(200, window.innerHeight - el.getBoundingClientRect().top - 130));
    };
    calc();
    window.addEventListener('resize', calc);
    return () => window.removeEventListener('resize', calc);
  }, [docs.length]);

  const setModelsStore = useGlobalStore((s) => s.setModels);
  const setModelConfigStatusStore = useGlobalStore((s) => s.setModelConfigStatus);

  const loadDocs = useCallback(async (kid: string) => {
    try {
      const list = await fetchDocuments(kid);
      setDocs(list);
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
    loadModelsData();
    if (kbId) loadDocs(kbId);
    return () => {
      runningTasks.current.forEach((t) => t.source.close());
      runningTasks.current.clear();
    };
  }, []);

  useEffect(() => {
    if (kbId) loadDocs(kbId);
  }, [kbId, loadDocs]);

  const checkModels = (): string | null => {
    const chatOk = parseModel || models.some((m) => m.model_type === 'llm' || !m.model_type);
    const embedOk = modelConfigStatus?.embedding_configured || false;
    if (!chatOk) return '未配置 Chat LLM 模型，请先在模型管理中配置';
    if (!embedOk) return '未配置 Embedding 模型，无法进行向量化';
    return null;
  };

  // 打开上传弹窗
  const openUploadModal = () => {
    setUploadFileList([]);
    setUploadModalOpen(true);
  };

  // 打开 Dify 对接弹窗
  const openDifyModal = async () => {
    setDifyModalOpen(true);
    setDifyLoading(true);
    try {
      const info = await fetchDifyServerInfo();
      setDifyInfo(info);
    } catch (e) {
      message.error(`获取 Dify 信息失败: ${(e as Error).message}`);
    } finally {
      setDifyLoading(false);
    }
  };

  // 上传弹窗内：添加文件（按 文件名+大小+修改时间 去重，避免重复选择同一文件）
  const handleUploadModalAdd = (files: File[]) => {
    setUploadFileList((prev) => {
      const seen = new Set(prev.map((f) => `${f.name}-${f.size}-${f.lastModified}`));
      const add = files.filter((f) => !seen.has(`${f.name}-${f.size}-${f.lastModified}`));
      return [...prev, ...add];
    });
  };

  // 上传弹窗内：移除文件
  const handleUploadModalRemove = (index: number) => {
    setUploadFileList((prev) => prev.filter((_, i) => i !== index));
  };

  // 确认上传（弹窗内）
  const handleConfirmUpload = async () => {
    if (!kbId) { message.warning('请先选择知识库'); return; }
    if (uploadFileList.length === 0) { message.warning('请选择要上传的文件'); return; }
    setUploading(true);
    try {
      const res = await uploadDocuments(kbId, uploadFileList);
      message.success(`上传成功 ${res.length} 个文件`);
      setUploadFileList([]);
      setUploadModalOpen(false);
      loadDocs(kbId);
    } catch (e) {
      message.error(`上传失败: ${(e as Error).message}`);
    } finally {
      setUploading(false);
    }
  };

  // 解析（单文件）
  const handleParse = async (doc: DocumentStatus) => {
    const tip = checkModels();
    if (tip) { message.error(tip); return; }
    if (!kbId) return;
    setDocs((prev) => prev.map((d) => (d.doc_id === doc.doc_id ? { ...d, status: 'parsing' as const } : d)));
    try {
      const res = await parseDocument(kbId, doc.doc_id, parseModel);
      startEventSource(doc.doc_id, res.task_id);
    } catch (e) {
      setDocs((prev) => prev.map((d) => (d.doc_id === doc.doc_id ? { ...d, status: 'failed' as const } : d)));
      message.error(`解析启动失败: ${(e as Error).message}`);
    }
  };

  const handleBatchParse = async () => {
    const tip = checkModels();
    if (tip) { message.error(tip); return; }
    if (!kbId) return;
    for (const key of selectedDocIds) {
      const [kid, did] = key.split('|');
      if (kid !== kbId) continue;
      const doc = docs.find((d) => d.doc_id === did);
      if (!doc) continue;
      setDocs((prev) => prev.map((d) => (d.doc_id === did ? { ...d, status: 'parsing' as const } : d)));
      try {
        const res = await parseDocument(kbId, did, parseModel);
        startEventSource(did, res.task_id);
      } catch (e) {
        setDocs((prev) => prev.map((d) => (d.doc_id === did ? { ...d, status: 'failed' as const } : d)));
      }
    }
    setSelectedDocIds([]);
    setBatchArmed(false);
    loadDocs(kbId);
  };

  const toggleAllDocs = (checked: boolean) => {
    if (checked) {
      setSelectedDocIds(
        sortedDocs
          .filter((d) => d.status !== 'parsing' && d.status !== 'generating_kg')
          .map((d) => `${kbId}|${d.doc_id}`),
      );
    } else {
      setSelectedDocIds([]);
    }
  };

  const handleCancelBatch = () => {
    setSelectedDocIds([]);
    setBatchArmed(false);
  };

  const handleBatchDelete = async () => {
    if (!kbId) return;
    if (selectedDocIds.length === 0) { message.warning('请先选择要删除的文档'); return; }
    for (const key of selectedDocIds) {
      const [kid, did] = key.split('|');
      if (kid !== kbId) continue;
      const task = runningTasks.current.get(key);
      if (task) { task.source.close(); runningTasks.current.delete(key); }
      try {
        await deleteDocument(kbId, did);
      } catch (e) {
        message.error(`删除 ${did} 失败: ${(e as Error).message}`);
      }
    }
    message.success(`已删除 ${selectedDocIds.length} 个文档`);
    setSelectedDocIds([]);
    setBatchArmed(false);
    loadDocs(kbId);
  };

  const startEventSource = (docId: string, taskId: string) => {
    const key = `${kbId}|${docId}`;
    const old = runningTasks.current.get(key);
    if (old) old.source.close();

    const logs: string[] = [];
    const source = new EventSource(`/api/v1/knowledge-bases/tasks/${taskId}/stream`);
    const entry: RunningTask = { source, taskId, logs, progress: 0 };
    runningTasks.current.set(key, entry);

    source.onmessage = (ev) => {
      let data: any;
      try { data = JSON.parse(ev.data); } catch { return; }
      if (data.type === 'progress') {
        entry.progress = data.progress;
        entry.logs.push(`[${formatDateTime()}] ${data.log || ''}`);
      } else if (data.type === 'log') {
        entry.logs.push(`[${formatDateTime()}] ${data.log}`);
      } else if (data.type === 'status' && data.status === 'cancelled') {
        entry.logs.push(`[${formatDateTime()}] ⚠️ 任务已取消`);
        cleanup();
      } else if (data.type === 'final') {
        cleanup();
      }
      forceUpdate((n) => n + 1);
    };
    source.onerror = () => {
      cleanup();
    };
    const cleanup = () => {
      source.close();
      runningTasks.current.delete(key);
      if (kbId) loadDocs(kbId);
      forceUpdate((n) => n + 1);
    };
  };

  const handleCancelTask = async (docId: string) => {
    const key = `${kbId}|${docId}`;
    const task = runningTasks.current.get(key);
    if (task) {
      await cancelTask(task.taskId);
    }
  };

  const showLogModal = (docId: string) => {
    const key = `${kbId}|${docId}`;
    const task = runningTasks.current.get(key);
    setLogContent({ logs: task?.logs || [], progress: task?.progress || 0 });
    setLogModal({ open: true, kbId: kbId!, docId, title: docId });
  };

  const handleDeleteDoc = async (docId: string) => {
    if (!kbId) return;
    const key = `${kbId}|${docId}`;
    const task = runningTasks.current.get(key);
    if (task) { task.source.close(); runningTasks.current.delete(key); }
    try {
      await deleteDocument(kbId, docId);
      message.success('文档已删除');
      loadDocs(kbId);
    } catch (e) {
      message.error(`删除失败: ${(e as Error).message}`);
    }
  };

  const getDocProgress = (docId: string) => {
    const task = runningTasks.current.get(`${kbId}|${docId}`);
    return task?.progress ?? 0;
  };

  const columns = [
    {
      title: '',
      width: 40,
      render: (_: unknown, r: DocumentStatus) =>
        batchArmed ? (
          <Checkbox
            className="chunk-checkbox"
            checked={selectedDocIds.includes(`${kbId}|${r.doc_id}`)}
            disabled={r.status === 'parsing' || r.status === 'generating_kg'}
            onChange={(e) => {
              setSelectedDocIds((prev) =>
                e.target.checked ? [...prev, `${kbId}|${r.doc_id}`] : prev.filter((x) => x !== `${kbId}|${r.doc_id}`),
              );
            }}
          />
        ) : null,
    },
    { title: '文件名', dataIndex: 'file_name', ellipsis: true },
    {
      title: '状态',
      dataIndex: 'status',
      render: (v: string, r: DocumentStatus) => (
        <Space direction="vertical" size={0}>
          <Tag color={DOC_STATUS_COLOR[v]}>{DOC_STATUS_LABEL[v] || v}</Tag>
          {(v === 'parsing' || v === 'generating_kg') && (
            <Progress percent={getDocProgress(r.doc_id)} size="small" style={{ width: 100 }} />
          )}
        </Space>
      ),
    },
    { title: '分块数', dataIndex: 'total_chunks', render: (v: number) => v ?? '-' },
    { title: '上传时间', dataIndex: 'created_at', render: (v: string) => (v ? formatDateTime(v) : '-') },
    {
      title: '操作',
      width: 260,
      render: (_: unknown, r: DocumentStatus) => (
        <Space size={4} className="kb-row-actions">
          {(r.status === 'uploaded' || r.status === 'failed') && (
            <GlassButton variant="primary" size="small" icon={<PlayCircleOutlined />} onClick={() => handleParse(r)}>解析</GlassButton>
          )}
          {(r.status === 'parsing' || r.status === 'generating_kg') && (
            <>
              <GlassButton variant="text" size="small" onClick={() => showLogModal(r.doc_id)}>查看日志</GlassButton>
              <GlassButton variant="text" size="small" onClick={() => handleCancelTask(r.doc_id)}>取消</GlassButton>
            </>
          )}
          <Popconfirm title="确定删除该文档？" onConfirm={() => handleDeleteDoc(r.doc_id)}>
            <GlassButton className="glass-btn-black" size="small" icon={<DeleteOutlined />}>删除</GlassButton>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  // 文档列表按上传时间倒序排列（最新在上）
  const sortedDocs = [...docs].sort((a, b) => (b.created_at || '').localeCompare(a.created_at || ''));

  return (
    <div className="glass-fade-in" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* 顶部紧凑工具栏：Dify 对接 + 解析模型 + 上传文件 */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16, flexWrap: 'wrap', flexShrink: 0 }}>
        {/* Dify 对接（缩小） */}
        <GlassButton variant="primary" size="small" icon={<ApiOutlined />} onClick={openDifyModal}>
          Dify 对接
        </GlassButton>

        {/* 解析模型胶囊：点击弹窗选择 */}
        <div
          className="agent-picker agent-picker-sm"
          onClick={() => setModelPickerOpen(true)}
          style={{ width: 'auto', maxWidth: 260 }}
          title="选择解析用 Chat 模型"
        >
          <span className="agent-picker-label">解析模型</span>
          <span className="agent-picker-value">
            {parseModel ? models.find((m) => m.model_id === parseModel)?.name || '已选择' : '使用默认'}
            <DownOutlined />
          </span>
        </div>

        {/* 上传文件胶囊：点击弹窗选择并上传 */}
        <div
          className="agent-picker agent-picker-sm"
          onClick={openUploadModal}
          style={{ width: 'auto', cursor: 'pointer' }}
          title="上传文件"
        >
          <span className="agent-picker-label" style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <CloudUploadOutlined /> 上传文件
          </span>
          <span className="agent-picker-value">
            <DownOutlined />
          </span>
        </div>

        {/* 模型未配置警告（紧凑展示） */}
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

      {/* 文档列表：占满剩余高度，内部滚动 */}
      <Card
        className="glass"
        style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}
        styles={{ body: { display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0, paddingTop: 12 } }}
        title={
          <Space>
            <span>文档列表</span>
            <GlassButton variant="text" size="small" icon={<SyncOutlined />} onClick={() => kbId && loadDocs(kbId)}>刷新</GlassButton>
          </Space>
        }
      >
        <div ref={tableWrapRef} style={{ flex: 1, minHeight: 0 }}>
          <Table<DocumentStatus>
            rowKey="doc_id"
            className="kb-doc-table"
            dataSource={sortedDocs}
            columns={columns}
            pagination={false}
            scroll={{ y: tableScrollY }}
            size="small"
            locale={{ emptyText: '暂无文档，请先上传' }}
          />
        </div>
      </Card>

      {/* 批量操作工具栏：固定于页面底部 */}
      <GlassToolbar style={{ marginTop: 16 }}>
        {!batchArmed ? (
          <GlassButton variant="primary" icon={<PlayCircleOutlined />} onClick={() => setBatchArmed(true)}>
            批量操作
          </GlassButton>
        ) : (
          <>
            <Typography.Text strong style={{ color: '#262626' }}>已选 {selectedDocIds.length} 个文档</Typography.Text>
            <GlassButton variant="primary" icon={<PlayCircleOutlined />} onClick={handleBatchParse}>
              批量解析
            </GlassButton>
            <Popconfirm
              title={`确定删除选中的 ${selectedDocIds.length} 个文档？`}
              okText="删除"
              cancelText="取消"
              disabled={selectedDocIds.length === 0}
              onConfirm={handleBatchDelete}
            >
              <GlassButton variant="danger" icon={<DeleteOutlined />} disabled={selectedDocIds.length === 0}>
                批量删除
              </GlassButton>
            </Popconfirm>
            <GlassButton variant="text" onClick={() => toggleAllDocs(true)}>全选</GlassButton>
            <GlassButton variant="text" onClick={() => toggleAllDocs(false)}>取消全选</GlassButton>
            <GlassButton variant="text" onClick={handleCancelBatch}>取消操作</GlassButton>
          </>
        )}
      </GlassToolbar>

      {/* 上传文件弹窗 */}
      <Modal
        title="上传文件"
        open={uploadModalOpen}
        onCancel={() => { setUploadModalOpen(false); setUploadFileList([]); }}
        footer={[
          <Button key="cancel" onClick={() => { setUploadModalOpen(false); setUploadFileList([]); }}>
            取消
          </Button>,
          <GlassButton
            key="confirm"
            variant="primary"
            icon={<CloudUploadOutlined />}
            loading={uploading}
            disabled={uploadFileList.length === 0}
            onClick={handleConfirmUpload}
          >
            {uploading ? '上传中…' : `保存并上传${uploadFileList.length > 0 ? `（${uploadFileList.length}）` : ''}`}
          </GlassButton>,
        ]}
        width={520}
        okText="保存"
        cancelText="取消"
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {/* 上传区：点击选择 + 拖拽 */}
          <AntdUpload.Dragger
            multiple
            accept=".docx,.txt,.md,.doc,.pdf,.xlsx,.pptx"
            beforeUpload={() => false}
            fileList={[]}
            style={{ padding: '12px 8px' }}
            onChange={({ fileList: fl }: { fileList: UploadFile[] }) => {
              const newFiles = fl
                .map((f) => f.originFileObj as File)
                .filter(Boolean);
              if (newFiles.length > 0) handleUploadModalAdd(newFiles);
            }}
          >
            <p className="ant-upload-drag-icon">
              <InboxOutlined style={{ fontSize: 42, color: 'rgba(0,0,0,0.4)' }} />
            </p>
            <p className="ant-upload-text">点击或拖拽文件到此处</p>
            <p className="ant-upload-hint" style={{ fontSize: 12 }}>
              支持单个或批量上传，格式：.docx .txt .md .pdf .xlsx .pptx
            </p>
          </AntdUpload.Dragger>

          {/* 文件列表预览 */}
          {uploadFileList.length > 0 && (
            <div style={{ marginTop: 8 }}>
              <Typography.Text strong style={{ fontSize: 13 }}>
                已选文件（{uploadFileList.length} 个）
              </Typography.Text>
              <div style={{ maxHeight: 200, overflow: 'auto', marginTop: 6, display: 'flex', flexDirection: 'column', gap: 4 }}>
                {uploadFileList.map((f, i) => (
                  <div
                    key={i}
                    className="glass-dense"
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      padding: '6px 10px',
                      borderRadius: 8,
                      border: '1px solid rgba(0,0,0,0.12)',
                      boxShadow: '0 2px 6px rgba(0,0,0,0.05)',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0, flex: 1 }}>
                      <FileOutlined style={{ color: 'rgba(0,0,0,0.5)', flexShrink: 0 }} />
                      <div style={{ minWidth: 0, flex: 1 }}>
                        <div style={{ fontSize: 13, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                          {f.name}
                        </div>
                        <div style={{ fontSize: 11, color: '#8c8c8c' }}>
                          {(f.size / 1024).toFixed(1)} KB
                        </div>
                      </div>
                    </div>
                    <GlassButton
                      variant="text"
                      size="small"
                      icon={<DeleteOutlined />}
                      onClick={() => handleUploadModalRemove(i)}
                    />
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </Modal>

      {/* 解析模型选择弹窗：胶囊点击后弹出卡片列表 */}
      <Modal
        title="选择解析模型"
        open={modelPickerOpen}
        onCancel={() => setModelPickerOpen(false)}
        onOk={() => setModelPickerOpen(false)}
        okText="确定"
        cancelText="取消"
        width={460}
        okButtonProps={{ disabled: models.length === 0 }}
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxHeight: 400, overflow: 'auto' }}>
          {models.length === 0 ? (
            <Empty description="暂无 Chat 模型，请先到模型管理配置" style={{ marginTop: 40 }} />
          ) : (
            <>
              {models.map((m) => {
                const selected = parseModel === m.model_id;
                return (
                  <div
                    key={m.model_id}
                    className={`agent-picker-option${selected ? ' selected' : ''}`}
                    onClick={() => setParseModel(m.model_id)}
                  >
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
              {/* 使用默认：清除选择 */}
              <div
                className={`agent-picker-option${!parseModel ? ' selected' : ''}`}
                onClick={() => setParseModel(undefined)}
              >
                <div className="agent-picker-option-name" style={{ color: '#8c8c8c' }}>使用默认模型</div>
                {!parseModel && <CheckOutlined style={{ color: '#8c8c8c', flexShrink: 0 }} />}
              </div>
            </>
          )}
        </div>
      </Modal>

      {/* Dify 对接弹窗 */}
      <Modal
        title="Dify 对接信息"
        open={difyModalOpen}
        onCancel={() => setDifyModalOpen(false)}
        footer={[
          <Button key="close" type="primary" onClick={() => setDifyModalOpen(false)}>
            知道了
          </Button>,
        ]}
        width={480}
      >
        {difyLoading ? (
          <div style={{ textAlign: 'center', padding: 24 }}>加载中…</div>
        ) : difyInfo ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div
              className="glass-dense"
              style={{ padding: 12, borderRadius: 10, border: '1px solid rgba(0,0,0,0.15)', boxShadow: '0 2px 8px rgba(0,0,0,0.06)' }}
            >
              <Typography.Text type="secondary" style={{ fontSize: 12 }}>API 地址</Typography.Text>
              <div style={{ wordBreak: 'break-all', marginTop: 4 }}>
                <Typography.Text copyable code>
                  http://{difyInfo.lan_ip}:9999/api/v1/dify/retrieval
                </Typography.Text>
              </div>
            </div>
            <div
              className="glass-dense"
              style={{ padding: 12, borderRadius: 10, border: '1px solid rgba(0,0,0,0.15)', boxShadow: '0 2px 8px rgba(0,0,0,0.06)' }}
            >
              <Typography.Text type="secondary" style={{ fontSize: 12 }}>知识库 ID</Typography.Text>
              <div style={{ marginTop: 4 }}>
                <Typography.Text copyable code>{kbId}</Typography.Text>
              </div>
            </div>
            <Alert
              type="info"
              showIcon
              message="外部知识库 API"
              description="请将上述信息填入 Dify 外部知识库配置中，选择 Compatible Knowledge API 类型"
            />
          </div>
        ) : (
          <div style={{ textAlign: 'center', padding: 24, color: '#8c8c8c' }}>
            获取 Dify 信息失败
          </div>
        )}
      </Modal>

      {/* 解析日志弹窗 */}
      <Modal
        title={`解析日志 - ${logModal?.title || ''}`}
        open={logModal?.open}
        footer={null}
        onCancel={() => setLogModal(null)}
        width={640}
      >
        <Progress percent={logContent.progress} />
        <div
          style={{
            maxHeight: 400,
            overflow: 'auto',
            background: 'rgba(31,45,78,0.04)',
            borderRadius: 8,
            padding: 12,
            fontFamily: 'monospace',
            fontSize: 12,
            whiteSpace: 'pre-wrap',
            marginTop: 8,
          }}
        >
          {logContent.logs.length === 0 ? '暂无日志' : logContent.logs.map((l, i) => <div key={i}>{l}</div>)}
        </div>
      </Modal>
    </div>
  );
}
