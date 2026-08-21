import { useEffect, useState, useCallback, useRef } from 'react';
import {
  Card, Button, Tag, Table, Space, Modal, Empty,
  Progress, Typography, message, Popconfirm, Alert,
} from 'antd';
import {
  CloudUploadOutlined, DeleteOutlined, PlayCircleOutlined, EyeOutlined,
  SyncOutlined, DownOutlined, CheckOutlined,
} from '@ant-design/icons';
import { useNavigate, useParams } from 'react-router-dom';
import {
  fetchDocumentType, fetchIcpDocuments,
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
    runningTasks.current.forEach((task) => task.source.close());
    runningTasks.current.clear();
  }, []);

  const checkModels = () => {
    if (!modelConfigStatus?.chat_configured) return '未配置 Chat LLM 模型，无法解析';
    if (!modelConfigStatus?.embedding_configured) return '未配置 Embedding 模型，无法向量化';
    return '';
  };

  const handleParse = async (doc: ICPDocument) => {
    const tip = checkModels();
    if (tip) { message.error(tip); return; }
    setDocs((prev) => prev.map((item) => (item.doc_id === doc.doc_id ? { ...item, status: 'parsing' } : item)));
    try {
      const result = await parseIcpDocument(docType, doc.doc_id, parseModel);
      startEventSource(doc.doc_id, result.task_id);
    } catch (error) {
      setDocs((prev) => prev.map((item) => (item.doc_id === doc.doc_id ? { ...item, status: 'failed' } : item)));
      message.error(`解析启动失败: ${(error as Error).message}`);
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
      forceUpdate((value) => value + 1);
    };

    source.onmessage = (event) => {
      let data: any;
      try { data = JSON.parse(event.data); } catch { return; }
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
      forceUpdate((value) => value + 1);
    };
    source.onerror = () => cleanup();
  };

  const handleCancelTask = async (docId: string) => {
    const task = runningTasks.current.get(`${docType}|${docId}`);
    if (task) await cancelTask(task.taskId);
  };

  const showLogModal = (docId: string) => {
    const task = runningTasks.current.get(`${docType}|${docId}`);
    const doc = docs.find((item) => item.doc_id === docId);
    setLogContent({ logs: task?.logs || [], progress: task?.progress || 0 });
    setLogModal({ open: true, docId, title: doc?.file_name || docId });
  };

  const handleDeleteDoc = async (docId: string) => {
    const key = `${docType}|${docId}`;
    const task = runningTasks.current.get(key);
    if (task) {
      task.source.close();
      runningTasks.current.delete(key);
    }
    try {
      await deleteIcpDocument(docType, docId);
      message.success('文档已删除');
      loadDocs(docType);
    } catch (error) {
      message.error(`删除失败: ${(error as Error).message}`);
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
      render: (value: string, record: ICPDocument) => (
        <Space direction="vertical" size={0}>
          <Tag color={DOC_STATUS_COLOR[value]}>{DOC_STATUS_LABEL[value] || value}</Tag>
          {RUNNING_STATUS.has(value) && (
            <Progress percent={getDocProgress(record.doc_id)} size="small" style={{ width: 110 }} />
          )}
        </Space>
      ),
    },
    {
      title: '解析次数',
      dataIndex: 'parse_count',
      width: 90,
      render: (value?: number) => value ?? 0,
    },
    { title: '更新时间', dataIndex: 'updated_at', width: 160, render: (value: string) => (value ? formatDateTime(value) : '-') },
    {
      title: '操作',
      width: 280,
      render: (_: unknown, record: ICPDocument) => (
        <Space size={4} className="kb-row-actions">
          {(record.status === 'uploaded' || record.status === 'failed' || record.status === 'indexed' ||
            record.status === 'kg_pending' || record.status === 'kg_done') && (
            <GlassButton variant="primary" size="small" icon={<PlayCircleOutlined />} onClick={() => handleParse(record)}>
              解析
            </GlassButton>
          )}
          {(record.status === 'parsing' || record.status === 'generating_kg') && (
            <>
              <GlassButton variant="text" size="small" onClick={() => showLogModal(record.doc_id)}>日志</GlassButton>
              <GlassButton variant="text" size="small" onClick={() => handleCancelTask(record.doc_id)}>取消</GlassButton>
            </>
          )}
          <GlassButton size="small" icon={<EyeOutlined />} onClick={() => navigate(`/type/${docType}/doc/${record.doc_id}`)}>
            查看
          </GlassButton>
          <Popconfirm title="确定删除该文档？" onConfirm={() => handleDeleteDoc(record.doc_id)}>
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
            文档管理与解析
          </Typography.Title>
          <Typography.Text type="secondary" style={{ fontSize: 12 }}>
            {typeInfo?.name || docType}
            {typeInfo?.input_formats ? ` · 支持格式: ${typeInfo.input_formats}` : ''}
          </Typography.Text>
        </div>
        <div style={{ flex: 1 }} />

        <GlassButton
          variant="text"
          icon={<CloudUploadOutlined />}
          onClick={() => navigate('/')}
          title="返回七类文档首页的统一上传入口"
        >
          统一上传入口
        </GlassButton>

        <div
          className="agent-picker agent-picker-sm"
          onClick={() => setModelPickerOpen(true)}
          style={{ width: 'auto', maxWidth: 260 }}
          title="选择解析用 Chat 模型"
        >
          <span className="agent-picker-label">解析模型</span>
          <span className="agent-picker-value">
            {parseModel ? models.find((model) => model.model_id === parseModel)?.name || '已选择' : '使用默认'}
            <DownOutlined />
          </span>
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

      <Alert
        type="info"
        showIcon
        style={{ marginBottom: 12, flexShrink: 0 }}
        message="文档上传已统一到七类文档知识库首页"
        description="当前页面只管理已归入该类型的文档，并负责解析、查看和删除。"
      />

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
          pagination={{ pageSize: 10, showSizeChanger: false, showTotal: (total) => `共 ${total} 份` }}
          size="small"
          locale={{ emptyText: '该类型暂无文档，请通过首页统一上传入口添加' }}
          scroll={{ y: 'calc(100vh - 360px)' }}
        />
      </Card>

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
              {models.map((model) => {
                const selected = parseModel === model.model_id;
                return (
                  <div
                    key={model.model_id}
                    className={`agent-picker-option${selected ? ' selected' : ''}`}
                    onClick={() => setParseModel(model.model_id)}
                  >
                    <div style={{ minWidth: 0 }}>
                      <div className="agent-picker-option-name">
                        {model.name}
                        <Tag style={{ marginLeft: 6 }}>Chat LLM</Tag>
                        {model.is_default && <Tag color="gold" style={{ marginLeft: 4 }}>★ 默认</Tag>}
                      </div>
                      <div className="agent-picker-option-sub">解析文档、表格和工程图</div>
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
