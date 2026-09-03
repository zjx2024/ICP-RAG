import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Alert,
  Button,
  Card,
  Drawer,
  Dropdown,
  Empty,
  Input,
  Modal,
  Progress,
  Select,
  Space,
  Table,
  Tag,
  Typography,
  message,
} from 'antd';
import type { ColumnsType } from 'antd/es/table';
import {
  CheckCircleOutlined,
  ClockCircleOutlined,
  CloudUploadOutlined,
  DeleteOutlined,
  ExclamationCircleOutlined,
  EyeOutlined,
  FileTextOutlined,
  HistoryOutlined,
  LoadingOutlined,
  MoreOutlined,
  PlayCircleOutlined,
  ReloadOutlined,
  SearchOutlined,
  StopOutlined,
} from '@ant-design/icons';
import { useNavigate, useParams, useSearchParams } from 'react-router-dom';
import { deleteIcpDocument, fetchDocumentType, fetchIcpDocuments, parseIcpDocument } from '@/api/icp';
import { cancelTask, openTaskStream } from '@/api/kb';
import { fetchCurrentModelConfig, fetchModels } from '@/api/models';
import { GlassButton } from '@/components/GlassUI';
import ParseRecordsPanel from '@/components/ParseRecordsPanel';
import UnifiedUploadModal from '@/components/UnifiedUploadModal';
import type { DocTypeInfo, ICPDocument } from '@/types';
import { DOC_STATUS_COLOR, DOC_STATUS_LABEL, formatDateTime } from '@/utils';

interface RunningTask {
  source: EventSource;
  taskId: string;
  logs: string[];
  progress: number;
}

const READY_STATUS = new Set(['indexed', 'kg_pending', 'kg_done', 'completed']);
const RUNNING_STATUS = new Set(['parsing', 'generating_kg', 'deleting']);
const ATTENTION_STATUS = new Set(['uploaded', 'failed', 'delete_failed', 'cancelled']);
const PARSEABLE_STATUS = new Set(['uploaded', 'failed', 'indexed', 'kg_pending', 'kg_done', 'completed', 'cancelled']);
const VISION_MODEL_TYPES = new Set(['function_locator', 'fta', 'rbd']);

function metadataText(document: ICPDocument) {
  const metadata = document.metadata || {};
  const values = [
    metadata.document_number,
    metadata.document_version,
    metadata.applicable_configuration,
  ].filter((value) => typeof value === 'string' && value.trim());
  return values.join(' · ');
}

export default function TypeDocs() {
  const { docType = '' } = useParams<{ docType: string }>();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const [docs, setDocs] = useState<ICPDocument[]>([]);
  const [typeInfo, setTypeInfo] = useState<DocTypeInfo | null>(null);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState('');
  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState('');
  const [models, setModels] = useState<any[]>([]);
  const [modelConfigStatus, setModelConfigStatus] = useState<any>(null);
  const [parseModel, setParseModel] = useState<string>();
  const [parseTarget, setParseTarget] = useState<ICPDocument | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<ICPDocument | null>(null);
  const [deleteLoading, setDeleteLoading] = useState(false);
  const [logModal, setLogModal] = useState<{ docId: string; title: string } | null>(null);
  const [logContent, setLogContent] = useState<{ logs: string[]; progress: number }>({ logs: [], progress: 0 });
  const runningTasks = useRef<Map<string, RunningTask>>(new Map());
  const [, forceUpdate] = useState(0);

  const supportsVisionModel = VISION_MODEL_TYPES.has(docType);
  const uploadTypes = useMemo(() => (typeInfo ? [typeInfo] : []), [typeInfo]);
  const recordsOpen = searchParams.get('records') === '1';
  const recordsDocumentId = searchParams.get('recordDoc') || undefined;

  const openParseRecords = useCallback((documentId?: string) => {
    const next = new URLSearchParams(searchParams);
    next.set('records', '1');
    if (documentId) next.set('recordDoc', documentId);
    else next.delete('recordDoc');
    setSearchParams(next, { replace: true });
  }, [searchParams, setSearchParams]);

  const closeParseRecords = useCallback(() => {
    const next = new URLSearchParams(searchParams);
    next.delete('records');
    next.delete('recordDoc');
    setSearchParams(next, { replace: true });
  }, [searchParams, setSearchParams]);

  const loadDocs = useCallback(async () => {
    if (!docType) return;
    setLoading(true);
    setLoadError('');
    try {
      setDocs(await fetchIcpDocuments(docType));
    } catch (error) {
      setLoadError((error as Error).message || '文档列表加载失败');
    } finally {
      setLoading(false);
    }
  }, [docType]);

  const loadModelsData = useCallback(async () => {
    if (!supportsVisionModel) {
      setModels([]);
      setModelConfigStatus(null);
      return;
    }
    try {
      const [modelResult, configResult] = await Promise.all([
        fetchModels(),
        fetchCurrentModelConfig(),
      ]);
      setModels((modelResult.models || []).filter((model) => model.model_type === 'llm' || !model.model_type));
      setModelConfigStatus(configResult.config_status);
    } catch {
      setModels([]);
      setModelConfigStatus(null);
    }
  }, [supportsVisionModel]);

  useEffect(() => {
    loadDocs();
    loadModelsData();
  }, [loadDocs, loadModelsData]);

  useEffect(() => {
    if (!docType) return;
    let active = true;
    setTypeInfo(null);
    fetchDocumentType(docType)
      .then((value) => { if (active) setTypeInfo(value); })
      .catch(() => { if (active) setTypeInfo(null); });
    return () => { active = false; };
  }, [docType]);

  useEffect(() => () => {
    runningTasks.current.forEach((task) => task.source.close());
    runningTasks.current.clear();
  }, []);

  const taskKey = (docId: string) => `${docType}|${docId}`;
  const getRunningTask = (docId: string) => runningTasks.current.get(taskKey(docId));

  const startEventSource = (docId: string, taskId: string) => {
    const key = taskKey(docId);
    runningTasks.current.get(key)?.source.close();

    const source = openTaskStream(taskId);
    const entry: RunningTask = { source, taskId, logs: [], progress: 0 };
    runningTasks.current.set(key, entry);

    const cleanup = () => {
      source.close();
      runningTasks.current.delete(key);
      loadDocs();
      forceUpdate((value) => value + 1);
    };

    source.onmessage = (event) => {
      let data: any;
      try { data = JSON.parse(event.data); } catch { return; }
      const now = formatDateTime(new Date().toISOString());
      if (data.type === 'progress') {
        entry.progress = Number(data.progress || 0);
        if (data.log) entry.logs.push(`[${now}] ${data.log}`);
      } else if (data.type === 'log' && data.log) {
        entry.logs.push(`[${now}] ${data.log}`);
      } else if (data.type === 'status' && data.status === 'cancelled') {
        entry.logs.push(`[${now}] 任务已取消`);
        cleanup();
      } else if (data.type === 'final') {
        cleanup();
      }
      if (logModal?.docId === docId) {
        setLogContent({ logs: [...entry.logs], progress: entry.progress });
      }
      forceUpdate((value) => value + 1);
    };
    source.onerror = cleanup;
  };

  const handleParse = async () => {
    const document = parseTarget;
    if (!document) return;
    setParseTarget(null);
    setDocs((previous) => previous.map((item) => (
      item.doc_id === document.doc_id ? { ...item, status: 'parsing' } : item
    )));
    try {
      const result = await parseIcpDocument(docType, document.doc_id, supportsVisionModel ? parseModel : undefined);
      startEventSource(document.doc_id, result.task_id);
      message.success('已开始解析');
    } catch (error) {
      setDocs((previous) => previous.map((item) => (
        item.doc_id === document.doc_id ? { ...item, status: 'failed' } : item
      )));
      message.error(`解析启动失败：${(error as Error).message}`);
    }
  };

  const handleCancelTask = async (docId: string) => {
    const task = getRunningTask(docId);
    if (!task) {
      openParseRecords(docId);
      return;
    }
    try {
      await cancelTask(task.taskId);
      message.info('已请求停止任务');
    } catch (error) {
      message.error(`停止失败：${(error as Error).message}`);
    }
  };

  const showProgress = (document: ICPDocument) => {
    const task = getRunningTask(document.doc_id);
    if (!task) {
      openParseRecords(document.doc_id);
      return;
    }
    setLogContent({ logs: [...task.logs], progress: task.progress });
    setLogModal({ docId: document.doc_id, title: document.file_name });
  };

  const handleDeleteDoc = async () => {
    if (!deleteTarget) return;
    const document = deleteTarget;
    setDeleteLoading(true);
    try {
      const task = getRunningTask(document.doc_id);
      task?.source.close();
      runningTasks.current.delete(taskKey(document.doc_id));
      await deleteIcpDocument(docType, document.doc_id);
      setDeleteTarget(null);
      message.success('文档已删除');
      loadDocs();
    } catch (error) {
      message.error(`删除失败：${(error as Error).message}`);
    } finally {
      setDeleteLoading(false);
    }
  };

  const sortedDocs = useMemo(
    () => [...docs].sort((a, b) => (b.updated_at || '').localeCompare(a.updated_at || '')),
    [docs],
  );
  const filteredDocs = useMemo(() => sortedDocs.filter((document) => {
    const keyword = search.trim().toLowerCase();
    const matchesSearch = !keyword || document.file_name.toLowerCase().includes(keyword)
      || metadataText(document).toLowerCase().includes(keyword);
    return matchesSearch && (!statusFilter || document.status === statusFilter);
  }), [sortedDocs, search, statusFilter]);

  const summary = useMemo(() => ({
    total: docs.length,
    ready: docs.filter((item) => READY_STATUS.has(item.status)).length,
    running: docs.filter((item) => RUNNING_STATUS.has(item.status)).length,
    attention: docs.filter((item) => ATTENTION_STATUS.has(item.status) || (!READY_STATUS.has(item.status) && !RUNNING_STATUS.has(item.status))).length,
  }), [docs]);

  const statusOptions = useMemo(() => Array.from(new Set(docs.map((item) => item.status))).map((status) => ({
    value: status,
    label: DOC_STATUS_LABEL[status] || status,
  })), [docs]);

  const renderActions = (document: ICPDocument) => {
    const localTask = getRunningTask(document.doc_id);
    const running = RUNNING_STATUS.has(document.status);
    const menuItems = [
      {
        key: 'logs',
        icon: <HistoryOutlined />,
        label: '查看解析记录',
      },
      ...(localTask ? [{ key: 'cancel', icon: <StopOutlined />, label: '停止当前任务', danger: true }] : []),
      {
        key: 'delete',
        icon: <DeleteOutlined />,
        label: '删除文档',
        danger: true,
        disabled: running,
      },
    ];
    return (
      <Space size={6} className="type-document-actions">
        <GlassButton
          size="small"
          icon={<EyeOutlined />}
          onClick={() => navigate(`/type/${docType}/doc/${document.doc_id}`)}
        >
          查看
        </GlassButton>
        {running ? (
          <GlassButton size="small" variant="primary" icon={<LoadingOutlined />} onClick={() => showProgress(document)}>
            进度
          </GlassButton>
        ) : PARSEABLE_STATUS.has(document.status) ? (
          <GlassButton size="small" variant="primary" icon={<PlayCircleOutlined />} onClick={() => setParseTarget(document)}>
            {(document.parse_count || 0) > 0 ? '重新解析' : '解析'}
          </GlassButton>
        ) : null}
        <Dropdown
          trigger={['click']}
          menu={{
            items: menuItems,
            onClick: ({ key, domEvent }) => {
              domEvent.stopPropagation();
              if (key === 'logs') openParseRecords(document.doc_id);
              if (key === 'cancel') handleCancelTask(document.doc_id);
              if (key === 'delete') setDeleteTarget(document);
            },
          }}
        >
          <Button type="text" size="small" icon={<MoreOutlined />} aria-label={`更多操作：${document.file_name}`} />
        </Dropdown>
      </Space>
    );
  };

  const columns: ColumnsType<ICPDocument> = [
    {
      title: '文档',
      dataIndex: 'file_name',
      render: (_value, document) => (
        <div className="type-file-cell">
          <span className="type-file-icon"><FileTextOutlined /></span>
          <div className="type-file-copy">
            <button
              type="button"
              className="type-file-link"
              onClick={() => navigate(`/type/${docType}/doc/${document.doc_id}`)}
            >
              {document.file_name}
            </button>
            <span className="type-file-meta">
              {metadataText(document) || `添加于 ${formatDateTime(document.created_at) || '-'}`}
            </span>
          </div>
        </div>
      ),
    },
    {
      title: '处理状态',
      dataIndex: 'status',
      width: 190,
      render: (status: string, document) => {
        const task = getRunningTask(document.doc_id);
        return (
          <div className="type-document-status">
            <Space size={5} wrap>
              <Tag color={DOC_STATUS_COLOR[status]}>{DOC_STATUS_LABEL[status] || status}</Tag>
              <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                解析 {document.parse_count || 0} 次
              </Typography.Text>
            </Space>
            {RUNNING_STATUS.has(status) && task ? (
              <Progress percent={task.progress} size="small" showInfo={false} />
            ) : null}
          </div>
        );
      },
    },
    {
      title: '最近更新',
      dataIndex: 'updated_at',
      width: 170,
      render: (value?: string) => <span className="type-document-time">{formatDateTime(value) || '-'}</span>,
    },
    {
      title: '操作',
      key: 'actions',
      width: 230,
      align: 'right',
      render: (_value, document) => renderActions(document),
    },
  ];

  return (
    <div className="type-page type-documents-page glass-fade-in">
      <div className="type-page-header">
        <div className="type-page-heading">
          <Typography.Title level={4}>文档</Typography.Title>
          <Typography.Text type="secondary">管理文件并查看处理结果</Typography.Text>
        </div>
        <Space size={8} wrap>
          <GlassButton icon={<HistoryOutlined />} onClick={() => openParseRecords()}>
            解析记录
          </GlassButton>
          <GlassButton icon={<CloudUploadOutlined />} onClick={() => setUploadOpen(true)}>
            上传文档
          </GlassButton>
        </Space>
      </div>

      <div className="type-summary-grid" aria-label="文档状态概览">
        <div className="type-summary-card">
          <FileTextOutlined />
          <span><strong>{summary.total}</strong><small>全部文档</small></span>
        </div>
        <div className="type-summary-card ready">
          <CheckCircleOutlined />
          <span><strong>{summary.ready}</strong><small>可用于查询</small></span>
        </div>
        <div className="type-summary-card running">
          <ClockCircleOutlined />
          <span><strong>{summary.running}</strong><small>正在处理</small></span>
        </div>
        <div className="type-summary-card attention">
          <ExclamationCircleOutlined />
          <span><strong>{summary.attention}</strong><small>需要处理</small></span>
        </div>
      </div>

      <Card
        className="glass type-section-card"
        title={<span>文档列表 <Typography.Text type="secondary">{filteredDocs.length}</Typography.Text></span>}
        styles={{ body: { padding: 0 } }}
      >
        <div className="type-table-toolbar">
          <Input
            allowClear
            prefix={<SearchOutlined />}
            placeholder="搜索文件名、编号或版本"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
          />
          <Select
            allowClear
            placeholder="全部状态"
            value={statusFilter || undefined}
            onChange={(value) => setStatusFilter(value || '')}
            options={statusOptions}
          />
          <GlassButton variant="text" icon={<ReloadOutlined />} loading={loading} onClick={loadDocs}>
            刷新
          </GlassButton>
        </div>

        {loadError ? (
          <div className="type-empty-state">
            <Empty description="文档列表加载失败">
              <GlassButton onClick={loadDocs}>重新加载</GlassButton>
            </Empty>
          </div>
        ) : (
          <>
            <div className="type-documents-desktop">
              <Table<ICPDocument>
                rowKey="doc_id"
                className="kb-doc-table type-documents-table"
                loading={loading}
                dataSource={filteredDocs}
                columns={columns}
                pagination={{
                  pageSize: 10,
                  showSizeChanger: false,
                  showTotal: (total) => `共 ${total} 份`,
                }}
                locale={{
                  emptyText: search || statusFilter
                    ? '没有符合条件的文档'
                    : '暂无文档，请上传文档',
                }}
                scroll={{ x: 920 }}
              />
            </div>

            <div className="type-documents-mobile">
              {loading ? (
                <div className="type-mobile-loading"><LoadingOutlined spin /> 正在加载</div>
              ) : filteredDocs.length === 0 ? (
                <Empty description={search || statusFilter ? '没有符合条件的文档' : '暂无文档，请上传文档'} />
              ) : filteredDocs.map((document) => {
                const task = getRunningTask(document.doc_id);
                return (
                  <article key={document.doc_id} className="type-document-card">
                    <div className="type-document-card-header">
                      <span className="type-file-icon"><FileTextOutlined /></span>
                      <div className="type-file-copy">
                        <button
                          type="button"
                          className="type-file-link"
                          onClick={() => navigate(`/type/${docType}/doc/${document.doc_id}`)}
                        >
                          {document.file_name}
                        </button>
                        <span className="type-file-meta">{metadataText(document) || formatDateTime(document.updated_at)}</span>
                      </div>
                      <Tag color={DOC_STATUS_COLOR[document.status]}>
                        {DOC_STATUS_LABEL[document.status] || document.status}
                      </Tag>
                    </div>
                    {RUNNING_STATUS.has(document.status) && task ? (
                      <Progress percent={task.progress} size="small" />
                    ) : null}
                    <div className="type-document-card-footer">
                      <span>解析 {document.parse_count || 0} 次 · {formatDateTime(document.updated_at) || '-'}</span>
                      {renderActions(document)}
                    </div>
                  </article>
                );
              })}
            </div>
          </>
        )}
      </Card>

      <UnifiedUploadModal
        open={uploadOpen}
        types={uploadTypes}
        initialDocType={docType}
        lockedDocType={docType}
        onClose={() => setUploadOpen(false)}
        onUploaded={async () => {
          await loadDocs();
        }}
      />

      <Drawer
        title="解析记录"
        open={recordsOpen}
        onClose={closeParseRecords}
        width="min(960px, 100vw)"
        rootClassName="parse-records-drawer"
        styles={{ body: { padding: 20 } }}
      >
        {recordsOpen ? (
          <ParseRecordsPanel
            docType={docType}
            documentId={recordsDocumentId}
            embedded
            onShowAll={() => openParseRecords()}
          />
        ) : null}
      </Drawer>

      <Modal
        title={(parseTarget?.parse_count || 0) > 0 ? '重新解析文档' : '解析文档'}
        open={Boolean(parseTarget)}
        onCancel={() => setParseTarget(null)}
        onOk={handleParse}
        okText="开始解析"
        cancelText="取消"
        width={520}
      >
        <Typography.Text strong>{parseTarget?.file_name}</Typography.Text>
        <Typography.Paragraph type="secondary" style={{ marginTop: 8 }}>
          处理完成后会更新该文档的查询内容和关系数据。
        </Typography.Paragraph>
        {supportsVisionModel ? (
          <div className="type-parse-model-field">
            <Typography.Text>工程图识别模型（可选）</Typography.Text>
            <Select
              value={parseModel || '__default__'}
              onChange={(value) => setParseModel(value === '__default__' ? undefined : value)}
              options={[
                { value: '__default__', label: '使用系统默认模型' },
                ...models.map((model) => ({ value: model.model_id, label: model.name })),
              ]}
            />
            {modelConfigStatus && !modelConfigStatus.chat_configured ? (
              <Alert
                type="warning"
                showIcon
                message="未配置工程图识别模型；文字和表格仍可处理，图片内容可能被跳过。"
                action={<Button size="small" onClick={() => navigate('/models')}>前往设置</Button>}
              />
            ) : null}
          </div>
        ) : null}
      </Modal>

      <Modal
        title="删除文档"
        open={Boolean(deleteTarget)}
        onCancel={() => setDeleteTarget(null)}
        onOk={handleDeleteDoc}
        okText="删除"
        cancelText="取消"
        okButtonProps={{ danger: true }}
        confirmLoading={deleteLoading}
      >
        <Typography.Paragraph>
          确定删除“{deleteTarget?.file_name}”吗？相关处理结果也会一并删除。
        </Typography.Paragraph>
      </Modal>

      <Modal
        title={`处理进度 · ${logModal?.title || ''}`}
        open={Boolean(logModal)}
        onCancel={() => setLogModal(null)}
        footer={<Button onClick={() => setLogModal(null)}>关闭</Button>}
        width={680}
      >
        <Progress percent={logContent.progress} size="small" style={{ marginBottom: 12 }} />
        <div className="type-task-log">
          {logContent.logs.length === 0 ? '正在等待任务输出…' : logContent.logs.join('\n')}
        </div>
      </Modal>
    </div>
  );
}
