import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Card,
  Empty,
  Input,
  Modal,
  Progress,
  Radio,
  Select,
  Space,
  Spin,
  Tag,
  Timeline,
  Typography,
  message,
} from 'antd';
import { FileTextOutlined, ReloadOutlined, SearchOutlined } from '@ant-design/icons';
import { fetchIcpDocuments } from '@/api/icp';
import { fetchLogEntries, fetchLogTasks } from '@/api/kb';
import type { LogEntry, LogTask } from '@/types';
import { formatDateTime } from '@/utils';
import { GlassButton } from '@/components/GlassUI';

const TASK_STATUS_LABEL: Record<string, string> = {
  pending: '等待中',
  queued: '排队中',
  running: '进行中',
  completed: '已完成',
  failed: '失败',
  cancelled: '已取消',
};

const taskTypeLabel = (taskType?: string) => (taskType === 'parse' ? '文档解析' : '关系数据');
const taskStatusLabel = (status?: string) => TASK_STATUS_LABEL[status || ''] || status || '未知';

const taskProgressStatus = (status?: string): 'normal' | 'exception' | 'active' | 'success' => {
  if (status === 'failed') return 'exception';
  if (status === 'running') return 'active';
  if (status === 'completed') return 'success';
  return 'normal';
};

interface ParseRecordsPanelProps {
  docType: string;
  documentId?: string;
  embedded?: boolean;
  onShowAll?: () => void;
}

export default function ParseRecordsPanel({
  docType,
  documentId,
  embedded = false,
  onShowAll,
}: ParseRecordsPanelProps) {
  const [taskType, setTaskType] = useState('');
  const [status, setStatus] = useState('');
  const [search, setSearch] = useState('');
  const [tasks, setTasks] = useState<LogTask[]>([]);
  const [loading, setLoading] = useState(false);
  const [docNameMap, setDocNameMap] = useState<Record<string, string>>({});
  const [detail, setDetail] = useState<{
    open: boolean;
    title: string;
    summary?: string | Record<string, unknown>;
    type: 'parse' | 'kg';
    logs: LogEntry[];
    docName?: string;
    status?: string;
    startTime?: string;
    endTime?: string;
  } | null>(null);
  const [logFilter, setLogFilter] = useState('all');
  const [logSearch, setLogSearch] = useState('');
  const debounceRef = useRef<ReturnType<typeof setTimeout>>();

  const loadTasks = useCallback(async () => {
    if (!docType) return;
    setLoading(true);
    try {
      const [res, docs] = await Promise.all([
        fetchLogTasks(docType),
        fetchIcpDocuments(docType).catch(() => []),
      ]);
      const nameMap: Record<string, string> = {};
      (docs || []).forEach((document) => {
        if (document.doc_id) nameMap[document.doc_id] = document.file_name;
      });
      setDocNameMap(nameMap);

      let merged: LogTask[] = (res.tasks || []).map((task) => ({
        task_type: task.task_type || 'parse',
        task_id: task.task_id,
        kb_id: task.kb_id,
        doc_id: task.doc_id,
        doc_name: task.doc_name,
        status: task.status,
        start_time: task.start_time,
        end_time: task.end_time,
        progress: task.progress,
        message: task.message,
      }));
      merged = merged.sort((a, b) => (
        (b.end_time || b.start_time || '').localeCompare(a.end_time || a.start_time || '')
      ));
      if (documentId) merged = merged.filter((task) => task.doc_id === documentId);
      if (taskType) merged = merged.filter((task) => task.task_type === taskType);
      if (status) merged = merged.filter((task) => task.status === status);
      if (search.trim()) {
        const keyword = search.trim().toLowerCase();
        merged = merged.filter((task) => (
          (nameMap[task.doc_id] || task.doc_name || '').toLowerCase().includes(keyword)
          || (task.doc_id || '').toLowerCase().includes(keyword)
          || (task.task_id || '').toLowerCase().includes(keyword)
        ));
      }
      setTasks(merged);
    } catch (error) {
      message.error(`加载解析记录失败：${(error as Error).message}`);
    } finally {
      setLoading(false);
    }
  }, [docType, documentId, taskType, status, search]);

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(loadTasks, 300);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [loadTasks]);

  const openDetail = async (task: LogTask) => {
    const type = task.task_type === 'parse' ? 'parse' : 'kg';
    try {
      const res = await fetchLogEntries(task.kb_id || docType, task.doc_id, task.task_id, type);
      setLogFilter('all');
      setLogSearch('');
      setDetail({
        open: true,
        title: `${type === 'parse' ? '文档解析' : '关系数据'}详情 · ${docNameMap[task.doc_id] || res.doc_name || task.doc_id}`,
        summary: res.summary,
        type,
        logs: res.logs || [],
        docName: docNameMap[task.doc_id] || res.doc_name,
        status: task.status,
        startTime: task.start_time,
        endTime: task.end_time,
      });
    } catch (error) {
      message.error(`加载记录详情失败：${(error as Error).message}`);
    }
  };

  const filteredLogs = useMemo(() => (detail?.logs || []).filter((entry) => {
    if (
      logFilter !== 'all'
      && entry.log_level?.toLowerCase() !== logFilter
      && entry.log_type !== logFilter
    ) return false;
    if (logSearch && !JSON.stringify(entry).includes(logSearch)) return false;
    return true;
  }), [detail?.logs, logFilter, logSearch]);

  const statusColor = (value?: string) => (
    value === 'completed' ? 'green'
      : value === 'failed' ? 'red'
        : value === 'running' ? 'blue'
          : value === 'cancelled' ? 'orange'
            : 'default'
  );

  const selectedDocumentName = documentId ? docNameMap[documentId] : '';
  const content = (
    <>
      {documentId ? (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14 }}>
          <Typography.Text type="secondary">当前文档</Typography.Text>
          <Tag>{selectedDocumentName || documentId}</Tag>
          {onShowAll ? (
            <GlassButton size="small" variant="text" onClick={onShowAll}>查看全部文档</GlassButton>
          ) : null}
        </div>
      ) : null}

      <div
        className="type-table-toolbar type-logs-toolbar"
        style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', marginBottom: 14 }}
      >
        <Select
          style={{ width: 150 }}
          placeholder="全部处理类型"
          allowClear
          value={taskType || undefined}
          onChange={(value) => setTaskType(value || '')}
          options={[
            { value: 'parse', label: '文档解析' },
            { value: 'kg', label: '关系数据' },
          ]}
        />
        <Select
          style={{ width: 140 }}
          placeholder="全部状态"
          allowClear
          value={status || undefined}
          onChange={(value) => setStatus(value || '')}
          options={[
            { value: 'running', label: '进行中' },
            { value: 'completed', label: '已完成' },
            { value: 'failed', label: '失败' },
            { value: 'cancelled', label: '已取消' },
          ]}
        />
        <Input
          placeholder="搜索文档"
          prefix={<SearchOutlined />}
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          allowClear
          className="type-logs-search"
          style={{ width: 240, maxWidth: '100%' }}
        />
        <span style={{ flex: 1 }} />
        <GlassButton variant="text" icon={<ReloadOutlined />} onClick={loadTasks} loading={loading}>
          刷新
        </GlassButton>
      </div>

      {loading ? (
        <div className="type-logs-loading" style={{ display: 'grid', placeItems: 'center', minHeight: 240 }}>
          <Spin size="large" />
        </div>
      ) : tasks.length === 0 ? (
        <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无解析记录" />
      ) : (
        <div className="type-log-task-list">
          {tasks.map((task) => {
            const progress = Math.max(0, Math.min(100, Number(task.progress) || 0));
            const documentName = docNameMap[task.doc_id] || task.doc_name || task.doc_id;
            return (
              <div
                key={task.task_id}
                className="type-log-task-row"
                role="button"
                tabIndex={0}
                aria-label={`查看${documentName}的${taskTypeLabel(task.task_type)}详情`}
                onClick={() => openDetail(task)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault();
                    openDetail(task);
                  }
                }}
              >
                <div className="type-log-task-header">
                  <div className="type-log-task-identity">
                    <Tag>{taskTypeLabel(task.task_type)}</Tag>
                    <div className="type-log-task-title">
                      <Typography.Text strong ellipsis title={documentName}>{documentName}</Typography.Text>
                    </div>
                  </div>
                  <div className="type-log-task-actions">
                    <Tag color={statusColor(task.status)}>{taskStatusLabel(task.status)}</Tag>
                    <GlassButton
                      size="small"
                      icon={<FileTextOutlined />}
                      onClick={(event) => {
                        event.stopPropagation();
                        openDetail(task);
                      }}
                    >
                      查看详情
                    </GlassButton>
                  </div>
                </div>

                <div className="type-log-task-progress">
                  <Progress
                    percent={progress}
                    size="small"
                    status={taskProgressStatus(task.status)}
                    showInfo={false}
                  />
                  <Typography.Text type="secondary">{progress}%</Typography.Text>
                </div>

                <div className="type-log-task-meta">
                  <Typography.Text type="secondary">
                    开始：{formatDateTime(task.start_time) || '-'}
                  </Typography.Text>
                  <Typography.Text type="secondary">
                    {task.end_time ? `结束：${formatDateTime(task.end_time)}` : '尚未结束'}
                  </Typography.Text>
                  {task.message ? (
                    <Typography.Text type="secondary" ellipsis title={task.message} className="type-log-task-message">
                      {task.message}
                    </Typography.Text>
                  ) : null}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </>
  );

  return (
    <>
      {embedded ? content : (
        <Card
          className="glass type-section-card type-logs-card"
          title={(
            <div className="type-logs-card-heading">
              <Typography.Text strong>解析记录</Typography.Text>
              <Typography.Text type="secondary" className="type-logs-card-description">
                查看文档解析过程和失败原因
              </Typography.Text>
            </div>
          )}
          extra={<Typography.Text type="secondary">共 {tasks.length} 条记录</Typography.Text>}
        >
          {content}
        </Card>
      )}

      <Modal
        title={detail?.title}
        open={detail?.open}
        footer={null}
        onCancel={() => setDetail(null)}
        width={760}
        rootClassName="parse-record-detail-modal"
      >
        {detail ? (
          <>
            <Space wrap style={{ marginBottom: 12 }}>
              <Tag color="blue">{detail.docName || '-'}</Tag>
              <Tag color={statusColor(detail.status)}>{taskStatusLabel(detail.status)}</Tag>
              <Typography.Text type="secondary">
                {formatDateTime(detail.startTime) || '-'} → {formatDateTime(detail.endTime) || '尚未结束'}
              </Typography.Text>
            </Space>
            <Space wrap style={{ marginBottom: 12 }}>
              <Radio.Group value={logFilter} onChange={(event) => setLogFilter(event.target.value)} size="small">
                <Radio.Button value="all">全部</Radio.Button>
                <Radio.Button value="error">错误</Radio.Button>
                <Radio.Button value="warning">警告</Radio.Button>
                <Radio.Button value="success">成功</Radio.Button>
                <Radio.Button value="step">步骤</Radio.Button>
              </Radio.Group>
              <Input
                placeholder="搜索记录"
                size="small"
                value={logSearch}
                onChange={(event) => setLogSearch(event.target.value)}
                style={{ width: 180 }}
              />
            </Space>
            <div style={{ maxHeight: 400, overflow: 'auto' }}>
              <Timeline
                items={filteredLogs.map((entry, index) => ({
                  key: entry.log_id || index,
                  color: entry.log_level === 'ERROR'
                    ? 'red'
                    : entry.log_level === 'WARNING'
                      ? 'orange'
                      : entry.log_type === 'step'
                        ? 'blue'
                        : 'green',
                  children: (
                    <div style={{ fontSize: 12 }}>
                      <span style={{ color: '#9aa7c3' }}>[{formatDateTime(entry.created_at)}]</span>{' '}
                      <Tag color={entry.log_level === 'ERROR' ? 'red' : entry.log_level === 'WARNING' ? 'orange' : 'blue'}>
                        {entry.log_level === 'ERROR' ? '错误' : entry.log_level === 'WARNING' ? '警告' : '信息'}
                      </Tag>
                      <span style={{ color: '#262626' }}>{entry.message}</span>
                    </div>
                  ),
                }))}
              />
            </div>
          </>
        ) : null}
      </Modal>
    </>
  );
}
