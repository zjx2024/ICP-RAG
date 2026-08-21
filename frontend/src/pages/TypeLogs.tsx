import { useEffect, useState, useCallback, useRef } from 'react';
import {
  Card, Select, Input, Button, Space, Tag, Timeline, Modal, Typography,
  Progress, message, Empty, Radio, Badge,
} from 'antd';
import { ReloadOutlined, SearchOutlined, FileTextOutlined } from '@ant-design/icons';
import { useParams } from 'react-router-dom';
import { fetchLogTasks, fetchLogEntries } from '@/api/kb';
import { fetchIcpDocuments } from '@/api/icp';
import type { LogTask, LogEntry } from '@/types';
import { formatDateTime } from '@/utils';
import { GlassButton } from '@/components/GlassUI';

export default function TypeLogs() {
  // docType 由类型框架路由提供（/type/:docType/logs）；ICP 任务日志 kb_id 即 doc_type，复用 knowledge-bases 日志接口
  const { docType = '' } = useParams<{ docType: string }>();
  const [taskType, setTaskType] = useState<string>('');
  const [status, setStatus] = useState<string>('');
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
    setLoading(true);
    try {
      const [res, docs] = await Promise.all([
        fetchLogTasks(docType),
        fetchIcpDocuments(docType).catch(() => []),
      ]);
      const nameMap: Record<string, string> = {};
      (docs || []).forEach((d) => { if (d.doc_id) nameMap[d.doc_id] = d.file_name; });
      setDocNameMap(nameMap);
      let merged: LogTask[] = (res.tasks || []).map((t) => ({
        task_type: t.task_type || 'parse',
        task_id: t.task_id,
        kb_id: t.kb_id,
        doc_id: t.doc_id,
        doc_name: t.doc_name,
        status: t.status,
        start_time: t.start_time,
        end_time: t.end_time,
        progress: t.progress,
        message: t.message,
      }));
      merged = merged.sort((a, b) => (b.end_time || '').localeCompare(a.end_time || ''));
      if (taskType) merged = merged.filter((t) => t.task_type === taskType);
      if (status) merged = merged.filter((t) => t.status === status);
      if (search) {
        merged = merged.filter(
          (t) =>
            (nameMap[t.doc_id] || t.doc_name || '').includes(search) ||
            (t.doc_id || '').includes(search) ||
            (t.task_id || '').includes(search),
        );
      }
      setTasks(merged);
    } catch (e) {
      message.error(`加载日志失败: ${(e as Error).message}`);
    } finally {
      setLoading(false);
    }
  }, [docType, taskType, status, search]);

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => loadTasks(), 300);
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current); };
  }, [docType, taskType, status, search, loadTasks]);

  const openDetail = async (t: LogTask) => {
    const type = t.task_type === 'parse' ? 'parse' : 'kg';
    try {
      const res = await fetchLogEntries(t.kb_id || docType, t.doc_id, t.task_id, type);
      setDetail({
        open: true,
        title: `${type === 'parse' ? '解析' : '图谱'}日志 - ${docNameMap[t.doc_id] || res.doc_name || t.doc_id}`,
        summary: res.summary,
        type,
        logs: res.logs || [],
        docName: docNameMap[t.doc_id] || res.doc_name,
        status: t.status,
        startTime: t.start_time,
        endTime: t.end_time,
      });
    } catch (e) {
      message.error(`加载日志失败: ${(e as Error).message}`);
    }
  };

  const filteredLogs = (detail?.logs || []).filter((l) => {
    if (logFilter !== 'all' && l.log_level?.toLowerCase() !== logFilter && l.log_type !== logFilter) return false;
    if (logSearch && !JSON.stringify(l).includes(logSearch)) return false;
    return true;
  });

  const statusColor = (s?: string) =>
    s === 'completed' ? 'green' : s === 'failed' ? 'red' : s === 'running' ? 'blue' : s === 'cancelled' ? 'orange' : 'default';

  return (
    <div className="glass-fade-in">
      <Card className="glass" style={{ marginBottom: 16 }} title="日志筛选">
        <Space wrap>
          <Select
            style={{ width: 140 }}
            placeholder="任务类型"
            allowClear
            value={taskType || undefined}
            onChange={(v) => setTaskType(v || '')}
            options={[
              { value: 'parse', label: '解析' },
              { value: 'kg', label: '图谱生成' },
            ]}
          />
          <Select
            style={{ width: 140 }}
            placeholder="状态"
            allowClear
            value={status || undefined}
            onChange={(v) => setStatus(v || '')}
            options={[
              { value: 'running', label: '运行中' },
              { value: 'completed', label: '已完成' },
              { value: 'failed', label: '失败' },
              { value: 'cancelled', label: '已取消' },
            ]}
          />
          <Input
            placeholder="搜索文档/任务"
            prefix={<SearchOutlined />}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{ width: 200 }}
          />
          <GlassButton variant="text" icon={<ReloadOutlined />} onClick={loadTasks} loading={loading}>刷新</GlassButton>
          <Badge count={tasks.length}><Typography.Text>共 {tasks.length} 个任务</Typography.Text></Badge>
        </Space>
      </Card>

      <Card className="glass" title={`任务列表 (${tasks.length})`}>
        {loading ? (
          <div style={{ textAlign: 'center', padding: 40 }}><Progress percent={100} status="active" /></div>
        ) : tasks.length === 0 ? (
          <Empty description="暂无任务日志（解析/图谱任务会记录于此）" />
        ) : (
          <div style={{ maxHeight: 600, overflow: 'auto' }}>
            {tasks.map((t) => (
              <div
                key={t.task_id}
                className="glass"
                style={{ padding: 12, marginBottom: 10, cursor: 'pointer' }}
                onClick={() => openDetail(t)}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Space>
                    <Tag color={t.task_type === 'parse' ? 'blue' : 'purple'}>
                      {t.task_type === 'parse' ? '解析' : '图谱'}
                    </Tag>
                    <Typography.Text strong>{docNameMap[t.doc_id] || t.doc_name || t.doc_id}</Typography.Text>
                    <Tag color={statusColor(t.status)}>{t.status}</Tag>
                  </Space>
                  <GlassButton size="small" icon={<FileTextOutlined />} onClick={(e) => { e.stopPropagation(); openDetail(t); }}>日志</GlassButton>
                </div>
                <Progress percent={t.progress || 0} size="small" status={t.status === 'failed' ? 'exception' : undefined} />
                <div style={{ fontSize: 12, color: '#595959' }}>
                  {formatDateTime(t.start_time)} → {formatDateTime(t.end_time)}
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>

      <Modal
        title={detail?.title}
        open={detail?.open}
        footer={null}
        onCancel={() => setDetail(null)}
        width={760}
      >
        {detail && (
          <>
            <Space wrap style={{ marginBottom: 12 }}>
              <Tag color="blue">{detail.docName || '-'}</Tag>
              <Tag color={statusColor(detail.status)}>{detail.status}</Tag>
              <Typography.Text type="secondary">{formatDateTime(detail.startTime)} → {formatDateTime(detail.endTime)}</Typography.Text>
            </Space>
            {detail.summary && (
              <div className="glass-dense" style={{ padding: 10, marginBottom: 12 }}>
                {typeof detail.summary === 'string' ? (
                  <>
                    <Typography.Text strong>汇总: </Typography.Text>
                    <Typography.Text>{detail.summary}</Typography.Text>
                  </>
                ) : (
                  <div style={{ fontSize: 12 }}>
                    <Typography.Text strong>汇总</Typography.Text>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 6 }}>
                      {Object.entries(detail.summary as Record<string, unknown>).map(([k, v]) => (
                        <Tag key={k} style={{ margin: 0 }}>{k}: {String(v ?? '')}</Tag>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
            <Space style={{ marginBottom: 12 }}>
              <Radio.Group value={logFilter} onChange={(e) => setLogFilter(e.target.value)} size="small">
                <Radio.Button value="all">全部</Radio.Button>
                <Radio.Button value="error">错误</Radio.Button>
                <Radio.Button value="warning">警告</Radio.Button>
                <Radio.Button value="success">成功</Radio.Button>
                <Radio.Button value="step">步骤</Radio.Button>
              </Radio.Group>
              <Input
                placeholder="搜索日志"
                size="small"
                value={logSearch}
                onChange={(e) => setLogSearch(e.target.value)}
                style={{ width: 180 }}
              />
            </Space>
            <div style={{ maxHeight: 400, overflow: 'auto' }}>
              <Timeline>
                {filteredLogs.map((l, i) => (
                  <Timeline.Item key={l.log_id || i} color={l.log_level === 'ERROR' ? 'red' : l.log_level === 'WARNING' ? 'orange' : l.log_type === 'step' ? 'blue' : 'green'}>
                    <div style={{ fontSize: 12 }}>
                      <span style={{ color: '#9aa7c3' }}>[{formatDateTime(l.created_at)}]</span>{' '}
                      <Tag color={l.log_level === 'ERROR' ? 'red' : l.log_level === 'WARNING' ? 'orange' : 'blue'}>{l.log_level}</Tag>
                      <span style={{ color: '#262626' }}>{l.message}</span>
                    </div>
                  </Timeline.Item>
                ))}
              </Timeline>
            </div>
          </>
        )}
      </Modal>
    </div>
  );
}
