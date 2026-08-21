import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Alert, Button, Card, Empty, Input, Layout, List, Popconfirm, Progress,
  Space, Spin, Tag, Tooltip, Typography, message,
} from 'antd';
import {
  DeleteOutlined, FileSearchOutlined, PlusOutlined, RobotOutlined,
  SendOutlined, StopOutlined, ToolOutlined,
} from '@ant-design/icons';
import {
  chatStream, createSession, deleteSession, fetchAgents, fetchSession,
  fetchSessions,
} from '@/api/agent';
import { GlassButton } from '@/components/GlassUI';
import { formatMarkdown, formatTimeShort } from '@/utils';
import type { AgentConfig, AgentMessage, AgentSession } from '@/types';

const { Sider, Content } = Layout;
const { TextArea } = Input;

interface ToolEvent {
  tool: string;
  count?: number;
  duration?: number;
  params?: Record<string, unknown>;
  results_preview?: Array<Record<string, unknown>>;
  error?: string;
}

const TOOL_LABEL: Record<string, string> = {
  function_locator: '功能定位',
  hybrid_search: '七类文档混合召回',
  hmc_query: 'HMC 查询',
  fmeca_query: 'FMECA 查询',
  fta_query: 'FTA 原因链',
  rbd_query: 'RBD 可靠性结构',
  mfl_query: 'MFL 维修定位',
  fault_history_query: '故障历史',
  fault_reasoning: '候选融合与置信度判断',
};

const SUGGESTIONS = [
  'ICP 上电后模板一直加载失败，重新上电仍不能恢复。',
  '设备运行几分钟后反复重启，偶尔伴随显示闪烁。',
  '查询 HMC 31810001-**-0001 的故障和维修方法。',
  'ICP 功能丧失可能由哪些原因引起？',
];

function parseSseChunk(buffer: string): { events: any[]; rest: string } {
  const parts = buffer.split('\n\n');
  const rest = parts.pop() || '';
  const events: any[] = [];
  for (const part of parts) {
    const line = part.split('\n').find((value) => value.startsWith('data:'));
    if (!line) continue;
    try {
      events.push(JSON.parse(line.slice(5).trim()));
    } catch {
      /* ignore malformed event */
    }
  }
  return { events, rest };
}

export default function Diagnosis() {
  const [agent, setAgent] = useState<AgentConfig | null>(null);
  const [sessions, setSessions] = useState<AgentSession[]>([]);
  const [sessionId, setSessionId] = useState<string>();
  const [messages, setMessages] = useState<AgentMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [streamingAnswer, setStreamingAnswer] = useState('');
  const [planText, setPlanText] = useState('');
  const [toolEvents, setToolEvents] = useState<ToolEvent[]>([]);
  const [liveCitations, setLiveCitations] = useState<any[]>([]);
  const [thinkingText, setThinkingText] = useState('');
  const abortRef = useRef<AbortController | null>(null);
  const chatBottomRef = useRef<HTMLDivElement | null>(null);

  const loadSession = useCallback(async (id: string) => {
    const detail = await fetchSession(id);
    setMessages(detail.messages || []);
    setSessionId(id);
  }, []);

  const bootstrap = useCallback(async () => {
    setLoading(true);
    try {
      const agents = await fetchAgents(true);
      const selected = agents.find((item) => item.name === 'ICP故障诊断智能体')
        || agents.find((item) => item.name.includes('ICP'))
        || agents[0];
      if (!selected) {
        setAgent(null);
        setSessions([]);
        setMessages([]);
        return;
      }
      setAgent(selected);
      let list = await fetchSessions(selected.agent_id);
      if (list.length === 0) {
        const created = await createSession(selected.agent_id, 'ICP 故障诊断');
        list = [created];
      }
      setSessions(list);
      await loadSession(list[0].id);
    } catch (error) {
      message.error(`诊断工作台初始化失败：${(error as Error).message}`);
    } finally {
      setLoading(false);
    }
  }, [loadSession]);

  useEffect(() => { bootstrap(); }, [bootstrap]);

  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages, streamingAnswer, thinkingText]);

  // formatMarkdown 生成可展开的内联引用，使用事件委托处理。
  useEffect(() => {
    const handler = (event: MouseEvent) => {
      const target = event.target as HTMLElement | null;
      const cite = target?.closest?.('.inline-cite') as HTMLElement | null;
      if (!cite) return;
      const key = cite.getAttribute('data-cite-key');
      if (!key) return;
      const body = document.getElementById(key);
      if (!body) return;
      const show = body.style.display === 'none';
      body.style.display = show ? 'block' : 'none';
      const caret = cite.querySelector('.cite-caret');
      if (caret) caret.textContent = show ? '▴' : '▾';
    };
    document.addEventListener('click', handler);
    return () => document.removeEventListener('click', handler);
  }, []);

  const currentSession = sessions.find((item) => item.id === sessionId);

  const createNewSession = async () => {
    if (!agent) return;
    const created = await createSession(agent.agent_id, '新诊断会话');
    setSessions((prev) => [created, ...prev]);
    setMessages([]);
    setSessionId(created.id);
    setToolEvents([]);
    setPlanText('');
  };

  const removeSession = async (id: string) => {
    await deleteSession(id);
    const next = sessions.filter((item) => item.id !== id);
    setSessions(next);
    if (sessionId === id) {
      if (next[0]) await loadSession(next[0].id);
      else if (agent) {
        const created = await createSession(agent.agent_id, 'ICP 故障诊断');
        setSessions([created]);
        setSessionId(created.id);
        setMessages([]);
      }
    }
  };

  const stopGeneration = () => {
    abortRef.current?.abort();
    abortRef.current = null;
    setGenerating(false);
  };

  const send = async (preset?: string) => {
    const query = (preset ?? input).trim();
    if (!query || !agent || !sessionId || generating) return;

    const tempUser: AgentMessage = {
      id: `temp-user-${Date.now()}`,
      role: 'user',
      content: query,
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, tempUser]);
    setInput('');
    setGenerating(true);
    setStreamingAnswer('');
    setLiveCitations([]);
    setToolEvents([]);
    setPlanText('');
    setThinkingText('');

    const controller = new AbortController();
    abortRef.current = controller;
    let answer = '';
    let citations: any[] = [];

    try {
      const response = await chatStream({
        agent_id: agent.agent_id,
        session_id: sessionId,
        query,
        user_id: 'default',
        chat_model_id: '',
        embedding_model_id: '',
      }, controller.signal);
      if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parsed = parseSseChunk(buffer);
        buffer = parsed.rest;
        for (const event of parsed.events) {
          if (event.type === 'plan') {
            setPlanText(event.reasoning || event.rewritten_query || '正在规划七类文档检索');
          } else if (event.type === 'thinking_delta') {
            setThinkingText((prev) => (prev + (event.content || '')).slice(-1200));
          } else if (event.type === 'tool_call') {
            setToolEvents((prev) => [...prev, event as ToolEvent]);
          } else if (event.type === 'answer_delta') {
            answer += event.content || '';
            setStreamingAnswer(answer);
          } else if (event.type === 'answer') {
            answer = event.content || answer;
            setStreamingAnswer(answer);
          } else if (event.type === 'citations') {
            citations = event.items || [];
            setLiveCitations(citations);
          } else if (event.type === 'error') {
            throw new Error(event.message || '诊断流程执行失败');
          } else if (event.type === 'done' || event.type === 'stopped') {
            // 服务端已完成持久化，稍后重新拉取会话。
          }
        }
      }

      if (answer) {
        setMessages((prev) => [...prev, {
          id: `temp-assistant-${Date.now()}`,
          role: 'assistant',
          content: answer,
          citations,
          created_at: new Date().toISOString(),
        }]);
      }
      // 后端 done 事件在消息保存后发送，重新拉取以取得真实消息 ID 与引用。
      await loadSession(sessionId);
      const refreshed = await fetchSessions(agent.agent_id);
      setSessions(refreshed);
    } catch (error) {
      if ((error as Error).name !== 'AbortError') {
        message.error(`诊断失败：${(error as Error).message}`);
      }
    } finally {
      abortRef.current = null;
      setGenerating(false);
      setStreamingAnswer('');
      setThinkingText('');
    }
  };

  const confidence = useMemo(() => {
    const event = [...toolEvents].reverse().find((item) => item.tool === 'fault_reasoning');
    const value = event?.params?.confidence;
    return typeof value === 'number' ? Math.round(value * 100) : undefined;
  }, [toolEvents]);

  if (loading) {
    return <div style={{ display: 'grid', placeItems: 'center', minHeight: '70vh' }}><Spin size="large" /></div>;
  }

  if (!agent) {
    return (
      <Card className="glass">
        <Empty
          description="未找到 ICP 故障诊断智能体。请重启后端以初始化默认智能体，或检查 PostgreSQL 连接。"
        />
      </Card>
    );
  }

  return (
    <Layout className="glass-fade-in" style={{ height: 'calc(100vh - 88px)', background: 'transparent', gap: 14 }}>
      <Sider
        width={240}
        theme="light"
        style={{ background: 'rgba(255,255,255,0.58)', borderRadius: 16, overflow: 'hidden' }}
      >
        <div style={{ padding: 16, borderBottom: '1px solid rgba(0,0,0,.06)' }}>
          <Space style={{ width: '100%', justifyContent: 'space-between' }}>
            <Typography.Text strong>诊断会话</Typography.Text>
            <Tooltip title="新建诊断会话">
              <Button type="text" size="small" icon={<PlusOutlined />} onClick={createNewSession} />
            </Tooltip>
          </Space>
        </div>
        <List
          dataSource={sessions}
          locale={{ emptyText: '暂无会话' }}
          style={{ height: 'calc(100% - 60px)', overflow: 'auto' }}
          renderItem={(item) => (
            <List.Item
              onClick={() => loadSession(item.id)}
              style={{
                cursor: 'pointer', padding: '11px 14px',
                background: item.id === sessionId ? 'rgba(22,119,255,.10)' : 'transparent',
              }}
              actions={[
                <Popconfirm key="delete" title="删除此诊断会话？" onConfirm={(event) => { event?.stopPropagation(); removeSession(item.id); }}>
                  <Button type="text" size="small" icon={<DeleteOutlined />} onClick={(event) => event.stopPropagation()} />
                </Popconfirm>,
              ]}
            >
              <List.Item.Meta
                avatar={<RobotOutlined style={{ color: item.id === sessionId ? '#1677ff' : '#8c8c8c' }} />}
                title={<Typography.Text ellipsis style={{ maxWidth: 150 }}>{item.title || '诊断会话'}</Typography.Text>}
                description={formatTimeShort(item.updated_at || item.created_at)}
              />
            </List.Item>
          )}
        />
      </Sider>

      <Content style={{ minWidth: 0, display: 'grid', gridTemplateColumns: 'minmax(440px, 1fr) 340px', gap: 14 }}>
        <Card
          className="glass"
          style={{ height: '100%', minWidth: 0 }}
          styles={{ body: { height: '100%', padding: 0, display: 'flex', flexDirection: 'column' } }}
          title={
            <Space>
              <RobotOutlined />
              <span>ICP 故障诊断</span>
              <Tag color="blue">七类文档闭环</Tag>
            </Space>
          }
          extra={<Typography.Text type="secondary">{currentSession?.title || '诊断会话'}</Typography.Text>}
        >
          <div style={{ flex: 1, minHeight: 0, overflow: 'auto', padding: '16px 20px' }}>
            {messages.length === 0 && !streamingAnswer ? (
              <div style={{ maxWidth: 720, margin: '70px auto 0' }}>
                <Typography.Title level={3}>描述你观察到的故障现象</Typography.Title>
                <Typography.Paragraph type="secondary">
                  不需要知道具体模块或故障码。系统会先做功能定位，再联合 HMC、FMECA、FTA、RBD、MFL 和历史案例检索。
                </Typography.Paragraph>
                <Space wrap>
                  {SUGGESTIONS.map((item) => (
                    <Button key={item} onClick={() => send(item)}>{item}</Button>
                  ))}
                </Space>
              </div>
            ) : null}

            {messages.map((item, index) => (
              <div
                key={item.id || index}
                style={{
                  display: 'flex', justifyContent: item.role === 'user' ? 'flex-end' : 'flex-start',
                  marginBottom: 16,
                }}
              >
                <div
                  style={{
                    maxWidth: '88%', padding: '11px 14px', borderRadius: 14,
                    background: item.role === 'user' ? '#1677ff' : 'rgba(255,255,255,.78)',
                    color: item.role === 'user' ? '#fff' : '#262626',
                    boxShadow: item.role === 'assistant' ? '0 4px 18px rgba(0,0,0,.06)' : undefined,
                  }}
                >
                  {item.role === 'assistant' ? (
                    <div dangerouslySetInnerHTML={{ __html: formatMarkdown(item.content, item.citations, index) }} />
                  ) : item.content}
                </div>
              </div>
            ))}

            {generating && (
              <div style={{ display: 'flex', justifyContent: 'flex-start', marginBottom: 16 }}>
                <div style={{ maxWidth: '88%', padding: '11px 14px', borderRadius: 14, background: 'rgba(255,255,255,.78)' }}>
                  {streamingAnswer ? (
                    <div dangerouslySetInnerHTML={{ __html: formatMarkdown(streamingAnswer, liveCitations, 99999) }} />
                  ) : (
                    <Space><Spin size="small" /><Typography.Text type="secondary">正在检索并分析七类知识…</Typography.Text></Space>
                  )}
                </div>
              </div>
            )}
            <div ref={chatBottomRef} />
          </div>

          <div style={{ padding: 14, borderTop: '1px solid rgba(0,0,0,.06)' }}>
            <Alert
              type="warning"
              showIcon
              message="诊断结论必须结合适用构型、有效文档版本和正式维修程序复核。"
              style={{ marginBottom: 10, padding: '5px 10px' }}
            />
            <Space.Compact style={{ width: '100%' }}>
              <TextArea
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onPressEnter={(event) => {
                  if (!event.shiftKey) { event.preventDefault(); send(); }
                }}
                autoSize={{ minRows: 2, maxRows: 5 }}
                placeholder="描述现象、发生阶段、告警文字、是否可复现、已执行操作等…"
                disabled={generating}
              />
              {generating ? (
                <GlassButton className="glass-btn-black" icon={<StopOutlined />} onClick={stopGeneration}>停止</GlassButton>
              ) : (
                <GlassButton variant="primary" icon={<SendOutlined />} onClick={() => send()} disabled={!input.trim()}>诊断</GlassButton>
              )}
            </Space.Compact>
          </div>
        </Card>

        <div style={{ minWidth: 0, overflow: 'auto', display: 'flex', flexDirection: 'column', gap: 12 }}>
          <Card className="glass" size="small" title={<Space><FileSearchOutlined />诊断状态</Space>}>
            <Space direction="vertical" style={{ width: '100%' }}>
              {confidence !== undefined ? (
                <>
                  <Typography.Text>当前证据置信度</Typography.Text>
                  <Progress percent={confidence} status={confidence < 65 ? 'exception' : 'active'} />
                </>
              ) : <Typography.Text type="secondary">发送故障描述后显示候选证据状态</Typography.Text>}
              {planText && <Alert type="info" showIcon message="检索规划" description={planText} />}
              {thinkingText && (
                <Typography.Paragraph type="secondary" ellipsis={{ rows: 5, expandable: true, symbol: '展开' }}>
                  {thinkingText}
                </Typography.Paragraph>
              )}
            </Space>
          </Card>

          <Card className="glass" size="small" title={<Space><ToolOutlined />工具与证据</Space>}>
            {toolEvents.length === 0 ? <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="尚无工具调用" /> : (
              <Space direction="vertical" style={{ width: '100%' }} size={10}>
                {toolEvents.map((item, index) => (
                  <Card key={`${item.tool}-${index}`} size="small" type="inner" title={TOOL_LABEL[item.tool] || item.tool}
                    extra={<Tag>{item.count ?? 0} 条</Tag>}>
                    {item.error ? <Alert type="error" message={item.error} /> : null}
                    {(item.results_preview || []).slice(0, 4).map((row, rowIndex) => (
                      <div key={rowIndex} style={{ marginBottom: 8, paddingBottom: 8, borderBottom: '1px dashed #eee' }}>
                        <Typography.Text strong>{String(row.title || row.doc_name || `结果 ${rowIndex + 1}`)}</Typography.Text>
                        {row.snippet ? (
                          <Typography.Paragraph type="secondary" ellipsis={{ rows: 2 }} style={{ margin: '3px 0 0' }}>
                            {String(row.snippet)}
                          </Typography.Paragraph>
                        ) : null}
                      </div>
                    ))}
                    {typeof item.duration === 'number' ? (
                      <Typography.Text type="secondary" style={{ fontSize: 11 }}>耗时 {item.duration}s</Typography.Text>
                    ) : null}
                  </Card>
                ))}
              </Space>
            )}
          </Card>
        </div>
      </Content>
    </Layout>
  );
}
