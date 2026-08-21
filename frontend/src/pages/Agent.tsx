import React, { useEffect, useState, useCallback, useRef, useMemo } from 'react';
import {
  Layout, Menu, Select, Button, Input, List, Modal, Form, Tag, Tooltip, Card,
  message, Space, Typography, Switch, InputNumber, Popconfirm, Drawer, Empty,
  Progress, Radio, Checkbox, Row, Col, Divider,
} from 'antd';
import {
  SendOutlined, StopOutlined, PlusOutlined, RobotOutlined, DeleteOutlined,
  EditOutlined, CompressOutlined, LikeOutlined, DislikeOutlined, BulbOutlined,
  DownOutlined, CheckOutlined,
} from '@ant-design/icons';
import {
  fetchAgents, fetchAgentTools, createAgent, updateAgent, deleteAgent,
  fetchAgent, fetchSessions, createSession, fetchSession, updateSession,
  deleteSession, generateSessionTitle, compressSession, bindAgentModel,
  unbindAgentModel, fetchAgentModels, chatStream, sendFeedback, fetchMemories, deleteMemory,
  clearMemories,
} from '@/api/agent';
import { fetchKnowledgeBases } from '@/api/kb';
import { fetchModels } from '@/api/models';
import { formatMarkdown, estimateTokens, tokenProgressColor, formatTimeShort } from '@/utils';
import { GlassButton } from '@/components/GlassUI';
import type { AgentConfig, AgentSession, AgentMessage, AgentToolDef } from '@/types';

// ===== localStorage agent_page_state =====
const AGENT_PAGE_STATE_KEY = 'agent_page_state';

interface PageState {
  agent_id?: string;
  chat_model_id?: string;
  embedding_model_id?: string;
  rerank_model_id?: string;
  session_id?: string;
}

function getPageState(): PageState | null {
  try {
    return JSON.parse(localStorage.getItem(AGENT_PAGE_STATE_KEY) || 'null');
  } catch {
    return null;
  }
}

function savePageState(fields: Partial<PageState>) {
  try {
    const prev = getPageState() || {};
    localStorage.setItem(AGENT_PAGE_STATE_KEY, JSON.stringify({ ...prev, ...fields }));
  } catch {
    /* ignore */
  }
}

// ===== 思考事件渲染辅助 =====
const TOOL_LABEL: Record<string, string> = {
  vector_search: '向量检索',
  kg_search: '知识图谱检索',
  sql_query: '数据库查询',
};

export default function Agent() {
  const [agents, setAgents] = useState<AgentConfig[]>([]);
  const [tools, setTools] = useState<AgentToolDef[]>([]);
  const [kbs, setKbs] = useState<any[]>([]);
  const [chatModels, setChatModels] = useState<any[]>([]);
  const [embeddingModels, setEmbeddingModels] = useState<any[]>([]);
  const [rerankModels, setRerankModels] = useState<any[]>([]);
  const [agentId, setAgentId] = useState<string>();
  const [sessions, setSessions] = useState<AgentSession[]>([]);
  const [sessionId, setSessionId] = useState<string>();
  const [messages, setMessages] = useState<AgentMessage[]>([]);
  const [inputText, setInputText] = useState('');
  const [generating, setGenerating] = useState(false);
  const [liveThinking, setLiveThinking] = useState<{ text: string; source: string }>({ text: '', source: '' });
  const [thinkingEvents, setThinkingEvents] = useState<any[]>([]);
  const [streamingContent, setStreamingContent] = useState('');
  const [streamingMsgId, setStreamingMsgId] = useState<string>();
  const [lastDoneData, setLastDoneData] = useState<{ message_id?: string; thinking_events?: unknown[]; citations?: unknown[] } | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const [agentModal, setAgentModal] = useState(false);
  const [editingAgent, setEditingAgent] = useState<AgentConfig | null>(null);
  const [memModal, setMemModal] = useState(false);
  const [memories, setMemories] = useState<any[]>([]);
  const [modelsModal, setModelsModal] = useState(false);
  const [modelsAgent, setModelsAgent] = useState<AgentConfig | null>(null);
  const [agentModels, setAgentModels] = useState<any[]>([]);
  const [manageOpen, setManageOpen] = useState(false); // 智能体管理抽屉
  const [contextInfo, setContextInfo] = useState<{ used: number; max: number; percent: number }>({ used: 0, max: 8192, percent: 0 });
  const [form] = Form.useForm();

  // 侧边栏配置项选择弹窗（替代原生下拉）：picker 标识当前打开的弹窗类型
  type PickerKind = 'agent' | 'chat' | 'embedding' | 'rerank';
  const [picker, setPicker] = useState<PickerKind | null>(null);
  const [pickerValue, setPickerValue] = useState<string>();

  // 引用展开/收起：formatMarkdown 生成 .inline-cite（data-cite-key）与隐藏的 .inline-cite-body，
  // 这里用事件委托统一处理点击切换（还原旧版 agent.html 的 toggleInlineCite 行为）
  useEffect(() => {
    const handler = (ev: MouseEvent) => {
      const target = ev.target as HTMLElement | null;
      if (!target || typeof target.closest !== 'function') return;
      const cite = target.closest('.inline-cite') as HTMLElement | null;
      if (!cite) return;
      const key = cite.getAttribute('data-cite-key');
      if (!key) return;
      const body = document.getElementById(key) as HTMLElement | null;
      if (!body) return;
      const show = body.style.display === 'none';
      body.style.display = show ? 'block' : 'none';
      const caret = cite.querySelector('.cite-caret');
      if (caret) caret.textContent = show ? '▴' : '▾';
    };
    document.addEventListener('click', handler);
    return () => document.removeEventListener('click', handler);
  }, []);

  // 侧边栏模型
  const [chatModelSel, setChatModelSel] = useState<string>();
  const [embeddingModelSel, setEmbeddingModelSel] = useState<string>();
  const [rerankModelSel, setRerankModelSel] = useState<string>();

  // ---------- 初始化 ----------
  useEffect(() => {
    (async () => {
      try {
        const [agentRes, toolRes, kbRes, chatRes, embedRes, rerankRes] = await Promise.all([
          fetchAgents(),
          fetchAgentTools(),
          fetchKnowledgeBases(),
          fetchModels('llm'),
          fetchModels('embedding'),
          fetchModels('rerank'),
        ]);
        setAgents(agentRes || []);
        setTools(toolRes || []);
        setKbs(kbRes || []);
        const chatList = (chatRes.models || []).filter((m) => m.model_type === 'llm' || !m.model_type);
        setChatModels(chatList);
        setEmbeddingModels(embedRes.models || []);
        setRerankModels(rerankRes.models || []);

        const saved = getPageState();
        let targetAgent: string | undefined;
        if (saved?.agent_id && agentRes.some((a) => a.agent_id === saved.agent_id)) {
          targetAgent = saved.agent_id;
        } else {
          targetAgent = agentRes[0]?.agent_id;
        }
        if (targetAgent) {
          setAgentId(targetAgent);
          await loadAgentModels(targetAgent, saved);
        }
      } catch (e) {
        message.error(`初始化失败: ${(e as Error).message}`);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const loadAgentModels = async (aid: string, saved?: PageState | null) => {
    const agent = await fetchAgent(aid).catch(() => null);
    if (agent?.model_roles) {
      const answer = agent.model_roles.find((r) => r.role === 'answer');
      const embedding = agent.model_roles.find((r) => r.role === 'embedding');
      const rerank = agent.model_roles.find((r) => r.role === 'rerank');
      const savedChat = saved?.chat_model_id;
      const chatModel = savedChat || answer?.model_id || agent.chat_model_id || '';
      setChatModelSel(chatModel || undefined);
      setEmbeddingModelSel(saved?.embedding_model_id || embedding?.model_id || undefined);
      setRerankModelSel(saved?.rerank_model_id || rerank?.model_id || undefined);
    }
    // 恢复会话
    const sessionRes = await fetchSessions(aid).catch(() => []);
    setSessions(sessionRes);
    const savedSession = saved?.session_id;
    if (savedSession && sessionRes.some((s) => s.id === savedSession)) {
      selectSession(savedSession, sessionRes);
    } else if (sessionRes.length > 0) {
      selectSession(sessionRes[0].id, sessionRes);
    }
  };

  // 侧边栏切换智能体
  const onAgentChange = async (aid: string) => {
    setAgentId(aid);
    savePageState({ agent_id: aid });
    setMessages([]);
    setSessionId(undefined);
    const agent = await fetchAgent(aid).catch(() => null);
    if (agent?.model_roles) {
      const answer = agent.model_roles.find((r) => r.role === 'answer');
      const embedding = agent.model_roles.find((r) => r.role === 'embedding');
      const rerank = agent.model_roles.find((r) => r.role === 'rerank');
      setChatModelSel(answer?.model_id || agent.chat_model_id || undefined);
      setEmbeddingModelSel(embedding?.model_id || undefined);
      setRerankModelSel(rerank?.model_id || undefined);
    }
    const list = await fetchSessions(aid).catch(() => []);
    setSessions(list);
    if (list.length > 0) selectSession(list[0].id, list);
  };

  // 侧边栏模型切换 → 写库
  const onChatModelChange = async (mid?: string) => {
    if (!agentId || !mid) return;
    setChatModelSel(mid);
    savePageState({ chat_model_id: mid });
    for (const role of ['plan', 'reflect', 'answer']) {
      try {
        await bindAgentModel(agentId, role, mid, true);
      } catch (e) {
        message.error(`绑定 ${role} 模型失败: ${(e as Error).message}`);
      }
    }
  };

  const onEmbeddingModelChange = async (mid?: string) => {
    if (!agentId) return;
    setEmbeddingModelSel(mid);
    savePageState({ embedding_model_id: mid });
    if (mid) {
      await bindAgentModel(agentId, 'embedding', mid, true);
    } else {
      await unbindAgentModel(agentId, 'embedding');
    }
  };

  const onRerankModelChange = async (mid?: string) => {
    if (!agentId) return;
    setRerankModelSel(mid);
    savePageState({ rerank_model_id: mid });
    if (mid) {
      await bindAgentModel(agentId, 'rerank', mid, true);
    } else {
      await unbindAgentModel(agentId, 'rerank');
    }
  };

  // ---------- 侧边栏配置项选择弹窗（智能体/对话模型/Embedding/Rerank） ----------
  const openPicker = (kind: PickerKind) => {
    const current =
      kind === 'agent' ? agentId
      : kind === 'chat' ? chatModelSel
      : kind === 'embedding' ? embeddingModelSel
      : rerankModelSel;
    setPicker(kind);
    setPickerValue(current);
  };

  const confirmPicker = async () => {
    if (!picker) return;
    const v = pickerValue;
    if (picker === 'agent') {
      if (v) await onAgentChange(v);
    } else if (picker === 'chat') {
      await onChatModelChange(v);
    } else if (picker === 'embedding') {
      await onEmbeddingModelChange(v);
    } else {
      await onRerankModelChange(v);
    }
    setPicker(null);
  };

  // ---------- 会话 ----------
  const selectSession = async (sid: string, sessionList?: AgentSession[]) => {
    setSessionId(sid);
    savePageState({ session_id: sid });
    const data = await fetchSession(sid).catch(() => null);
    if (data) {
      setMessages(data.messages || []);
      contextInfoFromMessages(data.messages || []);
    }
    if (sessionList) setSessions(sessionList);
  };

  const createNewSession = async () => {
    if (!agentId) { message.warning('请先选择智能体'); return; }
    const s = await createSession(agentId);
    setSessions((prev) => [s, ...prev]);
    await selectSession(s.id);
  };

  const removeSession = async (sid: string) => {
    await deleteSession(sid);
    setSessions((prev) => prev.filter((s) => s.id !== sid));
    if (sid === sessionId) {
      setMessages([]);
      setSessionId(undefined);
      setStreamingContent('');
      setThinkingEvents([]);
    }
  };

  const renameSession = async (sid: string, oldTitle: string) => {
    let title = oldTitle;
    Modal.confirm({
      title: '重命名会话',
      content: <Input defaultValue={oldTitle} onChange={(e) => (title = e.target.value)} />,
      onOk: async () => {
        await updateSession(sid, { title });
        setSessions((prev) => prev.map((s) => (s.id === sid ? { ...s, title } : s)));
      },
    });
  };

  const contextInfoFromMessages = (msgs: AgentMessage[]) => {
    const used = msgs.reduce((sum, m) => sum + estimateTokens(m.content || ''), 0);
    const max = 8192;
    setContextInfo({ used, max, percent: Math.min(100, Math.round((used / max) * 100)) });
  };

  // ---------- 发送消息（SSE 流式） ----------
  const sendMessage = async () => {
    const text = inputText.trim();
    if (!text || !agentId || !sessionId) {
      if (!sessionId) message.warning('请先创建会话');
      return;
    }
    setInputText('');
    // push 用户消息
    setMessages((prev) => [...prev, { id: 'temp-user-' + Date.now(), role: 'user', content: text }]);
    setGenerating(true);
    setStreamingContent('');
    setThinkingEvents([]);
    setLiveThinking({ text: '', source: '' });
    setLastDoneData(null);
    const abortController = new AbortController();
    abortRef.current = abortController;
    let finalAnswer = '';
    const accThinking: any[] = [];
    let liveText = '';
    let liveSource = '';

    try {
      const resp = await chatStream(
        {
          agent_id: agentId,
          session_id: sessionId,
          query: text,
          user_id: 'default',
          chat_model_id: '',
          embedding_model_id: '',
        },
        abortController.signal,
      );
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`);
      }
      const reader = resp.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split('\n\n');
        buffer = parts.pop() || '';
        for (const part of parts) {
          for (const line of part.split('\n')) {
            if (!line.startsWith('data: ')) continue;
            let data: any;
            try {
              data = JSON.parse(line.slice(6));
            } catch {
              continue;
            }
            switch (data.type) {
              case 'answer_delta': {
                finalAnswer += data.content || '';
                setStreamingContent(finalAnswer);
                break;
              }
              case 'answer': {
                finalAnswer = data.content || '';
                setStreamingContent(finalAnswer);
                break;
              }
              case 'thinking_delta': {
                liveText += data.content || '';
                liveSource = data.source || 'plan';
                setLiveThinking({ text: liveText, source: liveSource });
                break;
              }
              case 'plan':
              case 'reflect':
              case 'tool_call': {
                if (liveText) {
                  accThinking.push({ type: 'thinking_text', source: liveSource, content: liveText });
                  liveText = '';
                }
                accThinking.push(data);
                setThinkingEvents([...accThinking]);
                break;
              }
              case 'done': {
                if (liveText) accThinking.push({ type: 'thinking_text', source: liveSource, content: liveText });
                setLastDoneData({ message_id: data.message_id, thinking_events: data.thinking_events, citations: data.citations });
                break;
              }
              case 'error': {
                throw new Error(data.message || '未知错误');
              }
              case 'stopped': {
                if (liveText) accThinking.push({ type: 'thinking_text', source: liveSource, content: liveText });
                setLastDoneData({ message_id: data.message_id, thinking_events: data.thinking_events, citations: data.citations });
                message.info('已停止生成');
                break;
              }
            }
          }
        }
      }
      // 结束
      if (finalAnswer) {
        const msg: AgentMessage = {
          id: lastDoneData?.message_id || 'temp-' + Date.now(),
          role: 'assistant',
          content: finalAnswer,
          thinking_events: lastDoneData?.thinking_events || accThinking,
          citations: (lastDoneData?.citations as any) || undefined,
        };
        setMessages((prev) => [...prev, msg]);
        contextInfoFromMessages([...messages, { id: 'u', role: 'user' as const, content: text }, msg]);
        // 首条消息生成标题
        if (messages.filter((m) => m.role === 'user').length === 0) {
          generateSessionTitle(sessionId).catch(() => {});
        }
      }
      await loadSessionsOnly();
    } catch (e: any) {
      if (e.name === 'AbortError') {
        if (finalAnswer && lastDoneData?.message_id) {
          setMessages((prev) => [
            ...prev,
            { id: lastDoneData.message_id!, role: 'assistant', content: finalAnswer },
          ]);
        } else {
          setTimeout(() => selectSession(sessionId), 500);
        }
      } else {
        message.error(`对话出错: ${e.message}`);
      }
    } finally {
      setGenerating(false);
      abortRef.current = null;
    }
  };

  const loadSessionsOnly = async () => {
    if (agentId) {
      const list = await fetchSessions(agentId).catch(() => []);
      setSessions(list);
    }
  };

  const stopGeneration = () => {
    abortRef.current?.abort();
    setGenerating(false);
  };

  // ---------- 智能体 CRUD ----------
  const openCreateAgent = () => {
    setEditingAgent(null);
    form.resetFields();
    form.setFieldsValue({
      name: '',
      system_prompt: '',
      enabled_tools: (tools || []).map((t) => t.name),
      temperature: 0.7,
      top_p: 0.1,
      max_tokens: 8192,
      max_reflections: 3,
      retrieval_top_k: 10,
      kg_top_k: 5,
      sql_limit: 10,
      summary_threshold: 60,
      keep_recent_turns: 6,
      memory_enabled: true,
    });
    setAgentModal(true);
  };

  const openEditAgent = (a: AgentConfig) => {
    setEditingAgent(a);
    form.setFieldsValue({
      name: a.name,
      description: a.description,
      system_prompt: a.system_prompt,
      kb_id: a.kb_id,
      kb_ids: a.kb_ids ?? (a.kb_id ? [a.kb_id] : []),
      enabled_tools: a.enabled_tools,
      temperature: a.temperature ?? 0.7,
      top_p: a.top_p ?? 0.1,
      max_tokens: a.max_tokens ?? 8192,
      max_reflections: a.max_reflections ?? 3,
      retrieval_top_k: a.retrieval_top_k ?? 10,
      kg_top_k: a.kg_top_k ?? 5,
      sql_limit: a.sql_limit ?? 10,
      summary_threshold: a.summary_threshold ?? 60,
      keep_recent_turns: a.keep_recent_turns ?? 6,
      chat_model_id: a.chat_model_id,
      embedding_model_id: a.embedding_model_id,
      rerank_model_id: a.rerank_model_id,
      memory_enabled: a.memory_enabled ?? true,
    });
    setAgentModal(true);
  };

  // 从管理抽屉进入编辑/模型绑定：先收起抽屉，等抽屉关闭动画结束后再打开弹窗，
  // 避免弹窗叠在半收起的抽屉上，造成"管理页面没有缩回去"的观感
  const openEditFromManage = (a: AgentConfig) => {
    setManageOpen(false);
    setTimeout(() => openEditAgent(a), 300);
  };
  const openModelsFromManage = (a: AgentConfig) => {
    setManageOpen(false);
    setTimeout(() => showModelsModal(a), 300);
  };

  const saveAgent = async () => {
    const values = await form.validateFields();
    const payload: Record<string, unknown> = {
      name: values.name,
      description: values.description,
      system_prompt: values.system_prompt,
      kb_id: values.kb_ids?.[0],
      kb_ids: values.kb_ids,
      enabled_tools: values.enabled_tools,
      temperature: values.temperature,
      top_p: values.top_p,
      max_tokens: values.max_tokens,
      max_reflections: values.max_reflections,
      retrieval_top_k: values.retrieval_top_k,
      kg_top_k: values.kg_top_k,
      sql_limit: values.sql_limit,
      summary_threshold: values.summary_threshold,
      keep_recent_turns: values.keep_recent_turns,
      memory_enabled: values.memory_enabled,
    };
    try {
      if (editingAgent) {
        await updateAgent(editingAgent.agent_id, payload);
        message.success('智能体已更新');
      } else {
        const created = await createAgent(payload);
        message.success('智能体已创建');
        // 绑定模型
        if (values.chat_model_id) {
          for (const role of ['plan', 'reflect', 'answer']) {
            await bindAgentModel(created.agent_id, role, values.chat_model_id, true).catch(() => {});
          }
        }
        if (values.embedding_model_id) {
          await bindAgentModel(created.agent_id, 'embedding', values.embedding_model_id, true).catch(() => {});
        }
        if (values.rerank_model_id) {
          await bindAgentModel(created.agent_id, 'rerank', values.rerank_model_id, true).catch(() => {});
        }
      }
      setAgentModal(false);
      const list = await fetchAgents();
      setAgents(list);
    } catch (e) {
      message.error(`保存失败: ${(e as Error).message}`);
    }
  };

  const removeAgent = async (aid: string) => {
    await deleteAgent(aid);
    const list = await fetchAgents();
    setAgents(list);
    if (aid === agentId) setAgentId(list[0]?.agent_id);
  };

  // ---------- 模型绑定弹窗 ----------
  const showModelsModal = async (a: AgentConfig) => {
    setModelsAgent(a);
    const res = await fetchAgentModels(a.agent_id).catch(() => []);
    setAgentModels(res);
    setModelsModal(true);
  };

  const bindModel = async (role: string, mid: string) => {
    if (!modelsAgent) return;
    await bindAgentModel(modelsAgent.agent_id, role, mid, true);
    const res = await fetchAgentModels(modelsAgent.agent_id);
    setAgentModels(res);
    message.success('模型已绑定');
  };

  const unbindModel = async (role: string) => {
    if (!modelsAgent) return;
    await unbindAgentModel(modelsAgent.agent_id, role);
    const res = await fetchAgentModels(modelsAgent.agent_id);
    setAgentModels(res);
  };

  // ---------- 长期记忆 ----------
  const openMemory = async () => {
    const res = await fetchMemories().catch(() => []);
    setMemories(res || []);
    setMemModal(true);
  };

  // 长期记忆弹窗内：切换当前智能体的记忆开关
  const toggleCurrentAgentMemory = async (v: boolean) => {
    if (!agentId) return;
    const cur = agents.find((a) => a.agent_id === agentId);
    if (!cur) return;
    try {
      const updated = await updateAgent(agentId, { memory_enabled: v });
      setAgents((prev) => prev.map((x) => (x.agent_id === agentId ? { ...x, ...updated } : x)));
      message.success(`「${cur.name}」长期记忆已${v ? '开启' : '关闭'}`);
    } catch (e) {
      message.error(`操作失败: ${(e as Error).message}`);
    }
  };

  // ---------- 反馈 ----------
  const feedback = async (mid: string, fb: 'like' | 'dislike') => {
    if (mid.startsWith('temp-')) { message.info('临时消息不可反馈'); return; }
    await sendFeedback(mid, fb);
    setMessages((prev) => prev.map((m) => (m.id === mid ? { ...m, feedback: fb } : m)));
  };

  // ---------- 渲染 ----------
  const renderThinkingEvent = (e: any, idx: number) => {
    if (e.type === 'thinking_text') {
      const label = e.source === 'plan' ? '📋 规划思考' : e.source === 'reflect' ? '🤔 反思思考' : '💭 思考';
      return (
        <div key={idx} style={{ marginBottom: 8 }}>
          <Tag color="blue">{label}</Tag>
          <div style={{ whiteSpace: 'pre-wrap', fontSize: 12, color: '#595959' }}>{e.content}</div>
        </div>
      );
    }
    if (e.type === 'plan') {
      return (
        <div key={idx} style={{ marginBottom: 8 }}>
          <Tag color="purple">📝 规划</Tag>
          <div style={{ fontSize: 12 }}>
            {e.intent && <div>意图: {e.intent}</div>}
            {e.rewritten_query && <div>改写查询: {e.rewritten_query}</div>}
            {e.entity_keywords?.length > 0 && <div>实体关键词: {e.entity_keywords.join(', ')}</div>}
            {e.type_keywords?.length > 0 && <div>类型关键词: {e.type_keywords.join(', ')}</div>}
            {e.reasoning && <div style={{ color: '#595959' }}>{e.reasoning}</div>}
          </div>
        </div>
      );
    }
    if (e.type === 'tool_call') {
      // 复用移植自旧版 agent.html renderToolCallEvent 的详细渲染（query/条数/耗时/文档/实体/社区报告/表格预览）
      return (
        <div key={idx} style={{ marginBottom: 8 }}>
          <ToolCallEventView e={e} />
        </div>
      );
    }
    if (e.type === 'reflect') {
      const actionLabel: Record<string, string> = {
        done: '✅ 完成',
        retry_document: '🔄 重试文档检索',
        retry_graph: '🔄 重试图谱检索',
        retry_table: '🔄 重试数据库查询',
      };
      return (
        <div key={idx} style={{ marginBottom: 8 }}>
          <Tag color="orange">🤔 反思</Tag>
          <div style={{ fontSize: 12 }}>
            <div>是否足够: {e.need_more ? '是' : '否'} · 下一步: {actionLabel[e.next_action] || e.next_action}</div>
            {e.tried_tools?.length > 0 && <div>已尝试: {e.tried_tools.join(', ')}</div>}
            {e.result_summary && (
              <details style={{ marginTop: 4 }}>
                <summary style={{ cursor: 'pointer', fontSize: 11, color: '#595959' }}>查看结果摘要</summary>
                <pre style={{ marginTop: 4, padding: 6, background: 'rgba(255,255,255,0.6)', borderRadius: 4, whiteSpace: 'pre-wrap', fontSize: 11, color: '#555' }}>{e.result_summary}</pre>
              </details>
            )}
            {e.reflection && <div style={{ color: '#595959' }}>{e.reflection}</div>}
          </div>
        </div>
      );
    }
    return null;
  };

  const renderMessage = (msg: AgentMessage, idx: number) => {
    if (msg.role === 'user') {
      return (
        <div key={msg.id || idx} style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 12 }}>
          <div className="glass-dense agent-bubble" style={{ maxWidth: '75%', padding: '10px 14px', background: 'rgba(0,0,0,0.06)' }}>
            <Typography.Text style={{ color: '#262626' }}>{msg.content}</Typography.Text>
          </div>
        </div>
      );
    }
    return (
      <div key={msg.id || idx} style={{ marginBottom: 16 }}>
        <Space align="start" size={8}>
          <AvatarIcon />
          <div style={{ flex: 1 }}>
            {msg.thinking_events && msg.thinking_events.length > 0 && (
              <CollapsibleThinking events={msg.thinking_events as any[]} />
            )}
            <div
              className="glass agent-bubble"
              style={{ padding: '12px 16px', display: 'inline-block', maxWidth: '100%' }}
              dangerouslySetInnerHTML={{ __html: formatMarkdown(msg.content || '', msg.citations, idx) }}
            />
            <Space size={4} style={{ marginTop: 6 }}>
              <Tooltip title="有帮助">
                <GlassButton variant="text" size="small" icon={<LikeOutlined />} onClick={() => feedback(msg.id, 'like')} />
              </Tooltip>
              <Tooltip title="无帮助">
                <GlassButton variant="text" size="small" icon={<DislikeOutlined />} onClick={() => feedback(msg.id, 'dislike')} />
              </Tooltip>
            </Space>
          </div>
        </Space>
      </div>
    );
  };

  // 侧边栏配置项弹窗元数据（智能体/对话模型/Embedding/Rerank）
  const PICKER_META: Record<PickerKind, {
    title: string;
    emptyText: string;
    allowClear: boolean;
    options: { id: string; name: string; sub?: string; tag?: string; isDefault?: boolean }[];
  }> = {
    agent: {
      title: '选择智能体',
      emptyText: '暂无智能体，请先创建',
      allowClear: false,
      options: agents.map((a) => ({ id: a.agent_id, name: a.name, sub: a.description || '无描述' })),
    },
    chat: {
      title: '选择对话模型',
      emptyText: '暂无 Chat 模型，请先到模型管理配置',
      allowClear: false,
      options: chatModels.map((m) => ({ id: m.model_id, name: m.name, tag: 'Chat LLM', isDefault: m.is_default })),
    },
    embedding: {
      title: '选择 Embedding 模型',
      emptyText: '暂无 Embedding 模型，请先到模型管理配置',
      allowClear: true,
      options: embeddingModels.map((m) => ({ id: m.model_id, name: m.name, tag: 'Embedding', isDefault: m.is_default })),
    },
    rerank: {
      title: '选择 Rerank 模型',
      emptyText: '暂无 Rerank 模型，可在模型管理配置（可选）',
      allowClear: true,
      options: rerankModels.map((m) => ({ id: m.model_id, name: m.name, tag: 'Rerank', isDefault: m.is_default })),
    },
  };

  const showStreaming = generating && streamingContent;
  const showThinking = generating && (thinkingEvents.length > 0 || liveThinking.text);

  return (
    <div className="glass-fade-in" style={{ display: 'flex', gap: 16, height: 'calc(100vh - 120px)' }}>
      {/* 侧边栏 */}
      <Card
        className="glass"
        style={{ width: 260, flexShrink: 0, display: 'flex', flexDirection: 'column' }}
        bodyStyle={{ padding: 12, display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}
      >
        {/* 顶部固定：选择器 + 新建会话 */}
        <div style={{ flexShrink: 0 }}>
          <Space direction="vertical" style={{ width: '100%' }} size="middle">
            <div className="agent-picker" onClick={() => openPicker('agent')}>
              <span className="agent-picker-label">智能体</span>
              <span className="agent-picker-value">
                {agents.find((a) => a.agent_id === agentId)?.name || '未选择'}
                <DownOutlined />
              </span>
            </div>
            <div className="agent-picker" onClick={() => openPicker('chat')}>
              <span className="agent-picker-label">对话模型</span>
              <span className="agent-picker-value">
                {chatModels.find((m) => m.model_id === chatModelSel)?.name || '未选择'}
                <DownOutlined />
              </span>
            </div>
            <div className="agent-picker" onClick={() => openPicker('embedding')}>
              <span className="agent-picker-label">Embedding</span>
              <span className="agent-picker-value">
                {embeddingModels.find((m) => m.model_id === embeddingModelSel)?.name || '未配置'}
                <DownOutlined />
              </span>
            </div>
            <div className="agent-picker" onClick={() => openPicker('rerank')}>
              <span className="agent-picker-label">Rerank</span>
              <span className="agent-picker-value">
                {rerankModels.find((m) => m.model_id === rerankModelSel)?.name || '不启用'}
                <DownOutlined />
              </span>
            </div>
            {/* 主要操作 → 玻璃悬浮主按钮 */}
            <GlassButton variant="primary" block icon={<PlusOutlined />} onClick={createNewSession}>新建会话</GlassButton>
          </Space>
        </div>
        <Divider2 />
        {/* 会话列表：仅此区域可滚动 */}
        <div style={{ flex: 1, minHeight: 0, overflow: 'auto', margin: '0 -4px', padding: '0 4px' }}>
          <List
            size="small"
            dataSource={sessions}
            locale={{ emptyText: '暂无会话' }}
            renderItem={(s) => (
              <List.Item
                className="agent-session-item"
                style={{
                  cursor: 'pointer',
                  background: s.id === sessionId ? 'rgba(0,0,0,0.07)' : 'transparent',
                  borderRadius: 8,
                  padding: '6px 8px',
                }}
                onClick={() => selectSession(s.id)}
                actions={[
                  <Tooltip key="rename" title="重命名">
                    <GlassButton variant="text" size="small" icon={<EditOutlined />} onClick={(e) => { e.stopPropagation(); renameSession(s.id, s.title); }} />
                  </Tooltip>,
                  <Tooltip key="del" title="删除">
                    <GlassButton variant="danger" size="small" icon={<DeleteOutlined />} onClick={(e) => { e.stopPropagation(); removeSession(s.id); }} />
                  </Tooltip>,
                ]}
              >
                <div>
                  <div style={{ fontSize: 13, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 120 }}>
                    {s.title}
                  </div>
                  <div style={{ fontSize: 11, color: '#9aa7c3' }}>{s.message_count || 0} 条 · {formatTimeShort(s.updated_at)}</div>
                </div>
              </List.Item>
            )}
          />
        </div>
        <Divider2 />
        {/* 底部固定：长期记忆 + 智能体管理（不随会话列表滚动） */}
        <div style={{ flexShrink: 0 }}>
          <Space direction="vertical" style={{ width: '100%' }}>
            <GlassButton variant="text" block icon={<BulbOutlined />} onClick={openMemory}>长期记忆</GlassButton>
            {/* 智能体管理 → 打开管理抽屉（列出全部智能体：编辑/模型绑定/删除/新建） */}
            <GlassButton variant="text" block icon={<RobotOutlined />} onClick={() => setManageOpen(true)}>智能体管理</GlassButton>
          </Space>
        </div>
      </Card>

      {/* 对话区 */}
      <Card className="glass" style={{ flex: 1, display: 'flex', flexDirection: 'column' }} bodyStyle={{ display: 'flex', flexDirection: 'column', height: '100%', padding: 12 }}>
        <div style={{ flex: 1, overflow: 'auto', padding: '0 8px' }}>
          {messages.length === 0 && !generating ? (
            <Empty description="开始与智能体对话吧" style={{ marginTop: 80 }} />
          ) : (
            <>
              {messages.map((m, i) => renderMessage(m, i))}
              {generating && (
                <div style={{ marginBottom: 16 }}>
                  {showThinking && (
                    <div className="glass-dense agent-bubble" style={{ padding: 12, marginBottom: 8 }}>
                      <Tag color="blue">💭 思考中…</Tag>
                      {thinkingEvents.map((e, i) => renderThinkingEvent(e, i))}
                      {liveThinking.text && (
                        <div>
                          <Tag color="blue">{liveThinking.source === 'plan' ? '📋 规划思考' : '🤔 反思思考'}</Tag>
                          <div style={{ whiteSpace: 'pre-wrap', fontSize: 12, color: '#595959' }}>{liveThinking.text}…</div>
                        </div>
                      )}
                    </div>
                  )}
                  {showStreaming ? (
                    <div
                      className="glass agent-bubble"
                      style={{ padding: '12px 16px', display: 'inline-block', maxWidth: '100%' }}
                      dangerouslySetInnerHTML={{ __html: formatMarkdown(streamingContent) }}
                    />
                  ) : (
                    <div style={{ color: '#9aa7c3' }}>⏳ 思考中…</div>
                  )}
                </div>
              )}
            </>
          )}
        </div>

        {/* 上下文栏 */}
        <div className="glass-dense agent-context-bar" style={{ marginTop: 8, padding: '8px 12px' }}>
          <Space size="middle" style={{ width: '100%', justifyContent: 'space-between' }}>
            <Space>
              <Progress
                percent={contextInfo.percent}
                size="small"
                style={{ width: 180 }}
                strokeColor={tokenProgressColor(contextInfo.percent)}
                trailColor="rgba(0,0,0,0.10)"
              />
              <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                {contextInfo.used.toLocaleString()} / {contextInfo.max.toLocaleString()} tokens
              </Typography.Text>
            </Space>
            <GlassButton variant="text" size="small" icon={<CompressOutlined />} onClick={async () => {
              if (!sessionId) return;
              const res = await compressSession(sessionId);
              message.success(`已压缩: ${res.before_tokens} → ${res.after_tokens}`);
              const data = await fetchSession(sessionId);
              setMessages(data.messages || []);
              contextInfoFromMessages(data.messages || []);
            }}>压缩上下文</GlassButton>
          </Space>
        </div>

        {/* 输入区 */}
        <div style={{ marginTop: 8, display: 'flex', gap: 8, alignItems: 'flex-end' }}>
          <Input.TextArea
            className="agent-input"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="输入问题，Enter 发送，Shift+Enter 换行"
            autoSize={{ minRows: 1, maxRows: 4 }}
            disabled={generating}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
              }
            }}
            style={{ flex: 1 }}
          />
          {generating ? (
            <GlassButton variant="danger" icon={<StopOutlined />} onClick={stopGeneration}>停止</GlassButton>
          ) : (
            <GlassButton variant="primary" icon={<SendOutlined />} onClick={sendMessage}>发送</GlassButton>
          )}
        </div>
      </Card>

      {/* 智能体创建/编辑 */}
      <Modal
        title={editingAgent ? '编辑智能体' : '创建智能体'}
        open={agentModal}
        onOk={saveAgent}
        onCancel={() => setAgentModal(false)}
        width={640}
      >
        <Form form={form} layout="vertical">
          <Form.Item name="name" label="名称" rules={[{ required: true, message: '请输入名称' }]}>
            <Input />
          </Form.Item>
          <Form.Item name="description" label="描述">
            <Input.TextArea rows={2} />
          </Form.Item>
          <Form.Item name="kb_ids" label="关联知识库" extra="可关联多个知识库，智能体会在所有关联库中检索">
            <Select
              mode="multiple"
              allowClear
              options={kbs.map((k) => ({ value: k.kb_id, label: k.name }))}
              placeholder="选择知识库（可多选）"
              maxTagCount="responsive"
            />
          </Form.Item>
          <Form.Item name="system_prompt" label="系统提示词">
            <Input.TextArea rows={4} />
          </Form.Item>
          <Form.Item name="enabled_tools" label="启用工具" rules={[{ required: true, message: '请至少选择一个工具' }]}>
            <Checkbox.Group options={(tools || []).map((t) => ({ value: t.name, label: TOOL_LABEL[t.name] || t.name }))} />
          </Form.Item>
          <Form.Item name="memory_enabled" label="长期记忆" valuePropName="checked" extra="开启后智能体会跨会话记住用户偏好与关键事实，并在对话中引用">
            <Switch />
          </Form.Item>
          <Space size="large" wrap>
            <Form.Item name="temperature" label="Temperature"><InputNumber step={0.1} min={0} max={2} /></Form.Item>
            <Form.Item name="top_p" label="Top P"><InputNumber step={0.1} min={0} max={1} /></Form.Item>
            <Form.Item name="max_tokens" label="Max Tokens"><InputNumber min={1} /></Form.Item>
            <Form.Item name="max_reflections" label="最大反思次数"><InputNumber min={1} /></Form.Item>
          </Space>
          <Space size="large" wrap>
            <Form.Item name="retrieval_top_k" label="向量检索 Top K"><InputNumber min={1} /></Form.Item>
            <Form.Item name="kg_top_k" label="图谱检索 Top K"><InputNumber min={1} /></Form.Item>
            <Form.Item name="sql_limit" label="数据库查询条数"><InputNumber min={1} /></Form.Item>
          </Space>
          <Space size="large" wrap>
            <Form.Item name="summary_threshold" label="摘要阈值"><InputNumber min={1} /></Form.Item>
            <Form.Item name="keep_recent_turns" label="保留最近轮次"><InputNumber min={1} /></Form.Item>
          </Space>
          <Space size="large" wrap>
            <Form.Item name="chat_model_id" label="Chat 模型">
              <Select style={{ width: 220 }} options={chatModels.map((m) => ({ value: m.model_id, label: m.name }))} />
            </Form.Item>
            <Form.Item name="embedding_model_id" label="Embedding 模型">
              <Select style={{ width: 220 }} options={embeddingModels.map((m) => ({ value: m.model_id, label: m.name }))} />
            </Form.Item>
            <Form.Item name="rerank_model_id" label="Rerank 模型">
              <Select style={{ width: 220 }} allowClear options={rerankModels.map((m) => ({ value: m.model_id, label: m.name }))} />
            </Form.Item>
          </Space>
        </Form>
      </Modal>

      {/* 模型绑定 */}
      <Modal
        title={`模型绑定 - ${modelsAgent?.name || ''}`}
        open={modelsModal}
        onCancel={() => setModelsModal(false)}
        footer={null}
      >
        {agentModels.map((r: any) => (
          <div key={r.role} style={{ marginBottom: 12 }}>
            <Space style={{ width: '100%', justifyContent: 'space-between' }}>
              <Tag>{r.role}</Tag>
              <Select
                style={{ width: 200 }}
                value={r.model_id}
                onChange={(v) => bindModel(r.role, v)}
                options={
                  (r.role === 'embedding' ? embeddingModels : r.role === 'rerank' ? rerankModels : chatModels).map((m) => ({
                    value: m.model_id,
                    label: m.name,
                  }))
                }
                placeholder="未绑定"
              />
              <GlassButton variant="danger" size="small" onClick={() => unbindModel(r.role)}>解绑</GlassButton>
            </Space>
          </div>
        ))}
      </Modal>

      {/* 长期记忆 */}
      <Modal
        title="长期记忆"
        open={memModal}
        onCancel={() => setMemModal(false)}
        footer={null}
      >
        {/* 当前智能体记忆开关 */}
        {agentId && (() => {
          const cur = agents.find((a) => a.agent_id === agentId);
          if (!cur) return null;
          return (
            <div className="glass-dense" style={{ borderRadius: 10, padding: '10px 14px', marginBottom: 12, display: 'flex', alignItems: 'center', gap: 10 }}>
              <Switch
                size="small"
                checked={cur.memory_enabled !== false}
                onChange={toggleCurrentAgentMemory}
              />
              <span style={{ fontSize: 13, color: '#262626', flex: 1 }}>
                当前智能体「<b>{cur.name}</b>」长期记忆
              </span>
              <Tag color={cur.memory_enabled === false ? 'default' : 'green'}>
                {cur.memory_enabled === false ? '已关闭' : '已开启'}
              </Tag>
            </div>
          );
        })()}
        <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 12 }}>
          <GlassButton variant="danger" size="small" onClick={async () => {
            await clearMemories();
            setMemories([]);
            message.success('已清空');
          }}>清空全部记忆</GlassButton>
        </div>
        <List
          dataSource={memories}
          locale={{ emptyText: '暂无记忆' }}
          renderItem={(m: any, i: number) => (
            <List.Item
              actions={[
                <GlassButton key="del" variant="danger" size="small" onClick={async () => {
                  await deleteMemory(m.key);
                  setMemories((prev) => prev.filter((x) => x.key !== m.key));
                }}>删除</GlassButton>,
              ]}
            >
              <div>
                <Tag color="blue">{m.key}</Tag>
                <div style={{ fontSize: 12, color: '#595959' }}>{String(m.value || m.content || JSON.stringify(m)).slice(0, 200)}</div>
              </div>
            </List.Item>
          )}
        />
      </Modal>

      {/* 侧边栏配置项选择弹窗：胶囊选择器点击后弹出卡片列表 */}
      <Modal
        title={picker ? PICKER_META[picker].title : ''}
        open={!!picker}
        onCancel={() => setPicker(null)}
        onOk={confirmPicker}
        okText="确定"
        cancelText="取消"
        width={460}
        okButtonProps={{ disabled: !!picker && PICKER_META[picker].options.length === 0 }}
      >
        {picker && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxHeight: 440, overflow: 'auto' }}>
            {PICKER_META[picker].options.length === 0 ? (
              <Empty description={PICKER_META[picker].emptyText} style={{ marginTop: 40 }} />
            ) : (
              <>
                {PICKER_META[picker].options.map((opt) => {
                  const selected = opt.id === pickerValue;
                  return (
                    <div
                      key={opt.id}
                      className={`agent-picker-option${selected ? ' selected' : ''}`}
                      onClick={() => setPickerValue(opt.id)}
                    >
                      <div style={{ minWidth: 0 }}>
                        <div className="agent-picker-option-name">
                          {opt.name}
                          {opt.tag && <Tag style={{ marginLeft: 6 }}>{opt.tag}</Tag>}
                          {opt.isDefault && <Tag color="gold" style={{ marginLeft: 4 }}>★ 默认</Tag>}
                        </div>
                        {opt.sub && <div className="agent-picker-option-sub">{opt.sub}</div>}
                      </div>
                      {selected && <CheckOutlined style={{ color: '#1a1a1a', flexShrink: 0 }} />}
                    </div>
                  );
                })}
                {PICKER_META[picker].allowClear && (
                  <div
                    className={`agent-picker-option${!pickerValue ? ' selected' : ''}`}
                    onClick={() => setPickerValue(undefined)}
                  >
                    <div className="agent-picker-option-name" style={{ color: '#8c8c8c' }}>不启用（清除）</div>
                    {!pickerValue && <CheckOutlined style={{ color: '#8c8c8c', flexShrink: 0 }} />}
                  </div>
                )}
              </>
            )}
          </div>
        )}
      </Modal>

      {/* 智能体管理抽屉（全部智能体卡片：编辑/模型绑定/删除/新建） */}
      <Drawer
        title="智能体管理"
        open={manageOpen}
        onClose={() => setManageOpen(false)}
        width={500}
        extra={
          <GlassButton variant="primary" size="small" icon={<PlusOutlined />} onClick={() => { setManageOpen(false); setTimeout(openCreateAgent, 300); }}>
            新建智能体
          </GlassButton>
        }
      >
        {agents.length === 0 ? (
          <Empty description="暂无智能体，点击右上角新建" style={{ marginTop: 60 }} />
        ) : (
          <div>
            {agents.map((a) => {
              const toolLabels = (a.enabled_tools || []).map((t) => TOOL_LABEL[t] || t).join('、') || '无工具';
              return (
                <div
                  key={a.agent_id}
                  className="glass-dense"
                  style={{ borderRadius: 12, marginBottom: 12, padding: '12px 14px' }}
                >
                  {/* 头部：名称（超长省略）+ 操作按钮 */}
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
                    <Typography.Text
                      strong
                      style={{ fontSize: 14, color: '#262626', flex: 1, minWidth: 0 }}
                      ellipsis={{ tooltip: a.name }}
                    >
                      {a.name}
                    </Typography.Text>
                    <Space size={2} style={{ flexShrink: 0 }}>
                      <Tooltip title="编辑">
                        <GlassButton variant="text" size="small" icon={<EditOutlined />} onClick={() => openEditFromManage(a)} />
                      </Tooltip>
                      <Tooltip title="绑定各角色模型">
                        <GlassButton variant="text" size="small" icon={<RobotOutlined />} onClick={() => openModelsFromManage(a)} />
                      </Tooltip>
                      <Popconfirm
                        title={`删除智能体「${a.name}」？`}
                        description="删除后相关会话仍在，但无法再使用该智能体对话"
                        okText="删除"
                        okButtonProps={{ danger: true }}
                        cancelText="取消"
                        onConfirm={() => removeAgent(a.agent_id)}
                      >
                        <GlassButton variant="danger" size="small" icon={<DeleteOutlined />} />
                      </Popconfirm>
                    </Space>
                  </div>
                  {/* 描述 */}
                  <div style={{ fontSize: 12, color: '#595959', marginTop: 6, lineHeight: 1.6 }}>
                    {a.description || '无描述'}
                  </div>
                  {/* 参数标签 */}
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 8 }}>
                    <Tag color="blue">工具: {toolLabels}</Tag>
                    {(a.kb_ids && a.kb_ids.length > 0) && (
                      <Tag color="geekblue">关联 {a.kb_ids.length} 个知识库</Tag>
                    )}
                    <Tag color={a.memory_enabled === false ? 'default' : 'green'}>
                      记忆: {a.memory_enabled === false ? '关闭' : '开启'}
                    </Tag>
                    <Tag>Temp {a.temperature ?? 0.7}</Tag>
                    <Tag>反思 {a.max_reflections ?? 3}</Tag>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </Drawer>
    </div>
  );
}

function AvatarIcon() {
  return (
    <div style={{
      width: 32, height: 32, borderRadius: '50%', display: 'flex', alignItems: 'center',
      justifyContent: 'center', background: 'linear-gradient(135deg,#3a3a3a,#141414)', color: '#fff', fontSize: 14,
    }}>
      <RobotOutlined />
    </div>
  );
}

function CollapsibleThinking({ events }: { events: any[] }) {
  const [open, setOpen] = useState(false);
  // 与旧版 agent.html 一致：展示思考过程步数
  const label = open ? '收起思考过程 ▲' : `查看思考过程 (${events.length} 步) ▼`;
  return (
    <div style={{ marginBottom: 6 }}>
      <GlassButton variant="text" size="small" style={{ padding: 0 }} onClick={() => setOpen(!open)}>{label}</GlassButton>
      {open && (
        <div className="glass-dense" style={{ padding: 12, marginTop: 4 }}>
          {events.map((e, i) => (
            <EventView key={i} e={e} />
          ))}
        </div>
      )}
    </div>
  );
}

/* 历史消息思考过程渲染：与实时 renderThinkingEvent 保持同等详细度
   （plan 含实体/类型关键词与推理原文；tool_call 含完整结果预览与 rerank 分数；
    reflect 含已尝试工具与结果摘要），并用玻璃卡片分组展示 */
function EventView({ e }: { e: any }) {
  const style: React.CSSProperties = {
    marginBottom: 8,
    padding: '8px 10px',
    borderRadius: 10,
    background: 'rgba(255,255,255,0.5)',
    border: '1px solid rgba(255,255,255,0.6)',
  };
  if (e.type === 'thinking_text') {
    return (
      <div style={style}>
        <Tag color="blue">{e.source === 'plan' ? '📋 规划思考' : e.source === 'reflect' ? '🤔 反思思考' : '💭 思考'}</Tag>
        <div style={{ whiteSpace: 'pre-wrap', fontSize: 12, color: '#595959', marginTop: 2 }}>{e.content}</div>
      </div>
    );
  }
  if (e.type === 'plan') {
    return (
      <div style={style}>
        <Tag color="purple">📝 规划</Tag>
        <div style={{ fontSize: 12 }}>
          {e.intent && <div>意图: {e.intent}</div>}
          {e.rewritten_query && <div>改写查询: {e.rewritten_query}</div>}
          {e.entity_keywords?.length > 0 && <div>实体关键词: {e.entity_keywords.join(', ')}</div>}
          {e.type_keywords?.length > 0 && <div>类型关键词: {e.type_keywords.join(', ')}</div>}
          {Array.isArray(e.tools_plan) && e.tools_plan.length > 0 && (
            <div style={{ marginTop: 2 }}>
              <div style={{ color: '#8c6d1f', fontWeight: 500 }}>计划使用工具:</div>
              {e.tools_plan.map((t: any, i: number) => (
                <div key={i} style={{ padding: '2px 0' }}>
                  <Tag color="geekblue">{t.tool}</Tag>
                  {t.reason && <span style={{ color: '#595959' }}>{t.reason}</span>}
                </div>
              ))}
            </div>
          )}
          {e.reasoning && <div style={{ color: '#595959' }}>{e.reasoning}</div>}
        </div>
      </div>
    );
  }
  if (e.type === 'tool_call') {
    return (
      <div style={style}>
        <ToolCallEventView e={e} />
      </div>
    );
  }
  if (e.type === 'reflect') {
    const actionLabel: Record<string, string> = {
      done: '✅ 完成',
      retry_document: '🔄 重试文档检索',
      retry_graph: '🔄 重试图谱检索',
      retry_table: '🔄 重试数据库查询',
    };
    return (
      <div style={style}>
        <Tag color="orange">🤔 反思</Tag>
        <div style={{ fontSize: 12 }}>
          <div>是否足够: {e.need_more ? '是' : '否'} · 下一步: {actionLabel[e.next_action] || e.next_action}</div>
          {e.tried_tools?.length > 0 && <div>已尝试: {e.tried_tools.join(', ')}</div>}
          {e.result_summary && <div style={{ color: '#595959' }}>结果摘要: {e.result_summary}</div>}
          {e.reflection && <div style={{ color: '#595959' }}>{e.reflection}</div>}
        </div>
      </div>
    );
  }
  return null;
}

function Divider2() {
  return <div style={{ borderTop: '1px solid rgba(31,45,78,0.1)', margin: '4px 0' }} />;
}

/* 工具调用事件详细渲染：完整移植自旧版 agent.html 的 renderToolCallEvent。
   - 通用头：工具名（后端字段是 tool，兼容 name）、结果条数、耗时、rerank 徽标、查询词
   - vector_search：文档列表（标题/来源文档/片段 + score/rerank 分数）
   - kg_search：实体标签、关系示例（可折叠）、社区报告（含评分/关键发现，可折叠）
   - sql_query：图片/表格预览（含表格行数据）
   注：后端 tool_call 字段为 tool 而非 name，旧 React 版读 e.name 导致工具名恒为"工具" */
function ToolCallEventView({ e }: { e: any }) {
  const toolName = e.tool || e.name || '';
  const label = TOOL_LABEL[toolName] || toolName || '工具';
  return (
    <div>
      <div style={{ fontSize: 12 }}>
        <Tag color="green">🔧 {label}</Tag>
        {e.count !== undefined && <Tag color="blue">{e.count} 条结果</Tag>}
        {e.duration !== undefined && <Tag>{e.duration}s</Tag>}
        {toolName === 'vector_search' && (
          e.reranked
            ? <Tag color="gold">🎯 已使用重排序{e.rerank_model ? ` · ${e.rerank_model}` : ''}</Tag>
            : e.rerank_model
              ? <Tag color="orange">⚠️ 未使用重排序 · {e.rerank_model}</Tag>
              : <Tag>ℹ️ 未配置重排序模型</Tag>
        )}
        {toolName === 'kg_search' && e.fault_trees > 0 && (
          <Tag color="volcano">故障树因果链 {e.fault_trees} 条</Tag>
        )}
        {e.error && <div style={{ color: '#ff4d4f' }}>错误: {e.error}</div>}
        {e.rerank_error && <div style={{ color: '#8c8c8c' }}>重排序降级: {e.rerank_error}</div>}
      </div>
      {e.query && (
        <div style={{ fontSize: 11, color: '#595959', marginTop: 2 }}>查询: {e.query}</div>
      )}

      {toolName === 'vector_search' && Array.isArray(e.results_preview) && e.results_preview.length > 0 && (
        <div style={{ maxHeight: 260, overflow: 'auto', fontSize: 12, marginTop: 4 }}>
          {e.results_preview.map((r: any, i: number) => (
            <div key={i} style={{ margin: '4px 0', padding: '6px 8px', background: 'rgba(255,255,255,0.6)', borderRadius: 6, borderLeft: '2px solid #3a3a3a' }}>
              <div style={{ fontWeight: 500 }}>
                {i + 1}. {r.title || '(无标题)'}
                {r.rerank_score != null ? (
                  <span style={{ color: '#8c8c8c', fontSize: 11, marginLeft: 4 }}>(rerank: {Number(r.rerank_score).toFixed(3)})</span>
                ) : r.score != null ? (
                  <span style={{ color: '#888', fontSize: 11, marginLeft: 4 }}>(score: {Number(r.score).toFixed(3)})</span>
                ) : null}
              </div>
              {r.doc_name && <div style={{ fontSize: 11, color: '#888' }}>📄 {r.doc_name}</div>}
              {r.snippet && <div style={{ fontSize: 11, color: '#666', marginTop: 2 }}>{r.snippet}...</div>}
            </div>
          ))}
        </div>
      )}

      {toolName === 'kg_search' && (
        <div style={{ fontSize: 12, marginTop: 4 }}>
          {Array.isArray(e.entities_preview) && e.entities_preview.length > 0 && (
            <div>
              <div style={{ color: '#595959', fontSize: 11, marginBottom: 2 }}>相关实体:</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                {e.entities_preview.map((en: any, i: number) => (
                  <div key={i}>
                    <Tag color={en.source_type === 'fault_tree' ? 'volcano' : 'cyan'}>
                      {en.name}{en.type ? ` [${en.type}]` : ''}{en.source_type === 'fault_tree' ? ' 【故障树】' : ''}
                    </Tag>
                    {en.desc && <span style={{ fontSize: 11, color: '#666', marginLeft: 4 }}>{en.desc}</span>}
                  </div>
                ))}
              </div>
            </div>
          )}
          {Array.isArray(e.relations_preview) && e.relations_preview.length > 0 && (
            <details style={{ marginTop: 4 }}>
              <summary style={{ cursor: 'pointer', fontSize: 11, color: '#595959' }}>关系示例 ({e.relations_preview.length})</summary>
              <div style={{ marginTop: 4 }}>
                {e.relations_preview.map((r: any, i: number) => (
                  <div key={i} style={{ fontSize: 11, padding: '2px 0' }}>
                    {r.src} <span style={{ color: '#333333' }}>→</span> {r.tgt}
                    {r.type ? ` (${r.type})` : ''}
                    {r.gate_type ? <span style={{ color: '#d46b08' }}> [{r.gate_type}]</span> : null}
                    {r.desc && <span style={{ color: '#888', marginLeft: 4 }}>{r.desc}</span>}
                  </div>
                ))}
              </div>
            </details>
          )}
          {Array.isArray(e.reports_preview) && e.reports_preview.length > 0 && (
            <details style={{ marginTop: 4 }}>
              <summary style={{ cursor: 'pointer', fontSize: 11, color: '#595959' }}>社区报告 ({e.reports_preview.length})</summary>
              <div style={{ marginTop: 4 }}>
                {e.reports_preview.map((r: any, i: number) => (
                  <div key={i} style={{ margin: '6px 0', padding: '6px 8px', background: 'rgba(255,255,255,0.6)', borderRadius: 6, borderLeft: '2px solid #404040' }}>
                    <div style={{ fontWeight: 500, fontSize: 12 }}>
                      🏘️ {r.title || '未命名社区'}
                      {r.rating != null && <span style={{ color: '#8c8c8c' }}> ★{r.rating}</span>}
                      {r.entity_count ? <span style={{ color: '#888' }}> [{r.entity_count}实体]</span> : null}
                    </div>
                    {r.summary && <div style={{ fontSize: 11, color: '#666', marginTop: 2 }}>{r.summary}</div>}
                    {Array.isArray(r.findings_preview) && r.findings_preview.length > 0 && (
                      <div style={{ marginTop: 4, padding: '4px 6px', background: 'rgba(0,0,0,0.04)', borderRadius: 4 }}>
                        <div style={{ fontSize: 10, color: '#404040', fontWeight: 500 }}>💡 关键发现 ({r.findings_count}):</div>
                        {r.findings_preview.map((f: any, j: number) => (
                          <div key={j} style={{ fontSize: 11, color: '#555', marginTop: 2 }}>• {typeof f === 'string' ? f : f?.summary || ''}</div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </details>
          )}
        </div>
      )}

      {toolName === 'sql_query' && Array.isArray(e.results_preview) && e.results_preview.length > 0 && (
        <div style={{ fontSize: 12, marginTop: 4 }}>
          {e.results_preview.map((r: any, i: number) => (
            <div key={i} style={{ margin: '4px 0', padding: '6px 8px', background: 'rgba(255,255,255,0.6)', borderRadius: 6, borderLeft: '2px solid #8c8c8c' }}>
              <div style={{ fontWeight: 500 }}>{r.type === 'image' ? '🖼️' : '📊'} {i + 1}. {r.title || '(无标题)'}</div>
              {r.doc_name && <div style={{ fontSize: 11, color: '#888' }}>📄 {r.doc_name}</div>}
              {r.type === 'table' && r.rows_total !== undefined && (
                <div style={{ fontSize: 11, color: '#888' }}>共 {r.rows_total} 行数据</div>
              )}
              {Array.isArray(r.rows_preview) && r.rows_preview.length > 0 && (
                <pre style={{ marginTop: 4, padding: 4, background: 'rgba(255,255,255,0.7)', borderRadius: 4, fontSize: 10, whiteSpace: 'pre', overflowX: 'auto' }}>
                  {r.rows_preview.map((row: any[]) => row.map((c) => String(c)).join(' | ')).join('\n')}
                </pre>
              )}
            </div>
          ))}
        </div>
      )}

      {(!e.results_preview || e.results_preview.length === 0) && e.count === 0 && (
        <div style={{ fontSize: 11, color: '#8c8c8c', marginTop: 2 }}>⚠️ 未检索到相关内容</div>
      )}
    </div>
  );
}
