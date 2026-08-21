import { useEffect, useState, useCallback, useRef } from 'react';
import {
  Card, Table, Select, Tag, Space, Modal, Form, Input, InputNumber,
  Switch, Alert, message, Popconfirm, Typography, Divider,
} from 'antd';
import { PlusOutlined, ReloadOutlined } from '@ant-design/icons';
import { fetchModels, fetchModel, fetchCurrentModelConfig, createModel, updateModel, deleteModel, setDefaultModel, testModelConnection, MODEL_TYPES, PROVIDER_LABELS } from '@/api/models';
import type { LLMModel } from '@/types';
import { useGlobalStore } from '@/store';
import { GlassButton } from '@/components/GlassUI';

const DEFAULT_FORM = {
  temperature: 0.7,
  max_tokens: 262144,
  top_p: 0.1,
  timeout: 3000,
};

export default function Models() {
  const [models, setModels] = useState<LLMModel[]>([]);
  const [loading, setLoading] = useState(false);
  const [typeFilter, setTypeFilter] = useState<string>('llm');
  const [currentConfig, setCurrentConfig] = useState<any>(null);
  const [modalOpen, setModalOpen] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [testingId, setTestingId] = useState<string | null>(null);
  const [form] = Form.useForm();
  const setModelsStore = useGlobalStore((s) => s.setModels);
  const setModelConfigStatus = useGlobalStore((s) => s.setModelConfigStatus);

  const loadModels = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetchModels();
      const list = res.models || [];
      setModels(list);
      setModelsStore(list);
    } catch (e) {
      message.error(`加载模型失败: ${(e as Error).message}`);
    } finally {
      setLoading(false);
    }
  }, [setModelsStore]);

  const loadConfig = useCallback(async () => {
    try {
      const res = await fetchCurrentModelConfig();
      setCurrentConfig(res);
      setModelConfigStatus(res.config_status);
    } catch {
      /* ignore */
    }
  }, [setModelConfigStatus]);

  useEffect(() => {
    loadModels();
    loadConfig();
  }, [loadModels, loadConfig]);

  // 默认模型下拉
  const [defaultSel, setDefaultSel] = useState<Record<string, string | undefined>>({});

  useEffect(() => {
    if (currentConfig) {
      setDefaultSel({
        llm: currentConfig.chat_model?.model_id,
        embedding: currentConfig.embedding_model?.model_id,
        rerank: currentConfig.rerank_model?.model_id,
        vision: currentConfig.chat_model?.model_id && models.find((m) => m.model_type === 'vision') ? undefined : undefined,
      });
    }
  }, [currentConfig, models]);

  const saveDefaultModels = async () => {
    for (const type of ['llm', 'embedding', 'rerank']) {
      const mid = defaultSel[type];
      if (mid) {
        try {
          await setDefaultModel(type, mid);
        } catch (e) {
          message.error(`${type} 默认模型设置失败: ${(e as Error).message}`);
          return;
        }
      }
    }
    message.success('默认模型配置已保存');
    await Promise.all([loadModels(), loadConfig()]);
  };

  const openCreate = () => {
    setEditingId(null);
    form.resetFields();
    form.setFieldsValue({ model_type: 'llm', is_active: true, ...DEFAULT_FORM });
    setModalOpen(true);
  };

  const openEdit = async (m: LLMModel) => {
    setEditingId(m.model_id);
    form.setFieldsValue({
      name: m.name,
      model_type: m.model_type,
      provider: m.provider,
      model_name: m.model_name,
      api_key: m.api_key === '******' ? '******' : (m.api_key || ''),
      base_url: m.base_url,
      api_url: m.api_url,
      temperature: m.temperature ?? 0.7,
      max_tokens: m.max_tokens ?? 262144,
      top_p: m.top_p ?? 0.1,
      timeout: m.timeout ?? 3000,
      description: m.description,
      is_active: m.is_active,
    });
    setModalOpen(true);
    // 拉取详情获取真实 api_key：密码框默认圆点显示，点击"眼睛"可查看明文
    try {
      const detail = await fetchModel(m.model_id);
      if (detail?.api_key && detail.api_key !== '******') {
        form.setFieldValue('api_key', detail.api_key);
      }
    } catch {
      /* 拉取失败则保持掩码显示 */
    }
  };

  const handleSave = async () => {
    const values = await form.validateFields();
    // api_key 为掩码值（******）或空时表示"不修改"，不随请求提交，避免覆盖已存密钥
    if (values.api_key === '******' || !values.api_key) {
      delete values.api_key;
    }
    setSaving(true);
    try {
      if (editingId) {
        await updateModel(editingId, values);
        message.success('模型更新成功');
      } else {
        await createModel(values);
        message.success('模型创建成功');
      }
      setModalOpen(false);
      await Promise.all([loadModels(), loadConfig()]);
    } catch (e) {
      message.error(`保存失败: ${(e as Error).message}`);
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await deleteModel(id);
      message.success('模型删除成功');
      await Promise.all([loadModels(), loadConfig()]);
    } catch (e) {
      message.error(`删除失败: ${(e as Error).message}`);
    }
  };

  const handleTest = async (m: LLMModel) => {
    setTestingId(m.model_id);
    try {
      const res = await testModelConnection(m.model_id);
      if (res.ok) {
        message.success(res.message || '连接成功');
      } else {
        message.error(res.message || '连接失败');
      }
    } catch (e) {
      message.error(`测试连接失败: ${(e as Error).message}`);
    } finally {
      setTestingId(null);
    }
  };

  const watchType = Form.useWatch('model_type', form) || 'llm';
  const filtered = models.filter((m) => !typeFilter || m.model_type === typeFilter);

  // 模型列表表格：动态高度占满剩余可视区域
  const tableWrapRef = useRef<HTMLDivElement>(null);
  const [tableScrollY, setTableScrollY] = useState(400);
  useEffect(() => {
    const calc = () => {
      const el = tableWrapRef.current;
      if (!el) return;
      setTableScrollY(Math.max(200, window.innerHeight - el.getBoundingClientRect().top - 120));
    };
    calc();
    window.addEventListener('resize', calc);
    return () => window.removeEventListener('resize', calc);
  }, [models.length]);

  return (
    <div className="glass-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 12, height: '100%', overflow: 'hidden' }}>
      <Card className="glass" style={{ flexShrink: 0 }}>
        <Typography.Title level={5} style={{ marginTop: 0 }}>默认模型配置</Typography.Title>
        <Space size="large" wrap>
          <div>
            <Typography.Text type="secondary">Chat LLM</Typography.Text>
            <div style={{ marginTop: 4 }}>
              <Select
                style={{ width: 220 }}
                placeholder="选择默认 Chat 模型"
                value={defaultSel.llm}
                onChange={(v) => setDefaultSel((s) => ({ ...s, llm: v }))}
                options={models.filter((m) => m.model_type === 'llm' || !m.model_type).map((m) => ({ value: m.model_id, label: m.name }))}
              />
            </div>
          </div>
          <div>
            <Typography.Text type="secondary">Embedding</Typography.Text>
            <div style={{ marginTop: 4 }}>
              <Select
                style={{ width: 220 }}
                placeholder="选择默认 Embedding 模型"
                value={defaultSel.embedding}
                onChange={(v) => setDefaultSel((s) => ({ ...s, embedding: v }))}
                options={models.filter((m) => m.model_type === 'embedding').map((m) => ({ value: m.model_id, label: m.name }))}
              />
            </div>
          </div>
          <div>
            <Typography.Text type="secondary">Rerank</Typography.Text>
            <div style={{ marginTop: 4 }}>
              <Select
                style={{ width: 220 }}
                placeholder="选择默认 Rerank 模型（可选）"
                value={defaultSel.rerank}
                onChange={(v) => setDefaultSel((s) => ({ ...s, rerank: v }))}
                options={models.filter((m) => m.model_type === 'rerank').map((m) => ({ value: m.model_id, label: m.name }))}
              />
            </div>
          </div>
          {/* 主要操作 → 玻璃悬浮主按钮（方案 4） */}
          <GlassButton variant="primary" onClick={saveDefaultModels}>保存默认配置</GlassButton>
        </Space>
        <Divider />
        {currentConfig?.config_status?.ready_for_ai ? (
          <Alert type="success" showIcon message="AI 功能已就绪（对话 / 解析 / 图谱模型均已配置）" />
        ) : (
          <Alert
            type="warning"
            showIcon
            message="AI 功能未就绪"
            description={
              <span>
                {!currentConfig?.config_status?.chat_configured && '未配置 Chat LLM 模型；'}
                {!currentConfig?.config_status?.embedding_configured && '未配置 Embedding 模型；'}
              </span>
            }
          />
        )}
      </Card>

      <Card
        className="glass"
        style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}
        styles={{ body: { flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' } }}
        title={
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>模型列表</span>
            <Space>
              <Select
                style={{ width: 150 }}
                value={typeFilter}
                onChange={setTypeFilter}
                options={[{ value: '', label: '全部' }, ...MODEL_TYPES.map((t) => ({ value: t.value, label: t.label }))]}
              />
              <GlassButton variant="text" icon={<ReloadOutlined />} onClick={loadModels} loading={loading}>刷新</GlassButton>
              <GlassButton variant="primary" icon={<PlusOutlined />} onClick={openCreate}>添加模型</GlassButton>
            </Space>
          </div>
        }
      >
        <div ref={tableWrapRef} style={{ flex: 1, minHeight: 0 }}>
          <Table<LLMModel>
            rowKey="model_id"
            className="kb-doc-table"
            loading={loading}
            dataSource={filtered}
            scroll={{ y: tableScrollY }}
            pagination={false}
            columns={[
              { title: '名称', dataIndex: 'name', render: (v, r) => <Space>{v}{r.is_default && <Tag color="gold">默认</Tag>}</Space> },
              { title: '类型', dataIndex: 'model_type', render: (v) => <Tag color="blue">{v}</Tag> },
              { title: '提供方', dataIndex: 'provider', render: (v) => (v ? PROVIDER_LABELS[v] || v : '-') },
              { title: '模型名', dataIndex: 'model_name', render: (v) => v || '-' },
              { title: 'API 地址', dataIndex: 'base_url', ellipsis: true, render: (v) => v || '-' },
              { title: '活跃', dataIndex: 'is_active', render: (v) => (v ? <Tag color="green">是</Tag> : <Tag>否</Tag>) },
              {
                title: '操作',
                render: (_, r) => (
                  <Space className="kb-row-actions">
                    <GlassButton variant="text" size="small" loading={testingId === r.model_id} onClick={() => handleTest(r)}>测试连接</GlassButton>
                    <GlassButton variant="text" size="small" onClick={() => openEdit(r)}>编辑</GlassButton>
                    <Popconfirm title="确定删除该模型？" onConfirm={() => handleDelete(r.model_id)}>
                      <GlassButton variant="danger" size="small">删除</GlassButton>
                    </Popconfirm>
                  </Space>
                ),
              },
            ]}
          />
        </div>
      </Card>

      <Modal
        title={editingId ? '编辑模型' : '创建模型'}
        open={modalOpen}
        onOk={handleSave}
        onCancel={() => setModalOpen(false)}
        confirmLoading={saving}
        width={560}
      >
        <Form form={form} layout="vertical">
          <Form.Item name="name" label="名称" rules={[{ required: true, message: '请输入名称' }]}>
            <Input placeholder="例如: Ollama Qwen2.5" />
          </Form.Item>
          <Form.Item name="model_type" label="类型" rules={[{ required: true }]}>
            <Select
              options={MODEL_TYPES.map((t) => ({
                value: t.value,
                label: `${t.label}${t.value === 'llm' ? '，用于对话/解析/图谱' : t.value === 'embedding' ? '，用于向量化' : ''}`,
              }))}
            />
          </Form.Item>
          <Form.Item name="provider" label="提供方">
            <Select
              allowClear
              options={Object.entries(PROVIDER_LABELS).map(([v, label]) => ({ value: v, label }))}
            />
          </Form.Item>
          <Form.Item name="model_name" label="模型名称">
            <Input placeholder="如 qwen2.5:7b / text-embedding-v3" />
          </Form.Item>
          <Form.Item name="api_key" label="API Key">
            <Input.Password placeholder="已保存的 Key 默认圆点显示，点击眼睛可查看；留空表示不修改，输入新值可替换" />
          </Form.Item>
          <Form.Item name="base_url" label="Base URL">
            <Input placeholder="如 http://host:11434/v1" />
          </Form.Item>
          <Form.Item name="api_url" label="API URL">
            <Input placeholder="可选" />
          </Form.Item>
          {watchType === 'llm' && (
            <>
              <Space size="large" style={{ width: '100%' }}>
                <Form.Item name="temperature" label="Temperature">
                  <InputNumber step={0.1} min={0} max={2} />
                </Form.Item>
                <Form.Item name="max_tokens" label="Max Tokens">
                  <InputNumber min={1} style={{ width: 140 }} />
                </Form.Item>
                <Form.Item name="top_p" label="Top P">
                  <InputNumber step={0.1} min={0} max={1} />
                </Form.Item>
                <Form.Item name="timeout" label="Timeout(s)">
                  <InputNumber min={1} />
                </Form.Item>
              </Space>
            </>
          )}
          <Form.Item name="description" label="描述">
            <Input.TextArea rows={2} />
          </Form.Item>
          <Form.Item name="is_active" label="启用" valuePropName="checked">
            <Switch />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
}
