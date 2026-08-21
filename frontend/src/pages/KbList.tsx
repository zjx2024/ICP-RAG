import React, { useEffect, useState, useCallback } from 'react';
import {
  Row, Col, Card, Modal, Form, Input, Popconfirm, Typography,
  Space, Spin, message, Tag,
} from 'antd';
import {
  PlusOutlined, EditOutlined, DeleteOutlined, FolderOpenOutlined,
  ApartmentOutlined, FileTextOutlined,
} from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import {
  fetchKnowledgeBases, createKnowledgeBase, updateKnowledgeBase, deleteKnowledgeBase,
} from '@/api/kb';
import { fetchRetrievalStats } from '@/api/retrieval';
import { GlassButton, GlassEmpty, GlassStatCard } from '@/components/GlassUI';
import type { KnowledgeBase } from '@/types';
import { formatDateTime } from '@/utils';

interface KBStats {
  total_documents?: number;
  total_chunks?: number;
  total_entities?: number;
  total_relations?: number;
}

export default function KbList() {
  const navigate = useNavigate();
  const [kbs, setKbs] = useState<KnowledgeBase[]>([]);
  const [statsMap, setStatsMap] = useState<Record<string, KBStats>>({});
  const [loading, setLoading] = useState(false);
  const [modal, setModal] = useState<{ open: boolean; editing: KnowledgeBase | null }>({ open: false, editing: null });
  const [form] = Form.useForm();

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const list = await fetchKnowledgeBases();
      setKbs(list);
      // 并行拉取每个知识库的统计（文档数/实体数），失败不影响列表展示
      const map: Record<string, KBStats> = {};
      await Promise.all(
        list.map(async (kb) => {
          try {
            const s = await fetchRetrievalStats(kb.kb_id);
            map[kb.kb_id] = s as KBStats;
          } catch {
            map[kb.kb_id] = {};
          }
        }),
      );
      setStatsMap(map);
    } catch (e) {
      message.error(`加载知识库失败: ${(e as Error).message}`);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const openCreate = () => {
    form.resetFields();
    setModal({ open: true, editing: null });
  };

  const openEdit = (kb: KnowledgeBase) => {
    form.setFieldsValue({ name: kb.name, description: kb.description });
    setModal({ open: true, editing: kb });
  };

  const submit = async () => {
    const values = await form.validateFields();
    try {
      if (modal.editing) {
        await updateKnowledgeBase(modal.editing.kb_id, values);
        message.success('知识库已更新');
      } else {
        const kb = await createKnowledgeBase(values);
        message.success('知识库已创建');
        setModal({ open: false, editing: null });
        navigate(`/kb/${kb.kb_id}/upload`);
        return;
      }
      setModal({ open: false, editing: null });
      load();
    } catch (e) {
      message.error(`保存失败: ${(e as Error).message}`);
    }
  };

  const remove = async (kb: KnowledgeBase) => {
    try {
      await deleteKnowledgeBase(kb.kb_id);
      message.success(`知识库「${kb.name}」已删除`);
      load();
    } catch (e) {
      message.error(`删除失败: ${(e as Error).message}`);
    }
  };

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
        <div>
          <Typography.Title level={4} style={{ margin: 0, color: '#262626' }}>知识库</Typography.Title>
          <Typography.Text type="secondary">选择知识库进入，或新建一个知识库开始管理文档与图谱</Typography.Text>
        </div>
        {/* 玻璃悬浮主按钮（方案 4：Primary 采用玻璃悬浮，hover 浮起） */}
        <GlassButton variant="primary" icon={<PlusOutlined />} onClick={openCreate}>新建知识库</GlassButton>
      </div>

      {/* 大号玻璃角标统计（方案 3：统计数字用玻璃角标） */}
      <Row gutter={16} style={{ marginBottom: 20 }}>
        <Col xs={12} sm={6}>
          <GlassStatCard label="知识库总数" value={kbs.length} suffix="个" />
        </Col>
        <Col xs={12} sm={6}>
          <GlassStatCard label="文档总数" value={Object.values(statsMap).reduce((s, x) => s + (x.total_documents || 0), 0)} suffix="份" />
        </Col>
        <Col xs={12} sm={6}>
          <GlassStatCard label="图谱实体" value={Object.values(statsMap).reduce((s, x) => s + (x.total_entities || 0), 0)} suffix="个" />
        </Col>
        <Col xs={12} sm={6}>
          <GlassStatCard label="图谱关系" value={Object.values(statsMap).reduce((s, x) => s + (x.total_relations || 0), 0)} suffix="条" />
        </Col>
      </Row>

      {loading ? (
        <div style={{ textAlign: 'center', padding: 60 }}><Spin size="large" /></div>
      ) : kbs.length === 0 ? (
        /* 空状态：玻璃占位卡（方案 5） */
        <GlassEmpty description="还没有知识库，点击右上角新建">
          <GlassButton variant="primary" icon={<PlusOutlined />} onClick={openCreate}>新建知识库</GlassButton>
        </GlassEmpty>
      ) : (
        <Row gutter={[20, 20]}>
          {kbs.map((kb) => {
            const st = statsMap[kb.kb_id] || {};
            return (
              <Col key={kb.kb_id} xs={24} sm={12} lg={8} xl={6}>
                <Card
                  hoverable
                  onClick={() => navigate(`/kb/${kb.kb_id}/upload`)}
                  style={{ background: 'rgba(255,255,255,0.6)', backdropFilter: 'blur(14px) saturate(160%)', border: '1px solid rgba(255,255,255,0.65)', height: '100%' }}
                  styles={{ body: { display: 'flex', flexDirection: 'column', height: '100%' } }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                    <FolderOpenOutlined style={{ fontSize: 26, color: '#262626' }} />
                    <Typography.Title level={5} style={{ margin: 0, color: '#262626', flex: 1 }} ellipsis>
                      {kb.name}
                    </Typography.Title>
                  </div>
                  <Typography.Paragraph type="secondary" style={{ minHeight: 40, marginBottom: 12 }} ellipsis={{ rows: 2 }}>
                    {kb.description || '暂无描述'}
                  </Typography.Paragraph>
                  <Space size={[8, 8]} wrap style={{ marginBottom: 12 }}>
                    <Tag icon={<FileTextOutlined />} color="blue">文档 {st.total_documents ?? '-'}</Tag>
                    <Tag icon={<ApartmentOutlined />} color="purple">实体 {st.total_entities ?? '-'}</Tag>
                  </Space>
                  <div style={{ flex: 1 }} />
                  <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                    {kb.updated_at ? `更新于 ${formatDateTime(kb.updated_at)}` : kb.created_at ? `创建于 ${formatDateTime(kb.created_at)}` : ''}
                  </Typography.Text>
                  <div style={{ borderTop: '1px solid rgba(0,0,0,0.06)', marginTop: 10, paddingTop: 10, display: 'flex', justifyContent: 'space-between' }}>
                    {/* 文字+下划线次按钮 / 玻璃描边红危险按钮（方案 4） */}
                    <GlassButton variant="text" size="small" icon={<EditOutlined />} onClick={(e) => { e.stopPropagation(); openEdit(kb); }}>编辑</GlassButton>
                    <Popconfirm
                      title="删除知识库"
                      description={`将级联删除「${kb.name}」的全部文档、图谱与图片，不可恢复`}
                      okText="删除"
                      okButtonProps={{ danger: true }}
                      cancelText="取消"
                      onConfirm={() => remove(kb)}
                    >
                      <GlassButton variant="danger" size="small" icon={<DeleteOutlined />} onClick={(e) => e.stopPropagation()}>删除</GlassButton>
                    </Popconfirm>
                  </div>
                </Card>
              </Col>
            );
          })}
        </Row>
      )}

      <Modal
        title={modal.editing ? '编辑知识库' : '新建知识库'}
        open={modal.open}
        onOk={submit}
        onCancel={() => setModal({ open: false, editing: null })}
        okText={modal.editing ? '保存' : '创建'}
        destroyOnClose
      >
        <Form form={form} layout="vertical" initialValues={{ name: '', description: '' }}>
          <Form.Item name="name" label="知识库名称" rules={[{ required: true, message: '请输入知识库名称' }]}>
            <Input placeholder="例如：ICP故障知识库" maxLength={100} />
          </Form.Item>
          <Form.Item name="description" label="描述">
            <Input.TextArea placeholder="简要说明该知识库的用途（可选）" rows={3} maxLength={500} />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
}
