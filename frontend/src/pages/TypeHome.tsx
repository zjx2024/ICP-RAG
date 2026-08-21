import React, { useEffect, useState, useCallback } from 'react';
import { Row, Col, Card, Typography, Space, Tag, Spin, message } from 'antd';
import {
  TableOutlined, ApartmentOutlined, FileTextOutlined, ArrowRightOutlined,
} from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import { fetchDocumentTypes } from '@/api/icp';
import { GlassEmpty, GlassStatCard } from '@/components/GlassUI';
import type { DocTypeInfo } from '@/types';

// 按 data_form 映射图标与主题色
const FORM_META: Record<string, { icon: React.ReactNode; color: string; label: string }> = {
  table: { icon: <TableOutlined />, color: '#1677ff', label: '表格类' },
  image: { icon: <ApartmentOutlined />, color: '#722ed1', label: '图类' },
  text: { icon: <FileTextOutlined />, color: '#52c41a', label: '文本类' },
};

export default function TypeHome() {
  const navigate = useNavigate();
  const [types, setTypes] = useState<DocTypeInfo[]>([]);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const list = await fetchDocumentTypes();
      setTypes(list);
    } catch (e) {
      message.error(`加载文档类型失败: ${(e as Error).message}`);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const totalDocs = types.reduce((s, t) => s + (t.doc_count || 0), 0);
  const totalRecords = types.reduce((s, t) => s + (t.record_count || 0), 0);

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
        <div>
          <Typography.Title level={4} style={{ margin: 0, color: '#262626' }}>ICP 七类文档知识库</Typography.Title>
          <Typography.Text type="secondary">
            每种文档类型使用独立的解析方式，点击卡片进入该类型的管理页
          </Typography.Text>
        </div>
      </div>

      <Row gutter={16} style={{ marginBottom: 20 }}>
        <Col xs={12} sm={6}>
          <GlassStatCard label="文档类型" value={types.length} suffix="类" />
        </Col>
        <Col xs={12} sm={6}>
          <GlassStatCard label="文档总数" value={totalDocs} suffix="份" />
        </Col>
        <Col xs={12} sm={6}>
          <GlassStatCard label="数据记录" value={totalRecords} suffix="条" />
        </Col>
      </Row>

      {loading ? (
        <div style={{ textAlign: 'center', padding: 60 }}><Spin size="large" /></div>
      ) : types.length === 0 ? (
        <GlassEmpty description="暂无文档类型配置，请检查后端 document_types 表初始化" />
      ) : (
        <Row gutter={[20, 20]}>
          {types.map((t) => {
            const meta = FORM_META[t.data_form] || FORM_META.text;
            return (
              <Col key={t.doc_type} xs={24} sm={12} lg={8} xl={6}>
                <Card
                  hoverable
                  onClick={() => navigate(`/type/${t.doc_type}`)}
                  style={{
                    background: 'rgba(255,255,255,0.6)',
                    backdropFilter: 'blur(14px) saturate(160%)',
                    border: '1px solid rgba(255,255,255,0.65)',
                    height: '100%',
                  }}
                  styles={{ body: { display: 'flex', flexDirection: 'column', height: '100%' } }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                    <span style={{ fontSize: 26, color: meta.color }}>{meta.icon}</span>
                    <Typography.Title level={5} style={{ margin: 0, color: '#262626', flex: 1 }} ellipsis>
                      {t.name}
                    </Typography.Title>
                  </div>
                  <Typography.Text type="secondary" style={{ fontSize: 12, color: meta.color }}>
                    {t.doc_type}
                  </Typography.Text>
                  <Typography.Paragraph type="secondary" style={{ minHeight: 40, marginBottom: 12 }} ellipsis={{ rows: 2 }}>
                    {t.description || '暂无描述'}
                  </Typography.Paragraph>
                  <Space size={[8, 8]} wrap style={{ marginBottom: 12 }}>
                    <Tag color={meta.color}>{meta.label}</Tag>
                    {t.input_formats ? <Tag>支持 {t.input_formats}</Tag> : null}
                    <Tag>文档 {t.doc_count ?? 0}</Tag>
                    {t.record_count ? <Tag>记录 {t.record_count}</Tag> : null}
                  </Space>
                  <div style={{ flex: 1 }} />
                  <div style={{ borderTop: '1px solid rgba(0,0,0,0.06)', marginTop: 10, paddingTop: 10, textAlign: 'right' }}>
                    <Typography.Text type="secondary" style={{ fontSize: 13 }}>
                      进入管理 <ArrowRightOutlined />
                    </Typography.Text>
                  </div>
                </Card>
              </Col>
            );
          })}
        </Row>
      )}
    </div>
  );
}
