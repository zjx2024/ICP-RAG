import React, { useEffect, useState } from 'react';
import { Layout, Menu, Typography, Button, Space, Tag } from 'antd';
import {
  ArrowLeftOutlined, CloudUploadOutlined, SearchOutlined, EyeOutlined,
  ApartmentOutlined, ShareAltOutlined, FileTextOutlined, PictureOutlined,
  LoadingOutlined,
} from '@ant-design/icons';
import { useParams, useNavigate, useLocation, Outlet } from 'react-router-dom';
import { getKnowledgeBase } from '@/api/kb';
import type { KnowledgeBase } from '@/types';

const { Sider, Content } = Layout;

const SUB_MENU = [
  { key: 'upload', icon: <CloudUploadOutlined />, label: '文档上传与解析' },
  { key: 'retrieval', icon: <SearchOutlined />, label: '知识检索' },
  { key: 'view', icon: <EyeOutlined />, label: '内容查看' },
  { key: 'graph', icon: <ApartmentOutlined />, label: '知识图谱构建' },
  { key: 'visualize', icon: <ShareAltOutlined />, label: '图谱可视化' },
  { key: 'logs', icon: <FileTextOutlined />, label: '知识库日志' },
  { key: 'gallery', icon: <PictureOutlined />, label: '图片画廊' },
];

export default function KbLayout() {
  const { kbId } = useParams<{ kbId: string }>();
  const navigate = useNavigate();
  const location = useLocation();
  const [kb, setKb] = useState<KnowledgeBase | null>(null);
  const [notFound, setNotFound] = useState(false);

  useEffect(() => {
    if (!kbId) return;
    setKb(null);
    setNotFound(false);
    getKnowledgeBase(kbId)
      .then(setKb)
      .catch(() => setNotFound(true));
  }, [kbId]);

  // 解析当前子页 key（/kb/:kbId/upload -> upload）
  const segments = location.pathname.split('/').filter(Boolean); // ['kb', kbId, 'upload']
  const sub = segments.length >= 3 ? segments[2] : 'upload';
  const selected = SUB_MENU.find((m) => m.key === sub)?.key || 'upload';

  if (notFound) {
    return (
      <div style={{ textAlign: 'center', padding: 80 }}>
        <Typography.Title level={4}>知识库不存在或已被删除</Typography.Title>
        <Button type="primary" icon={<ArrowLeftOutlined />} onClick={() => navigate('/')}>返回知识库列表</Button>
      </div>
    );
  }

  return (
    <Layout style={{ height: 'calc(100vh - 64px)', overflow: 'hidden', background: 'transparent' }}>
      <Sider
        width={200}
        style={{
          height: '100%',
          overflow: 'auto',
          backdropFilter: 'blur(14px) saturate(160%)',
          WebkitBackdropFilter: 'blur(14px) saturate(160%)',
          borderRight: '1px solid rgba(255,255,255,0.65)',
          paddingTop: 16,
          background: 'rgba(255,255,255,0.45)',
        }}
      >
        <div style={{ padding: '4px 16px 12px', borderBottom: '1px solid rgba(0,0,0,0.06)', marginBottom: 8 }}>
          <Button type="text" icon={<ArrowLeftOutlined />} onClick={() => navigate('/')} style={{ padding: 0, marginBottom: 8 }}>
            返回知识库列表
          </Button>
          <Space style={{ display: 'flex' }}>
            {kb ? (
              <Typography.Text strong style={{ color: '#262626', fontSize: 14 }} ellipsis>
                {kb.name}
              </Typography.Text>
            ) : (
              <LoadingOutlined />
            )}
          </Space>
          {kb?.description ? (
            <Typography.Paragraph type="secondary" style={{ fontSize: 12, margin: '4px 0 0' }} ellipsis={{ rows: 2 }}>
              {kb.description}
            </Typography.Paragraph>
          ) : null}
          {kb ? <Tag color="geekblue" style={{ marginTop: 6 }}>知识库</Tag> : null}
        </div>
        <Menu
          mode="inline"
          selectedKeys={[selected]}
          items={SUB_MENU}
          onClick={({ key }) => navigate(`/kb/${kbId}/${key}`)}
          style={{ background: 'transparent', borderInlineEnd: 'none' }}
        />
      </Sider>
      <Content style={{ overflow: 'auto', padding: '0 24px 24px' }}>
        <Outlet />
      </Content>
    </Layout>
  );
}
