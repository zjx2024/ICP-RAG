import React, { useEffect, useState } from 'react';
import { Layout, Menu, Typography, Space, Tag } from 'antd';
import {
  ArrowLeftOutlined, CloudUploadOutlined, EyeOutlined,
  ApartmentOutlined, ShareAltOutlined, FileTextOutlined,
  LoadingOutlined,
} from '@ant-design/icons';
import { useParams, useNavigate, useLocation, Outlet } from 'react-router-dom';
import { fetchDocumentType } from '@/api/icp';
import type { DocTypeInfo } from '@/types';

const { Sider, Content } = Layout;

// 全部类型共有的侧边栏项
const BASE_MENU = [
  { key: 'upload', icon: <CloudUploadOutlined />, label: '文档上传' },
  { key: 'view', icon: <EyeOutlined />, label: '内容查看' },
  { key: 'logs', icon: <FileTextOutlined />, label: '日志管理' },
];

// 图类（fta/rbd）额外侧边栏项：两阶段流程（确认三元组→生成图谱 / 可视化查看）
const IMAGE_MENU = [
  { key: 'graph', icon: <ApartmentOutlined />, label: '知识图谱生成' },
  { key: 'visualize', icon: <ShareAltOutlined />, label: '知识图谱可视化' },
];

export default function TypeLayout() {
  const { docType = '' } = useParams<{ docType: string }>();
  const navigate = useNavigate();
  const location = useLocation();
  const [typeInfo, setTypeInfo] = useState<DocTypeInfo | null>(null);
  const [notFound, setNotFound] = useState(false);

  useEffect(() => {
    if (!docType) return;
    setTypeInfo(null);
    setNotFound(false);
    fetchDocumentType(docType)
      .then(setTypeInfo)
      .catch(() => setNotFound(true));
  }, [docType]);

  // 解析当前子页 key（/type/:docType/upload -> upload；doc/:docId 视为内容查看）
  const segments = location.pathname.split('/').filter(Boolean); // ['type', docType, 'upload']
  const raw = segments.length >= 3 ? segments[2] : 'upload';
  const selected = raw === 'doc' ? 'view' : raw;

  // 图类额外显示图谱生成/可视化菜单
  const isImage = typeInfo?.data_form === 'image';
  const SUB_MENU = isImage ? [...BASE_MENU, ...IMAGE_MENU] : BASE_MENU;

  if (notFound) {
    return (
      <div style={{ textAlign: 'center', padding: 80 }}>
        <Typography.Title level={4}>文档类型不存在或已被停用</Typography.Title>
        <button type="button" className="glass-tabbar-item" onClick={() => navigate('/')}>返回首页</button>
      </div>
    );
  }

  return (
    <Layout style={{ height: 'calc(100vh - 64px)', overflow: 'hidden', background: 'transparent' }}>
      <Sider
        width={210}
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
          <button type="button" className="glass-tabbar-item" style={{ marginBottom: 10 }} onClick={() => navigate('/')}>
            <ArrowLeftOutlined /> 返回类型列表
          </button>
          <Space style={{ display: 'flex' }}>
            {typeInfo ? (
              <Typography.Text strong style={{ color: '#262626', fontSize: 14 }} ellipsis>
                {typeInfo.name}
              </Typography.Text>
            ) : (
              <LoadingOutlined />
            )}
          </Space>
          {typeInfo?.description ? (
            <Typography.Paragraph type="secondary" style={{ fontSize: 12, margin: '4px 0 0' }} ellipsis={{ rows: 2 }}>
              {typeInfo.description}
            </Typography.Paragraph>
          ) : null}
          <Space size={4} wrap style={{ marginTop: 6 }}>
            <Tag color={isImage ? 'purple' : typeInfo?.data_form === 'table' ? 'geekblue' : 'green'}>
              {typeInfo?.data_form === 'image' ? '图类' : typeInfo?.data_form === 'table' ? '表格类' : '文本类'}
            </Tag>
            {typeInfo?.input_formats ? <Tag>{typeInfo.input_formats}</Tag> : null}
            {isImage ? <Tag color="gold">两阶段图谱</Tag> : null}
          </Space>
        </div>
        <Menu
          mode="inline"
          selectedKeys={[selected]}
          items={SUB_MENU}
          onClick={({ key }) => navigate(`/type/${docType}/${key}`)}
          style={{ background: 'transparent', borderInlineEnd: 'none' }}
        />
      </Sider>
      <Content style={{ overflow: 'auto', padding: '0 24px 24px' }}>
        <Outlet />
      </Content>
    </Layout>
  );
}
