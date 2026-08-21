import React from 'react';
import { Layout } from 'antd';
import {
  FolderOpenOutlined,
  SettingOutlined,
  RobotOutlined,
  PictureOutlined,
} from '@ant-design/icons';
import { useNavigate, useLocation, Outlet } from 'react-router-dom';

/* ============================================================
   主布局：一级导航改为「顶部居中玻璃标签栏」（macOS 风格）
   改造决策：一级菜单仅 4 项（文档类型/模型管理/智能体/图片画廊），
   按方案「3-4 项 → 居中玻璃标签栏」由侧边栏改为顶部标签栏；
   文档类型详情（/type/...）视为「文档类型」下的子页面，高亮对应标签。
   ============================================================ */

const { Header, Content } = Layout;

const NAV = [
  { key: '/', icon: <FolderOpenOutlined />, label: '文档类型' },
  { key: '/models', icon: <SettingOutlined />, label: '模型管理' },
  { key: '/agent', icon: <RobotOutlined />, label: '智能体' },
  { key: '/gallery', icon: <PictureOutlined />, label: '图片画廊' },
];

export default function MainLayout() {
  const navigate = useNavigate();
  const location = useLocation();
  // 文档类型详情（/type/...）与知识库详情（/kb/...）视为「文档类型」下的子页面，高亮对应标签
  let selected = '/';
  if (location.pathname.startsWith('/models')) selected = '/models';
  else if (location.pathname.startsWith('/agent')) selected = '/agent';
  else if (location.pathname.startsWith('/gallery')) selected = '/gallery';

  return (
    <Layout style={{ minHeight: '100vh', height: '100vh', overflow: 'hidden', background: 'transparent' }}>
      <Header
        style={{
          background: 'rgba(255,255,255,0.35)',
          backdropFilter: 'blur(18px) saturate(170%)',
          WebkitBackdropFilter: 'blur(18px) saturate(170%)',
          borderBottom: '1px solid rgba(255,255,255,0.55)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '0 24px',
          position: 'sticky',
          top: 0,
          zIndex: 100,
          height: 64,
        }}
      >
        <div className="glass-tabbar" style={{ flexWrap: 'wrap' }}>
          {NAV.map((item) => (
            <button
              key={item.key}
              type="button"
              className={`glass-tabbar-item${selected === item.key ? ' active' : ''}`}
              onClick={() => navigate(item.key)}
            >
              {item.icon}
              <span>{item.label}</span>
            </button>
          ))}
        </div>
      </Header>
      <Content style={{ padding: location.pathname.startsWith('/kb/') || location.pathname.startsWith('/type/') ? 0 : '24px 24px 0', overflow: 'auto' }}>
        <Outlet />
      </Content>
    </Layout>
  );
}
