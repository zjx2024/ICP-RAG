import { Layout } from 'antd';
import {
  FolderOpenOutlined,
  SettingOutlined,
  RobotOutlined,
} from '@ant-design/icons';
import { useNavigate, useLocation, Outlet } from 'react-router-dom';

const { Header, Content } = Layout;

const NAV = [
  { key: '/', icon: <FolderOpenOutlined />, label: '七类文档知识库' },
  { key: '/diagnosis', icon: <RobotOutlined />, label: 'ICP 故障诊断' },
  { key: '/models', icon: <SettingOutlined />, label: '系统设置' },
];

export default function MainLayout() {
  const navigate = useNavigate();
  const location = useLocation();

  let selected = '/';
  if (location.pathname.startsWith('/diagnosis') || location.pathname.startsWith('/agent')) {
    selected = '/diagnosis';
  } else if (location.pathname.startsWith('/models')) {
    selected = '/models';
  }

  const isFullWorkspace = location.pathname.startsWith('/type/') || location.pathname.startsWith('/diagnosis');

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
          justifyContent: 'space-between',
          padding: '0 24px',
          position: 'sticky',
          top: 0,
          zIndex: 100,
          height: 64,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, minWidth: 260 }}>
          <div
            style={{
              width: 34,
              height: 34,
              borderRadius: 10,
              display: 'grid',
              placeItems: 'center',
              background: '#111',
              color: '#fff',
              fontWeight: 700,
              letterSpacing: 1,
            }}
          >
            ICP
          </div>
          <div style={{ lineHeight: 1.25 }}>
            <div style={{ fontWeight: 700, color: '#1f1f1f' }}>ICP 故障诊断知识系统</div>
            <div style={{ fontSize: 11, color: '#8c8c8c' }}>七类文档 · 混合检索 · 多源证据</div>
          </div>
        </div>

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

        <div style={{ minWidth: 260, textAlign: 'right', color: '#8c8c8c', fontSize: 12 }}>
          诊断结果需结合正式维修程序复核
        </div>
      </Header>
      <Content style={{ padding: isFullWorkspace ? 0 : '24px 24px 0', overflow: 'auto' }}>
        <Outlet />
      </Content>
    </Layout>
  );
}
