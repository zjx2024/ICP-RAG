import { useEffect, useState } from 'react';
import type { ReactNode } from 'react';
import { Typography } from 'antd';
import {
  ArrowLeftOutlined,
  DatabaseOutlined,
  EyeOutlined,
  FileTextOutlined,
  LoadingOutlined,
  ShareAltOutlined,
  TableOutlined,
} from '@ant-design/icons';
import { Outlet, useLocation, useNavigate, useParams } from 'react-router-dom';
import { fetchDocumentType } from '@/api/icp';
import type { DocTypeInfo } from '@/types';

const TYPE_NAV = [
  { key: 'documents', icon: <DatabaseOutlined />, label: '文档' },
  { key: 'view', icon: <EyeOutlined />, label: '内容' },
  { key: 'visualize', icon: <ShareAltOutlined />, label: '关系图' },
];

const FORM_LABEL: Record<DocTypeInfo['data_form'], string> = {
  table: '表格文档',
  image: '工程图文档',
  text: '图文文档',
};

const FORM_ICON: Record<DocTypeInfo['data_form'], ReactNode> = {
  table: <TableOutlined />,
  image: <ShareAltOutlined />,
  text: <FileTextOutlined />,
};

function formatInputFormats(formats?: string) {
  if (!formats) return '';
  const values = new Set<string>();
  formats.split(',').forEach((raw) => {
    const ext = raw.trim().toLowerCase();
    if (['.doc', '.docx'].includes(ext)) values.add('Word');
    else if (['.xls', '.xlsx', '.xlsm'].includes(ext)) values.add('Excel');
    else if (['.txt', '.md'].includes(ext)) values.add('文本');
    else if (['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'].includes(ext)) values.add('图片');
  });
  return Array.from(values).join(' / ');
}

export default function TypeLayout() {
  const { docType = '' } = useParams<{ docType: string }>();
  const navigate = useNavigate();
  const location = useLocation();
  const [typeInfo, setTypeInfo] = useState<DocTypeInfo | null>(null);
  const [notFound, setNotFound] = useState(false);

  useEffect(() => {
    if (!docType) return;
    let active = true;
    setTypeInfo(null);
    setNotFound(false);
    fetchDocumentType(docType)
      .then((value) => { if (active) setTypeInfo(value); })
      .catch(() => { if (active) setNotFound(true); });
    return () => { active = false; };
  }, [docType]);

  const segments = location.pathname.split('/').filter(Boolean);
  const raw = segments.length >= 3 ? segments[2] : 'documents';
  const selected = raw === 'doc' || raw === 'upload' || raw === 'logs' ? 'documents' : raw;
  const formatLabel = typeInfo ? FORM_LABEL[typeInfo.data_form] : '';
  const inputLabel = formatInputFormats(typeInfo?.input_formats);

  if (notFound) {
    return (
      <div className="type-workspace-error">
        <Typography.Title level={4}>文档类型不存在或已停用</Typography.Title>
        <button type="button" className="type-back-button" onClick={() => navigate('/')}>
          <ArrowLeftOutlined /> 返回七类文档
        </button>
      </div>
    );
  }

  return (
    <div className="type-workspace">
      <header className="type-workspace-header">
        <div className="type-workspace-heading">
          <button
            type="button"
            className="type-back-button"
            onClick={() => navigate('/')}
            aria-label="返回七类文档知识库"
          >
            <ArrowLeftOutlined />
            <span>七类文档</span>
          </button>

          <div className="type-workspace-identity">
            <span className="type-workspace-icon" aria-hidden="true">
              {typeInfo ? FORM_ICON[typeInfo.data_form] : <LoadingOutlined />}
            </span>
            <div className="type-workspace-copy">
              <div className="type-workspace-title-row">
                <Typography.Title level={4} ellipsis style={{ margin: 0 }}>
                  {typeInfo?.name || '正在加载文档类型'}
                </Typography.Title>
                {typeInfo && <span className="type-workspace-kind">{formatLabel}</span>}
              </div>
            </div>
          </div>

          {inputLabel ? (
            <div className="type-workspace-format" title={typeInfo?.input_formats}>
              <span>支持格式</span>
              <strong>{inputLabel}</strong>
            </div>
          ) : null}
        </div>

        <nav className="type-workspace-nav" aria-label="文档类型工作区导航">
          {TYPE_NAV.map((item) => (
            <button
              key={item.key}
              type="button"
              className={`type-workspace-nav-item${selected === item.key ? ' active' : ''}`}
              aria-current={selected === item.key ? 'page' : undefined}
              onClick={() => navigate(`/type/${docType}/${item.key}`)}
            >
              {item.icon}
              <span>{item.label}</span>
            </button>
          ))}
        </nav>
      </header>

      <main className="type-workspace-content">
        <Outlet />
      </main>
    </div>
  );
}
