import { useCallback, useEffect, useState } from 'react';
import { Alert, Card, Empty, Space, Spin, Tag, Typography, message } from 'antd';
import { ArrowLeftOutlined, FileTextOutlined, ReloadOutlined } from '@ant-design/icons';
import { useNavigate, useParams } from 'react-router-dom';
import { fetchDocumentType, fetchIcpDocuments } from '@/api/icp';
import DocContentView, {
  DocumentContentModeSwitch,
  type DocumentContentMode,
} from '@/components/DocContentView';
import { GlassButton } from '@/components/GlassUI';
import type { DocTypeInfo, ICPDocument } from '@/types';
import { DOC_STATUS_COLOR, DOC_STATUS_LABEL, formatDateTime } from '@/utils';

export default function DocDetail() {
  const { docType = '', docId = '' } = useParams<{ docType: string; docId: string }>();
  const navigate = useNavigate();
  const [typeInfo, setTypeInfo] = useState<DocTypeInfo | null>(null);
  const [document, setDocument] = useState<ICPDocument | null>(null);
  const [loading, setLoading] = useState(true);
  const [contentMode, setContentMode] = useState<DocumentContentMode>('document');

  const load = useCallback(async () => {
    if (!docType || !docId) return;
    setLoading(true);
    try {
      const [type, documents] = await Promise.all([
        fetchDocumentType(docType),
        fetchIcpDocuments(docType),
      ]);
      setTypeInfo(type);
      setDocument(documents.find((item) => item.doc_id === docId) || null);
    } catch (error) {
      message.error(`加载文档详情失败：${(error as Error).message}`);
      setDocument(null);
    } finally {
      setLoading(false);
    }
  }, [docType, docId]);

  useEffect(() => { load(); }, [load]);

  if (loading) {
    return <div style={{ display: 'grid', placeItems: 'center', minHeight: '60vh' }}><Spin size="large" /></div>;
  }

  if (!document) {
    return (
      <Card className="glass">
        <Empty description="文档不存在或已被删除；需要添加文档时，请从七类文档知识库首页统一上传">
          <GlassButton onClick={() => navigate(`/type/${docType}/documents`)}>返回文档列表</GlassButton>
        </Empty>
      </Card>
    );
  }

  const parseMessage = document.last_result && typeof document.last_result === 'object'
    ? String((document.last_result as Record<string, unknown>).message || '')
    : '';

  return (
    <div className="glass-fade-in type-page type-document-detail-page">
      <Card
        className="glass"
        style={{ marginBottom: 16 }}
        title={
          <Space wrap>
            <FileTextOutlined style={{ color: '#1677ff' }} />
            <Typography.Text strong>{document.file_name}</Typography.Text>
            <Tag color={DOC_STATUS_COLOR[document.status]}>{DOC_STATUS_LABEL[document.status] || document.status}</Tag>
            <Tag>{typeInfo?.name || docType}</Tag>
          </Space>
        }
        extra={
          <Space>
            <GlassButton variant="text" size="small" icon={<ReloadOutlined />} onClick={load}>刷新</GlassButton>
            <GlassButton variant="text" size="small" icon={<ArrowLeftOutlined />} onClick={() => navigate(`/type/${docType}/documents`)}>
              返回列表
            </GlassButton>
          </Space>
        }
      >
        <Space size={[8, 8]} wrap>
          <Typography.Text type="secondary">更新时间：{formatDateTime(document.updated_at)}</Typography.Text>
          <Typography.Text type="secondary">解析次数：{document.parse_count || 0}</Typography.Text>
          {parseMessage ? <Typography.Text type="secondary">{parseMessage}</Typography.Text> : null}
        </Space>
      </Card>

      {document.status === 'uploaded' ? (
        <Alert
          type="info"
          showIcon
          style={{ marginBottom: 16 }}
          message="该文档还没有完成处理"
          description="返回文档列表，点击“解析”后即可查看提取结果。"
        />
      ) : null}

      <div className="document-view-mode-bar">
        <DocumentContentModeSwitch value={contentMode} onChange={setContentMode} />
      </div>

      <Card className="glass document-detail-content-card">
        <DocContentView
          docType={docType}
          docId={docId}
          file_name={document.file_name}
          mode={contentMode}
          showTabs={false}
        />
      </Card>
    </div>
  );
}
