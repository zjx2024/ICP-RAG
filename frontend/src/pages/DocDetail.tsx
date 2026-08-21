import { useCallback, useEffect, useState } from 'react';
import { Alert, Card, Empty, Space, Spin, Tag, Typography, message } from 'antd';
import { ArrowLeftOutlined, FileTextOutlined, ReloadOutlined } from '@ant-design/icons';
import { useNavigate, useParams } from 'react-router-dom';
import { fetchDocumentType, fetchIcpDocuments } from '@/api/icp';
import DocContentView from '@/components/DocContentView';
import { GlassButton } from '@/components/GlassUI';
import type { DocTypeInfo, ICPDocument } from '@/types';
import { DOC_STATUS_COLOR, DOC_STATUS_LABEL, formatDateTime } from '@/utils';

export default function DocDetail() {
  const { docType = '', docId = '' } = useParams<{ docType: string; docId: string }>();
  const navigate = useNavigate();
  const [typeInfo, setTypeInfo] = useState<DocTypeInfo | null>(null);
  const [document, setDocument] = useState<ICPDocument | null>(null);
  const [loading, setLoading] = useState(true);

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
        <Empty description="文档不存在或已被删除">
          <GlassButton onClick={() => navigate(`/type/${docType}/upload`)}>返回文档列表</GlassButton>
        </Empty>
      </Card>
    );
  }

  return (
    <div className="glass-fade-in">
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
            <GlassButton variant="text" size="small" icon={<ArrowLeftOutlined />} onClick={() => navigate(`/type/${docType}/upload`)}>
              返回列表
            </GlassButton>
          </Space>
        }
      >
        <Space size={[8, 8]} wrap>
          <Typography.Text type="secondary">更新时间：{formatDateTime(document.updated_at)}</Typography.Text>
          <Typography.Text type="secondary">解析次数：{document.parse_count || 0}</Typography.Text>
          {document.last_result?.message ? <Typography.Text type="secondary">{document.last_result.message}</Typography.Text> : null}
        </Space>
      </Card>

      {document.status === 'uploaded' ? (
        <Alert
          type="info"
          showIcon
          style={{ marginBottom: 16 }}
          message="该文档尚未解析"
          description="可以查看原始内容；结构化记录、语义索引和图结构需要先在文档列表中执行解析。"
        />
      ) : null}

      <Card className="glass">
        <DocContentView docType={docType} docId={docId} file_name={document.file_name} />
      </Card>
    </div>
  );
}
