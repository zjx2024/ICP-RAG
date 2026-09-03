import { useCallback, useEffect, useMemo, useState } from 'react';
import { Card, Empty, Select, Space, Typography, message } from 'antd';
import { FileTextOutlined } from '@ant-design/icons';
import { useNavigate, useParams } from 'react-router-dom';
import { fetchDocumentType, fetchIcpDocuments } from '@/api/icp';
import DocContentView, {
  DocumentContentModeSwitch,
  type DocumentContentMode,
} from '@/components/DocContentView';
import { GlassButton } from '@/components/GlassUI';
import type { ICPDocument, DocTypeInfo } from '@/types';

const CONTENT_HINT: Record<string, string> = {
  image: '选择一份或多份文档，查看原始内容、表格和图片',
  table: '选择一份或多份文档，查看原始内容、表格和图片',
  text: '选择一份或多份文档，查看原始内容、表格和图片',
};

const ALL_DOCUMENTS = '__all_documents__';

function sortByUpdatedAt(documents: ICPDocument[]): ICPDocument[] {
  return [...documents].sort((left, right) => (
    String(right.updated_at || '').localeCompare(String(left.updated_at || ''))
  ));
}

export default function TypeView() {
  const { docType = '' } = useParams<{ docType: string }>();
  const navigate = useNavigate();
  const [typeInfo, setTypeInfo] = useState<DocTypeInfo | null>(null);
  const [docs, setDocs] = useState<ICPDocument[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [contentMode, setContentMode] = useState<DocumentContentMode>('document');
  const [loading, setLoading] = useState(false);

  const loadType = useCallback(async (type: string) => {
    try {
      setTypeInfo(await fetchDocumentType(type));
    } catch {
      setTypeInfo(null);
    }
  }, []);

  const loadDocs = useCallback(async (type: string, preserveSelection = true) => {
    setLoading(true);
    try {
      const sorted = sortByUpdatedAt(await fetchIcpDocuments(type));
      setDocs(sorted);
      setSelectedIds((current) => {
        if (preserveSelection && current.includes(ALL_DOCUMENTS) && sorted.length) {
          return [ALL_DOCUMENTS];
        }
        const available = preserveSelection
          ? current.filter((id) => sorted.some((document) => document.doc_id === id))
          : [];
        return available.length ? available : (sorted[0] ? [sorted[0].doc_id] : []);
      });
    } catch (error) {
      message.error(`加载文档失败: ${(error as Error).message}`);
      setDocs([]);
      setSelectedIds([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!docType) return;
    setSelectedIds([]);
    setContentMode('document');
    loadType(docType);
    loadDocs(docType, false);
  }, [docType, loadDocs, loadType]);

  const selectedDocs = useMemo(() => {
    if (selectedIds.includes(ALL_DOCUMENTS)) return docs;
    const byId = new Map(docs.map((document) => [document.doc_id, document]));
    return selectedIds
      .map((id) => byId.get(id))
      .filter((document): document is ICPDocument => Boolean(document));
  }, [docs, selectedIds]);

  const handleSelectionChange = (nextIds: string[]) => {
    if (nextIds.includes(ALL_DOCUMENTS)) {
      // “全部文档”与具体文档互斥：选择全部时清除已有项；在全部状态下
      // 再选具体文档，则自动退出全部状态并保留新的选择顺序。
      setSelectedIds(selectedIds.includes(ALL_DOCUMENTS)
        ? nextIds.filter((id) => id !== ALL_DOCUMENTS)
        : [ALL_DOCUMENTS]);
      return;
    }
    setSelectedIds(nextIds);
  };

  const contentHint = CONTENT_HINT[typeInfo?.data_form || 'text'] || CONTENT_HINT.text;
  return (
    <div className="glass-fade-in type-page type-content-page" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <Card
        className="glass document-view-picker"
        styles={{ body: { padding: 16 } }}
        title={
          <div>
            <Typography.Text strong>内容查看</Typography.Text>
            <Typography.Text type="secondary" style={{ display: 'block', marginTop: 2, fontSize: 12 }}>
              {contentHint}
            </Typography.Text>
          </div>
        }
      >
        {docs.length === 0 && !loading ? (
          <Empty description="该类型暂无文档，请从七类文档知识库首页统一上传并解析">
            <GlassButton variant="primary" onClick={() => navigate('/')}>返回首页上传</GlassButton>
          </Empty>
        ) : (
          <div className="document-view-selector">
            <div>
              <Typography.Text type="secondary" style={{ display: 'block', marginBottom: 6, fontSize: 12 }}>
                选择要查看的文档
              </Typography.Text>
              <Select
                mode="multiple"
                value={selectedIds}
                onChange={handleSelectionChange}
                loading={loading}
                showSearch
                allowClear
                optionFilterProp="label"
                placeholder="请选择文档"
                maxTagCount="responsive"
                style={{ width: '100%' }}
                options={[
                  {
                    value: ALL_DOCUMENTS,
                    label: '全部文档',
                  },
                  ...docs.map((document) => ({
                    value: document.doc_id,
                    label: document.file_name,
                  })),
                ]}
              />
            </div>
          </div>
        )}
      </Card>

      {selectedDocs.length > 0 ? (
        <div className="document-view-mode-bar">
          <DocumentContentModeSwitch value={contentMode} onChange={setContentMode} />
        </div>
      ) : null}

      {selectedDocs.map((document) => (
        <Card
          key={document.doc_id}
          className="glass document-view-card"
          title={
            <Space wrap>
              <FileTextOutlined style={{ color: '#1677ff' }} />
              <Typography.Text strong ellipsis style={{ maxWidth: 'min(720px, 68vw)' }}>
                {document.file_name}
              </Typography.Text>
            </Space>
          }
          styles={{ body: { paddingTop: 12 } }}
        >
          <DocContentView
            docType={docType}
            docId={document.doc_id}
            file_name={document.file_name}
            mode={contentMode}
            showTabs={false}
          />
        </Card>
      ))}

      {docs.length > 0 && selectedDocs.length === 0 ? (
        <Card className="glass">
          <Empty description="请选择要查看的文档" />
        </Card>
      ) : null}
    </div>
  );
}
