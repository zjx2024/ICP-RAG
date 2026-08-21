import { useEffect, useState, useCallback } from 'react';
import { Card, Tag, Typography, Space, Empty, Modal, Button, Checkbox, Divider, message } from 'antd';
import { SyncOutlined, DownOutlined, FileTextOutlined, CheckOutlined } from '@ant-design/icons';
import { useParams } from 'react-router-dom';
import { fetchDocumentType, fetchIcpDocuments } from '@/api/icp';
import DocContentView from '@/components/DocContentView';
import { GlassButton } from '@/components/GlassUI';
import type { ICPDocument, DocTypeInfo } from '@/types';
import { DOC_STATUS_LABEL, DOC_STATUS_COLOR, formatDateTime } from '@/utils';

export default function TypeView() {
  const { docType = '' } = useParams<{ docType: string }>();
  const [typeInfo, setTypeInfo] = useState<DocTypeInfo | null>(null);
  const [docs, setDocs] = useState<ICPDocument[]>([]);
  const [loading, setLoading] = useState(false);
  // 默认全选该类型所有文档
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [previewId, setPreviewId] = useState<string>(); // 是否限制只展示某一份（null=全部）

  const loadType = useCallback(async (dt: string) => {
    try {
      setTypeInfo(await fetchDocumentType(dt));
    } catch {
      setTypeInfo(null);
    }
  }, []);

  const loadDocs = useCallback(async (dt: string, selectAll = true) => {
    setLoading(true);
    try {
      const list = await fetchIcpDocuments(dt);
      setDocs(list);
      if (selectAll) setSelectedIds(list.map((d) => d.doc_id));
    } catch (e) {
      message.error(`加载文档失败: ${(e as Error).message}`);
      setDocs([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!docType) return;
    loadType(docType);
    loadDocs(docType, true);
  }, [docType, loadType, loadDocs]);

  // 刷新时保持当前选择
  const handleRefresh = async () => {
    const list = await fetchIcpDocuments(docType).catch(() => []);
    setDocs(list);
    setSelectedIds((prev) => (prev.length ? prev.filter((id) => list.some((d) => d.doc_id === id)) : list.map((d) => d.doc_id)));
  };

  const selectedDocs = previewId ? docs.filter((d) => d.doc_id === previewId) : docs.filter((d) => selectedIds.includes(d.doc_id));
  const sortedSelected = [...selectedDocs].sort((a, b) => (b.updated_at || '').localeCompare(a.updated_at || ''));

  const toggleDoc = (id: string) => {
    setSelectedIds((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
  };

  const docNameMap = new Map(docs.map((d) => [d.doc_id, d.file_name]));

  return (
    <div className="glass-fade-in">
      <Card
        className="glass"
        style={{ marginBottom: 16 }}
        title={
          <Space>
            <span>内容查看</span>
            <Typography.Text type="secondary" style={{ fontSize: 12 }}>
              {typeInfo?.data_form === 'image'
                ? '图类：原文图片 / 向量化 / 三元组 / 图谱'
                : typeInfo?.data_form === 'table'
                  ? '表格类：原文 / 向量化 / PG 结构'
                  : '文本类：原文结构 / 向量化分块'}
            </Typography.Text>
          </Space>
        }
        extra={
          <Space>
            <GlassButton variant="text" size="small" icon={<SyncOutlined />} onClick={handleRefresh} loading={loading}>刷新</GlassButton>
          </Space>
        }
      >
        {docs.length === 0 ? (
          <Empty description="暂无已上传文档，请先到「文档上传」上传并解析" />
        ) : (
          <>
            <Space wrap style={{ marginBottom: 12, alignItems: 'center' }}>
              {/* 胶囊：选择要查看的文档（可单选/多选，默认全部） */}
              <div
                className="agent-picker agent-picker-sm"
                style={{ width: 'auto', maxWidth: 420, cursor: 'pointer' }}
                onClick={() => setPickerOpen(true)}
              >
                <span className="agent-picker-label">查看文档</span>
                <span className="agent-picker-value">
                  {previewId
                    ? docNameMap.get(previewId) || previewId
                    : selectedIds.length === 0
                      ? '点击选择文档'
                      : `已选 ${selectedIds.length} / ${docs.length} 份文档`}
                  <DownOutlined style={{ fontSize: 10 }} />
                </span>
              </div>
              {selectedIds.length > 0 && !previewId && (
                <Space size={4} wrap style={{ maxWidth: 420 }}>
                  {selectedIds.slice(0, 3).map((id) => (
                    <Tag key={id} closable onClose={(e) => { e.preventDefault(); toggleDoc(id); }} style={{ maxWidth: 140, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {docNameMap.get(id) || id}
                    </Tag>
                  ))}
                  {selectedIds.length > 3 ? <Tag>+{selectedIds.length - 3}</Tag> : null}
                </Space>
              )}
              {previewId && (
                <Tag color="blue" closable onClose={() => setPreviewId(undefined)}>单文档预览：{docNameMap.get(previewId) || previewId}</Tag>
              )}
            </Space>
            <Divider style={{ margin: '8px 0 16px' }} />
          </>
        )}
      </Card>

      {/* 文档内容区：默认展示全部选中文档，每份文档一个四视图卡片 */}
      {sortedSelected.length === 0 ? (
        <Card className="glass"><Empty description="未选择任何文档，请点击上方胶囊选择" /></Card>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {sortedSelected.map((d) => (
            <Card
              key={d.doc_id}
              className="glass"
              title={
                <Space wrap>
                  <FileTextOutlined style={{ color: '#722ed1' }} />
                  <Typography.Text strong>{d.file_name}</Typography.Text>
                  <Tag color={DOC_STATUS_COLOR[d.status]}>{DOC_STATUS_LABEL[d.status] || d.status}</Tag>
                  {d.parse_count ? <Tag>解析 {d.parse_count} 次</Tag> : null}
                  <Typography.Text type="secondary" style={{ fontSize: 12 }}>{formatDateTime(d.updated_at)}</Typography.Text>
                </Space>
              }
              extra={
                <GlassButton variant="text" size="small" onClick={() => setPreviewId(d.doc_id)}>单独查看</GlassButton>
              }
            >
              <DocContentView docType={docType} docId={d.doc_id} file_name={d.file_name} />
            </Card>
          ))}
        </div>
      )}

      {/* 文档选择弹窗（胶囊）：多选 */}
      <Modal
        title="选择要查看的文档（默认全部，可单选/多选）"
        open={pickerOpen}
        onCancel={() => setPickerOpen(false)}
        width={560}
        footer={[
          <Button key="all" onClick={() => { setSelectedIds(docs.map((d) => d.doc_id)); setPreviewId(undefined); }}>全选</Button>,
          <Button key="none" onClick={() => { setSelectedIds([]); setPreviewId(undefined); }}>清空</Button>,
          <Button key="ok" type="primary" onClick={() => setPickerOpen(false)}>确定</Button>,
        ]}
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxHeight: 420, overflow: 'auto' }}>
          {docs.map((d) => {
            const checked = selectedIds.includes(d.doc_id);
            return (
              <div
                key={d.doc_id}
                className={['agent-picker-option', checked ? 'selected' : ''].filter(Boolean).join(' ')}
                onClick={() => toggleDoc(d.doc_id)}
              >
                <Checkbox checked={checked} onClick={(e) => e.stopPropagation()} onChange={() => toggleDoc(d.doc_id)} />
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, flex: 1, minWidth: 0 }}>
                  <span style={{ fontWeight: 600, fontSize: 14, minWidth: 0, flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{d.file_name}</span>
                  <Tag color={DOC_STATUS_COLOR[d.status]}>{DOC_STATUS_LABEL[d.status] || d.status}</Tag>
                  {checked && <CheckOutlined style={{ color: '#000' }} />}
                </div>
              </div>
            );
          })}
        </div>
      </Modal>
    </div>
  );
}
