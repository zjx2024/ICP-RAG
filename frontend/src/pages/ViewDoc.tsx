import { useEffect, useState, useCallback } from 'react';
import { Card, Tabs, Tag, Spin, Typography, Space, Empty, Alert, Modal, Button, Popconfirm, message } from 'antd';
import { DownOutlined, ClearOutlined } from '@ant-design/icons';
import { useParams } from 'react-router-dom';
import { fetchDocuments, fetchOriginalChunks, fetchVectorStoreContent, fetchPostgresContent, dedupIndex } from '@/api/kb';
import { formatMarkdown } from '@/utils';
import { GlassButton } from '@/components/GlassUI';

interface ImgItem {
  image_id: string;
  title: string;
  description: string;
  image_type: string;
  minio_path: string;
  image_url: string;
  chunk_id: string;
}

interface TblItem {
  table_id: string;
  title: string;
  description: string;
  meta_data: any;
  chunk_id: string;
}

export default function ViewDoc() {
  // kbId 由知识库详情路由提供（/kb/:kbId/view），仅展示本库文档
  const { kbId } = useParams<{ kbId: string }>();
  const [docs, setDocs] = useState<Array<{ doc_id: string; file_name: string; status: string }>>([]);
  const [selected, setSelected] = useState<{ doc_id: string; file_name: string }>();
  const [pickerOpen, setPickerOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [content, setContent] = useState<{ original?: any; vector?: any; postgres?: any }>({});
  const [preview, setPreview] = useState<ImgItem | null>(null);
  const [deduping, setDeduping] = useState(false);

  // 拉取当前知识库的 indexed/completed 文档
  const loadDocOptions = useCallback(async () => {
    if (!kbId) return;
    const d = await fetchDocuments(kbId).catch(() => []);
    setDocs(d.filter((x) => x.status === 'indexed' || x.status === 'completed'));
  }, [kbId]);

  useEffect(() => {
    loadDocOptions();
  }, [loadDocOptions]);

  const loadContent = async (docId: string) => {
    if (!kbId) return;
    setLoading(true);
    setContent({});
    try {
      const [original, vector, postgres] = await Promise.all([
        fetchOriginalChunks(kbId, docId).catch((e) => ({ error: e.message })),
        fetchVectorStoreContent(kbId, docId).catch((e) => ({ error: e.message })),
        fetchPostgresContent(kbId, docId).catch((e) => ({ error: e.message })),
      ]);
      setContent({ original, vector, postgres });
    } finally {
      setLoading(false);
    }
  };

  // 清除向量库重复索引块（以 PG 文档为有效依据，删除历史重复解析产生的冗余索引）
  const handleDedup = async () => {
    if (!kbId) return;
    setDeduping(true);
    try {
      const res = await dedupIndex(kbId);
      if (res.total_deleted > 0) {
        message.success(res.message);
        if (selected) await loadContent(selected.doc_id);
      } else {
        message.info(res.message || '没有发现重复索引块');
      }
    } catch (e) {
      message.error(`清除重复索引失败: ${(e as Error).message}`);
    } finally {
      setDeduping(false);
    }
  };

  // 表格：黑色描边 + 黑底表头 + 斑马纹
  const renderTable = (meta: any) => {
    const data = meta?.data || meta?.rows;
    if (!Array.isArray(data) || !Array.isArray(data[0])) return null;
    const headers = data[0] as string[];
    const rows = data.slice(1) as string[][];
    return (
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
        <thead>
          <tr>
            {headers.map((h, i) => (
              <th key={i} style={{ border: '1px solid #1f1f1f', padding: '6px 10px', background: '#1f1f1f', color: '#fff', fontWeight: 600, textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, ri) => (
            <tr key={ri} style={{ background: ri % 2 ? 'rgba(0,0,0,0.04)' : '#fff' }}>
              {r.map((c, ci) => (
                <td key={ci} style={{ border: '1px solid rgba(0,0,0,0.45)', padding: '6px 10px', color: '#262626', whiteSpace: 'nowrap' }}>{c}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    );
  };

  // 图片卡片：点击放大
  const renderImageCard = (img: ImgItem, width = 200) => (
    <div
      key={img.image_id}
      onClick={() => setPreview(img)}
      style={{ width, cursor: 'zoom-in', border: '1px solid rgba(0,0,0,0.5)', borderRadius: 8, padding: 8, background: 'rgba(255,255,255,0.65)', boxShadow: '0 2px 8px rgba(0,0,0,0.12)', transition: 'all 0.2s' }}
      onMouseEnter={(e) => { e.currentTarget.style.boxShadow = '0 4px 14px rgba(0,0,0,0.25)'; e.currentTarget.style.borderColor = '#000'; }}
      onMouseLeave={(e) => { e.currentTarget.style.boxShadow = '0 2px 8px rgba(0,0,0,0.12)'; e.currentTarget.style.borderColor = 'rgba(0,0,0,0.5)'; }}
    >
      <img src={img.image_url} alt={img.title} style={{ width: '100%', height: 140, objectFit: 'cover', borderRadius: 6, display: 'block', border: '1px solid rgba(0,0,0,0.15)' }} />
      <div style={{ fontSize: 12, marginTop: 6, color: '#262626', fontWeight: 600 }}>{img.title || '图片'}</div>
      <div style={{ fontSize: 11, color: '#8c8c8c', display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
        {img.description || '暂无描述'}
      </div>
    </div>
  );

  // 表格卡片：标题 + 描述 + 表格
  const renderTableCard = (t: TblItem) => (
    <div key={t.table_id} style={{ marginBottom: 12, border: '1px solid rgba(0,0,0,0.5)', borderRadius: 8, padding: 12, background: 'rgba(255,255,255,0.65)', boxShadow: '0 2px 8px rgba(0,0,0,0.12)' }}>
      <Space wrap>
        <Tag color="purple">表格</Tag>
        <Typography.Text strong style={{ fontSize: 14 }}>{t.title || '未命名表格'}</Typography.Text>
      </Space>
      {t.description && <div style={{ fontSize: 12, color: '#595959', margin: '6px 0' }}>{t.description}</div>}
      <div style={{ overflowX: 'auto', marginTop: 6 }}>{renderTable(t.meta_data)}</div>
    </div>
  );

  // 原文档分块：原文 + 关联的图片/表格（按 chunk_id 关联）
  const renderOriginalChunks = () => {
    if (content.original?.error) return <Alert type="error" message={content.original.error} />;
    const chunks: any[] = content.original?.chunks || [];
    if (!chunks.length) return <Empty description="暂无分块" />;
    const images: ImgItem[] = content.postgres?.images || [];
    const tables: TblItem[] = content.postgres?.tables || [];
    const matchedImgIds = new Set<string>();
    const matchedTblIds = new Set<string>();
    return (
      <div>
        {chunks.map((c, i) => {
          const imgs = images.filter((img) => img.chunk_id === c.chunk_id);
          const tbls = tables.filter((t) => t.chunk_id === c.chunk_id);
          imgs.forEach((img) => matchedImgIds.add(img.image_id));
          tbls.forEach((t) => matchedTblIds.add(t.table_id));
          return (
            <div key={c.chunk_id || i} style={{ marginBottom: 16, border: '1px solid rgba(0,0,0,0.45)', borderRadius: 8, padding: 12, background: 'rgba(255,255,255,0.55)' }}>
              <Space wrap>
                <Tag color="blue">分块 {c.chunk_index}</Tag>
                <Typography.Text strong>{c.title}</Typography.Text>
                {c.selected_for_kg && <Tag color="gold">已选入图谱</Tag>}
              </Space>
              <div style={{ fontSize: 13, color: '#262626', marginTop: 6 }} dangerouslySetInnerHTML={{ __html: formatMarkdown(c.content || '') }} />
              {(imgs.length > 0 || tbls.length > 0) && (
                <div style={{ marginTop: 10, borderTop: '1px dashed rgba(0,0,0,0.3)', paddingTop: 10 }}>
                  <Space wrap>
                    {imgs.map((img) => renderImageCard(img))}
                    {tbls.map((t) => renderTableCard(t))}
                  </Space>
                </div>
              )}
            </div>
          );
        })}
        {(() => {
          const orphanImgs = images.filter((img) => !matchedImgIds.has(img.image_id));
          const orphanTbls = tables.filter((t) => !matchedTblIds.has(t.table_id));
          if (!orphanImgs.length && !orphanTbls.length) return null;
          return (
            <div style={{ border: '1px dashed rgba(0,0,0,0.4)', borderRadius: 8, padding: 12 }}>
              <Typography.Text type="secondary">未关联到分块的图片与表格</Typography.Text>
              <Space wrap style={{ marginTop: 8 }}>
                {orphanImgs.map((img) => renderImageCard(img))}
                {orphanTbls.map((t) => renderTableCard(t))}
              </Space>
            </div>
          );
        })()}
      </div>
    );
  };

  // 结构化数据：图片网格 + 表格列表
  const renderPostgres = () => {
    if (content.postgres?.error) return <Alert type="error" message={content.postgres.error} />;
    const images: ImgItem[] = content.postgres?.images || [];
    const tables: TblItem[] = content.postgres?.tables || [];
    return (
      <div>
        <div style={{ fontWeight: 600, marginBottom: 10 }}>图片 <span style={{ color: '#999', fontWeight: 400 }}>({images.length})</span></div>
        {images.length ? (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>{images.map((img) => renderImageCard(img))}</div>
        ) : (
          <Typography.Text type="secondary">无图片</Typography.Text>
        )}
        <div style={{ borderTop: '1px dashed rgba(0,0,0,0.3)', margin: '16px 0' }} />
        <div style={{ fontWeight: 600, marginBottom: 10 }}>表格 <span style={{ color: '#999', fontWeight: 400 }}>({tables.length})</span></div>
        {tables.length ? tables.map((t) => renderTableCard(t)) : <Typography.Text type="secondary">无表格</Typography.Text>}
      </div>
    );
  };

  return (
    <div className="glass-fade-in">
      <Card
        className="glass glass-card-bordered"
        title="内容查看"
        style={{ marginBottom: 16 }}
        extra={(
          <Popconfirm
            title="确定清除重复索引块？"
            description="将删除向量库中因重复解析产生的冗余索引（有效文档不受影响）"
            okText="清除"
            cancelText="取消"
            onConfirm={handleDedup}
          >
            <GlassButton variant="primary" size="small" loading={deduping} icon={<ClearOutlined />}>清除重复索引块</GlassButton>
          </Popconfirm>
        )}
      >
        <Space>
          <div
            className="agent-picker agent-picker-sm"
            style={{ width: 'auto', maxWidth: 380, cursor: 'pointer' }}
            onClick={() => setPickerOpen(true)}
          >
            <span className="agent-picker-label">选择文档</span>
            <span className="agent-picker-value">
              {selected?.file_name || '点击选择要查看的文档'}
              <DownOutlined style={{ fontSize: 10 }} />
            </span>
          </div>
          {selected && <Tag color="green">已选择</Tag>}
          {loading && <Spin />}
        </Space>
      </Card>

      {selected && (
        <Card className="glass glass-card-bordered">
          <Tabs
            defaultActiveKey="original"
            items={[
              {
                key: 'original',
                label: `原文档分块 (${content.original?.chunks?.length || 0})`,
                children: renderOriginalChunks(),
              },
              {
                key: 'vector',
                label: `向量库内容 (${content.vector?.contents?.length || 0})`,
                children: content.vector?.error ? (
                  <Alert type="error" message={content.vector.error} />
                ) : content.vector?.contents?.length ? (
                  content.vector.contents.map((c: any, i: number) => (
                    <div key={i} style={{ marginBottom: 12, border: '1px solid rgba(0,0,0,0.4)', borderRadius: 8, padding: 8, background: 'rgba(255,255,255,0.5)' }}>
                      <Space wrap>
                        <Tag color={c.content_type === 'image' ? 'blue' : c.content_type === 'table' ? 'purple' : 'green'}>{c.content_type}</Tag>
                        <Typography.Text code style={{ fontSize: 12 }}>{c.chunk_id}</Typography.Text>
                      </Space>
                      <div style={{ fontSize: 13, marginTop: 4 }} dangerouslySetInnerHTML={{ __html: formatMarkdown(c.content || '') }} />
                    </div>
                  ))
                ) : (
                  <Empty description="暂无向量内容" />
                ),
              },
              {
                key: 'postgres',
                label: `结构化数据 (${(content.postgres?.images?.length || 0) + (content.postgres?.tables?.length || 0)})`,
                children: renderPostgres(),
              },
            ]}
          />
        </Card>
      )}

      {/* 文档选择弹窗 */}
      <Modal
        title="选择文档"
        open={pickerOpen}
        onCancel={() => setPickerOpen(false)}
        width={520}
        footer={[<Button key="c" onClick={() => setPickerOpen(false)}>取消</Button>]}
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxHeight: 460, overflow: 'auto' }}>
          {docs.length ? (
            docs.map((d) => (
              <div
                key={d.doc_id}
                className={['agent-picker-option', selected?.doc_id === d.doc_id ? 'selected' : ''].filter(Boolean).join(' ')}
                onClick={() => { setSelected({ doc_id: d.doc_id, file_name: d.file_name }); setPickerOpen(false); loadContent(d.doc_id); }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ fontWeight: 600, fontSize: 14 }}>{d.file_name}</span>
                  <Tag color={d.status === 'indexed' ? 'blue' : 'green'}>{d.status === 'indexed' ? '已索引' : '已完成'}</Tag>
                  {selected?.doc_id === d.doc_id && <span style={{ color: '#000' }}>✓</span>}
                </div>
              </div>
            ))
          ) : (
            <Empty description="暂无已索引/已完成的文档" />
          )}
        </div>
      </Modal>

      {/* 图片放大预览弹窗 */}
      <Modal
        title={preview?.title || '图片预览'}
        open={!!preview}
        onCancel={() => setPreview(null)}
        footer={null}
        width={760}
      >
        {preview && (
          <div>
            <img src={preview.image_url} alt={preview.title} style={{ width: '100%', borderRadius: 8, border: '1px solid rgba(0,0,0,0.4)' }} />
            <div style={{ marginTop: 12 }}>
              <Space wrap>
                <Tag color="blue">{preview.image_type}</Tag>
                <Typography.Text strong>{preview.title}</Typography.Text>
              </Space>
              <div style={{ marginTop: 8, padding: '10px 12px', background: 'rgba(0,0,0,0.04)', borderRadius: 6, fontSize: 13, color: '#262626' }}>
                <Typography.Text type="secondary" style={{ display: 'block' }}>图片描述</Typography.Text>
                {preview.description || '暂无描述'}
              </div>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
}
