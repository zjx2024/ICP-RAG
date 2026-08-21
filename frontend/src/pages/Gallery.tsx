import React, { useEffect, useState, useCallback } from 'react';
import { Card, Input, Select, Button, Empty, Spin, Modal, Tag, Statistic, Row, Col, Space, Typography } from 'antd';
import { ReloadOutlined } from '@ant-design/icons';
import { useParams } from 'react-router-dom';
import { fetchImageDocs, fetchDocMediaDetail, resolveImageSrc, getTypeLabel } from '@/api/images';
import type { ImageDocsResponse, ImageItem, TableItem, DocMediaDetail } from '@/types';
import { GlassButton } from '@/components/GlassUI';

// meta_data.data / rows 二维数组 → 表格
function renderTableFromMeta(meta: Record<string, unknown>): { headers: string[]; rows: string[][] } {
  const data = (meta.data || meta.rows) as unknown;
  if (Array.isArray(data) && Array.isArray(data[0])) {
    const arr = data as string[][];
    if (arr.length === 0) return { headers: [], rows: [] };
    return { headers: arr[0], rows: arr.slice(1) };
  }
  return { headers: [], rows: [] };
}

export default function Gallery() {
  // kbId 可选：/gallery 为全局画廊（无 kbId），/kb/:kbId/gallery 为该知识库画廊
  const { kbId } = useParams<{ kbId: string }>();
  const isKbScoped = !!kbId;
  const [docs, setDocs] = useState<ImageDocsResponse['documents']>([]);
  const [filterDoc, setFilterDoc] = useState<string>('');
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(false);
  const [details, setDetails] = useState<Record<string, DocMediaDetail>>({});
  const [modalImg, setModalImg] = useState<{ src: string; title: string; desc: string } | null>(null);

  const totalImages = docs.reduce((s, d) => s + (d.image_count || 0), 0);
  const totalTables = Object.values(details).reduce((s, d) => s + (d.table_count || 0), 0);

  const loadImages = useCallback(async () => {
    setLoading(true);
    setDetails({});
    try {
      const res = await fetchImageDocs(kbId);
      setDocs(res.documents || []);
    } catch (e) {
      // 显示空
    } finally {
      setLoading(false);
    }
  }, [kbId]);

  useEffect(() => {
    loadImages();
  }, [loadImages]);

  // 串行拉取详情（保持原行为，避免后端压力）
  useEffect(() => {
    let cancelled = false;
    const filtered = docs.filter((d) => {
      if (filterDoc && d.doc_id !== filterDoc) return false;
      if (search && !d.doc_id.includes(search)) return false;
      return true;
    });
    (async () => {
      const acc: Record<string, DocMediaDetail> = {};
      for (const doc of filtered) {
        try {
          const detail = await fetchDocMediaDetail(doc.doc_id);
          if (!cancelled) {
            acc[doc.doc_id] = detail;
            setDetails((prev) => ({ ...prev, [doc.doc_id]: detail }));
          }
        } catch {
          /* ignore */
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [docs, filterDoc, search]);

  const filtered = docs.filter((d) => {
    if (filterDoc && d.doc_id !== filterDoc) return false;
    if (search && !d.doc_id.toLowerCase().includes(search.toLowerCase())) return false;
    return true;
  });

  return (
    <div className="glass-fade-in">
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={8}><Card className="glass"><Statistic title="文档数" value={docs.length} /></Card></Col>
        <Col span={8}><Card className="glass"><Statistic title="图片数" value={totalImages} /></Card></Col>
        <Col span={8}><Card className="glass"><Statistic title="表格数" value={totalTables} /></Card></Col>
      </Row>

      <Card
        className="glass"
        title={isKbScoped ? '知识库图片画廊' : '全局图片画廊'}
        extra={
          <Space>
            <Input.Search
              placeholder="搜索 doc_id"
              allowClear
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              style={{ width: 220 }}
            />
            <Select
              placeholder="按文档过滤"
              allowClear
              value={filterDoc || undefined}
              onChange={(v) => setFilterDoc(v || '')}
              style={{ width: 220 }}
              options={docs.map((d) => ({ value: d.doc_id, label: `${d.doc_id} (${d.image_count})` }))}
            />
            <GlassButton variant="text" icon={<ReloadOutlined />} onClick={loadImages} loading={loading}>刷新</GlassButton>
          </Space>
        }
      >
        {loading && filtered.length === 0 ? (
          <div style={{ textAlign: 'center', padding: 40 }}><Spin /></div>
        ) : filtered.length === 0 ? (
          <Empty description="没有匹配的文档" />
        ) : (
          filtered.map((doc) => {
            const detail = details[doc.doc_id];
            return (
              <div key={doc.doc_id} style={{ marginBottom: 24 }}>
                <Typography.Title level={5} style={{ color: '#262626' }}>
                  📄 {doc.doc_id}
                  <Typography.Text type="secondary" style={{ fontSize: 13, marginLeft: 12 }}>
                    图片 {doc.image_count} · 表格 {detail ? detail.table_count : '--'}
                  </Typography.Text>
                </Typography.Title>
                {!detail ? (
                  <div style={{ textAlign: 'center', padding: 20 }}><Spin size="small" /></div>
                ) : (
                  <div>
                    {detail.images.length > 0 && (
                      <div
                        style={{
                          display: 'grid',
                          gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
                          gap: 16,
                        }}
                      >
                        {detail.images.map((img: ImageItem, i: number) => {
                          const src = resolveImageSrc(img, doc.doc_id);
                          const desc = img.description || '';
                          return (
                            <Card
                              key={img.image_id || i}
                              size="small"
                              hoverable
                              className="glass"
                              cover={
                                <div style={{ height: 150, overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(255,255,255,0.4)' }}>
                                  <img
                                    src={src}
                                    alt={img.title}
                                    style={{ maxWidth: '100%', maxHeight: 150, objectFit: 'cover' }}
                                    onClick={() => setModalImg({ src, title: img.title, desc })}
                                    onError={(e) => {
                                      (e.target as HTMLImageElement).src =
                                        'data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%22120%22 height=%22120%22%3E%3Crect fill=%22%23ccc%22 width=%22120%22 height=%22120%22/%3E%3C/svg%3E';
                                    }}
                                  />
                                </div>
                              }
                            >
                              <div style={{ fontSize: 13, color: '#262626', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                {img.title || '未命名图片'}
                              </div>
                              <Tag color="blue" style={{ marginTop: 4 }}>{getTypeLabel(img.image_type)}</Tag>
                              {desc && (
                                <div style={{ fontSize: 12, color: '#595959', marginTop: 4, maxHeight: 60, overflow: 'hidden' }}>
                                  {desc.length > 120 ? `${desc.slice(0, 120)}…` : desc}
                                </div>
                              )}
                            </Card>
                          );
                        })}
                      </div>
                    )}
                    {detail.tables.length > 0 && (
                      <div style={{ marginTop: 16 }}>
                        {detail.tables.map((tbl: TableItem, i: number) => {
                          const { headers, rows } = renderTableFromMeta(tbl.meta_data);
                          if (headers.length === 0) return null;
                          return (
                            <Card key={tbl.table_id || i} size="small" className="glass-dense" style={{ marginBottom: 12 }}>
                              <div style={{ fontWeight: 600, marginBottom: 8 }}>{tbl.title || `表格 ${i + 1}`}</div>
                              {tbl.description && (
                                <Typography.Text type="secondary" style={{ fontSize: 12 }}>{tbl.description}</Typography.Text>
                              )}
                              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13, marginTop: 8 }}>
                                <thead>
                                  <tr>{headers.map((h, hi) => <th key={hi} style={{ border: '1px solid #ddd', padding: '4px 8px', background: 'rgba(0,0,0,0.05)' }}>{h}</th>)}</tr>
                                </thead>
                                <tbody>
                                  {rows.map((r, ri) => (
                                    <tr key={ri}>{r.map((c, ci) => <td key={ci} style={{ border: '1px solid #ddd', padding: '4px 8px' }}>{c}</td>)}</tr>
                                  ))}
                                </tbody>
                              </table>
                            </Card>
                          );
                        })}
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })
        )}
      </Card>

      <Modal
        open={!!modalImg}
        footer={null}
        onCancel={() => setModalImg(null)}
        width={800}
        title={modalImg?.title}
      >
        {modalImg && (
          <div>
            <img src={modalImg.src} alt={modalImg.title} style={{ maxWidth: '100%' }} />
            {modalImg.desc && modalImg.desc !== '暂无解析描述' && (
              <Typography.Paragraph style={{ marginTop: 12 }}>{modalImg.desc}</Typography.Paragraph>
            )}
          </div>
        )}
      </Modal>
    </div>
  );
}
