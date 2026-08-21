import React, { useEffect, useState } from 'react';
import {
  Card, Input, InputNumber, Radio, Space, Slider, Typography, Tag,
  Divider, Row, Col, Alert, message, Tooltip, Empty, Modal, Button,
} from 'antd';
import { SearchOutlined, ClearOutlined, QuestionCircleOutlined, DownOutlined } from '@ant-design/icons';
import { Link, useParams } from 'react-router-dom';
import { fetchRetrievalModes, fetchRetrievalStats, search } from '@/api/retrieval';
import { GlassButton, GlassStatCard } from '@/components/GlassUI';
import type { SearchResponse, SearchResultItem } from '@/types';
import { useGlobalStore } from '@/store';
import { formatMarkdown } from '@/utils';

const VECTOR_MODES = ['es_hybrid', 'es_vector', 'graph_enhanced', 'hybrid_all', 'rag_fusion'];

interface ModeDef {
  mode: string;
  name: string;
  description: string;
}

export default function Retrieval() {
  // kbId 由知识库详情路由提供（/kb/:kbId/retrieval）
  const { kbId } = useParams<{ kbId: string }>();
  const [modes, setModes] = useState<ModeDef[]>([]);
  const [mode, setMode] = useState('es_hybrid');
  const [modePickerOpen, setModePickerOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [topK, setTopK] = useState(10);
  const [keywordWeight, setKeywordWeight] = useState(0.5);
  const [vectorWeight, setVectorWeight] = useState(0.5);
  const [depth, setDepth] = useState(2);
  const [minWeight, setMinWeight] = useState(1);
  const [allocMode, setAllocMode] = useState<'auto' | 'manual'>('auto');
  const [alloc, setAlloc] = useState({ chunks_count: 5, entities_count: 5, reports_count: 3 });
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [stats, setStats] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [error, setError] = useState('');
  const [compare, setCompare] = useState<{ total: number; chunks: number; entities: number; relations: number; reports: number } | null>(null);
  const [modelConfigStatus, setModelConfigStatus] = useState<any>(useGlobalStore.getState().modelConfigStatus);

  useEffect(() => {
    const unsub = useGlobalStore.subscribe((s) => setModelConfigStatus(s.modelConfigStatus));
    return unsub;
  }, []);

  useEffect(() => {
    if (kbId) {
      fetchRetrievalModes(kbId).then((r) => setModes(r.modes || [])).catch(() => {});
      fetchRetrievalStats(kbId).then(setStats).catch(() => setStats(null));
    }
  }, [kbId]);

  const checkModels = (): string | null => {
    const embedOk = modelConfigStatus?.embedding_configured;
    if (!embedOk) return '未配置 Embedding 模型，向量检索不可用';
    return null;
  };

  const doSearch = async (modeOverride?: string, paramsOverride?: Record<string, string | number | boolean>) => {
    if (!kbId) { message.warning('请选择知识库'); return; }
    if (!query.trim()) { message.warning('请输入查询内容'); return; }
    const m = modeOverride || mode;
    if (VECTOR_MODES.includes(m)) {
      const tip = checkModels();
      if (tip) { message.error(tip); return; }
    }
    setLoading(true);
    setError('');
    setResult(null);
    setCompare(null);
    try {
      const params: Record<string, string | number | boolean> = { query: query.trim(), top_k: topK };
      if (m === 'es_hybrid') { params.keyword_weight = keywordWeight; params.vector_weight = vectorWeight; }
      if (m === 'kg_graph') { params.depth = depth; params.top_k = topK; }
      if (m === 'kg_relation_weighted') { params.depth = depth; params.min_weight = minWeight; }
      if (m === 'rag_fusion' || m === 'hybrid_all') {
        if (allocMode === 'manual') {
          if (alloc.chunks_count) params.chunks_count = alloc.chunks_count;
          if (alloc.entities_count) params.entities_count = alloc.entities_count;
          if (alloc.reports_count) params.reports_count = alloc.reports_count;
        }
      }
      const res = await search(kbId, m, paramsOverride || params);
      setResult(res);
      if (res.total_chunks !== undefined || res.total_entities !== undefined) {
        setCompare({
          total: res.total_results,
          chunks: res.total_chunks || 0,
          entities: res.total_entities || 0,
          relations: 0,
          reports: res.total_reports || 0,
        });
      } else {
        const counts = { text: 0, entity: 0, graph: 0, community_report: 0 };
        (res.results || []).forEach((item) => {
          counts[detectResultType(item) as keyof typeof counts]++;
        });
        setCompare({
          total: res.total_results,
          chunks: counts.text,
          entities: counts.entity,
          relations: counts.graph,
          reports: counts.community_report,
        });
      }
    } catch (e) {
      setError((e as Error).message);
      message.error(`检索失败: ${(e as Error).message}`);
    } finally {
      setLoading(false);
    }
  };

  // ---- 结果类型识别（与旧版一致）：社区报告 / 关系 / 实体 / 文本 ----
  const detectResultType = (item: SearchResultItem): string => {
    if (item.title && (item.findings || item.entities)) return 'community_report';
    if (item.relation || item.source_entity) return 'graph';
    if (item.entity_id) return 'entity';
    return 'text';
  };

  const TYPE_CONFIG: Record<string, { color: string; label: string }> = {
    community_report: { color: 'green', label: '社区报告' },
    graph: { color: 'purple', label: '关系' },
    entity: { color: 'blue', label: '实体' },
    text: { color: 'default', label: '文本' },
  };

  const renderResultItem = (item: SearchResultItem, idx: number, mode: string) => {
    const type = detectResultType(item);
    const cfg = TYPE_CONFIG[type] || TYPE_CONFIG.text;
    let title = String(item.title || item.section || `结果 ${idx + 1}`);
    let content = String(item.content || '');
    let meta: Array<[string, unknown]> = [];
    if (type === 'community_report') {
      title = String(item.title || `社区报告 ${idx + 1}`);
      content = String(item.summary || '');
      const stars = '⭐'.repeat((item.rating as number) || 0) + '☆'.repeat(Math.max(0, 5 - ((item.rating as number) || 0)));
      meta = [
        ['社区ID', item.community_id || '-'],
        ['评级', stars],
        ['实体数', Array.isArray(item.entities) ? item.entities.length : '-'],
        ['报告ID', item.report_id || '-'],
      ];
    } else if (type === 'graph') {
      title = String(item.entity_name || item.entity_id || '关系路径');
      content = String(item.description || '');
      meta = [
        ['源实体', item.source_entity || '-'],
        ['关系', item.relation || '-'],
        ['目标类型', item.entity_type || '-'],
        ['目标ID', item.entity_id || '-'],
      ];
    } else if (type === 'entity') {
      title = String(item.entity_name || item.entity_id);
      content = String(item.description || '无描述');
      meta = [
        ['类型', item.entity_type || '-'],
        ['权重', item.weight ?? '-'],
        ['PageRank', typeof item.pagerank === 'number' ? item.pagerank.toFixed(4) : '-'],
        ['文档ID', item.doc_id || '-'],
      ];
    } else {
      meta = [
        ['文档', item.doc_name || '-'],
        ['分块', item.chunk_index != null ? item.chunk_index : '-'],
        ['类型', item.content_type || '-'],
        ['分块ID', item.chunk_id || '-'],
      ];
    }
    if (item.retrieval_mode || mode) meta.push(['来源', item.retrieval_mode || mode]);
    const sources = Array.isArray(item.sources) && item.sources.length > 0 ? item.sources : null;
    return (
      <div key={idx} className="glass-dense" style={{ padding: 12, marginBottom: 12 }}>
        <Space wrap>
          <Tag color="blue">#{idx + 1}</Tag>
          <Typography.Text strong style={{ fontSize: 14 }}>{title}</Typography.Text>
          <Tag color={cfg.color}>{cfg.label}</Tag>
          {item.score !== undefined && (
            <Tag color={Number(item.score) > 0.6 ? 'green' : 'orange'}>得分: {Number(item.score).toFixed(4)}</Tag>
          )}
        </Space>
        {content && (
          <div style={{ marginTop: 8, fontSize: 13, color: '#262626' }} dangerouslySetInnerHTML={{ __html: formatMarkdown(content) }} />
        )}
        {type === 'community_report' && Array.isArray(item.findings) && item.findings.length > 0 && (
          <div style={{ marginTop: 8, padding: '8px 12px', background: '#f6ffed', borderRadius: 6 }}>
            <div style={{ fontWeight: 600, fontSize: 13 }}>关键发现</div>
            {item.findings.map((f, i) => (
              <div key={i} style={{ fontSize: 13, marginTop: 4 }}>
                {typeof f === 'object' && f && (f as any).summary ? (
                  <>
                    <strong>{(f as any).summary}</strong>
                    {(f as any).explanation ? <div style={{ fontSize: 12, color: '#888' }}>{(f as any).explanation}</div> : null}
                  </>
                ) : (
                  <div>• {String(f)}</div>
                )}
              </div>
            ))}
          </div>
        )}
        <div style={{ marginTop: 8, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          {meta.map(([k, v]) => (
            <Tag key={k} style={{ margin: 0 }}>{k}: {typeof v === 'object' ? JSON.stringify(v).slice(0, 60) : String(v).slice(0, 60)}</Tag>
          ))}
        </div>
        {sources && <div style={{ marginTop: 4, fontSize: 12, color: '#999' }}>融合来源: {sources.join(', ')}</div>}
      </div>
    );
  };

  // ---- 按检索模式分区渲染结果（与旧版一致） ----
  const renderResults = (res: SearchResponse) => {
    const m = res.mode;
    // RAG 融合：分块列表 + 命中的实体/社区报告提示
    if (m === 'rag_fusion') {
      const chunks = res.chunks || res.results || [];
      const hitEntities = res.total_entities || (Array.isArray(res.entities) ? res.entities.length : 0);
      const hitReports = res.total_reports || (Array.isArray(res.community_reports) ? res.community_reports.length : 0);
      if (chunks.length === 0 && !hitEntities && !hitReports) return <Empty description="无检索结果" />;
      return (
        <div>
          {chunks.length > 0 && (
            <>
              <div style={{ fontWeight: 600, marginBottom: 8 }}>📄 文本分块 <span style={{ color: '#999', fontWeight: 400 }}>({chunks.length})</span></div>
              {chunks.map((item, i) => renderResultItem(item, i, m))}
            </>
          )}
          {(hitEntities > 0 || hitReports > 0) && (
            <Alert
              type="info" showIcon style={{ marginTop: 12 }}
              message="检索同时命中其他类型结果"
              description={
                <span>
                  同时命中 {hitEntities > 0 && <b>{hitEntities}</b>}{hitEntities > 0 && hitReports > 0 && ' 个实体 和 '}{hitReports > 0 && <><b>{hitReports}</b> 条社区报告</>}，
                  前往 <Link to={`/kb/${kbId}/visualize`}>图谱可视化</Link> 查看详情
                </span>
              }
            />
          )}
        </div>
      );
    }
    // 全源融合：分块列表 + 命中的实体提示
    if (m === 'hybrid_all') {
      const chunks = res.results || res.chunks || [];
      const entities = Array.isArray(res.kg_context) ? res.kg_context : [];
      if (chunks.length === 0 && entities.length === 0) return <Empty description="无检索结果" />;
      return (
        <div>
          {chunks.length > 0 && (
            <>
              <div style={{ fontWeight: 600, marginBottom: 8 }}>📄 文本分块 <span style={{ color: '#999', fontWeight: 400 }}>({chunks.length})</span></div>
              {chunks.map((item, i) => renderResultItem(item, i, m))}
            </>
          )}
          {entities.length > 0 && (
            <Alert
              type="info" showIcon style={{ marginTop: 12 }}
              message={`命中的知识图谱实体（${entities.length}）`}
              description={
                <div>
                  {entities.slice(0, 20).map((e, i) => (
                    <Tag key={i} color="purple" style={{ marginBottom: 4 }}>
                      {String((e as any).entity_name || (e as any).entity_id || (e as any).name || e)}
                    </Tag>
                  ))}
                  {entities.length > 20 ? ` 等 ${entities.length} 个实体` : ''}
                </div>
              }
            />
          )}
        </div>
      );
    }
    // 图增强检索：过滤非文本结果并提示
    let list = res.results || [];
    let hint: React.ReactNode = null;
    if (m === 'graph_enhanced') {
      const counts = { entity: 0, graph: 0, community_report: 0 };
      const texts = list.filter((item) => {
        const t = detectResultType(item);
        if (t !== 'text') { counts[t as keyof typeof counts]++; return false; }
        return true;
      });
      if (texts.length === 0 && (counts.entity > 0 || counts.graph > 0 || counts.community_report > 0)) {
        return (
          <Alert
            type="info" showIcon
            message="检索命中了非文本类型的结果"
            description={
              <span>
                命中 {counts.entity > 0 && <b>{counts.entity} 个实体</b>}
                {counts.entity > 0 && counts.graph > 0 && '、'}
                {counts.graph > 0 && <b>{counts.graph} 个关系</b>}
                {(counts.entity > 0 || counts.graph > 0) && counts.community_report > 0 && '、'}
                {counts.community_report > 0 && <b>{counts.community_report} 条社区报告</b>}，
                前往 <Link to={`/kb/${kbId}/visualize`}>图谱可视化</Link> 查看详情
              </span>
            }
          />
        );
      }
      list = texts;
      if (counts.entity > 0 || counts.graph > 0 || counts.community_report > 0) {
        hint = (
          <Alert
            type="info" showIcon style={{ marginBottom: 12 }}
            message="同时命中其他类型结果"
            description="部分实体/关系/社区报告结果请在 图谱可视化 页面查看"
          />
        );
      }
    }
    if (list.length === 0) return <Empty description="无检索结果" />;
    return (
      <div>
        {hint}
        {list.map((item, i) => renderResultItem(item, i, m))}
      </div>
    );
  };

  return (
    <div className="glass-fade-in">
      <Card className="glass glass-card-bordered" style={{ marginBottom: 16 }} title="检索条件">
        <Space direction="vertical" style={{ width: '100%' }} size="middle">
          <Space wrap>
            {/* 检索方法：胶囊按钮 → 弹出卡片列表选择 */}
            <div
              className="agent-picker agent-picker-sm"
              style={{ width: 'auto', maxWidth: 380, cursor: 'pointer' }}
              onClick={() => setModePickerOpen(true)}
            >
              <span className="agent-picker-label">检索方法</span>
              <span className="agent-picker-value">
                {modes.find((m) => m.mode === mode)?.name || mode}
                <DownOutlined style={{ fontSize: 10 }} />
              </span>
            </div>
            <Tooltip title="查看每种检索方法的介绍">
              <Button type="text" shape="circle" icon={<QuestionCircleOutlined />} onClick={() => showModalHelp(modes)} />
            </Tooltip>
          </Space>
          <Space.Compact style={{ width: '100%' }}>
            <Input.TextArea
              placeholder="请输入检索问题，如: 系统故障如何排查？"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              autoSize={{ minRows: 1, maxRows: 4 }}
            />
            <GlassButton variant="primary" icon={<SearchOutlined />} loading={loading} onClick={() => doSearch()}>
              检索
            </GlassButton>
            <GlassButton variant="text" icon={<ClearOutlined />} onClick={() => { setQuery(''); setResult(null); setCompare(null); }}>
              清空
            </GlassButton>
          </Space.Compact>

          <Space wrap>
            <span>Top K: <b>{topK}</b></span>
            <Slider style={{ width: 150 }} min={1} max={50} value={topK} onChange={setTopK} />

            {/* 文字+下划线次按钮：高级选项折叠 */}
            <GlassButton variant="text" onClick={() => setShowAdvanced(!showAdvanced)}>
              {showAdvanced ? '收起高级选项 ▲' : '高级选项 ▼'}
            </GlassButton>
          </Space>

          {showAdvanced && (
            <div className="glass-dense" style={{ padding: 12 }}>
              <Space size="large" wrap>
                {['es_hybrid', 'hybrid_all'].includes(mode) && (
                  <>
                    <div>关键词权重: <b>{keywordWeight}</b>
                      <Slider style={{ width: 120 }} min={0} max={1} step={0.1} value={keywordWeight} onChange={setKeywordWeight} />
                    </div>
                    <div>向量权重: <b>{vectorWeight}</b>
                      <Slider style={{ width: 120 }} min={0} max={1} step={0.1} value={vectorWeight} onChange={setVectorWeight} />
                    </div>
                  </>
                )}
                {['kg_graph', 'kg_relation_weighted'].includes(mode) && (
                  <>
                    <div>
                      图谱深度: <b>{depth}</b>
                      <Slider style={{ width: 120 }} min={1} max={5} value={depth} onChange={setDepth} />
                    </div>
                    {mode === 'kg_relation_weighted' && (
                      <div>
                        最小权重: <b>{minWeight}</b>
                        <Slider style={{ width: 120 }} min={1} max={10} value={minWeight} onChange={setMinWeight} />
                      </div>
                    )}
                  </>
                )}
                {['rag_fusion', 'hybrid_all'].includes(mode) && (
                  <div>
                    <Radio.Group value={allocMode} onChange={(e) => setAllocMode(e.target.value)}>
                      <Radio.Button value="auto">自动分配</Radio.Button>
                      <Radio.Button value="manual">手动分配</Radio.Button>
                    </Radio.Group>
                    {allocMode === 'manual' && (
                      <Space style={{ marginTop: 8 }} wrap>
                        {mode === 'rag_fusion' && (
                          <>
                            <span>chunks: <InputNumber size="small" min={0} value={alloc.chunks_count} onChange={(v) => setAlloc({ ...alloc, chunks_count: v || 0 })} /></span>
                            <span>entities: <InputNumber size="small" min={0} value={alloc.entities_count} onChange={(v) => setAlloc({ ...alloc, entities_count: v || 0 })} /></span>
                            <span>reports: <InputNumber size="small" min={0} value={alloc.reports_count} onChange={(v) => setAlloc({ ...alloc, reports_count: v || 0 })} /></span>
                          </>
                        )}
                        {mode === 'hybrid_all' && (
                          <>
                            <span>chunks: <InputNumber size="small" min={0} value={alloc.chunks_count} onChange={(v) => setAlloc({ ...alloc, chunks_count: v || 0 })} /></span>
                            <span>entities: <InputNumber size="small" min={0} value={alloc.entities_count} onChange={(v) => setAlloc({ ...alloc, entities_count: v || 0 })} /></span>
                          </>
                        )}
                      </Space>
                    )}
                  </div>
                )}
              </Space>
            </div>
          )}
        </Space>
      </Card>

      {kbId && stats && (
        <Row gutter={16} style={{ marginBottom: 16 }}>
          <Col xs={12} sm={6}><GlassStatCard label="文档数" value={stats.total_documents} suffix="份" /></Col>
          <Col xs={12} sm={6}><GlassStatCard label="分块数" value={stats.total_chunks} suffix="块" /></Col>
          <Col xs={12} sm={6}><GlassStatCard label="实体数" value={stats.total_entities} suffix="个" /></Col>
          <Col xs={12} sm={6}><GlassStatCard label="关系数" value={stats.total_relations} suffix="条" /></Col>
        </Row>
      )}

      {mode === 'community_report' && (
        <Alert type="info" showIcon message="社区报告检索结果请在" description="图谱可视化页面中查看社区报告面板" style={{ marginBottom: 16 }} />
      )}

      {error && <Alert type="error" showIcon message={error} style={{ marginBottom: 16 }} />}

      {result && (
        <Card
          className="glass glass-card-bordered"
          style={{ marginBottom: 16 }}
          title={`检索结果（${result.total_results} 条${mode === 'rag_fusion' ? `文本分块${result.total_chunks || 0}条` : ''}）`}
        >
          {renderResults(result)}
        </Card>
      )}

      {compare && (
        <Card className="glass glass-card-bordered" title="对比统计">
          <Row gutter={[12, 12]}>
            <Col xs={12} sm={5}><GlassStatCard label="总结果" value={compare.total} /></Col>
            <Col xs={12} sm={5}><GlassStatCard label="文本分块" value={compare.chunks} /></Col>
            <Col xs={12} sm={4}><GlassStatCard label="实体" value={compare.entities} /></Col>
            <Col xs={12} sm={5}><GlassStatCard label="关系" value={compare.relations} /></Col>
            <Col xs={12} sm={5}><GlassStatCard label="社区报告" value={compare.reports} /></Col>
          </Row>
          <Divider />
          <Typography.Text type="secondary">
            当前模式: {modes.find((m) => m.mode === mode)?.name || mode} — {modes.find((m) => m.mode === mode)?.description}
          </Typography.Text>
        </Card>
      )}

      {/* 检索方法选择弹窗 */}
      <Modal
        title="选择检索方法"
        open={modePickerOpen}
        onCancel={() => setModePickerOpen(false)}
        width={560}
        footer={[
          <Button key="close" onClick={() => setModePickerOpen(false)}>取消</Button>,
          <Button key="ok" type="primary" onClick={() => setModePickerOpen(false)}>确定</Button>,
        ]}
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxHeight: 460, overflow: 'auto' }}>
          {modes.map((m) => (
            <div
              key={m.mode}
              className={['agent-picker-option', mode === m.mode ? 'selected' : ''].filter(Boolean).join(' ')}
              onClick={() => { setMode(m.mode); setModePickerOpen(false); }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ fontWeight: 600, fontSize: 14 }}>{m.name}</span>
                {m.mode === 'rag_fusion' && <Tag color="green">推荐</Tag>}
                {mode === m.mode && <span style={{ color: '#000', fontSize: 13 }}>✓</span>}
              </div>
              <div style={{ fontSize: 12, color: '#595959', marginTop: 4 }}>{m.description}</div>
            </div>
          ))}
        </div>
      </Modal>
    </div>
  );
}

// 检索帮助弹窗（命令式直接调用）
function showModalHelp(modes: ModeDef[]) {
  Modal.info({
    title: '检索方式说明',
    width: 640,
    content: (
      <div>
        {modes.map((m) => (
          <div key={m.mode} style={{ marginBottom: 12 }}>
            <Space>
              <Tag color="blue">{m.name}</Tag>
              <Typography.Text code>{m.mode}</Typography.Text>
            </Space>
            <div style={{ fontSize: 13, color: '#595959', marginTop: 4 }}>{m.description}</div>
          </div>
        ))}
      </div>
    ),
  });
}
