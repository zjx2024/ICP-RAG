import { useCallback, useEffect, useState } from 'react';
import { Row, Col, Card, Typography, Space, Tag, Spin, message, Button } from 'antd';
import {
  TableOutlined, ApartmentOutlined, FileTextOutlined, ArrowRightOutlined,
  RobotOutlined, ReloadOutlined, CloudUploadOutlined,
} from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import { fetchDiagnosisReadiness, fetchDocumentTypes } from '@/api/icp';
import { GlassEmpty, GlassStatCard } from '@/components/GlassUI';
import UnifiedUploadModal from '@/components/UnifiedUploadModal';
import type { DocTypeInfo } from '@/types';

const FORM_META: Record<string, { icon: React.ReactNode; color: string }> = {
  table: { icon: <TableOutlined />, color: '#1677ff' },
  image: { icon: <ApartmentOutlined />, color: '#722ed1' },
  text: { icon: <FileTextOutlined />, color: '#52c41a' },
};

const TYPE_SUMMARY: Record<string, string> = {
  function_locator: '系统、模块、功能与接口信息',
  hmc: '故障代码、故障现象与维修方法',
  fta: '故障事件及其逻辑关系',
  rbd: '系统可靠性结构与计算参数',
  fmeca: '故障模式、原因、影响与危害等级',
  mfl: '维修故障清单、上报接口与定位信息',
  fault_history: '历史故障、处理步骤与最终方案',
};

interface TypeCardInfo extends DocTypeInfo {
  ready_count?: number;
  failed_count?: number;
}

interface Readiness {
  ready_type_count: number;
  total_records: number;
}

export default function TypeHome() {
  const navigate = useNavigate();
  const [types, setTypes] = useState<TypeCardInfo[]>([]);
  const [readiness, setReadiness] = useState<Readiness>({
    ready_type_count: 0,
    total_records: 0,
  });
  const [loading, setLoading] = useState(false);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [recentlyUploadedType, setRecentlyUploadedType] = useState('');

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [typeList, ready] = await Promise.all([
        fetchDocumentTypes(),
        fetchDiagnosisReadiness(),
      ]);
      const readyMap = new Map(ready.types.map((item) => [item.doc_type, item]));
      setTypes(typeList.map((item) => {
        const status = readyMap.get(item.doc_type);
        return {
          ...item,
          doc_count: status?.doc_count ?? item.doc_count,
          record_count: status?.record_count ?? item.record_count,
          ready_count: status?.ready_count ?? 0,
          failed_count: status?.failed_count ?? 0,
        };
      }));
      setReadiness({
        ready_type_count: ready.ready_type_count,
        total_records: ready.total_records,
      });
    } catch (error) {
      message.error(`文档状态加载失败：${(error as Error).message}`);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const totalDocs = types.reduce((sum, item) => sum + (item.doc_count || 0), 0);

  const handleUploaded = async (docType: string) => {
    setRecentlyUploadedType(docType);
    await load();
    window.setTimeout(() => setRecentlyUploadedType(''), 2200);
  };

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20, gap: 12 }}>
        <div>
          <Typography.Title level={4} style={{ margin: 0, color: '#262626' }}>ICP 七类文档知识库</Typography.Title>
          <Typography.Text type="secondary">
            集中管理故障诊断所需的工程文档。
          </Typography.Text>
        </div>
        <Space wrap>
          <Button type="primary" icon={<CloudUploadOutlined />} onClick={() => setUploadOpen(true)}>
            上传文档
          </Button>
          <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>刷新</Button>
          <Button className="glass-btn-black" icon={<RobotOutlined />} onClick={() => navigate('/diagnosis')}>
            进入故障诊断
          </Button>
        </Space>
      </div>

      <Row gutter={16} style={{ marginBottom: 20 }}>
        <Col xs={12} sm={6}>
          <GlassStatCard label="文档类型" value={types.length} suffix="类" />
        </Col>
        <Col xs={12} sm={6}>
          <GlassStatCard label="已有内容" value={readiness.ready_type_count} suffix="/ 7" />
        </Col>
        <Col xs={12} sm={6}>
          <GlassStatCard label="文档总数" value={totalDocs} suffix="份" />
        </Col>
        <Col xs={12} sm={6}>
          <GlassStatCard label="提取结果" value={readiness.total_records} suffix="条" />
        </Col>
      </Row>

      {loading ? (
        <div style={{ textAlign: 'center', padding: 60 }}><Spin size="large" /></div>
      ) : types.length === 0 ? (
        <GlassEmpty description="暂未获取到文档类型，请稍后刷新" />
      ) : (
        <Row gutter={[20, 20]}>
          {types.map((item) => {
            const meta = FORM_META[item.data_form] || FORM_META.text;
            const ready = (item.ready_count || 0) > 0;
            const justUploaded = recentlyUploadedType === item.doc_type;
            return (
              <Col key={item.doc_type} xs={24} sm={12} lg={8} xl={6}>
                <Card
                  hoverable
                  onClick={() => navigate(`/type/${item.doc_type}/documents`)}
                  style={{
                    background: justUploaded ? 'rgba(230,247,255,.88)' : 'rgba(255,255,255,0.6)',
                    backdropFilter: 'blur(14px) saturate(160%)',
                    border: justUploaded
                      ? '1px solid rgba(22,119,255,.72)'
                      : `1px solid ${ready ? 'rgba(82,196,26,.35)' : 'rgba(255,255,255,.65)'}`,
                    boxShadow: justUploaded ? '0 0 0 3px rgba(22,119,255,.10), 0 12px 30px rgba(22,119,255,.14)' : undefined,
                    height: '100%',
                    transition: 'all .25s ease',
                  }}
                  styles={{ body: { display: 'flex', flexDirection: 'column', height: '100%' } }}
                  role="link"
                  tabIndex={0}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' || event.key === ' ') {
                      event.preventDefault();
                      navigate(`/type/${item.doc_type}/documents`);
                    }
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                    <span style={{ fontSize: 26, color: meta.color }}>{meta.icon}</span>
                    <Typography.Title level={5} style={{ margin: 0, color: '#262626', flex: 1 }} ellipsis>
                      {item.name}
                    </Typography.Title>
                    <Tag color={ready ? 'success' : 'default'}>{ready ? '可用于诊断' : '待解析'}</Tag>
                  </div>
                  <Typography.Paragraph type="secondary" style={{ minHeight: 40, marginBottom: 12 }} ellipsis={{ rows: 2 }}>
                    {TYPE_SUMMARY[item.doc_type] || '管理和查看该类型的工程文档'}
                  </Typography.Paragraph>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 20, flexWrap: 'wrap', marginBottom: 12 }}>
                    {[
                      { value: item.doc_count ?? 0, label: '份文档' },
                      { value: item.ready_count ?? 0, label: '份可用' },
                      { value: item.record_count ?? 0, label: '条记录' },
                    ].map((metric) => (
                      <span key={metric.label} style={{ display: 'inline-flex', alignItems: 'baseline', gap: 4 }}>
                        <Typography.Text strong style={{ fontSize: 16 }}>{metric.value}</Typography.Text>
                        <Typography.Text type="secondary" style={{ fontSize: 12 }}>{metric.label}</Typography.Text>
                      </span>
                    ))}
                    {(item.failed_count || 0) > 0 ? <Tag color="error">失败 {item.failed_count}</Tag> : null}
                  </div>
                  <div style={{ flex: 1 }} />
                  <div style={{ borderTop: '1px solid rgba(0,0,0,0.06)', marginTop: 10, paddingTop: 10, textAlign: 'right' }}>
                    <Typography.Text type="secondary" style={{ fontSize: 13 }}>
                      进入管理 <ArrowRightOutlined />
                    </Typography.Text>
                  </div>
                </Card>
              </Col>
            );
          })}
        </Row>
      )}

      <UnifiedUploadModal
        open={uploadOpen}
        types={types}
        onClose={() => setUploadOpen(false)}
        onUploaded={handleUploaded}
      />
    </div>
  );
}
