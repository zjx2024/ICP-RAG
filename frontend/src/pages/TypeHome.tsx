import { useCallback, useEffect, useState } from 'react';
import { Row, Col, Card, Typography, Space, Tag, Spin, message, Button, Alert } from 'antd';
import {
  TableOutlined, ApartmentOutlined, FileTextOutlined, ArrowRightOutlined,
  RobotOutlined, ReloadOutlined,
} from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import { fetchDiagnosisReadiness, fetchDocumentTypes } from '@/api/icp';
import { GlassEmpty, GlassStatCard } from '@/components/GlassUI';
import type { DocTypeInfo } from '@/types';

const FORM_META: Record<string, { icon: React.ReactNode; color: string; label: string }> = {
  table: { icon: <TableOutlined />, color: '#1677ff', label: '表格类' },
  image: { icon: <ApartmentOutlined />, color: '#722ed1', label: '图类' },
  text: { icon: <FileTextOutlined />, color: '#52c41a', label: '图文/文本类' },
};

interface TypeCardInfo extends DocTypeInfo {
  ready_count?: number;
  failed_count?: number;
}

interface Readiness {
  all_types_present: boolean;
  ready_type_count: number;
  total_records: number;
}

export default function TypeHome() {
  const navigate = useNavigate();
  const [types, setTypes] = useState<TypeCardInfo[]>([]);
  const [readiness, setReadiness] = useState<Readiness>({
    all_types_present: false,
    ready_type_count: 0,
    total_records: 0,
  });
  const [loading, setLoading] = useState(false);

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
        all_types_present: ready.all_types_present,
        ready_type_count: ready.ready_type_count,
        total_records: ready.total_records,
      });
    } catch (error) {
      message.error(`加载七类知识状态失败: ${(error as Error).message}`);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const totalDocs = types.reduce((sum, item) => sum + (item.doc_count || 0), 0);

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20, gap: 12 }}>
        <div>
          <Typography.Title level={4} style={{ margin: 0, color: '#262626' }}>ICP 七类文档知识库</Typography.Title>
          <Typography.Text type="secondary">
            每类文档使用独立的预处理和存储方式；解析完成后共同服务于 ICP 故障诊断。
          </Typography.Text>
        </div>
        <Space>
          <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>刷新</Button>
          <Button type="primary" icon={<RobotOutlined />} onClick={() => navigate('/diagnosis')}>
            进入故障诊断
          </Button>
        </Space>
      </div>

      {!readiness.all_types_present && readiness.ready_type_count > 0 ? (
        <Alert
          type="warning"
          showIcon
          style={{ marginBottom: 16 }}
          message={`当前已有 ${readiness.ready_type_count}/7 类知识可用于诊断`}
          description="缺失类型不会阻止系统运行，但候选定位、原因链、影响或维修证据可能不完整。"
        />
      ) : readiness.all_types_present ? (
        <Alert
          type="success"
          showIcon
          style={{ marginBottom: 16 }}
          message="七类文档均已有可用解析结果，诊断闭环已就绪"
        />
      ) : null}

      <Row gutter={16} style={{ marginBottom: 20 }}>
        <Col xs={12} sm={6}>
          <GlassStatCard label="文档类型" value={types.length} suffix="类" />
        </Col>
        <Col xs={12} sm={6}>
          <GlassStatCard label="诊断就绪" value={readiness.ready_type_count} suffix="/ 7" />
        </Col>
        <Col xs={12} sm={6}>
          <GlassStatCard label="文档总数" value={totalDocs} suffix="份" />
        </Col>
        <Col xs={12} sm={6}>
          <GlassStatCard label="结构化记录" value={readiness.total_records} suffix="条" />
        </Col>
      </Row>

      {loading ? (
        <div style={{ textAlign: 'center', padding: 60 }}><Spin size="large" /></div>
      ) : types.length === 0 ? (
        <GlassEmpty description="暂无七类文档配置，请检查后端 document_types 初始化" />
      ) : (
        <Row gutter={[20, 20]}>
          {types.map((item) => {
            const meta = FORM_META[item.data_form] || FORM_META.text;
            const ready = (item.ready_count || 0) > 0;
            return (
              <Col key={item.doc_type} xs={24} sm={12} lg={8} xl={6}>
                <Card
                  hoverable
                  onClick={() => navigate(`/type/${item.doc_type}`)}
                  style={{
                    background: 'rgba(255,255,255,0.6)',
                    backdropFilter: 'blur(14px) saturate(160%)',
                    border: `1px solid ${ready ? 'rgba(82,196,26,.35)' : 'rgba(255,255,255,.65)'}`,
                    height: '100%',
                  }}
                  styles={{ body: { display: 'flex', flexDirection: 'column', height: '100%' } }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                    <span style={{ fontSize: 26, color: meta.color }}>{meta.icon}</span>
                    <Typography.Title level={5} style={{ margin: 0, color: '#262626', flex: 1 }} ellipsis>
                      {item.name}
                    </Typography.Title>
                    <Tag color={ready ? 'success' : 'default'}>{ready ? '可用于诊断' : '待解析'}</Tag>
                  </div>
                  <Typography.Text type="secondary" style={{ fontSize: 12, color: meta.color }}>
                    {item.doc_type}
                  </Typography.Text>
                  <Typography.Paragraph type="secondary" style={{ minHeight: 40, marginBottom: 12 }} ellipsis={{ rows: 2 }}>
                    {item.description || '暂无描述'}
                  </Typography.Paragraph>
                  <Space size={[8, 8]} wrap style={{ marginBottom: 12 }}>
                    <Tag color={meta.color}>{meta.label}</Tag>
                    {item.input_formats ? <Tag>支持 {item.input_formats}</Tag> : null}
                    <Tag>文档 {item.doc_count ?? 0}</Tag>
                    <Tag>可用 {item.ready_count ?? 0}</Tag>
                    <Tag>记录 {item.record_count ?? 0}</Tag>
                    {(item.failed_count || 0) > 0 ? <Tag color="error">失败 {item.failed_count}</Tag> : null}
                  </Space>
                  <div style={{ flex: 1 }} />
                  <div style={{ borderTop: '1px solid rgba(0,0,0,0.06)', marginTop: 10, paddingTop: 10, textAlign: 'right' }}>
                    <Typography.Text type="secondary" style={{ fontSize: 13 }}>
                      上传、解析与查看 <ArrowRightOutlined />
                    </Typography.Text>
                  </div>
                </Card>
              </Col>
            );
          })}
        </Row>
      )}
    </div>
  );
}
