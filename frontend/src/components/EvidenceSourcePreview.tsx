import { useCallback, useEffect, useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import { Button, Image, Spin, Typography } from 'antd';
import {
  FileTextOutlined,
  PictureOutlined,
  ReloadOutlined,
  TableOutlined,
  ZoomInOutlined,
} from '@ant-design/icons';
import { fetchEvidencePreview } from '@/api/icp';
import type { AgentEvidence, ICPEvidencePreview } from '@/types';

const previewCache = new Map<string, Promise<ICPEvidencePreview>>();

function scalar(value: unknown): string {
  return typeof value === 'string' || typeof value === 'number'
    ? String(value).trim()
    : '';
}

function isPlaceholderSummary(value: string): boolean {
  return /(?:结构化关系证据|结构化数据).*?(?:来源定位|内部字段)|详细机器结构请按来源定位查看/.test(value);
}

function previewKey(evidence: AgentEvidence): string {
  return [
    scalar(evidence.doc_type),
    scalar(evidence.doc_id),
    scalar(evidence.source_ref),
    scalar(evidence.record_id),
    scalar(evidence.title),
  ].join('|');
}

function loadPreview(evidence: AgentEvidence): Promise<ICPEvidencePreview> {
  const key = previewKey(evidence);
  const cached = previewCache.get(key);
  if (cached) return cached;
  const request = fetchEvidencePreview(
    scalar(evidence.doc_type),
    scalar(evidence.doc_id),
    {
      source_ref: scalar(evidence.source_ref),
      record_id: scalar(evidence.record_id),
      title: scalar(evidence.title),
    },
  ).catch((error) => {
    previewCache.delete(key);
    throw error;
  });
  previewCache.set(key, request);
  return request;
}

function PreviewSection({
  icon,
  label,
  children,
}: {
  icon: ReactNode;
  label: string;
  children: ReactNode;
}) {
  return (
    <section className="diagnosis-source-preview-section">
      <div className="diagnosis-source-preview-heading">
        {icon}
        <span>{label}</span>
      </div>
      {children}
    </section>
  );
}

export default function EvidenceSourcePreview({
  evidence,
  summaryLabel,
  summaryText,
  autoLoad = false,
}: {
  evidence: AgentEvidence;
  summaryLabel: string;
  summaryText: string;
  autoLoad?: boolean;
}) {
  const docType = scalar(evidence.doc_type);
  const docId = scalar(evidence.doc_id);
  const key = previewKey(evidence);
  const [requested, setRequested] = useState(autoLoad);
  const [loading, setLoading] = useState(false);
  const [preview, setPreview] = useState<ICPEvidencePreview>();
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    setRequested(autoLoad);
    setPreview(undefined);
    setFailed(false);
  }, [autoLoad, key]);

  const requestPreview = useCallback(() => {
    if (!docType || !docId) return;
    setRequested(true);
    setLoading(true);
    setFailed(false);
    loadPreview(evidence)
      .then(setPreview)
      .catch(() => setFailed(true))
      .finally(() => setLoading(false));
  }, [docId, docType, evidence, key]);

  useEffect(() => {
    if (requested && !preview && !loading && !failed) requestPreview();
  }, [failed, loading, preview, requestPreview, requested]);

  const usefulSummary = useMemo(() => {
    const value = summaryText.trim();
    return value && !isPlaceholderSummary(value) ? value : '';
  }, [summaryText]);
  const structureSummary = scalar(preview?.structure_summary);
  const showFallbackSummary = usefulSummary
    && usefulSummary.replace(/\s+/g, '') !== structureSummary.replace(/\s+/g, '');
  const sourceRef = scalar(preview?.resolved_source_ref) || scalar(evidence.source_ref)
    || scalar(evidence.record_id) || scalar(evidence.chunk_id);

  if (!docType || !docId) {
    return usefulSummary ? (
      <>
        <div className="diagnosis-citation-preview-label">{summaryLabel}</div>
        <Typography.Paragraph className="diagnosis-citation-preview">
          {usefulSummary}
        </Typography.Paragraph>
      </>
    ) : null;
  }

  return (
    <div className="diagnosis-source-preview">
      {loading ? (
        <div className="diagnosis-source-preview-loading">
          <Spin size="small" />
          <span>正在读取原始证据…</span>
        </div>
      ) : null}

      {preview?.excerpt ? (
        <PreviewSection icon={<FileTextOutlined />} label="原文">
          <div className="diagnosis-source-preview-excerpt">{preview.excerpt}</div>
        </PreviewSection>
      ) : null}

      {preview?.table?.row?.length ? (
        <PreviewSection icon={<TableOutlined />} label="表格命中行">
          <div className="diagnosis-source-preview-table-wrap">
            <table className="diagnosis-source-preview-table">
              <thead>
                <tr>
                  {preview.table.headers.map((header, index) => <th key={`${header}-${index}`}>{header}</th>)}
                </tr>
              </thead>
              <tbody>
                <tr>
                  {preview.table.row.map((cell, index) => <td key={`${cell}-${index}`}>{cell}</td>)}
                </tr>
              </tbody>
            </table>
          </div>
        </PreviewSection>
      ) : null}

      {preview?.image?.image_url ? (
        <PreviewSection icon={<PictureOutlined />} label="原图">
          <figure className="diagnosis-source-preview-image">
            <Image
              src={preview.image.image_url}
              alt={preview.image.title || scalar(evidence.title) || '证据原图'}
              loading="lazy"
              preview={{
                mask: <span><ZoomInOutlined /> 查看大图</span>,
              }}
            />
            {preview.image.title ? <figcaption>{preview.image.title}</figcaption> : null}
          </figure>
        </PreviewSection>
      ) : null}

      {structureSummary ? (
        <PreviewSection icon={<FileTextOutlined />} label="结构概览">
          <div className="diagnosis-source-preview-summary">{structureSummary}</div>
        </PreviewSection>
      ) : null}

      {showFallbackSummary ? (
        <PreviewSection icon={<FileTextOutlined />} label={summaryLabel}>
          <div className="diagnosis-source-preview-summary">{usefulSummary}</div>
        </PreviewSection>
      ) : null}

      {!requested && !preview ? (
        <Button type="link" size="small" onClick={requestPreview} className="diagnosis-source-preview-action">
          查看原始证据
        </Button>
      ) : null}
      {failed ? (
        <Button
          type="link"
          size="small"
          icon={<ReloadOutlined />}
          onClick={requestPreview}
          className="diagnosis-source-preview-action"
        >
          重新读取原始证据
        </Button>
      ) : null}

      {sourceRef ? (
        <div className="diagnosis-citation-source-ref">
          <span>定位</span>
          <code>{sourceRef}</code>
        </div>
      ) : null}
    </div>
  );
}
