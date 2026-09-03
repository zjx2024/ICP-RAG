import { useCallback, useEffect, useMemo, useState } from 'react';
import { Empty, Image, Spin, Typography } from 'antd';
import {
  FileTextOutlined,
  PictureOutlined,
  TableOutlined,
  ZoomInOutlined,
} from '@ant-design/icons';
import { fetchOriginal } from '@/api/icp';
import type { ICPOriginalBlock, ICPOriginalResponse } from '@/types';

interface Props {
  docType: string;
  docId: string;
  file_name?: string;
  mode?: DocumentContentMode;
  showTabs?: boolean;
}

export type DocumentContentMode = 'document' | 'tables' | 'images';

const CONTENT_MODES: Array<{
  key: DocumentContentMode;
  label: string;
  shortLabel: string;
  icon: React.ReactNode;
}> = [
  { key: 'document', label: '全部文档内容', shortLabel: '全文', icon: <FileTextOutlined /> },
  { key: 'tables', label: '表格提取结果', shortLabel: '表格', icon: <TableOutlined /> },
  { key: 'images', label: '图片提取结果', shortLabel: '图片', icon: <PictureOutlined /> },
];

export function DocumentContentModeSwitch({
  value,
  onChange,
}: {
  value: DocumentContentMode;
  onChange: (mode: DocumentContentMode) => void;
}) {
  return (
    <div className="document-content-mode-switch" role="group" aria-label="文档内容显示方式">
      {CONTENT_MODES.map((item) => (
        <button
          key={item.key}
          type="button"
          aria-pressed={value === item.key}
          onClick={() => onChange(item.key)}
        >
          {item.icon}
          <span className="document-mode-label-full">{item.label}</span>
          <span className="document-mode-label-short">{item.shortLabel}</span>
        </button>
      ))}
    </div>
  );
}

function blockKey(block: ICPOriginalBlock, index: number): string {
  return block.id || block.source_ref || `${block.kind || 'block'}-${block.sequence ?? index}-${index}`;
}

function titleLevel(level?: number): 1 | 2 | 3 | 4 | 5 {
  return Math.min(5, Math.max(1, level || 4)) as 1 | 2 | 3 | 4 | 5;
}

function hasTable(block: ICPOriginalBlock): boolean {
  return block.kind === 'table' || Boolean(block.table?.length);
}

function hasImage(block: ICPOriginalBlock): boolean {
  return block.kind === 'image' || Boolean(block.image_url);
}

const imagePreviewMask = (
  <span className="document-image-preview-mask">
    <ZoomInOutlined />
    查看大图
  </span>
);

function DocumentTable({ block }: { block: ICPOriginalBlock }) {
  if (!block.table?.length) return null;
  return (
    <div className="document-content-table-wrap">
      <table className="document-content-table">
        <tbody>
          {block.table.map((row, rowIndex) => (
            <tr key={rowIndex}>
              {row.map((cell, cellIndex) => (
                <td key={cellIndex} className={rowIndex === 0 ? 'is-heading' : undefined}>
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DocumentHeading({ block }: { block: ICPOriginalBlock }) {
  if (!block.title) return null;
  if ((block.level || 0) > 0 && !hasTable(block) && !hasImage(block)) {
    return (
      <Typography.Title
        level={titleLevel(block.level)}
        className={`document-content-title level-${block.level}`}
      >
        {block.title}
      </Typography.Title>
    );
  }
  return <div className="document-content-caption">{block.title}</div>;
}

function FullDocument({ blocks }: { blocks: ICPOriginalBlock[] }) {
  if (!blocks.length) return <Empty description="暂未提取到可显示的文档内容" />;
  return (
    <Image.PreviewGroup>
      <article className="document-flow" aria-label="全部文档内容">
        {blocks.map((block, index) => (
          <section key={blockKey(block, index)} className="document-flow-block">
            <DocumentHeading block={block} />
            {block.text && !hasTable(block) ? (
              <div className="document-content-text">{block.text}</div>
            ) : null}
            <DocumentTable block={block} />
            {block.image_url ? (
              <figure className="document-content-image">
                <Image
                  src={block.image_url}
                  alt={block.title || `文档图片 ${block.image_index || index + 1}`}
                  loading="lazy"
                  preview={{ mask: imagePreviewMask }}
                />
              </figure>
            ) : null}
          </section>
        ))}
      </article>
    </Image.PreviewGroup>
  );
}

function ExtractedTables({ blocks }: { blocks: ICPOriginalBlock[] }) {
  const tables = blocks.filter((block) => hasTable(block) && Boolean(block.table?.length));
  if (!tables.length) return <Empty description="该文档中没有提取到表格" />;
  return (
    <div className="document-table-results">
      {tables.map((block, index) => (
        <section key={blockKey(block, index)} className="document-table-result">
          <div className="document-result-heading">
            <Typography.Text strong>{block.title || `表格 ${index + 1}`}</Typography.Text>
          </div>
          <DocumentTable block={block} />
        </section>
      ))}
    </div>
  );
}

function ExtractedImages({ blocks }: { blocks: ICPOriginalBlock[] }) {
  const images = blocks.filter((block) => hasImage(block) && Boolean(block.image_url));
  if (!images.length) return <Empty description="该文档中没有提取到图片" />;
  return (
    <Image.PreviewGroup>
      <div className="document-image-grid">
        {images.map((block, index) => (
          <figure key={blockKey(block, index)} className="document-image-result">
            <div className="document-image-frame">
              <Image
                src={block.image_url}
                alt={block.title || `图片 ${block.image_index || index + 1}`}
                loading="lazy"
                preview={{ mask: imagePreviewMask }}
              />
            </div>
            <figcaption>{block.title || `图片 ${block.image_index || index + 1}`}</figcaption>
          </figure>
        ))}
      </div>
    </Image.PreviewGroup>
  );
}

/** 按原文件顺序展示原文，并为表格和图片提供各自的集中查看入口。 */
export default function DocContentView({
  docType,
  docId,
  mode,
  showTabs = true,
}: Props) {
  const [activeTab, setActiveTab] = useState<DocumentContentMode>('document');
  const [originalData, setOriginalData] = useState<ICPOriginalResponse | null>(null);
  const [loading, setLoading] = useState(true);

  const loadOriginal = useCallback(async () => {
    setLoading(true);
    try {
      // 内容查看需要忠实呈现整份文件，不在阅读过程中做分页或截断。
      const firstPage = await fetchOriginal(docType, docId, { page: 1, size: 10000 });
      const blocks = [...firstPage.blocks];
      let page = 2;
      while (blocks.length < firstPage.total) {
        const nextPage = await fetchOriginal(docType, docId, { page, size: 10000 });
        if (!nextPage.blocks.length) break;
        blocks.push(...nextPage.blocks);
        page += 1;
      }
      setOriginalData({ ...firstPage, blocks });
    } catch {
      setOriginalData({
        doc_type: docType,
        file_name: '',
        total: 0,
        page: 1,
        size: 10000,
        blocks: [],
      });
    } finally {
      setLoading(false);
    }
  }, [docId, docType]);

  useEffect(() => {
    setActiveTab('document');
    setOriginalData(null);
    loadOriginal();
  }, [docId, docType, loadOriginal]);

  const blocks = originalData?.blocks || [];
  const tabs = useMemo(() => [
    {
      key: 'document',
      label: '全部文档内容',
      icon: <FileTextOutlined />,
      children: <FullDocument blocks={blocks} />,
    },
    {
      key: 'tables',
      label: '表格提取结果',
      icon: <TableOutlined />,
      children: <ExtractedTables blocks={blocks} />,
    },
    {
      key: 'images',
      label: '图片提取结果',
      icon: <PictureOutlined />,
      children: <ExtractedImages blocks={blocks} />,
    },
  ], [blocks]);

  const visibleMode = mode || activeTab;
  const visibleContent = tabs.find((tab) => tab.key === visibleMode)?.children;

  if (loading && !originalData) return <Spin className="document-content-loading" />;

  if (!showTabs) {
    return <div className="document-content-pane">{visibleContent}</div>;
  }

  return (
    <div className="document-content-tabs">
      <DocumentContentModeSwitch value={visibleMode} onChange={setActiveTab} />
      <div className="document-content-pane">{visibleContent}</div>
    </div>
  );
}
