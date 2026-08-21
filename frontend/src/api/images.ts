import { request, buildQuery } from './request';
import type { ImageDocsResponse, DocMediaDetail } from '@/types';

// ---------- 图片/画廊 ----------
// kbId 可选：不传为全局画廊，传入则只列出该知识库下有图片的文档
export const fetchImageDocs = (kbId?: string) =>
  request<ImageDocsResponse>(kbId ? `/images${buildQuery({ kb_id: kbId })}` : '/images');

export const fetchDocMediaDetail = (docId: string) =>
  request<DocMediaDetail>(`/images/${encodeURIComponent(docId)}`);

// 图片 URL 处理：http 绝对地址取 path，相对路径原样返回，空值占位
export function resolveImageSrc(img: { image_url?: string }, docId: string): string {
  if (img.image_url) {
    if (img.image_url.startsWith('http')) {
      try {
        return new URL(img.image_url).pathname;
      } catch {
        return img.image_url;
      }
    }
    return img.image_url;
  }
  return `/images/${docId}/placeholder`;
}

export function getTypeLabel(type: string): string {
  const map: Record<string, string> = {
    normal: '普通图片',
    fault_tree: '故障树',
    flowchart: '流程图',
    table: '表格',
  };
  return map[type] || '普通图片';
}
