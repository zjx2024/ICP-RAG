// ===== formatMarkdown 处理函数 =====
// 正则顺序不能乱（契约）：提取 img → escapeHtml → 还原 img → 引用替换 → code block →
// markdown 图片 → 表格（换行替换前） → inline code → bold → \n→<br>

export interface Citation {
  doc_name?: string;
  title?: string;
  content?: string;
  [k: string]: unknown;
}

export function escapeHtml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// 图片地址归一化：绝对 URL 若指向本站 /images 静态资源，转成相对路径
// （开发环境走 vite 代理、生产环境同源，避免 host 不可达导致图片加载失败）
function toRelImageSrc(u: string): string {
  try {
    const parsed = new URL(u);
    if (parsed.pathname.startsWith('/images/')) {
      return parsed.pathname + parsed.search;
    }
  } catch {
    /* 非 URL，原样返回 */
  }
  return u;
}

// 渲染内联引用 HTML（React 中需要事件委托，这里保留 data 属性由组件绑定）
export function formatMarkdown(
  text: string,
  citations?: Citation[],
  msgKey = 0,
): string {
  let result = text || '';

  // 1. 提取 <img> 标签，用占位符替换避免被转义
  const imgPlaceholders: string[] = [];
  result = result.replace(/<img\s+[^>]*src=["']([^"']+)["'][^>]*\/?>/gi, (match, src) => {
    const idx = imgPlaceholders.length;
    imgPlaceholders.push(src);
    return `\x00IMG${idx}\x00`;
  });

  // 2. 转义剩余文本
  result = escapeHtml(result);

  // 3. 还原 <img>
  result = result.replace(/\x00IMG(\d+)\x00/g, (_, idx) => {
    let src = imgPlaceholders[parseInt(idx, 10)] || '';
    src = toRelImageSrc(src).replace(/[`\s]/g, '');
    return `<img src="${src}" style="max-width:100%;border-radius:8px;margin:8px 0;" loading="lazy" alt="图片">`;
  });

  // 4. 引用标记 [^n] → 内联引用
  if (citations && citations.length > 0) {
    let citeSeq = 0;
    result = result.replace(/\[\^(\d+)\]/g, (_m, n) => {
      const idx = parseInt(n, 10) - 1;
      const c = citations[idx];
      if (!c) return '';
      citeSeq += 1;
      const label = c.doc_name || c.title || `引用${n}`;
      const body = escapeHtml(c.content || c.title || '');
      const title = escapeHtml(c.title || '');
      return (
        `<span class="inline-cite" data-cite-key="inline-cite-${msgKey}-${citeSeq}">📄 ${label} <span class="cite-caret">▾</span></span>` +
        `<span class="inline-cite-body" id="inline-cite-${msgKey}-${citeSeq}" style="display:none;">` +
        `<strong>${title}</strong><br>${body}</span>`
      );
    });
  } else {
    result = result.replace(/\[\^(\d+)\]/g, '');
  }

  // 5. 代码块
  result = result.replace(/```(\w*)\n([\s\S]*?)```/g, (_m, lang, code) => {
    return `<pre><code class="lang-${lang || ''}">${escapeHtml(code)}</code></pre>`;
  });

  // 6. markdown 图片
  result = result.replace(/!\[([^\]]*)\]\((https?:\/\/[^)\s]+)\)/g, (_m, alt, url) => {
    return `<img src="${toRelImageSrc(url)}" alt="${escapeHtml(alt)}" style="max-width:100%;border-radius:8px;">`;
  });

  // 7. markdown 表格（必须在内联 code / 换行处理之前）
  result = result.replace(/(^\|[^\n]+\|\s*$\n)(^\|[\s:|-]+\|\s*$\n)?((?:\|[^\n]+\|\s*$\n)*)/gm, (whole) => {
    const lines = whole.trim().split('\n').filter((l: string) => l.trim().startsWith('|'));
    if (lines.length < 2) return whole;
    const parseRow = (row: string) =>
      row
        .trim()
        .replace(/^\|/, '')
        .replace(/\|$/, '')
        .split('|')
        .map((cell) => cell.trim());
    const header = parseRow(lines[0]);
    const bodyRows = lines.slice(2).map(parseRow);
    let html = '<table style="width:100%;border-collapse:collapse;margin:8px 0;font-size:13px;">';
    html += '<thead><tr>' + header.map((h) => `<th style="border:1px solid #ddd;padding:6px 10px;background:rgba(0,0,0,0.05);">${h}</th>`).join('') + '</tr></thead>';
    html += '<tbody>';
    for (const row of bodyRows) {
      html += '<tr>' + row.map((cell) => `<td style="border:1px solid #ddd;padding:6px 10px;">${cell}</td>`).join('') + '</tr>';
    }
    html += '</tbody></table>';
    return html;
  });

  // 8. 行内代码
  result = result.replace(/`([^`\n]+)`/g, (_m, code) => `<code style="background:rgba(0,0,0,0.06);padding:1px 5px;border-radius:4px;font-size:0.92em;">${code}</code>`);

  // 9. 加粗
  result = result.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

  // 10. 换行
  result = result.replace(/\n/g, '<br>');

  return result;
}

// ===== token 估算（中文 1.5 字/token、英文 4 字符/token） =====
export function estimateTokens(text: string): number {
  const chineseChars = (text.match(/[\u4e00-\u9fa5]/g) || []).length;
  const otherChars = text.length - chineseChars;
  return Math.ceil(chineseChars / 1.5 + otherChars / 4);
}

// 上下文进度条三色阈值
export function tokenProgressColor(percent: number): string {
  if (percent < 60) return '#52c41a';
  if (percent < 85) return '#fa8c16';
  return '#ff4d4f';
}

// ===== 时间格式化 =====
export function formatDateTime(iso?: string): string {
  if (!iso) return '';
  const d = new Date(iso);
  if (isNaN(d.getTime())) return String(iso);
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

export function formatTimeShort(iso?: string): string {
  if (!iso) return '';
  const d = new Date(iso);
  if (isNaN(d.getTime())) return String(iso);
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

// ===== 文档状态映射（含 ICP 类型驱动文档状态） =====
export const DOC_STATUS_LABEL: Record<string, string> = {
  uploaded: '已上传',
  parsing: '解析中',
  indexed: '已索引',
  kg_pending: '待生成图谱',
  generating_kg: '图谱生成中',
  kg_done: '图谱已生成',
  completed: '已完成',
  failed: '失败',
  deleting: '删除中',
  delete_failed: '删除失败',
  cancelled: '已取消',
};

export const DOC_STATUS_COLOR: Record<string, string> = {
  uploaded: 'default',
  parsing: 'processing',
  indexed: 'cyan',
  kg_pending: 'gold',
  generating_kg: 'processing',
  kg_done: 'success',
  completed: 'success',
  failed: 'error',
  deleting: 'warning',
  delete_failed: 'error',
  cancelled: 'default',
};
