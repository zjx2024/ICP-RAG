import DOMPurify from 'dompurify';
import { Marked } from 'marked';

// ===== formatMarkdown 处理函数 =====

export interface Citation {
  evidence_id?: string;
  status?: string;
  answer_citation_id?: number;
  doc_id?: string;
  doc_name?: string;
  title?: string;
  display_summary?: string;
  content?: string;
  snippet?: string;
  doc_type?: string;
  source_ref?: string;
  record_id?: string;
  image_url?: string;
  document_metadata?: Record<string, unknown>;
  [k: string]: unknown;
}

/** 按正文首次出现顺序返回本轮引用编号，兼容后端的 `[^n]` 标记。 */
export function extractCitationNumbers(text?: string): number[] {
  const numbers: number[] = [];
  const seen = new Set<number>();
  for (const match of String(text || '').matchAll(/\[\^(\d+)\]/g)) {
    const number = Number.parseInt(match[1], 10);
    if (number < 1 || seen.has(number)) continue;
    seen.add(number);
    numbers.push(number);
  }
  return numbers;
}

export function escapeHtml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
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

function safeImageSrc(value: string): string {
  const raw = String(value || '').trim();
  if (!raw || /[\u0000-\u001f\u007f\s]/.test(raw)) return '';
  const normalized = toRelImageSrc(raw);
  if (normalized.startsWith('/images/')) return normalized;
  try {
    const parsed = new URL(normalized);
    return parsed.protocol === 'http:' || parsed.protocol === 'https:' ? parsed.toString() : '';
  } catch {
    return '';
  }
}

function renderImage(src: string, alt = ''): string {
  const safeSrc = safeImageSrc(src);
  if (!safeSrc) return escapeHtml(alt || '[图片地址无效]');
  return (
    `<img class="diagnosis-markdown-image" src="${escapeHtml(safeSrc)}" `
    + `alt="${escapeHtml(alt)}" loading="lazy">`
  );
}

const markdown = new Marked({
  gfm: true,
  breaks: true,
  renderer: {
    // 工程资料内容不能通过回答文本注入任意 HTML；允许的图片单独走白名单。
    html({ text }) {
      return escapeHtml(text);
    },
    image({ href, text }) {
      return renderImage(href, text);
    },
  },
});

/**
 * 兼容旧会话中偶发的 JSON 风格结构转义，只修复行首 Markdown 标记。
 * 代码块、Windows 路径和 HMC 中间的 ``**`` 等业务字符保持原样。
 */
export function normalizeLegacyMarkdown(text: string): string {
  const lines = String(text || '').replace(/\r\n?/g, '\n').split('\n');
  const continuationLines = lines.filter((line) => /\\\s*$/.test(line)).length;
  const stripLegacyContinuations = continuationLines >= 2;
  let fenced = false;

  return lines.map((line) => {
    const isFence = /^\s*(?:```|~~~)/.test(line);
    if (isFence) {
      fenced = !fenced;
      return line;
    }
    if (fenced) return line;

    let normalized = line
      .replace(/^(\s*)\\(#{1,6}(?:\s|$))/, '$1$2')
      .replace(/^(\s*)\\([-+*]\s+)/, '$1$2')
      .replace(/^(\s*\d+)\\([.)]\s+)/, '$1$2');
    if (stripLegacyContinuations) normalized = normalized.replace(/\\\s*$/, '');
    // Some local-model answers contain literal ``**标题**`` markers (or an
    // escaped variant) instead of rendering them as emphasis.  They make the
    // diagnosis card look like an internal template.  Remove only paired
    // markers whose body contains no ``*``; an HMC wildcard such as
    // ``91860001**0001`` has no closing pair and is therefore preserved.
    normalized = normalized.replace(
      /(?<![A-Za-z0-9])\\?\*\\?\*([^*\n]+?\S)\\?\*\\?\*(?![A-Za-z0-9])/g,
      '$1',
    );
    return normalized;
  }).join('\n');
}

// 渲染内联引用 HTML（React 中需要事件委托，这里保留 data 属性由组件绑定）
export function formatMarkdown(
  text: string,
  citations?: Citation[],
  msgKey: string | number = 0,
): string {
  let result = normalizeLegacyMarkdown(text);

  // 新回答使用紧凑的 answer_citation_id，旧回答可能保留稀疏的 id。
  // 不能只按数组下标查找，否则 [^9] 会被错误绑定到数组第 9 项或直接丢失。
  const citationByNumber = new Map<number, Citation>();
  (citations || []).forEach((citation, index) => {
    const explicitNumbers = [citation.answer_citation_id, citation.id]
      .map((value) => Number(value))
      .filter((value) => Number.isInteger(value) && value > 0);
    const numbers = explicitNumbers.length ? explicitNumbers : [index + 1];
    numbers.forEach((number) => {
      if (!citationByNumber.has(number)) citationByNumber.set(number, citation);
    });
  });

  // 允许工程图证据使用历史 <img> 形式，其余原始 HTML 由 renderer 转义。
  const rawImages: string[] = [];
  result = result.replace(/<img\s+[^>]*src=["']([^"']+)["'][^>]*\/?>/gi, (_match, src) => {
    const index = rawImages.push(String(src)) - 1;
    return `\uE100RAWIMAGE${index}\uE101`;
  });

  // Marked 不负责项目特有的 [^n] 引用；先换为不可碰撞占位符，解析后再还原。
  const citationHtml: string[] = [];
  result = result.replace(/\[\^(\d+)\]/g, (_match, rawNumber) => {
    const number = Number.parseInt(rawNumber, 10);
    const citation = citationByNumber.get(number);
    if (!citation) return '';
    const sequence = citationHtml.length + 1;
    const label = escapeHtml(String(citation.doc_name || citation.title || `引用${number}`));
    citationHtml.push(
      `<button class="inline-cite" type="button" data-cite-message-key="${escapeHtml(String(msgKey))}" `
      + `data-cite-number="${number}" aria-expanded="false" aria-controls="diagnosis-evidence-rail" `
      + `title="在证据栏定位：${label}" aria-label="在证据栏查看来源 ${number}：${label}">[${number}]</button>`,
    );
    return `\uE100CITATION${sequence - 1}\uE101`;
  });

  let html = markdown.parse(result, { async: false });
  html = html.replace(/\uE100RAWIMAGE(\d+)\uE101/g, (_match, rawIndex) => (
    renderImage(rawImages[Number.parseInt(rawIndex, 10)] || '', '图片')
  ));
  html = html.replace(/\uE100CITATION(\d+)\uE101/g, (_match, rawIndex) => (
    citationHtml[Number.parseInt(rawIndex, 10)] || ''
  ));

  return DOMPurify.sanitize(html, {
    ALLOW_DATA_ATTR: true,
    ADD_ATTR: ['loading', 'role', 'tabindex', 'type', 'aria-controls'],
    FORBID_TAGS: ['script', 'style', 'iframe', 'object', 'embed', 'form'],
    FORBID_ATTR: ['style'],
  });
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
function parseServerDateTime(value?: string): Date | null {
  if (!value) return null;
  const raw = String(value).trim();
  // 后端部分 TIMESTAMP WITHOUT TIME ZONE 字段实际保存 UTC。无时区值必须先补
  // UTC 标识，再统一转换为北京时间；已有偏移的时间继续遵循其绝对时间语义。
  const hasTimezone = /(?:z|[+-]\d{2}:?\d{2})$/i.test(raw);
  const looksLikeServerTimestamp = /^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?$/.test(raw);
  const normalized = looksLikeServerTimestamp && !hasTimezone
    ? `${raw.replace(' ', 'T')}Z`
    : raw;
  const date = new Date(normalized);
  return Number.isNaN(date.getTime()) ? null : date;
}

function shanghaiDateParts(date: Date, includeSeconds: boolean) {
  const parts = new Intl.DateTimeFormat('zh-CN', {
    timeZone: 'Asia/Shanghai',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    ...(includeSeconds ? { second: '2-digit' } : {}),
    hourCycle: 'h23',
  }).formatToParts(date);
  return (type: Intl.DateTimeFormatPartTypes) => (
    parts.find((part) => part.type === type)?.value || ''
  );
}

export function formatDateTime(iso?: string): string {
  if (!iso) return '';
  const date = parseServerDateTime(iso);
  if (!date) return String(iso);
  const get = shanghaiDateParts(date, true);
  return `${get('year')}-${get('month')}-${get('day')} ${get('hour')}:${get('minute')}:${get('second')}`;
}

export function formatTimeShort(iso?: string): string {
  if (!iso) return '';
  const date = parseServerDateTime(iso);
  if (!date) return String(iso);
  const get = shanghaiDateParts(date, false);
  return `${get('month')}-${get('day')} ${get('hour')}:${get('minute')}`;
}

// ===== 文档状态映射（含 ICP 类型驱动文档状态） =====
export const DOC_STATUS_LABEL: Record<string, string> = {
  uploaded: '已上传',
  parsing: '解析中',
  indexed: '已索引',
  kg_pending: '待自动图处理',
  generating_kg: '自动图处理中',
  kg_done: '图数据已就绪',
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
