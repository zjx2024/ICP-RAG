// ===== fetch 统一封装（service 层基础） =====

const BASE_URL = '/api/v1';

// 业务接口默认挂在 /api/v1 下；以 /api/ 开头的路径（如 /api/agent、/api/v1/...）视为完整前缀，不再叠加
function resolveUrl(path: string): string {
  return path.startsWith('/api/') ? path : `${BASE_URL}${path}`;
}

export class ApiError extends Error {
  status: number;
  data?: unknown;
  constructor(status: number, message: string, data?: unknown) {
    super(message);
    this.status = status;
    this.data = data;
  }
}

export async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(resolveUrl(path), {
    headers: options?.body instanceof FormData ? undefined : { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    let message = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      message = body?.detail || body?.message || message;
    } catch {
      /* ignore */
    }
    throw new ApiError(res.status, message);
  }
  return res.json() as Promise<T>;
}

// 支持 text 响应（如 SSE 等）
export async function requestText(path: string, options?: RequestInit): Promise<string> {
  const res = await fetch(resolveUrl(path), options);
  if (!res.ok) {
    throw new ApiError(res.status, `HTTP ${res.status}`);
  }
  return res.text();
}

export function buildQuery(params: Record<string, string | number | boolean | undefined | null>): string {
  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== '') sp.append(k, String(v));
  }
  const s = sp.toString();
  return s ? `?${s}` : '';
}
