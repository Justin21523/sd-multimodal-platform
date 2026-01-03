export type ApiError = {
  message: string;
  status?: number;
  detail?: unknown;
};

const DEFAULT_API_BASE = "http://localhost:8000";

export function getApiBaseUrl(): string {
  const raw = (import.meta as any).env?.VITE_API_BASE_URL as string | undefined;
  return (raw && raw.trim()) || DEFAULT_API_BASE;
}

function joinUrl(base: string, path: string): string {
  const baseTrimmed = base.replace(/\/+$/, "");
  const pathTrimmed = path.startsWith("/") ? path : `/${path}`;
  return `${baseTrimmed}${pathTrimmed}`;
}

export function apiUrl(path: string): string {
  return joinUrl(getApiBaseUrl(), path);
}

export function withQuery(path: string, params: Record<string, string | number | boolean | undefined | null>): string {
  const entries = Object.entries(params).filter(([, v]) => v !== undefined && v !== null && String(v).length > 0);
  if (!entries.length) return path;
  const query = entries
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`)
    .join("&");
  return `${path}${path.includes("?") ? "&" : "?"}${query}`;
}

export function resolveMediaUrl(maybePath: string): string {
  if (!maybePath) return "";
  if (maybePath.startsWith("http://") || maybePath.startsWith("https://")) return maybePath;
  if (maybePath.startsWith("/")) return apiUrl(maybePath);
  return maybePath;
}

async function parseJsonSafe(response: Response) {
  const text = await response.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

export async function apiGet<T>(path: string): Promise<T> {
  const response = await fetch(apiUrl(path), {
    method: "GET",
    headers: { Accept: "application/json" }
  });
  if (!response.ok) {
    const detail = await parseJsonSafe(response);
    throw { message: "Request failed", status: response.status, detail } satisfies ApiError;
  }
  return (await response.json()) as T;
}

export async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(apiUrl(path), {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(body)
  });
  if (!response.ok) {
    const detail = await parseJsonSafe(response);
    throw { message: "Request failed", status: response.status, detail } satisfies ApiError;
  }
  return (await response.json()) as T;
}

export async function apiPatch<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(apiUrl(path), {
    method: "PATCH",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(body)
  });
  if (!response.ok) {
    const detail = await parseJsonSafe(response);
    throw { message: "Request failed", status: response.status, detail } satisfies ApiError;
  }
  return (await response.json()) as T;
}

export async function apiDelete<T>(path: string): Promise<T> {
  const response = await fetch(apiUrl(path), {
    method: "DELETE",
    headers: { Accept: "application/json" }
  });
  if (!response.ok) {
    const detail = await parseJsonSafe(response);
    throw { message: "Request failed", status: response.status, detail } satisfies ApiError;
  }
  return (await response.json()) as T;
}

export async function apiPostForm<T>(path: string, form: FormData): Promise<T> {
  const response = await fetch(apiUrl(path), {
    method: "POST",
    headers: { Accept: "application/json" },
    body: form
  });
  if (!response.ok) {
    const detail = await parseJsonSafe(response);
    throw { message: "Request failed", status: response.status, detail } satisfies ApiError;
  }
  return (await response.json()) as T;
}

export async function apiPostEmpty<T>(path: string): Promise<T> {
  const response = await fetch(apiUrl(path), {
    method: "POST",
    headers: { Accept: "application/json" }
  });
  if (!response.ok) {
    const detail = await parseJsonSafe(response);
    throw { message: "Request failed", status: response.status, detail } satisfies ApiError;
  }
  return (await response.json()) as T;
}
