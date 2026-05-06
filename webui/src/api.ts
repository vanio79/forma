/** API client for Forma Web UI */

import type { Stats, RequestListItem, RequestFullDetail, Upstream } from "./types";

const API_BASE = "/ui";

async function fetchJSON<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

export async function getStats(): Promise<Stats> {
  return fetchJSON<Stats>(`${API_BASE}/stats`);
}

export async function getRequests(
  limit = 100,
  offset = 0
): Promise<{ requests: RequestListItem[]; limit: number; offset: number }> {
  return fetchJSON<{ requests: RequestListItem[]; limit: number; offset: number }>(
    `${API_BASE}/requests?limit=${limit}&offset=${offset}`
  );
}

export async function getRequestDetail(requestId: string): Promise<RequestFullDetail> {
  return fetchJSON<RequestFullDetail>(`${API_BASE}/requests/${requestId}`);
}

export async function clearData(): Promise<{ status: string; message: string }> {
  const response = await fetch(`${API_BASE}/clear`, { method: "DELETE" });
  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

// === Upstream Management ===

export async function getUpstreams(): Promise<{ upstreams: Upstream[] }> {
  return fetchJSON<{ upstreams: Upstream[] }>(`${API_BASE}/upstreams`);
}

export async function getUpstream(upstreamId: string): Promise<{ upstream: Upstream }> {
  return fetchJSON<{ upstream: Upstream }>(`${API_BASE}/upstreams/${upstreamId}`);
}

export async function createUpstream(params: {
  name: string;
  upstream_model?: string;
  base_url: string;
  api_key?: string;
  timeout?: number;
  is_enabled?: boolean;
}): Promise<{ status: string; message: string; upstream: Upstream }> {
  const query = new URLSearchParams();
  query.set("name", params.name);
  if (params.upstream_model) query.set("upstream_model", params.upstream_model);
  query.set("base_url", params.base_url);
  if (params.api_key) query.set("api_key", params.api_key);
  if (params.timeout) query.set("timeout", params.timeout.toString());
  if (params.is_enabled !== undefined) query.set("is_enabled", params.is_enabled ? "true" : "false");
  
  const response = await fetch(`${API_BASE}/upstreams?${query.toString()}`, { method: "POST" });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(errorData.detail || `API error: ${response.status}`);
  }
  return response.json();
}

export async function updateUpstream(
  upstreamId: string,
  params: {
    name?: string;
    upstream_model?: string;
    base_url?: string;
    api_key?: string;
    timeout?: number;
    is_enabled?: boolean;
  }
): Promise<{ status: string; message: string; upstream: Upstream }> {
  const query = new URLSearchParams();
  if (params.name) query.set("name", params.name);
  if (params.upstream_model !== undefined) query.set("upstream_model", params.upstream_model);
  if (params.base_url) query.set("base_url", params.base_url);
  if (params.api_key !== undefined) query.set("api_key", params.api_key);
  if (params.timeout !== undefined) query.set("timeout", params.timeout.toString());
  if (params.is_enabled !== undefined) query.set("is_enabled", params.is_enabled ? "true" : "false");
  
  const response = await fetch(`${API_BASE}/upstreams/${upstreamId}?${query.toString()}`, { method: "PUT" });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(errorData.detail || `API error: ${response.status}`);
  }
  return response.json();
}

export async function deleteUpstream(upstreamId: string): Promise<{ status: string; message: string }> {
  const response = await fetch(`${API_BASE}/upstreams/${upstreamId}`, { method: "DELETE" });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(errorData.detail || `API error: ${response.status}`);
  }
  return response.json();
}