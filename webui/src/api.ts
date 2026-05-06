/** API client for Forma Web UI */

import type { Stats, RequestListItem, RequestFullDetail } from "./types";

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

export async function clearTrackingData(): Promise<{ status: string; message: string }> {
  const response = await fetch(`${API_BASE}/clear`, { method: "DELETE" });
  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`);
  }
  return response.json();
}