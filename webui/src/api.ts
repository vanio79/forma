/** API client for Forma Web UI */

import type { Stats, RequestListItem, RequestFullDetail, Upstream, ChatMessage, ChatCompletionResponse } from "./types";

const API_BASE = "/ui";
const CHAT_API_BASE = "/v1";

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

// === Chat ===

/**
 * Stream a chat completion request.
 * 
 * @param model - Model name to use
 * @param messages - Chat messages
 * @param onChunk - Callback for each chunk of content
 * @param onComplete - Callback when stream completes
 * @param onError - Callback for errors
 */
export async function streamChatCompletion(
  model: string,
  messages: ChatMessage[],
  onChunk: (chunk: string) => void,
  onComplete: () => void,
  onError: (error: string) => void
): Promise<void> {
  const response = await fetch(`${CHAT_API_BASE}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model,
      messages,
      stream: true,
    }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: "Unknown error" }));
    onError(errorData.detail || `API error: ${response.status}`);
    return;
  }

  const reader = response.body?.getReader();
  if (!reader) {
    onError("No response body");
    return;
  }

  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      
      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      
      // Process SSE events
      const lines = buffer.split("\n");
      buffer = ""; // Reset buffer, will add back incomplete line

      for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        
        if (i === lines.length - 1 && !line.endsWith("\n")) {
          // Incomplete line, keep in buffer
          buffer = line;
          continue;
        }

        if (line.startsWith("data: ")) {
          const data = line.slice(6).trim();
          
          if (data === "[DONE]") {
            onComplete();
            return;
          }

          try {
            const parsed = JSON.parse(data);
            const content = parsed.choices?.[0]?.delta?.content;
            if (content) {
              onChunk(content);
            }
          } catch {
            // Ignore parse errors for malformed chunks
          }
        }
      }
    }

    onComplete();
  } catch (e) {
    onError(e instanceof Error ? e.message : "Stream error");
  }
}

/**
 * Send a non-streaming chat completion request.
 * Returns the full response with usage stats.
 * 
 * @param model - Model name to use
 * @param messages - Chat messages
 * @returns Promise with the completion response including usage stats
 */
export async function nonStreamingChatCompletion(
  model: string,
  messages: ChatMessage[]
): Promise<ChatCompletionResponse> {
  const response = await fetch(`${CHAT_API_BASE}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model,
      messages,
      stream: false,
    }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(errorData.detail || `API error: ${response.status}`);
  }

  return response.json();
}