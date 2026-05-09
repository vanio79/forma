/** API client for Forma Web UI */

import type { Stats, RequestListItem, RequestFullDetail, Upstream, ChatMessage, ChatCompletionResponse, ToolEvent, AgentEvent, Agent } from "./types";

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
 * @param onReasoningChunk - Callback for each chunk of reasoning content (optional)
 * @param onToolEvent - Callback for tool execution events (optional)
 * @param onAgentEvent - Callback for agent start/end events (optional)
 * @param onComplete - Callback when stream completes
 * @param onError - Callback for errors
 */
export async function streamChatCompletion(
  model: string,
  messages: ChatMessage[],
  onChunk: (chunk: string) => void,
  onComplete: () => void,
  onError: (error: string) => void,
  onReasoningChunk?: (chunk: string) => void,
  onToolEvent?: (event: ToolEvent) => void,
  onAgentEvent?: (event: AgentEvent) => void
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

        // Check for agent markers in raw lines (not wrapped in data:)
        if (line.includes("__AGENT_START__") || line.includes("__AGENT_END__")) {
          const agentStartRegex = /__AGENT_START__(.+?)__END__/g;
          const agentEndRegex = /__AGENT_END__(.+?)__END__/g;
          let match;

          // Parse agent start markers
          while ((match = agentStartRegex.exec(line)) !== null) {
            try {
              const eventData = JSON.parse(match[1]) as AgentEvent;
              eventData.type = "agent_start";
              if (onAgentEvent) {
                onAgentEvent(eventData);
              }
            } catch {
              // Ignore parse errors for agent events
            }
          }

          // Parse agent end markers
          while ((match = agentEndRegex.exec(line)) !== null) {
            try {
              const eventData = JSON.parse(match[1]) as AgentEvent;
              eventData.type = "agent_end";
              if (onAgentEvent) {
                onAgentEvent(eventData);
              }
            } catch {
              // Ignore parse errors for agent events
            }
          }
          continue;  // Skip further processing for marker lines
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
            const reasoningContent = parsed.choices?.[0]?.delta?.reasoning_content;

            if (content) {
              // Check for agent markers inside content (legacy fallback)
              const agentStartRegex = /__AGENT_START__(.+?)__END__/g;
              const agentEndRegex = /__AGENT_END__(.+?)__END__/g;
              let match;
              let remainingContent = content;
              
              // Parse agent start markers
              while ((match = agentStartRegex.exec(content)) !== null) {
                try {
                  const eventData = JSON.parse(match[1]) as AgentEvent;
                  eventData.type = "agent_start";
                  if (onAgentEvent) {
                    onAgentEvent(eventData);
                  }
                } catch {
                  // Ignore parse errors for agent events
                }
                remainingContent = remainingContent.replace(match[0], "");
              }
              
              // Parse agent end markers
              while ((match = agentEndRegex.exec(content)) !== null) {
                try {
                  const eventData = JSON.parse(match[1]) as AgentEvent;
                  eventData.type = "agent_end";
                  if (onAgentEvent) {
                    onAgentEvent(eventData);
                  }
                } catch {
                  // Ignore parse errors for agent events
                }
                remainingContent = remainingContent.replace(match[0], "");
              }
              
              // Check for tool event markers
              const toolEventRegex = /__TOOL_EVENT__(.+?)__END__/g;
              
              while ((match = toolEventRegex.exec(content)) !== null) {
                // Parse the tool event JSON
                try {
                  const eventData = JSON.parse(match[1]) as ToolEvent;
                  if (onToolEvent) {
                    onToolEvent(eventData);
                  }
                } catch {
                  // Ignore parse errors for tool events
                }
                // Remove the marker from remaining content
                remainingContent = remainingContent.replace(match[0], "");
              }
              
              // Send remaining content (without agent or tool event markers)
              // Don't trim - spaces are valid content
              if (remainingContent) {
                onChunk(remainingContent);
              }
            }
            
            // Handle reasoning content (DeepSeek R1 and similar models)
            if (reasoningContent && onReasoningChunk) {
              onReasoningChunk(reasoningContent);
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

// === Agent Management ===

export async function getAgents(): Promise<{ agents: Agent[] }> {
  return fetchJSON<{ agents: Agent[] }>(`${API_BASE}/agents`);
}

export async function getAgent(agentId: string): Promise<{ agent: Agent }> {
  return fetchJSON<{ agent: Agent }>(`${API_BASE}/agents/${agentId}`);
}

export async function createAgent(params: {
  name: string;
  purpose: string;
  instruction_prompt: string;
  upstream_id?: string;
  tools_enabled?: boolean;
  tool_whitelist?: string[];
  max_iterations?: number;
  is_enabled?: boolean;
}): Promise<{ status: string; message: string; agent: Agent }> {
  const query = new URLSearchParams();
  query.set("name", params.name);
  query.set("purpose", params.purpose);
  query.set("instruction_prompt", params.instruction_prompt);
  if (params.upstream_id) query.set("upstream_id", params.upstream_id);
  if (params.tools_enabled !== undefined) query.set("tools_enabled", params.tools_enabled ? "true" : "false");
  if (params.tool_whitelist) query.set("tool_whitelist", JSON.stringify(params.tool_whitelist));
  if (params.max_iterations !== undefined) query.set("max_iterations", params.max_iterations.toString());
  if (params.is_enabled !== undefined) query.set("is_enabled", params.is_enabled ? "true" : "false");
  
  const response = await fetch(`${API_BASE}/agents?${query.toString()}`, { method: "POST" });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(errorData.detail || `API error: ${response.status}`);
  }
  return response.json();
}

export async function updateAgent(
  agentId: string,
  params: {
    name?: string;
    purpose?: string;
    instruction_prompt?: string;
    upstream_id?: string;
    tools_enabled?: boolean;
    tool_whitelist?: string[];
    max_iterations?: number;
    is_enabled?: boolean;
  }
): Promise<{ status: string; message: string; agent: Agent }> {
  const query = new URLSearchParams();
  if (params.name) query.set("name", params.name);
  if (params.purpose) query.set("purpose", params.purpose);
  if (params.instruction_prompt) query.set("instruction_prompt", params.instruction_prompt);
  if (params.upstream_id !== undefined) query.set("upstream_id", params.upstream_id);
  if (params.tools_enabled !== undefined) query.set("tools_enabled", params.tools_enabled ? "true" : "false");
  if (params.tool_whitelist !== undefined) query.set("tool_whitelist", JSON.stringify(params.tool_whitelist));
  if (params.max_iterations !== undefined) query.set("max_iterations", params.max_iterations.toString());
  if (params.is_enabled !== undefined) query.set("is_enabled", params.is_enabled ? "true" : "false");
  
  const response = await fetch(`${API_BASE}/agents/${agentId}?${query.toString()}`, { method: "PUT" });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(errorData.detail || `API error: ${response.status}`);
  }
  return response.json();
}

export async function deleteAgent(agentId: string): Promise<{ status: string; message: string }> {
  const response = await fetch(`${API_BASE}/agents/${agentId}`, { method: "DELETE" });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(errorData.detail || `API error: ${response.status}`);
  }
  return response.json();
}