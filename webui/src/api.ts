/** API client for Forma Web UI */

import type { Stats, RequestListItem, RequestFullDetail, Upstream, ChatMessage, ChatCompletionResponse, ToolEvent, AgentEvent, EvaluationEvent, SummaryEvent, Agent } from "./types";

const API_BASE = "/ui";
const CHAT_API_BASE = "/v1";  // Use vite proxy

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

// === Semantic Block Parser ===

/**
 * Parse semantic blocks from text content.
 * 
 * Semantic blocks are human-readable markers:
 * [BLOCK_TYPE: name]
 * key: value
 * [/BLOCK_TYPE]
 */
function parseSemanticBlocks(text: string): Array<{ type: string; name: string; data: Record<string, string>; raw: string }> {
  const blocks: Array<{ type: string; name: string; data: Record<string, string>; raw: string }> = [];
  
  // Regex to match [TYPE: name]...[/TYPE] blocks
  // Handles both single-line and multi-line content
  const blockRegex = /\[(\w+)(?:\s*:\s*([^\]]+))?\]([\s\S]*?)\[\/\1\]/g;
  
  let match;
  while ((match = blockRegex.exec(text)) !== null) {
    const blockType = match[1];
    const blockName = match[2]?.trim() || "";
    const blockContent = match[3].trim();
    const rawBlock = match[0];
    
    // Parse key: value pairs from block content
    const data: Record<string, string> = {};
    const lines = blockContent.split('\n');
    for (const line of lines) {
      const colonIndex = line.indexOf(':');
      if (colonIndex > 0) {
        const key = line.slice(0, colonIndex).trim();
        const value = line.slice(colonIndex + 1).trim();
        if (key && value) {
          data[key] = value;
        }
      }
    }
    
    blocks.push({
      type: blockType,
      name: blockName,
      data,
      raw: rawBlock,
    });
  }
  
  return blocks;
}

/**
 * Convert a parsed semantic block to a ToolEvent.
 */
function blockToToolEvent(block: { type: string; name: string; data: Record<string, string>; raw: string }): ToolEvent | null {
  const timestamp = parseFloat(block.data.timestamp || "0") || Date.now();
  
  if (block.type === "TOOL_START") {
    return {
      type: "tool_call_start",
      timestamp,
      id: block.data.id || "",
      name: block.name,
      arguments: block.data.args ? JSON.parse(block.data.args) : {},
    };
  }
  
  if (block.type === "TOOL_END") {
    const success = block.data.status === "success";
    return {
      type: "tool_call_end",
      timestamp,
      id: block.data.id || "",
      name: block.name,
      success,
      duration_ms: parseFloat(block.data.duration?.replace("ms", "") || "0"),
      result_preview: block.data.result,
    };
  }
  
  if (block.type === "TOOL_LOOP_COMPLETE") {
    return {
      type: "tool_loop_complete",
      timestamp,
      total_tool_calls: parseInt(block.data.total_calls || "0"),
      total_tool_time_ms: parseFloat(block.data.total_time?.replace("ms", "") || "0"),
    };
  }
  
  if (block.type === "TOOL_PROGRESS") {
    const [iteration, max] = (block.data.iteration || "0/0").split("/").map(n => parseInt(n.trim()));
    return {
      type: "tool_loop_progress",
      timestamp,
      iteration: iteration || 0,
      max_iterations: max || 0,
    };
  }
  
  if (block.type === "TOOL_CALLS_RECEIVED") {
    const toolsList = block.data.tools?.split(",").map(t => t.trim()) || [];
    return {
      type: "tool_calls_received",
      timestamp,
      count: parseInt(block.data.count || "0"),
      tools: toolsList.map(name => ({ name, arguments: {} })),
    };
  }
  
  return null;
}

/**
 * Convert a parsed semantic block to an AgentEvent.
 */
function blockToAgentEvent(block: { type: string; name: string; data: Record<string, string>; raw: string }): AgentEvent | null {
  const timestamp = Date.now();
  
  if (block.type === "AGENT_START") {
    const depth = parseInt(block.data.depth || "0");
    const chain = block.data.chain?.split(" → ").map(a => a.trim()) || [block.name];
    return {
      type: "agent_start",
      timestamp,
      agent: block.name,
      depth,
      chain,
    };
  }
  
  if (block.type === "AGENT_END") {
    const depth = parseInt(block.data.depth || "0");
    const chain = block.data.chain?.split(" → ").map(a => a.trim()) || [block.name];
    return {
      type: "agent_end",
      timestamp,
      agent: block.name,
      depth,
      chain,
    };
  }
  
  return null;
}

/**
 * Convert a parsed semantic block to an EvaluationEvent.
 */
function blockToEvaluationEvent(block: { type: string; name: string; data: Record<string, string>; raw: string }): EvaluationEvent | null {
  if (block.type === "EVALUATION") {
    const confidenceStr = block.data.confidence?.replace("%", "") || "0";
    const confidence = parseFloat(confidenceStr) / 100;  // Convert percentage to decimal
    
    return {
      type: "evaluation_result",
      timestamp: Date.now(),
      agent: block.name,
      status: block.data.status as "complete" | "incomplete" | "failed",
      reason: block.data.reason,
      confidence,
      retry_instructions: block.data.retry_instructions,
    };
  }
  
  return null;
}

/**
 * Convert a parsed semantic block to a SummaryEvent.
 * 
 * NOTE: We extract content from block.raw because parseSemanticBlocks'
 * key:value parser only captures the first line of multi-line values.
 * This keeps the backend format human-readable for non-parsing frontends.
 */
function blockToSummaryEvent(block: { type: string; name: string; data: Record<string, string>; raw: string }): SummaryEvent | null {
  if (block.type === "SUMMARY") {
    // Extract everything between the opening [SUMMARY: name] and closing [/SUMMARY] tags
    const contentMatch = block.raw.match(/\[SUMMARY:[^\]]*\]\n?([\s\S]*?)\n?\[\/SUMMARY\]/);
    let content = contentMatch ? contentMatch[1].trim() : block.data.content || "";
    
    // Strip the "content:" prefix if present (backend adds it for consistency)
    if (content.startsWith("content:")) {
      content = content.slice(8).trim();
    }
    
    return {
      type: "summary_result",
      timestamp: Date.now(),
      agent: block.name,
      content,
    };
  }
  
  return null;
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
 * @param onEvaluationEvent - Callback for evaluation events (optional)
 * @param onSummaryEvent - Callback for summary events (optional)
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
  onAgentEvent?: (event: AgentEvent) => void,
  onEvaluationEvent?: (event: EvaluationEvent) => void,
  onSummaryEvent?: (event: SummaryEvent) => void
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
      
      // FIRST PASS: Parse raw semantic blocks that are sent directly (not wrapped in JSON)
      // AGENT_START/END and EVALUATION blocks are sent as raw text, not in SSE JSON
      // So we parse them from the raw buffer first
      const rawBlocks = parseSemanticBlocks(buffer);
      
      if (rawBlocks.length > 0) {
        // Process each semantic block found
        for (const block of rawBlocks) {
          // Only process AGENT and EVALUATION events in first pass
          // TOOL events come wrapped in SSE JSON, so they're processed in second pass
          
          // Process agent events (AGENT_START/END are sent as raw text)
          const agentEvent = blockToAgentEvent(block);
          if (agentEvent && onAgentEvent) {
            onAgentEvent(agentEvent);
          }
          
          // Process evaluation events (EVALUATION blocks are sent as raw text)
          const evalEvent = blockToEvaluationEvent(block);
          if (evalEvent && onEvaluationEvent) {
            onEvaluationEvent(evalEvent);
          }
          
          // Process summary events (SUMMARY blocks are sent as raw text)
          const summaryEvent = blockToSummaryEvent(block);
          if (summaryEvent && onSummaryEvent) {
            onSummaryEvent(summaryEvent);
          }
          
          // Only remove block from buffer if it's NOT a TOOL block
          // (TOOL blocks are inside JSON and will be removed in second pass)
          if (!block.type.startsWith('TOOL_')) {
            buffer = buffer.replace(block.raw, "");
          }
        }
      }
      
      // SECOND PASS: Process SSE events (line-by-line)
      // This is where TOOL events come from (wrapped in JSON content)
      const lines = buffer.split("\n");
      buffer = ""; // Reset buffer, will add back incomplete line

      for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        
        if (i === lines.length - 1 && !line.endsWith("\n")) {
          // Incomplete line, keep in buffer
          buffer = line;
          continue;
        }

        // Skip empty lines
        if (!line.trim()) {
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
            const reasoningContent = parsed.choices?.[0]?.delta?.reasoning_content;

            if (content) {
              // Check for semantic blocks inside content
              const blocks = parseSemanticBlocks(content);
              
              if (blocks.length > 0) {
                let remainingContent = content;
                
                for (const block of blocks) {
                  // Process agent events
                  const agentEvent = blockToAgentEvent(block);
                  if (agentEvent && onAgentEvent) {
                    onAgentEvent(agentEvent);
                  }
                  
                  // Process tool events
                  const toolEvent = blockToToolEvent(block);
                  if (toolEvent && onToolEvent) {
                    onToolEvent(toolEvent);
                  }
                  
                  // Process evaluation events
                  const evalEvent = blockToEvaluationEvent(block);
                  if (evalEvent && onEvaluationEvent) {
                    onEvaluationEvent(evalEvent);
                  }
                  
                  // Process summary events
                  const summaryEvent = blockToSummaryEvent(block);
                  if (summaryEvent && onSummaryEvent) {
                    onSummaryEvent(summaryEvent);
                  }
                  
                  // Remove the block from content
                  remainingContent = remainingContent.replace(block.raw, "");
                }
                
                // Send remaining content (without semantic blocks)
                if (remainingContent.trim()) {
                  onChunk(remainingContent);
                }
              } else {
                // No semantic blocks, just send the content
                onChunk(content);
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