/** TypeScript types for Forma Web UI API */

export interface Stats {
  total_requests: number;
  total_extractions: number;
  total_retrievals: number;
  avg_extraction_ms: number;
  extractions_by_type: {
    entities: number;
    relationships: number;
    facts: number;
    recipes: number;
  };
  upstream_count: number;
}

export interface RequestListItem {
  id: string;
  model: string;
  user_prompt: string;
  timestamp: number;
  timestamp_formatted: string;
  extraction_ms: number;
  has_extraction: boolean;
  has_augmentation: boolean;
}

export interface RequestDetail {
  id: string;
  model: string;
  user_prompt: string;
  history: string;
  extraction_response: string;
  extraction_prompt: string;
  extraction_ms: number;
  augmented_prompt: string;
  agent_response: string;
  timestamp: number;
  timestamp_formatted: string;
}

export interface ExtractionItem {
  id: string;
  data: string;
  confidence: number;
}

export interface RetrievalItem {
  id: string;
  data: string;
  confidence: number;
  score: number;
}

export interface ExtractionsByType {
  relationship?: ExtractionItem[];
  fact?: ExtractionItem[];
  recipe?: ExtractionItem[];
}

export interface RetrievalsByType {
  relationship?: RetrievalItem[];
  fact?: RetrievalItem[];
  recipe?: RetrievalItem[];
}

export interface RequestFullDetail {
  request: RequestDetail;
  extractions: ExtractionsByType;
  retrievals: RetrievalsByType;
}

export interface Upstream {
  id: string;
  name: string; // Local model name used for routing
  upstream_model: string; // Model name to send to upstream API
  base_url: string;
  api_key: string;
  timeout: number;
  is_enabled: boolean;
  created_at: number;
  updated_at: number;
}

export interface CreateUpstreamRequest {
  name: string;
  upstream_model?: string;
  base_url: string;
  api_key?: string;
  timeout?: number;
  is_enabled?: boolean;
}

export interface UpdateUpstreamRequest {
  name?: string;
  upstream_model?: string;
  base_url?: string;
  api_key?: string;
  timeout?: number;
  is_enabled?: boolean;
}

// === Chat Types ===

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
  reasoning?: string;  // Reasoning content (separate from main content)
  timestamp?: number;
  isStreaming?: boolean;  // For messages being streamed (assistant responses or compaction summaries)
  isCompacting?: boolean; // For compaction progress messages
  showReasoning?: boolean; // Whether reasoning section is expanded (UI state)
  agentName?: string;  // Which agent generated this response (for multi-agent scenarios)
  agentChain?: string[];  // Delegation chain showing how this agent was reached (e.g., ["assistant", "researcher"])
  // Tool execution state
  toolExecution?: ToolExecutionState;
  toolExecutionExpanded?: boolean; // Whether tool execution details are expanded (UI state)
}

// Tool event types
export type ToolEventType = 
  | "tool_loop_progress"
  | "tool_calls_received"
  | "tool_call_start"
  | "tool_call_end"
  | "tool_loop_complete";

export interface ToolEvent {
  type: ToolEventType;
  timestamp: number;
  // For tool_loop_progress
  iteration?: number;
  max_iterations?: number;
  // For tool_calls_received
  count?: number;
  tools?: { name: string; arguments: Record<string, unknown> }[];
  // For tool_call_start
  id?: string;
  name?: string;
  arguments?: Record<string, unknown>;
  // For tool_call_end
  success?: boolean;
  duration_ms?: number;
  result_preview?: string;
  // For tool_loop_complete
  total_tool_calls?: number;
  total_tool_time_ms?: number;
}

export interface ToolCallInfo {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
  status: "pending" | "running" | "success" | "failed";
  duration_ms?: number;
  result?: string;
  error?: string;
  expanded?: boolean;  // Whether result is expanded (UI state)
}

export interface ToolExecutionState {
  iteration: number;
  maxIterations: number;
  toolCalls: ToolCallInfo[];
  isComplete: boolean;
  totalTimeMs: number;
}

export interface ChatCompletionRequest {
  model: string;
  messages: ChatMessage[];
  stream?: boolean;
  max_tokens?: number;
  temperature?: number;
}

export interface ChatCompletionChunk {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: {
    index: number;
    delta: {
      role?: string;
      content?: string;
      reasoning_content?: string;  // DeepSeek R1 and similar models output reasoning separately
    };
    finish_reason: string | null;
  }[];
}

export interface TokenUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface ChatCompletionResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: {
    index: number;
    message: {
      role: string;
      content: string;
    };
    finish_reason: string;
  }[];
  usage: TokenUsage;
}

// === Agent Types ===

export type AgentEventType = "agent_start" | "agent_end";

export interface AgentEvent {
  type: AgentEventType;
  agent: string;  // Agent name
  depth?: number;  // Nesting depth (0 = primary agent, 1 = sub-agent, etc.)
  chain?: string[];  // Delegation chain (e.g., ["assistant", "researcher", "coder"])
}

export interface Agent {
  id: string;
  name: string;
  purpose: string;
  instruction_prompt: string;
  upstream_id: string | null;
  tools_enabled: boolean;
  tool_whitelist: string[];
  max_iterations: number;
  is_enabled: boolean;
  created_at: number;
  updated_at: number;
}