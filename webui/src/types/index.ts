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
  entity?: ExtractionItem[];
  relationship?: ExtractionItem[];
  fact?: ExtractionItem[];
  recipe?: ExtractionItem[];
}

export interface RetrievalsByType {
  entity?: RetrievalItem[];
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
  timestamp?: number;
  isStreaming?: boolean;  // For messages being streamed (assistant responses or compaction summaries)
  isCompacting?: boolean; // For compaction progress messages
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