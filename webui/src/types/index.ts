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