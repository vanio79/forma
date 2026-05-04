# Autonomous Cognitive Proxy and Hybrid RAG System
## Design Document and Implementation Roadmap

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Core Subsystems](#3-core-subsystems)
   - 3.1 [Proxy and Streaming Infrastructure](#31-proxy-and-streaming-infrastructure)
   - 3.2 [Reinforcement Learning Decomposition Engine](#32-reinforcement-learning-decomposition-engine)
   - 3.3 [Automated Hybrid Ingestion Engine](#33-automated-hybrid-ingestion-engine)
   - 3.4 [Agentic Retrieval and Verification Loop](#34-agentic-retrieval-and-verification-loop)
4. [Technical Specifications](#4-technical-specifications)
5. [Implementation Roadmap](#5-implementation-roadmap)
6. [Appendices](#6-appendices)

---

## 1. Executive Summary

### 1.1 Project Vision

This document outlines the architecture for an advanced, OpenAI-compatible proxy service that operates as an **autonomous cognitive middleware layer**. Unlike traditional proxy servers that merely intercept and forward requests, this system:

- Dynamically decomposes complex tasks
- Extracts structured data to build ephemeral/persistent hybrid datastores
- Iteratively retrieves and verifies contextual information
- Synthesizes optimized payloads before forwarding to upstream LLMs

### 1.2 Key Differentiators

| Feature | Traditional Proxy | Cognitive Proxy |
|---------|------------------|-----------------|
| Request Handling | Pass-through | Multi-agent orchestration |
| Task Processing | None | Dynamic decomposition via RL |
| Data Management | None | Hybrid vector + graph stores |
| Context Retrieval | None | Iterative verification loops |
| Client Awareness | N/A | Completely transparent |

### 1.3 Design Principles

1. **Transparency**: Client applications require zero modifications
2. **Autonomy**: Self-improving via reinforcement learning
3. **Locality**: All datastores run as Python-native dependencies
4. **Verifiability**: Every retrieval step is validated

---

## 2. System Architecture Overview

### 2.1 High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CLIENT APPLICATIONS                                │
│                    (No code changes required)                                │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      COGNITIVE PROXY MIDDLEWARE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                    1. PROXY & STREAMING LAYER                      │     │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │     │
│  │  │   FastAPI    │  │   LiteLLM    │  │  Telemetry   │             │     │
│  │  │   Server     │  │   Router     │  │   Middleware │             │     │
│  │  └──────────────┘  └──────────────┘  └──────────────┘             │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                  │                                           │
│                                  ▼                                           │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │               2. RL DECOMPOSITION ENGINE                           │     │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │     │
│  │  │  Small LM    │  │    PPO       │  │  Evaluator   │             │     │
│  │  │  (Qwen/LLaMA)│  │  Trainer     │  │   Model      │             │     │
│  │  └──────────────┘  └──────────────┘  └──────────────┘             │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                  │                                           │
│                                  ▼                                           │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │               3. HYBRID INGESTION ENGINE                           │     │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │     │
│  │  │   Text       │  │  FastEmbed   │  │  Extraction  │             │     │
│  │  │  Chunker     │  │  (CPU-opt)   │  │   + Conf.    │             │     │
│  │  └──────────────┘  └──────────────┘  └──────────────┘             │     │
│  │                                                                     │     │
│  │  ┌────────────────────────────┐  ┌────────────────────────────┐   │     │
│  │  │     LanceDB (Vector)        │  │     Kùzu (Graph)           │   │     │
│  │  │     Embedded Store         │  │     Embedded Store         │   │     │
│  │  └────────────────────────────┘  └────────────────────────────┘   │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                  │                                           │
│                                  ▼                                           │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │               4. AGENTIC RETRIEVAL LOOP                            │     │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐     │     │
│  │  │  Query    │───▶│ Retrieve │───▶│  Verify  │───▶│  Rerank  │     │     │
│  │  │  Gen      │    │  Hybrid  │    │  & Eval  │    │  & Fuse  │     │     │
│  │  └──────────┘    └──────────┘    └──────────┘    └──────────┘     │     │
│  │       ▲                                                │            │     │
│  │       └────────────────────────────────────────────────┘            │     │
│  │                      (Retry if needed)                               │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        UPSTREAM LLM PROVIDERS                                │
│         (OpenAI, Anthropic, Google, Azure, AWS Bedrock, etc.)                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow Overview

```
User Request
     │
     ▼
┌─────────────────┐
│ Proxy Intercept │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│         TASK DECOMPOSITION PHASE            │
│  ┌───────────────────────────────────────┐ │
│  │ 1. Analyze prompt complexity          │ │
│  │ 2. Decompose into sub-tasks           │ │
│  │ 3. Rewrite for clarity                │ │
│  │ 4. Evaluate & score decomposition     │ │
│  │ 5. Update RL policy if needed          │ │
│  └───────────────────────────────────────┘ │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│         INGESTION PHASE (if needed)          │
│  ┌───────────────────────────────────────┐   │
│  │ 1. Chunk text (paragraph-level)       │   │
│  │ 2. Generate embeddings (FastEmbed)     │   │
│  │ 3. Extract entities/relations          │   │
│  │ 4. Calculate confidence scores        │   │
│  │ 5. Store in LanceDB + Kùzu            │   │
│  └───────────────────────────────────────┘   │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│         RETRIEVAL PHASE (Agentic Loop)       │
│  ┌───────────────────────────────────────┐   │
│  │ Loop until verified:                   │   │
│  │   1. Generate queries (vector+graph)   │   │
│  │   2. Execute hybrid retrieval           │   │
│  │   3. Verify relevance                   │   │
│  │   4. If failed, refine & retry          │   │
│  │   5. Fuse results (RRF + confidence)    │   │
│  │   6. Enforce token budget               │   │
│  └───────────────────────────────────────┘   │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│         FORWARDING PHASE                     │
│  ┌───────────────────────────────────────┐  │
│  │ 1. Assemble enriched prompt            │  │
│  │ 2. Route to optimal upstream provider │  │
│  │ 3. Stream response back to client     │  │
│  │ 4. Log telemetry & costs              │  │
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

---

## 3. Core Subsystems

### 3.1 Proxy and Streaming Infrastructure

#### 3.1.1 Technology Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| Web Framework | FastAPI | Native ASGI, Pydantic integration, streaming support |
| API Abstraction | LiteLLM | Multi-provider routing, unified interface |
| Streaming | Server-Sent Events (SSE) | Real-time chunk delivery |
| Async Runtime | asyncio + uvicorn | High-concurrency handling |

#### 3.1.2 OpenAI API Compatibility

The proxy must emulate these endpoints:

```
POST /v1/chat/completions
POST /v1/completions
POST /v1/embeddings
GET  /v1/models
```

#### 3.1.3 Streaming Implementation

```python
# Pseudo-architecture for SSE streaming
async def stream_response(upstream_generator):
    """Yield chunks in SSE format."""
    async for chunk in upstream_generator:
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"
```

#### 3.1.4 Telemetry & Middleware Hooks

| Metric | Collection Point | Storage |
|--------|------------------|---------|
| Request Latency | Pre/Post middleware | SQLite/PostgreSQL |
| Token Consumption | Model inference calls | SQLite/PostgreSQL |
| Cost Tracking | Per-provider rates | SQLite/PostgreSQL |
| Error Rates | Exception handlers | SQLite/PostgreSQL |

### 3.2 Reinforcement Learning Decomposition Engine

#### 3.2.1 Small Language Model Selection

| Model | Parameters | Use Case | Advantages |
|-------|------------|----------|------------|
| **Qwen3-8B** | 8B | Primary decomposition | Highest task extraction accuracy |
| **Llama-3.2-3B-Instruct** | 3B | High-throughput scenarios | Memory-efficient, fast |
| **LFM2-2.6B-Exp** | 2.6B | Tunable scenarios | Highest fine-tuning delta |

#### 3.2.2 PPO Framework Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PPO TRAINING LOOP                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────────┐      ┌──────────────┐      ┌──────────────┐│
│   │    State     │      │   Policy     │      │    Action    ││
│   │  (Prompt +   │─────▶│   Network    │─────▶│  (Rewritten  ││
│   │   Context)   │      │   (SLM)      │      │    Prompt)   ││
│   └──────────────┘      └──────────────┘      └──────────────┘│
│          │                                            │        │
│          │                                            │        │
│          │              ┌──────────────┐              │        │
│          │              │              │              │        │
│          └─────────────▶│  Evaluator   │◀─────────────┘        │
│                         │   Model      │                       │
│                         │              │                       │
│                         └──────┬───────┘                       │
│                                │                               │
│                                ▼                               │
│                         ┌──────────────┐                       │
│                         │    Reward    │                       │
│                         │   (Score)    │                       │
│                         └──────┬───────┘                       │
│                                │                               │
│                                ▼                               │
│                         ┌──────────────┐                       │
│                         │  KL Diver-   │                       │
│                         │  gence Pen.  │                       │
│                         └──────┬───────┘                       │
│                                │                               │
│                                ▼                               │
│                         ┌──────────────┐                       │
│                         │   Policy     │                       │
│                         │   Update     │                       │
│                         └──────────────┘                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 3.2.3 Evaluator Model Criteria

**Evaluation Dimensions:**

| Dimension | Definition | Weight |
|-----------|------------|--------|
| **Precision** | Does rewritten prompt maintain original intent without hallucinations? | 0.4 |
| **Recall** | Does rewritten prompt cover all sub-tasks and constraints? | 0.4 |
| **Clarity** | Is the rewritten prompt structurally clear? | 0.2 |

**Reward Calculation:**

```
Reward = (Precision × 0.4) + (Recall × 0.4) + (Clarity × 0.2) - KL_Penalty
```

#### 3.2.4 Fallback Mechanism

```
IF evaluation_score < threshold:
    revert_to_original_prompt()
    log_decomposition_failure()
    disable_rl_update()
```

### 3.3 Automated Hybrid Ingestion Engine

#### 3.3.1 Text Segmentation Strategy

**Paragraph-Level Chunking Algorithm:**

```python
def chunk_text(text: str, 
               chunk_size: int = 512,
               overlap: float = 0.15) -> List[Chunk]:
    """
    Recursive character-based chunking preserving semantic boundaries.
    
    Priority order:
    1. Paragraph breaks (\n\n)
    2. Sentence boundaries (. ! ?)
    3. Clause boundaries (, ; :)
    4. Word boundaries
    """
    pass
```

**Overlap Calculation:**

```
overlap_tokens = chunk_size × overlap_percentage
actual_overlap = min(overlap_tokens, len(previous_chunk) × 0.5)
```

#### 3.3.2 Embedding Generation

**Technology: FastEmbed**

| Feature | Specification |
|---------|---------------|
| Runtime | CPU-optimized (no GPU required) |
| Model | BAAI/bge-small-en-v1.5 or similar |
| Dimension | 384-768 (configurable) |
| Batch Size | 256 (parallel encoding) |

```python
from fastembed import TextEmbedding

embedding_model = TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

embeddings = list(embedding_model.embed(chunks))
```

#### 3.3.3 Vector Store: LanceDB

**Why LanceDB:**

| Requirement | LanceDB | ChromaDB | Pinecone |
|-------------|---------|----------|----------|
| Embedded/Local | ✅ | ⚠️ Issues | ❌ |
| Zero Dependencies | ✅ | ❌ | ❌ |
| Multi-modal | ✅ | ⚠️ Limited | ✅ |
| Zero-copy Reads | ✅ | ❌ | ❌ |
| File-based | ✅ | ✅ | ❌ |

**Schema Definition:**

```python
import lancedb
from lancedb.pydantic import LanceModel, Vector

class Document(LanceModel):
    id: str
    text: str
    vector: Vector(384)
    confidence: float
    source: str
    timestamp: datetime
    metadata: dict
```

#### 3.3.4 Graph Store: Kùzu

**Why Kùzu:**

| Feature | Kùzu | FalkorDBLite | CogDB |
|---------|------|--------------|-------|
| Query Language | Cypher | Custom | Custom |
| Embedded | ✅ | ⚠️ Subprocess | ✅ |
| Analytics | ✅ Vectorized | ⚠️ Limited | ❌ |
| Scalability | ✅ High | ⚠️ Medium | ❌ Low |

**Graph Schema:**

```cypher
// Node definitions
CREATE NODE TABLE Entity (
    id STRING,
    name STRING,
    type STRING,
    confidence FLOAT,
    PRIMARY KEY (id)
);

CREATE NODE TABLE Document (
    id STRING,
    content STRING,
    confidence FLOAT,
    PRIMARY KEY (id)
);

// Relationship definitions
CREATE REL TABLE RELATES_TO (
    FROM Entity TO Entity,
    relation_type STRING,
    confidence FLOAT
);

CREATE REL TABLE MENTIONED_IN (
    FROM Entity TO Document,
    confidence FLOAT
);
```

#### 3.3.5 Confidence Metric Calculation

**Mathematical Formulation:**

For extracted sequence T = {t₁, t₂, ..., tₙ}:

```
log_prob_avg = (1/n) × Σ log P(tᵢ | t₁...tᵢ₋₁)

confidence_score = exp(log_prob_avg)
```

**Implementation Approach:**

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def calculate_confidence(model, tokenizer, text: str) -> float:
    """
    Extract log probabilities from model and compute confidence.
    """
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=False)
        logits = outputs.logits
        
    # Get log probabilities
    log_probs = torch.log_softmax(logits, dim=-1)
    
    # Extract token log probs
    token_log_probs = log_probs[0, range(len(inputs["input_ids"][0])), inputs["input_ids"][0]]
    
    # Average and exponentiate
    avg_log_prob = token_log_probs.mean()
    confidence = torch.exp(avg_log_prob).item()
    
    return confidence
```

### 3.4 Agentic Retrieval and Verification Loop

#### 3.4.1 Control Loop Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AGENTIC CONTROL LOOP                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│    ┌──────────────┐                                             │
│    │   START      │                                             │
│    │  (Query In)  │                                             │
│    └──────┬───────┘                                             │
│           │                                                     │
│           ▼                                                     │
│    ┌──────────────┐                                             │
│    │    Query     │                                             │
│    │  Generation  │◀──────────────────┐                        │
│    └──────┬───────┘                   │                        │
│           │                           │                        │
│           ▼                           │                        │
│    ┌──────────────┐                   │                        │
│    │   Hybrid     │                   │                        │
│    │  Retrieval   │                   │                        │
│    │ (Vector+Graph)                   │                        │
│    └──────┬───────┘                   │                        │
│           │                           │                        │
│           ▼                           │                        │
│    ┌──────────────┐                   │                        │
│    │ Verification │                   │                        │
│    │   & Eval     │─────┐             │                        │
│    └──────┬───────┘     │             │                        │
│           │             │             │                        │
│           │        ┌────▼────┐        │                        │
│           │        │ Score < │        │                        │
│           │        │Threshold?│       │                        │
│           │        └────┬────┘        │                        │
│           │             │             │                        │
│           │        Yes  │  No         │                        │
│           │             │             │                        │
│           │             ▼             │                        │
│           │      ┌──────────┐        │                        │
│           │      │ Refine   │        │                        │
│           │      │ Queries  │────────┘                        │
│           │      └──────────┘                                 │
│           │                                                   │
│           ▼                                                   │
│    ┌──────────────┐                                           │
│    │  Reciprocal  │                                           │
│    │  Rank Fusion │                                           │
│    │ (+ Conf.)    │                                           │
│    └──────┬───────┘                                           │
│           │                                                     │
│           ▼                                                     │
│    ┌──────────────┐                                           │
│    │    Token     │                                           │
│    │   Budget     │                                           │
│    │   Check      │                                           │
│    └──────┬───────┘                                           │
│           │                                                     │
│           ▼                                                     │
│    ┌──────────────┐                                           │
│    │  Summarize   │ (if over budget)                          │
│    │    if needed │                                           │
│    └──────┬───────┘                                           │
│           │                                                     │
│           ▼                                                     │
│    ┌──────────────┐                                           │
│    │     END      │                                           │
│    │ (Context Out)│                                           │
│    └──────────────┘                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 3.4.2 Hybrid Query Generation

**Vector Query:**

```python
def generate_vector_query(decomposed_prompt: Prompt) -> List[str]:
    """Generate semantic search queries from decomposed prompt."""
    queries = []
    for subtask in decomposed_prompt.subtasks:
        query = f"Find information about: {subtask.key_concepts}"
        queries.append(query)
    return queries
```

**Graph Query (Cypher):**

```python
def generate_graph_query(entities: List[Entity]) -> str:
    """Generate multi-hop graph traversal query."""
    return f"""
    MATCH path = (e1:Entity)-[r:RELATES_TO*1..3]-(e2:Entity)
    WHERE e1.name IN {entity_names}
    AND r.confidence > 0.7
    RETURN path, 
           sum(r.confidence) as path_confidence
    ORDER BY path_confidence DESC
    LIMIT 10
    """
```

#### 3.4.3 Verification Node

**Verification Criteria:**

| Criterion | Threshold | Action if Failed |
|-----------|-----------|------------------|
| Relevance Score | > 0.65 | Refine queries |
| Coverage Score | > 0.70 | Expand search |
| Contradiction | < 0.20 conflict | Remove conflicting sources |
| Recency (if applicable) | Within window | Re-fetch recent data |

```python
class VerificationResult:
    score: float
    relevance: float
    coverage: float
    contradictions: List[Contradiction]
    missing_entities: List[str]
    retry_needed: bool
    failure_reasons: List[str]
```

#### 3.4.4 Reciprocal Rank Fusion with Confidence

**Standard RRF:**

```
RRF_score(D) = Σ (1 / (k + rank(D, L)))
```

**Enhanced RRF with Confidence:**

```
Enhanced_RRF_score(D) = Σ (confidence(D) / (k + rank(D, L)))
```

Where:
- D = document/data point
- L = retrieval list
- k = smoothing constant (typically 60)
- confidence(D) = extraction confidence metric

```python
def reciprocal_rank_fusion(
    results: Dict[str, List[Result]],
    k: int = 60
) -> List[RankedResult]:
    """
    Fuse results from multiple retrievers with confidence weighting.
    """
    fused_scores = {}
    
    for source, ranked_list in results.items():
        for rank, result in enumerate(ranked_list, 1):
            doc_id = result.id
            confidence = result.confidence
            
            score = confidence / (k + rank)
            
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {
                    'score': 0,
                    'result': result
                }
            fused_scores[doc_id]['score'] += score
    
    # Sort by fused score
    sorted_results = sorted(
        fused_scores.items(),
        key=lambda x: x[1]['score'],
        reverse=True
    )
    
    return [
        RankedResult(
            id=doc_id,
            score=data['score'],
            content=data['result'].content,
            confidence=data['result'].confidence
        )
        for doc_id, data in sorted_results
    ]
```

#### 3.4.5 Token Budget Management

```python
class TokenBudgetManager:
    def __init__(
        self,
        max_tokens: int = 4096,
        tokenizer_model: str = "cl100k_base"
    ):
        self.max_tokens = max_tokens
        self.encoding = tiktoken.get_encoding(tokenizer_model)
        self.summarizer = SummarizationModel()
    
    def enforce_budget(
        self,
        context_blocks: List[ContextBlock],
        buffer: int = 512
    ) -> List[ContextBlock]:
        """
        Ensure context fits within token budget.
        
        Strategy:
        1. Calculate cumulative tokens
        2. If over budget, summarize lowest-ranked blocks
        3. Recursively check until within budget
        """
        available = self.max_tokens - buffer
        
        while True:
            total_tokens = sum(
                len(self.encoding.encode(block.content))
                for block in context_blocks
            )
            
            if total_tokens <= available:
                return context_blocks
            
            # Summarize lowest-ranked block
            context_blocks = self._summarize_lowest(context_blocks)
    
    def _summarize_lowest(
        self,
        blocks: List[ContextBlock]
    ) -> List[ContextBlock]:
        """Summarize the lowest-ranked context block."""
        blocks_sorted = sorted(blocks, key=lambda x: x.rank)
        lowest = blocks_sorted[-1]
        
        summary = self.summarizer.summarize(
            lowest.content,
            compression_ratio=0.5
        )
        
        lowest.content = summary
        return blocks
```

---

## 4. Technical Specifications

### 4.1 Technology Stack Summary

| Layer | Technology | Version | Purpose |
|-------|------------|---------|---------|
| **Web Framework** | FastAPI | 0.109+ | Async HTTP server |
| **API Router** | LiteLLM | 1.0+ | Multi-provider abstraction |
| **RL Framework** | TRL (Transformer Reinforcement Learning) | 0.7+ | PPO training |
| **Validation** | Pydantic | 2.0+ | Schema enforcement |
| **Extraction** | Instructor | 1.0+ | Structured outputs |
| **Embeddings** | FastEmbed | 0.2+ | CPU-optimized vectors |
| **Vector DB** | LanceDB | 0.4+ | Embedded columnar store |
| **Graph DB** | Kùzu | 0.4+ | Embedded graph engine |
| **Tokenizer** | tiktoken | 0.5+ | Token counting |
| **Orchestration** | LangGraph | 0.1+ | State machine control |

### 4.2 Model Specifications

| Model Role | Recommended Model | Parameters | Quantization |
|------------|------------------|------------|--------------|
| Task Decomposition | Qwen3-8B | 8B | 4-bit (optional) |
| Alternative SLM | Llama-3.2-3B-Instruct | 3B | 4-bit (optional) |
| Evaluator | GPT-4o-mini / Claude Haiku | - | N/A |
| Summarization | Phi-3-mini | 3.8B | 4-bit |
| Extraction | Mistral-7B-Instruct | 7B | 4-bit |

### 4.3 Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Request Latency (P50) | < 500ms | Before upstream |
| Request Latency (P99) | < 2000ms | Before upstream |
| Internal Processing | < 300ms | Decomposition + Retrieval |
| Memory Footprint | < 16GB | Runtime |
| Concurrent Requests | 100+ | Per instance |
| Retrieval Accuracy | > 90% | Verification pass rate |

### 4.4 Data Schemas

#### 4.4.1 Request Schema

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class ProxyRequest(BaseModel):
    """Incoming request to the cognitive proxy."""
    request_id: str
    timestamp: datetime
    endpoint: str
    headers: Dict[str, str]
    body: Dict[str, Any]
    client_id: Optional[str] = None
    
class DecomposedPrompt(BaseModel):
    """Output of the decomposition engine."""
    original_prompt: str
    rewritten_prompt: str
    subtasks: List[SubTask]
    confidence: float
    requires_ingestion: bool
    requires_retrieval: bool

class SubTask(BaseModel):
    """Individual decomposed subtask."""
    id: str
    description: str
    priority: int
    entities: List[str]
    constraints: Dict[str, Any]

class ExtractionResult(BaseModel):
    """Output from extraction pipeline."""
    entities: List[Entity]
    relations: List[Relation]
    confidence_map: Dict[str, float]
    
class Entity(BaseModel):
    """Extracted entity."""
    id: str
    name: str
    type: str
    properties: Dict[str, Any]
    confidence: float

class Relation(BaseModel):
    """Extracted relationship."""
    source_id: str
    target_id: str
    relation_type: str
    confidence: float
    evidence: str

class RetrievalContext(BaseModel):
    """Final context from agentic loop."""
    vector_results: List[VectorResult]
    graph_results: List[GraphPath]
    fused_rank: List[RankedResult]
    token_count: int
    verification_score: float
    iterations: int

class VectorResult(BaseModel):
    """Result from vector search."""
    id: str
    content: str
    embedding: List[float]
    confidence: float
    metadata: Dict[str, Any]

class GraphPath(BaseModel):
    """Result from graph traversal."""
    path: List[str]
    relations: List[str]
    confidence: float
    depth: int

class RankedResult(BaseModel):
    """Fused and ranked result."""
    id: str
    content: str
    score: float
    confidence: float
    sources: List[str]
```

#### 4.4.2 Telemetry Schema

```python
class TelemetryLog(BaseModel):
    """Telemetry entry for request processing."""
    request_id: str
    timestamp: datetime
    client_id: Optional[str]
    
    # Timing
    total_latency_ms: float
    decomposition_ms: float
    ingestion_ms: float
    retrieval_ms: float
    upstream_ms: float
    
    # Tokens
    prompt_tokens: int
    completion_tokens: int
    internal_tokens: int
    total_tokens: int
    
    # Costs
    internal_cost_usd: float
    upstream_cost_usd: float
    total_cost_usd: float
    
    # Quality
    decomposition_score: float
    verification_score: float
    retrieval_iterations: int
    
    # Models used
    decomposition_model: str
    upstream_model: str
    embedding_model: str
    
    # Status
    success: bool
    error_message: Optional[str]
    fallback_triggered: bool
```

---

## 5. Implementation Roadmap

### 5.1 Phase 1: Core Proxy Infrastructure (Month 1)

#### Week 1-2: Framework & Telemetry Setup

**Objectives:**
- Implement foundational FastAPI server
- Configure LiteLLM routing
- Establish telemetry middleware

**Deliverables:**

| Deliverable | Description | Owner |
|-------------|-------------|-------|
| FastAPI Server | Async server with SSE support | Backend |
| LiteLLM Integration | Multi-provider routing | Backend |
| Middleware Layer | Token counting, latency tracking | Backend |
| Configuration System | Environment-based config | DevOps |
| Logging Infrastructure | Structured JSON logging | DevOps |

**Key Tasks:**

```markdown
- [ ] Initialize FastAPI project structure
- [ ] Implement /v1/chat/completions endpoint
- [ ] Implement /v1/completions endpoint
- [ ] Implement /v1/embeddings endpoint
- [ ] Implement /v1/models endpoint
- [ ] Integrate LiteLLM for provider abstraction
- [ ] Create streaming response handlers (SSE)
- [ ] Implement token counting middleware
- [ ] Set up SQLite database for telemetry
- [ ] Create configuration management (Pydantic Settings)
- [ ] Implement health check endpoints
- [ ] Set up development environment (Docker, docker-compose)
```

**Technical Specifications:**

```python
# Project structure
forma/
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── chat.py
│   │   │   ├── completions.py
│   │   │   ├── embeddings.py
│   │   │   └── models.py
│   │   └── dependencies.py
│   ├── middleware/
│   │   ├── telemetry.py
│   │   ├── auth.py
│   │   └── rate_limit.py
│   ├── core/
│   │   ├── config.py
│   │   └── logging.py
│   └── main.py
├── tests/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
└── pyproject.toml
```

#### Week 3: Embedded Datastore Provisioning

**Objectives:**
- Integrate LanceDB for vector storage
- Integrate Kùzu for graph storage
- Implement session-based isolation

**Deliverables:**

| Deliverable | Description | Owner |
|-------------|-------------|-------|
| LanceDB Integration | Embedded vector store | Backend |
| Kùzu Integration | Embedded graph store | Backend |
| Session Manager | Isolated datastore instances | Backend |
| Schema Definitions | Database schemas | Backend |

**Key Tasks:**

```markdown
- [ ] Add LanceDB dependency
- [ ] Implement LanceDB connection manager
- [ ] Define vector schema (Document model)
- [ ] Create CRUD operations for vectors
- [ ] Add Kùzu dependency
- [ ] Implement Kùzu connection manager
- [ ] Define graph schema (Cypher DDL)
- [ ] Create graph CRUD operations
- [ ] Implement session-based isolation
- [ ] Create database migration utilities
- [ ] Write integration tests
- [ ] Document database schemas
```

**Schema Implementation:**

```python
# src/storage/vector.py
import lancedb
from lancedb.pydantic import LanceModel, Vector
from datetime import datetime
from typing import Optional, Dict, Any

class DocumentChunk(LanceModel):
    id: str
    session_id: str
    text: str
    vector: Vector(384)  # FastEmbed dimension
    confidence: float
    source: str
    chunk_index: int
    timestamp: datetime = datetime.now()
    metadata: Dict[str, Any] = {}

class VectorStore:
    def __init__(self, db_path: str, session_id: str):
        self.db = lancedb.connect(db_path)
        self.session_id = session_id
        self.table = self._get_or_create_table()
    
    def _get_or_create_table(self):
        try:
            return self.db.open_table(f"chunks_{self.session_id}")
        except:
            return self.db.create_table(
                f"chunks_{self.session_id}",
                schema=DocumentChunk
            )
    
    async def add_chunks(self, chunks: List[DocumentChunk]):
        self.table.add(chunks)
    
    async def search(self, query_vector: List[float], k: int = 10):
        return (
            self.table.search(query_vector)
            .where(f"session_id = '{self.session_id}'")
            .limit(k)
            .to_pydantic(DocumentChunk)
        )
```

```python
# src/storage/graph.py
import kuzu

class GraphStore:
    def __init__(self, db_path: str, session_id: str):
        self.db = kuzu.Database(db_path)
        self.conn = kuzu.Connection(self.db)
        self.session_id = session_id
        self._initialize_schema()
    
    def _initialize_schema(self):
        self.conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS Entity (
                id STRING,
                name STRING,
                type STRING,
                confidence FLOAT,
                session_id STRING,
                PRIMARY KEY (id)
            )
        """)
        
        self.conn.execute("""
            CREATE REL TABLE IF NOT EXISTS RELATES_TO (
                FROM Entity TO Entity,
                relation_type STRING,
                confidence FLOAT,
                evidence STRING
            )
        """)
    
    async def add_entity(self, entity: Entity):
        self.conn.execute(
            "CREATE (e:Entity $props)",
            {"props": entity.dict()}
        )
    
    async def add_relation(self, relation: Relation):
        self.conn.execute(
            """
            MATCH (e1:Entity {id: $source}), (e2:Entity {id: $target})
            CREATE (e1)-[r:RELATES_TO $props]->(e2)
            """,
            {
                "source": relation.source_id,
                "target": relation.target_id,
                "props": {
                    "relation_type": relation.relation_type,
                    "confidence": relation.confidence,
                    "evidence": relation.evidence
                }
            }
        )
```

#### Week 4: Text Segmentation & Embedding Pipeline

**Objectives:**
- Implement recursive character chunking
- Integrate FastEmbed for embeddings
- Create ingestion pipeline

**Deliverables:**

| Deliverable | Description | Owner |
|-------------|-------------|-------|
| Text Chunker | Paragraph-level splitting | Backend |
| Embedding Generator | FastEmbed integration | Backend |
| Ingestion Pipeline | End-to-end ingestion | Backend |
| Pipeline Tests | Unit & integration tests | QA |

**Key Tasks:**

```markdown
- [ ] Implement RecursiveCharacterTextSplitter
- [ ] Add paragraph boundary detection
- [ ] Implement overlap calculation
- [ ] Integrate FastEmbed library
- [ ] Create batch embedding generator
- [ ] Implement ingestion orchestrator
- [ ] Add progress tracking
- [ ] Create error handling
- [ ] Write unit tests for chunker
- [ ] Write integration tests for pipeline
- [ ] Benchmark performance
```

**Implementation:**

```python
# src/ingestion/chunker.py
from typing import List, Optional
from dataclasses import dataclass
import re

@dataclass
class Chunk:
    id: str
    text: str
    start_index: int
    end_index: int
    overlap_with_previous: int

class RecursiveCharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: float = 0.15,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or [
            "\n\n",  # Paragraph
            "\n",    # Line
            ". ",    # Sentence
            "! ",
            "? ",
            ", ",    # Clause
            " ",     # Word
            ""       # Character
        ]
    
    def split_text(self, text: str) -> List[Chunk]:
        chunks = []
        self._split_recursive(
            text, 
            self.separators, 
            chunks
        )
        return self._add_overlap(chunks)
    
    def _split_recursive(
        self,
        text: str,
        separators: List[str],
        chunks: List[Chunk],
        start_index: int = 0
    ):
        if len(text) <= self.chunk_size:
            chunks.append(Chunk(
                id=str(len(chunks)),
                text=text,
                start_index=start_index,
                end_index=start_index + len(text),
                overlap_with_previous=0
            ))
            return
        
        # Find best separator
        separator = separators[-1]
        for sep in separators:
            if sep and sep in text:
                separator = sep
                break
        
        # Split by separator
        splits = text.split(separator)
        
        current_chunk = ""
        current_start = start_index
        
        for i, split in enumerate(splits):
            candidate = current_chunk + split + separator if i < len(splits) - 1 else current_chunk + split
            
            if len(candidate) > self.chunk_size:
                if current_chunk:
                    chunks.append(Chunk(
                        id=str(len(chunks)),
                        text=current_chunk.strip(),
                        start_index=current_start,
                        end_index=current_start + len(current_chunk),
                        overlap_with_previous=0
                    ))
                current_chunk = split + separator if i < len(splits) - 1 else split
                current_start = start_index + text.find(split)
            else:
                current_chunk = candidate
        
        if current_chunk.strip():
            chunks.append(Chunk(
                id=str(len(chunks)),
                text=current_chunk.strip(),
                start_index=current_start,
                end_index=current_start + len(current_chunk),
                overlap_with_previous=0
            ))
    
    def _add_overlap(self, chunks: List[Chunk]) -> List[Chunk]:
        if len(chunks) <= 1:
            return chunks
        
        overlap_size = int(self.chunk_size * self.chunk_overlap)
        
        for i in range(1, len(chunks)):
            prev_text = chunks[i-1].text
            overlap_len = min(overlap_size, len(prev_text))
            chunks[i].overlap_with_previous = overlap_len
        
        return chunks
```

```python
# src/ingestion/embeddings.py
from fastembed import TextEmbedding
from typing import List
import numpy as np

class EmbeddingGenerator:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 256
    ):
        self.model = TextEmbedding(model_name=model_name)
        self.batch_size = batch_size
    
    async def generate(self, texts: List[str]) -> List[np.ndarray]:
        embeddings = list(self.model.embed(texts, batch_size=self.batch_size))
        return embeddings
    
    def generate_sync(self, texts: List[str]) -> List[np.ndarray]:
        return list(self.model.embed(texts, batch_size=self.batch_size))
```

---

### 5.2 Phase 2: Extraction & Agentic Loop (Months 2-3)

#### Month 2, Week 1-2: Schema-Enforced Extraction Engine

**Objectives:**
- Implement structured output enforcement
- Create confidence metric calculation
- Build extraction pipeline

**Deliverables:**

| Deliverable | Description | Owner |
|-------------|-------------|-------|
| Instructor Integration | Structured output enforcement | Backend |
| Confidence Calculator | Log-prob extraction | Backend |
| Extraction Pipeline | Entity/relation extraction | Backend |
| Schema Library | Predefined schemas | Backend |

**Key Tasks:**

```markdown
- [ ] Add Instructor dependency
- [ ] Define extraction schemas (Pydantic)
- [ ] Implement entity extraction model
- [ ] Implement relation extraction model
- [ ] Create log-probability extraction logic
- [ ] Implement confidence calculation
- [ ] Build extraction orchestrator
- [ ] Add batch processing
- [ ] Create fallback handling
- [ ] Write unit tests
- [ ] Write integration tests
- [ ] Benchmark extraction accuracy
```

**Implementation:**

```python
# src/extraction/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class EntityType(str, Enum):
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    CONCEPT = "concept"
    EVENT = "event"

class RelationType(str, Enum):
    WORKS_FOR = "works_for"
    LOCATED_IN = "located_in"
    RELATED_TO = "related_to"
    PART_OF = "part_of"
    CAUSED_BY = "caused_by"

class ExtractedEntity(BaseModel):
    name: str = Field(..., description="Entity name")
    type: EntityType = Field(..., description="Entity type")
    description: Optional[str] = Field(None, description="Entity description")
    
class ExtractedRelation(BaseModel):
    source: str = Field(..., description="Source entity name")
    target: str = Field(..., description="Target entity name")
    relation: RelationType = Field(..., description="Relation type")
    context: str = Field(..., description="Context sentence")

class ExtractionResult(BaseModel):
    entities: List[ExtractedEntity] = Field(default_factory=list)
    relations: List[ExtractedRelation] = Field(default_factory=list)

class ExtractionWithConfidence(BaseModel):
    result: ExtractionResult
    entity_confidences: List[float] = Field(default_factory=list)
    relation_confidences: List[float] = Field(default_factory=list)
    overall_confidence: float
```

```python
# src/extraction/extractor.py
import instructor
from openai import OpenAI
from typing import List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .schemas import ExtractionResult, ExtractionWithConfidence

class StructuredExtractor:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        self.client = instructor.from_openai(
            OpenAI(
                base_url="http://localhost:8000/v1",  # Local model
                api_key="dummy"
            )
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    async def extract(
        self,
        text: str,
        schema: type = ExtractionResult
    ) -> ExtractionWithConfidence:
        # Get structured extraction
        result = self.client.chat.completions.create(
            model="local",
            response_model=schema,
            messages=[
                {
                    "role": "system",
                    "content": "Extract entities and relations from the text."
                },
                {
                    "role": "user",
                    "content": text
                }
            ]
        )
        
        # Calculate confidence from log probs
        confidences = self._calculate_confidence(text, result)
        
        return ExtractionWithConfidence(
            result=result,
            entity_confidences=confidences[0],
            relation_confidences=confidences[1],
            overall_confidence=sum(confidences[0] + confidences[1]) / 
                               (len(confidences[0]) + len(confidences[1]))
        )
    
    def _calculate_confidence(
        self,
        text: str,
        result: ExtractionResult
    ) -> Tuple[List[float], List[float]]:
        entity_confidences = []
        relation_confidences = []
        
        for entity in result.entities:
            conf = self._get_token_confidence(entity.name)
            entity_confidences.append(conf)
        
        for relation in result.relations:
            conf = self._get_token_confidence(
                f"{relation.source} {relation.relation} {relation.target}"
            )
            relation_confidences.append(conf)
        
        return entity_confidences, relation_confidences
    
    def _get_token_confidence(self, text: str) -> float:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=False)
            logits = outputs.logits
            log_probs = torch.log_softmax(logits, dim=-1)
            
            token_log_probs = log_probs[0, range(len(inputs["input_ids"][0])), inputs["input_ids"][0]]
            avg_log_prob = token_log_probs.mean()
            confidence = torch.exp(avg_log_prob).item()
        
        return confidence
```

#### Month 2, Week 3-4: State Machine Orchestration

**Objectives:**
- Implement LangGraph control flow
- Create node definitions
- Build verification system

**Deliverables:**

| Deliverable | Description | Owner |
|-------------|-------------|-------|
| LangGraph Setup | State machine framework | Backend |
| Query Generator Node | Query creation logic | Backend |
| Retriever Node | Hybrid retrieval | Backend |
| Verifier Node | Context verification | Backend |
| Control Flow | Conditional edges | Backend |

**Key Tasks:**

```markdown
- [ ] Add LangGraph dependency
- [ ] Define state schema
- [ ] Implement QueryGenerator node
- [ ] Implement Retriever node
- [ ] Implement Verifier node
- [ ] Implement Reranker node
- [ ] Create conditional edges
- [ ] Build retry logic
- [ ] Add max iteration limit
- [ ] Implement fallback to original prompt
- [ ] Write unit tests
- [ ] Write integration tests
```

**Implementation:**

```python
# src/retrieval/state.py
from typing import TypedDict, List, Annotated
import operator

class RetrievalState(TypedDict):
    # Input
    original_query: str
    decomposed_query: str
    
    # Iteration tracking
    iteration: int
    max_iterations: int
    
    # Query state
    vector_queries: List[str]
    graph_queries: List[str]
    
    # Results
    vector_results: List[dict]
    graph_results: List[dict]
    fused_results: List[dict]
    
    # Verification
    verification_score: float
    verification_passed: bool
    failure_reasons: List[str]
    
    # Control
    needs_retry: Annotated[bool, operator.or_]
    context_finalized: bool
    
    # Final output
    final_context: str
    token_count: int
```

```python
# src/retrieval/graph.py
from langgraph.graph import StateGraph, END
from .state import RetrievalState

class RetrievalGraph:
    def __init__(
        self,
        vector_store,
        graph_store,
        verifier,
        max_iterations: int = 5
    ):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.verifier = verifier
        self.max_iterations = max_iterations
        
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(RetrievalState)
        
        # Add nodes
        workflow.add_node("query_generator", self._query_generator)
        workflow.add_node("retriever", self._retriever)
        workflow.add_node("verifier", self._verifier)
        workflow.add_node("reranker", self._reranker)
        workflow.add_node("budget_enforcer", self._budget_enforcer)
        
        # Set entry point
        workflow.set_entry_point("query_generator")
        
        # Add edges
        workflow.add_edge("query_generator", "retriever")
        workflow.add_edge("retriever", "verifier")
        
        # Conditional edges from verifier
        workflow.add_conditional_edges(
            "verifier",
            self._should_retry,
            {
                "retry": "query_generator",
                "continue": "reranker"
            }
        )
        
        workflow.add_edge("reranker", "budget_enforcer")
        workflow.add_edge("budget_enforcer", END)
        
        return workflow.compile()
    
    async def _query_generator(self, state: RetrievalState) -> dict:
        """Generate vector and graph queries from decomposed prompt."""
        # Implementation
        pass
    
    async def _retriever(self, state: RetrievalState) -> dict:
        """Execute hybrid retrieval."""
        # Implementation
        pass
    
    async def _verifier(self, state: RetrievalState) -> dict:
        """Verify retrieval results."""
        # Implementation
        pass
    
    async def _reranker(self, state: RetrievalState) -> dict:
        """Fuse and rerank results."""
        # Implementation
        pass
    
    async def _budget_enforcer(self, state: RetrievalState) -> dict:
        """Enforce token budget."""
        # Implementation
        pass
    
    def _should_retry(self, state: RetrievalState) -> str:
        """Determine if retry is needed."""
        if state["iteration"] >= self.max_iterations:
            return "continue"
        if state["verification_passed"]:
            return "continue"
        return "retry"
```

#### Month 3, Week 1-2: Hybrid Ranking & Budget Constraints

**Objectives:**
- Implement RRF algorithm
- Add confidence weighting
- Build token budget system

**Deliverables:**

| Deliverable | Description | Owner |
|-------------|-------------|-------|
| RRF Implementation | Reciprocal Rank Fusion | Backend |
| Confidence Weighting | Enhanced RRF | Backend |
| Token Counter | tiktoken integration | Backend |
| Summarizer | Context compression | Backend |
| Budget Manager | Token limit enforcement | Backend |

**Key Tasks:**

```markdown
- [ ] Implement standard RRF
- [ ] Add confidence weighting
- [ ] Integrate tiktoken
- [ ] Create token counting utility
- [ ] Implement context block ranking
- [ ] Create summarization pipeline
- [ ] Build budget enforcement logic
- [ ] Add recursive summarization
- [ ] Write unit tests
- [ ] Write integration tests
- [ ] Benchmark token accuracy
```

**Implementation:**

```python
# src/retrieval/reranking.py
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class RetrievalResult:
    id: str
    content: str
    rank: int
    confidence: float
    source: str

class ReciprocalRankFusion:
    def __init__(self, k: int = 60):
        self.k = k
    
    def fuse(
        self,
        results_by_source: Dict[str, List[RetrievalResult]]
    ) -> List[RetrievalResult]:
        """
        Fuse results from multiple retrievers using RRF with confidence.
        
        Formula: RRF(D) = Σ confidence(D) / (k + rank(D))
        """
        fused_scores: Dict[str, float] = {}
        result_map: Dict[str, RetrievalResult] = {}
        
        for source, results in results_by_source.items():
            for rank, result in enumerate(results, start=1):
                if result.id not in fused_scores:
                    fused_scores[result.id] = 0.0
                    result_map[result.id] = result
                
                # Enhanced RRF with confidence weighting
                score = result.confidence / (self.k + rank)
                fused_scores[result.id] += score
        
        # Sort by fused score
        sorted_ids = sorted(
            fused_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            RetrievalResult(
                id=rid,
                content=result_map[rid].content,
                rank=idx + 1,
                confidence=fused_scores[rid],
                source="fused"
            )
            for idx, (rid, _) in enumerate(sorted_ids)
        ]
```

```python
# src/retrieval/budget.py
import tiktoken
from typing import List
from dataclasses import dataclass

@dataclass
class ContextBlock:
    id: str
    content: str
    rank: int
    score: float
    compressed: bool = False

class TokenBudgetManager:
    def __init__(
        self,
        max_tokens: int = 4096,
        buffer_tokens: int = 512,
        encoding_name: str = "cl100k_base"
    ):
        self.max_tokens = max_tokens
        self.buffer_tokens = buffer_tokens
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.available_tokens = max_tokens - buffer_tokens
    
    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))
    
    def enforce_budget(
        self,
        blocks: List[ContextBlock],
        summarizer
    ) -> List[ContextBlock]:
        """
        Enforce token budget by summarizing lowest-ranked blocks.
        """
        while True:
            total_tokens = sum(
                self.count_tokens(block.content)
                for block in blocks
            )
            
            if total_tokens <= self.available_tokens:
                return blocks
            
            # Summarize lowest-ranked block
            blocks = self._compress_lowest(blocks, summarizer)
    
    def _compress_lowest(
        self,
        blocks: List[ContextBlock],
        summarizer
    ) -> List[ContextBlock]:
        """Compress the lowest-ranked block."""
        sorted_blocks = sorted(blocks, key=lambda x: x.rank, reverse=True)
        
        # Compress lowest ranked
        lowest = sorted_blocks[-1]
        compressed_content = summarizer.summarize(
            lowest.content,
            target_ratio=0.5
        )
        
        lowest.content = compressed_content
        lowest.compressed = True
        
        return sorted_blocks
```

---

### 5.3 Phase 3: RL & Evaluation Frameworks (Months 4-5)

#### Month 4, Week 1-2: Lightweight Model Deployment

**Objectives:**
- Deploy SLM for decomposition
- Create inference pipeline
- Implement caching

**Deliverables:**

| Deliverable | Description | Owner |
|-------------|-------------|-------|
| SLM Deployment | Qwen/Llama deployment | ML Eng |
| Inference Server | vLLM/TGI setup | ML Eng |
| Decomposition API | Decomposition endpoint | Backend |
| Response Cache | Caching layer | Backend |

**Key Tasks:**

```markdown
- [ ] Select SLM (Qwen3-8B or Llama-3.2-3B)
- [ ] Set up model quantization (4-bit)
- [ ] Deploy vLLM or TGI server
- [ ] Create inference client
- [ ] Implement decomposition prompt template
- [ ] Add response caching (Redis/in-memory)
- [ ] Create fallback mechanisms
- [ ] Optimize batch inference
- [ ] Write unit tests
- [ ] Benchmark latency
- [ ] Document deployment
```

**Implementation:**

```python
# src/decomposition/model.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Optional
from dataclasses import dataclass

@dataclass
class DecompositionResult:
    original_prompt: str
    rewritten_prompt: str
    subtasks: List[str]
    confidence: float
    model_used: str

class DecompositionModel:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "auto",
        load_in_4bit: bool = True
    ):
        self.model_name = model_name
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
        else:
            quantization_config = None
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            quantization_config=quantization_config,
            torch_dtype=torch.float16
        )
    
    async def decompose(
        self,
        prompt: str,
        max_new_tokens: int = 1024
    ) -> DecompositionResult:
        messages = [
            {
                "role": "system",
                "content": self._get_system_prompt()
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs.shape[-1]:],
            skip_special_tokens=True
        )
        
        return self._parse_response(prompt, response)
    
    def _get_system_prompt(self) -> str:
        return """You are a task decomposition expert. Your job is to:
1. Analyze complex user prompts
2. Break them into clear, actionable subtasks
3. Rewrite the prompt for maximum clarity
4. Identify any missing information

Output format:
REWRITTEN_PROMPT: [clear, rewritten version]
SUBTASKS:
1. [subtask 1]
2. [subtask 2]
...
CONFIDENCE: [0.0-1.0]"""
    
    def _parse_response(
        self,
        original: str,
        response: str
    ) -> DecompositionResult:
        # Parse the structured response
        # Implementation details...
        pass
```

#### Month 4, Week 3-4: Criteria-Based Evaluation Implementation

**Objectives:**
- Implement evaluator model
- Create evaluation criteria
- Build scoring system

**Deliverables:**

| Deliverable | Description | Owner |
|-------------|-------------|-------|
| Evaluator Model | GPT-4o-mini / Claude Haiku | Backend |
| Evaluation Criteria | Precision, Recall, Clarity | Backend |
| Scoring System | Weighted scoring | Backend |
| Evaluation API | Evaluation endpoint | Backend |

**Key Tasks:**

```markdown
- [ ] Design evaluation prompt template
- [ ] Implement precision evaluation
- [ ] Implement recall evaluation
- [ ] Implement clarity evaluation
- [ ] Create weighted scoring
- [ ] Build threshold detection
- [ ] Add logging for evaluations
- [ ] Write unit tests
- [ ] Benchmark against human evaluations
- [ ] Document evaluation criteria
```

**Implementation:**

```python
# src/evaluation/evaluator.py
from pydantic import BaseModel, Field
from typing import List
import instructor
from openai import OpenAI

class EvaluationScores(BaseModel):
    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    clarity: float = Field(..., ge=0.0, le=1.0)
    overall: float = Field(..., ge=0.0, le=1.0)
    issues: List[str] = Field(default_factory=list)

class DecompositionEvaluator:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = instructor.from_openai(OpenAI())
        self.model = model
    
    async def evaluate(
        self,
        original_prompt: str,
        decomposed_prompt: str,
        subtasks: List[str]
    ) -> EvaluationScores:
        evaluation = self.client.chat.completions.create(
            model=self.model,
            response_model=EvaluationScores,
            messages=[
                {
                    "role": "system",
                    "content": self._get_system_prompt()
                },
                {
                    "role": "user",
                    "content": f"""
                    ORIGINAL PROMPT:
                    {original_prompt}
                    
                    DECOMPOSED PROMPT:
                    {decomposed_prompt}
                    
                    SUBTASKS:
                    {chr(10).join(f'- {task}' for task in subtasks)}
                    
                    Evaluate the decomposition.
                    """
                }
            ]
        )
        
        # Calculate weighted overall score
        evaluation.overall = (
            evaluation.precision * 0.4 +
            evaluation.recall * 0.4 +
            evaluation.clarity * 0.2
        )
        
        return evaluation
    
    def _get_system_prompt(self) -> str:
        return """You are an expert evaluator for prompt decomposition.

Evaluate based on:

1. PRECISION (0.0-1.0):
   - Does the rewritten prompt accurately reflect the original?
   - Are there hallucinated requirements?
   - Is the intent preserved?

2. RECALL (0.0-1.0):
   - Are all original subtasks covered?
   - Are all constraints preserved?
   - Is nothing important lost?

3. CLARITY (0.0-1.0):
   - Is the structure clear?
   - Are subtasks well-defined?
   - Is it actionable?

Output issues as a list of problems found."""
```

#### Month 5, Week 1-2: Policy Optimization & Declarative Tuning

**Objectives:**
- Implement PPO training loop
- Configure reward function
- Integrate DSPy optimization

**Deliverables:**

| Deliverable | Description | Owner |
|-------------|-------------|-------|
| PPO Trainer | RL training loop | ML Eng |
| Reward Function | Evaluation-based reward | ML Eng |
| DSPy Integration | Declarative optimization | Backend |
| Training Pipeline | End-to-end training | ML Eng |

**Key Tasks:**

```markdown
- [ ] Add TRL dependency
- [ ] Configure PPO trainer
- [ ] Implement reward function
- [ ] Create KL divergence penalty
- [ ] Set up training loop
- [ ] Add checkpoint saving
- [ ] Add DSPy dependency
- [ ] Define DSPy signatures
- [ ] Configure DSPy optimizer
- [ ] Create prompt template optimization
- [ ] Write training scripts
- [ ] Document training process
```

**Implementation:**

```python
# src/rl/ppo_trainer.py
from trl import PPOTrainer, PPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List
import torch

class DecompositionPPOTrainer:
    def __init__(
        self,
        model_name: str,
        evaluator: DecompositionEvaluator,
        learning_rate: float = 1e-5,
        kl_penalty: float = 0.1
    ):
        self.evaluator = evaluator
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # PPO configuration
        config = PPOConfig(
            model_name=model_name,
            learning_rate=learning_rate,
            kl_penalty=kl_penalty,
            batch_size=16,
            mini_batch_size=4,
            gradient_accumulation_steps=4
        )
        
        self.trainer = PPOTrainer(
            config=config,
            model=self.model,
            tokenizer=self.tokenizer
        )
    
    async def train_step(
        self,
        original_prompts: List[str]
    ) -> dict:
        # Generate decompositions
        query_tensors = []
        response_tensors = []
        rewards = []
        
        for prompt in original_prompts:
            # Tokenize
            query_tensor = self.tokenizer.encode(
                prompt,
                return_tensors="pt"
            ).to(self.model.device)
            
            # Generate
            response = self.model.generate(
                query_tensor,
                max_new_tokens=512,
                temperature=0.7
            )
            
            # Evaluate
            decomposed = self._parse_decomposition(response)
            evaluation = await self.evaluator.evaluate(
                prompt,
                decomposed.rewritten,
                decomposed.subtasks
            )
            
            # Calculate reward
            reward = self._calculate_reward(evaluation)
            
            query_tensors.append(query_tensor)
            response_tensors.append(response)
            rewards.append(reward)
        
        # PPO training step
        stats = self.trainer.step(
            query_tensors,
            response_tensors,
            rewards
        )
        
        return stats
    
    def _calculate_reward(self, evaluation: EvaluationScores) -> float:
        # Base reward from evaluation
        reward = evaluation.overall
        
        # Penalty for low scores
        if evaluation.precision < 0.5:
            reward -= 0.5
        if evaluation.recall < 0.5:
            reward -= 0.5
        
        return reward
```

```python
# src/rl/dspy_optimizer.py
import dspy
from dspy import Signature, Module, Example
from typing import List

class DecompositionSignature(Signature):
    """Decompose a complex prompt into clear subtasks."""
    original_prompt: str = dspy.InputField(desc="The original user prompt")
    rewritten_prompt: str = dspy.OutputField(desc="Clear, rewritten prompt")
    subtasks: List[str] = dspy.OutputField(desc="List of subtasks")

class DecompositionModule(Module):
    def __init__(self):
        super().__init__()
        self.decompose = dspy.ChainOfThought(DecompositionSignature)
    
    def forward(self, original_prompt: str):
        return self.decompose(original_prompt=original_prompt)

class DSPyOptimizer:
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        num_threads: int = 4
    ):
        # Configure DSPy
        dspy.settings.configure(
            lm=dspy.LM(model_name),
            num_threads=num_threads
        )
        
        self.module = DecompositionModule()
        self.optimizer = dspy.BootstrapFewShot(
            metric=self._evaluation_metric,
            max_bootstrapped_demos=10,
            max_labeled_demos=16
        )
    
    def optimize(
        self,
        train_examples: List[Example]
    ) -> DecompositionModule:
        optimized = self.optimizer.compile(
            self.module,
            trainset=train_examples
        )
        return optimized
    
    def _evaluation_metric(
        self,
        example: Example,
        pred: dspy.Prediction,
        trace=None
    ) -> float:
        # Evaluate decomposition quality
        # Implementation details...
        pass
```

#### Month 5, Week 3-4: Integration & Testing

**Objectives:**
- Integrate all components
- End-to-end testing
- Performance benchmarking

**Deliverables:**

| Deliverable | Description | Owner |
|-------------|-------------|-------|
| Integration Tests | End-to-end tests | QA |
| Performance Benchmarks | Latency/throughput metrics | DevOps |
| Bug Fixes | Issue resolution | All |
| Documentation | API docs, runbooks | All |

**Key Tasks:**

```markdown
- [ ] Write integration test suite
- [ ] Create end-to-end test scenarios
- [ ] Benchmark latency
- [ ] Benchmark throughput
- [ ] Test memory usage
- [ ] Test concurrent requests
- [ ] Fix identified bugs
- [ ] Update documentation
- [ ] Create runbooks
- [ ] Performance profiling
```

---

### 5.4 Phase 4: System Hardening (Month 6)

#### Week 1: Edge Case & Loop Deterrence

**Objectives:**
- Implement circuit breakers
- Add fallback mechanisms
- Handle error states

**Deliverables:**

| Deliverable | Description | Owner |
|-------------|-------------|-------|
| Circuit Breakers | Retry limits | Backend |
| Fallback System | Pass-through mode | Backend |
| Error Handling | Comprehensive error handling | Backend |
| Graceful Degradation | Feature fallbacks | Backend |

**Key Tasks:**

```markdown
- [ ] Implement max iteration limits
- [ ] Add timeout handling
- [ ] Create pass-through fallback
- [ ] Implement model failure handling
- [ ] Add rate limiting
- [ ] Create error categorization
- [ ] Build recovery mechanisms
- [ ] Write unit tests for edge cases
- [ ] Create integration tests
- [ ] Document error codes
```

#### Week 2: Concurrency & Thread Safety

**Objectives:**
- Stress testing
- Deadlock prevention
- Thread safety verification

**Deliverables:**

| Deliverable | Description | Owner |
|-------------|-------------|-------|
| Stress Tests | Load testing scripts | QA |
| Thread Safety Audit | Code review findings | Backend |
| Performance Fixes | Optimizations | Backend |
| Monitoring Setup | Metrics collection | DevOps |

**Key Tasks:**

```markdown
- [ ] Write load testing scripts (Locust/k6)
- [ ] Test 100+ concurrent requests
- [ ] Identify deadlocks
- [ ] Fix race conditions
- [ ] Optimize database access
- [ ] Add connection pooling
- [ ] Implement async locks where needed
- [ ] Test under various loads
- [ ] Document scaling limits
- [ ] Create performance baseline
```

#### Week 3: Telemetry Visualization & Analytics

**Objectives:**
- Build monitoring dashboards
- Create analytics queries
- Set up alerting

**Deliverables:**

| Deliverable | Description | Owner |
|-------------|-------------|-------|
| Dashboard | Grafana/visualization | DevOps |
| Analytics API | Metrics endpoints | Backend |
| Alerting | Alert rules | DevOps |
| Reports | Usage reports | Backend |

**Key Tasks:**

```markdown
- [ ] Set up Grafana dashboards
- [ ] Create Prometheus metrics
- [ ] Build analytics queries
- [ ] Create token consumption reports
- [ ] Build cost tracking reports
- [ ] Create RL reward curve visualization
- [ ] Set up alerting rules
- [ ] Create usage reports
- [ ] Document metrics
- [ ] Create runbooks for alerts
```

#### Week 4: Final Integration & Release

**Objectives:**
- Final testing
- Documentation completion
- Production deployment

**Deliverables:**

| Deliverable | Description | Owner |
|-------------|-------------|-------|
| Final Test Suite | Complete test coverage | QA |
| API Documentation | OpenAPI docs | Backend |
| User Guide | End-user documentation | All |
| Deployment Guide | Production setup | DevOps |
| Release Notes | Version notes | All |

**Key Tasks:**

```markdown
- [ ] Complete test coverage
- [ ] Generate OpenAPI documentation
- [ ] Write user guide
- [ ] Create deployment guide
- [ ] Finalize runbooks
- [ ] Create release notes
- [ ] Production deployment
- [ ] Smoke testing
- [ ] Go-live monitoring
- [ ] Post-deployment validation
```

---

## 6. Appendices

### Appendix A: Project Structure

```
forma/
├── docs/
│   ├── design-document.md
│   ├── api-reference.md
│   ├── deployment-guide.md
│   └── user-guide.md
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── chat.py
│   │   │   ├── completions.py
│   │   │   ├── embeddings.py
│   │   │   └── models.py
│   │   └── dependencies.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── logging.py
│   │   └── exceptions.py
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── telemetry.py
│   │   ├── auth.py
│   │   └── rate_limit.py
│   ├── decomposition/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   └── prompts.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── chunker.py
│   │   ├── embeddings.py
│   │   └── pipeline.py
│   ├── extraction/
│   │   ├── __init__.py
│   │   ├── schemas.py
│   │   ├── extractor.py
│   │   └── confidence.py
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── vector.py
│   │   ├── graph.py
│   │   └── session.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── graph.py
│   │   ├── state.py
│   │   ├── nodes.py
│   │   ├── reranking.py
│   │   └── budget.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py
│   │   └── criteria.py
│   ├── rl/
│   │   ├── __init__.py
│   │   ├── ppo_trainer.py
│   │   └── dspy_optimizer.py
│   └── main.py
├── tests/
│   ├── __init__.py
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── scripts/
│   ├── train.py
│   ├── benchmark.py
│   └── evaluate.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── config/
│   ├── default.yaml
│   ├── development.yaml
│   └── production.yaml
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

### Appendix B: Configuration Schema

```yaml
# config/default.yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4

upstream:
  default_provider: "openai"
  providers:
    openai:
      api_key: "${OPENAI_API_KEY}"
      base_url: "https://api.openai.com/v1"
    anthropic:
      api_key: "${ANTHROPIC_API_KEY}"
      base_url: "https://api.anthropic.com/v1"

decomposition:
  model:
    name: "Qwen/Qwen2.5-7B-Instruct"
    device: "auto"
    quantization: "4bit"
  cache:
    enabled: true
    ttl: 3600

embedding:
  model: "BAAI/bge-small-en-v1.5"
  batch_size: 256

storage:
  vector:
    type: "lancedb"
    path: "./data/lancedb"
  graph:
    type: "kuzu"
    path: "./data/kuzu"

retrieval:
  max_iterations: 5
  verification_threshold: 0.65
  token_budget: 4096
  buffer_tokens: 512

evaluation:
  model: "gpt-4o-mini"
  precision_weight: 0.4
  recall_weight: 0.4
  clarity_weight: 0.2

rl:
  enabled: true
  learning_rate: 1e-5
  kl_penalty: 0.1
  batch_size: 16

telemetry:
  enabled: true
  database: "./data/telemetry.db"
  log_level: "INFO"
```

### Appendix C: API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions (streaming supported) |
| `/v1/completions` | POST | Legacy completions |
| `/v1/embeddings` | POST | Generate embeddings |
| `/v1/models` | GET | List available models |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |
| `/admin/sessions` | GET | List active sessions |
| `/admin/sessions/{id}` | DELETE | Terminate session |
| `/admin/telemetry` | GET | Telemetry dashboard data |

### Appendix D: Dependencies

```toml
# pyproject.toml
[project]
name = "forma"
version = "0.1.0"
description = "Autonomous Cognitive Proxy and Hybrid RAG System"
requires-python = ">=3.10"

dependencies = [
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "litellm>=1.0.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    
    # Models & Inference
    "transformers>=4.36.0",
    "torch>=2.1.0",
    "accelerate>=0.25.0",
    "bitsandbytes>=0.41.0",
    
    # RL & Optimization
    "trl>=0.7.0",
    "dspy-ai>=2.0.0",
    "instructor>=1.0.0",
    
    # Embeddings
    "fastembed>=0.2.0",
    "tiktoken>=0.5.0",
    
    # Storage
    "lancedb>=0.4.0",
    "kuzu>=0.4.0",
    
    # Orchestration
    "langgraph>=0.1.0",
    
    # Utilities
    "httpx>=0.26.0",
    "tenacity>=8.2.0",
    "structlog>=24.1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.1.0",
    "black>=24.1.0",
    "ruff>=0.1.0",
    "mypy>=1.8.0",
]

monitoring = [
    "prometheus-client>=0.19.0",
    "grafana-api>=1.0.0",
]
```

### Appendix E: Glossary

| Term | Definition |
|------|------------|
| **Agentic RAG** | Retrieval-Augmented Generation with autonomous decision-making |
| **Cognitive Proxy** | Middleware that processes requests intelligently before forwarding |
| **Confidence Metric** | Probability score derived from model log-probabilities |
| **Hybrid RAG** | RAG combining vector and graph datastores |
| **PPO** | Proximal Policy Optimization - RL algorithm |
| **RRF** | Reciprocal Rank Fusion - algorithm for combining ranked lists |
| **SLM** | Small Language Model |
| **SSE** | Server-Sent Events - streaming protocol |

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-05-04 | Initial | Initial design document |

---

*End of Document*