# Forma

Autonomous Cognitive Proxy and Hybrid RAG System

## Overview

Forma is an OpenAI-compatible proxy that augments conversations with retrieved context from a hybrid memory system. It automatically extracts entities, facts, and procedural knowledge from conversations, stores them in dual storage (graph + vector), and retrieves relevant context to augment future queries.

### How It Works

For each chat completion request, Forma executes this pipeline:

1. **Extract** - Analyzes the conversation to extract:
   - Entities (people, organizations, concepts, technologies, etc.)
   - Relationships between entities
   - Factual statements
   - Procedural knowledge (recipes/how-to guides)
   - Queries for retrieval

2. **Retrieve** - Uses extracted queries to fetch relevant context from storage:
   - Entity relationships from CogDB (graph database)
   - Facts from ChromaDB (vector search)
   - Recipes from ChromaDB (vector search)

3. **Augment** - Injects retrieved context into the prompt, giving the model access to information from previous conversations that may have been lost due to context window limits

4. **Forward** - Sends the augmented request to the upstream model

5. **Store** - Persists newly extracted data in background (async, fire-and-forget)

### Storage Architecture

Forma uses dual storage for different data types:

| Data Type | Storage | Why |
|-----------|---------|-----|
| Entities & Relationships | CogDB (Graph) | Efficient traversal of entity connections |
| Facts & Recipes | ChromaDB (Vector) | Semantic similarity search |

### Retrieval Scoring

Context items are ranked by a composite score:

- **ChromaDB (facts, recipes)**: `confidence × similarity × time_decay`
- **CogDB (relationships)**: `confidence × time_decay`

Time decay uses exponential decay (30-day half-life by default), prioritizing recent information.

## Quick Start

```bash
# Clone and enter directory
git clone https://github.com/your-org/forma.git
cd forma

# Install dependencies using uv (recommended)
uv sync

# Or using pip with virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Copy environment file and configure
cp .env.example .env
# Edit .env with your upstream API settings

# Run the proxy
uv run forma
# Or directly:
uv run python -m forma.main
```

The server runs at `http://localhost:8000` by default.

## Configuration

Edit `.env` to configure. All settings are optional except the upstream endpoint.

### Server

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
| `DEBUG` | `false` | Enable debug mode with auto-reload |

### Upstream API (Required)

The OpenAI-compatible API that Forma proxies to:

| Variable | Default | Description |
|----------|---------|-------------|
| `UPSTREAM_BASE_URL` | `http://localhost:8080/v1` | Upstream API URL |
| `UPSTREAM_API_KEY` | `""` | API key (leave empty for local servers) |
| `UPSTREAM_TIMEOUT` | `300.0` | Request timeout in seconds |
| `MODEL_MAPPING` | `""` | Map local model names to upstream (format: `local:upstream,local2:upstream2`) |

### Embedding Endpoint (Optional)

Separate endpoint for embeddings (e.g., LM Studio with local embedding model):

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_BASE_URL` | `""` | Embedding endpoint URL (empty = use upstream) |
| `EMBEDDING_API_KEY` | `""` | API key for embedding endpoint |
| `EMBEDDING_MODEL_NAME` | `""` | Default embedding model |
| `EMBEDDING_TIMEOUT` | `60.0` | Embedding request timeout |

### Extraction LLM (Optional)

LLM used internally for extracting entities/facts/recipes:

| Variable | Default | Description |
|----------|---------|-------------|
| `EXTRACTOR_BASE_URL` | `""` | Extraction endpoint URL (empty = use upstream) |
| `EXTRACTOR_API_KEY` | `""` | API key for extraction endpoint |
| `EXTRACTOR_MODEL_NAME` | `""` | Model for extraction tasks |
| `EXTRACTOR_TIMEOUT` | `120.0` | Extraction timeout (may need time for complex extraction) |

### Storage

| Variable | Default | Description |
|----------|---------|-------------|
| `CHROMADB_HOST` | `localhost` | ChromaDB server host (used only if persist_directory is empty) |
| `CHROMADB_PORT` | `8001` | ChromaDB server port (used only if persist_directory is empty) |
| `CHROMADB_PERSIST_DIRECTORY` | `""` | Path for persistent ChromaDB (empty = in-memory or server mode) |
| `COGDB_HOME` | `forma_graph` | CogDB graph name |
| `COGDB_PATH_PREFIX` | `./cog_data` | CogDB storage directory |

### Example: LM Studio + OpenAI

```env
# Use OpenAI for chat completions
UPSTREAM_BASE_URL=https://api.openai.com/v1
UPSTREAM_API_KEY=sk-your-key

# Use LM Studio for embeddings and extraction (no auth needed)
EMBEDDING_BASE_URL=http://localhost:1234/v1
EMBEDDING_MODEL_NAME=text-embedding-all-minilm-l6-v2-embedding

EXTRACTOR_BASE_URL=http://localhost:1234/v1
EXTRACTOR_MODEL_NAME=gemma-4-e4b-it
```

### Example: Fully Local (LM Studio)

```env
UPSTREAM_BASE_URL=http://localhost:1234/v1
UPSTREAM_API_KEY=

EMBEDDING_BASE_URL=http://localhost:1234/v1
EMBEDDING_MODEL_NAME=text-embedding-all-minilm-l6-v2-embedding

EXTRACTOR_BASE_URL=http://localhost:1234/v1
EXTRACTOR_MODEL_NAME=smollm3-3b-128k

# Persistent storage (embedded mode - port ignored)
CHROMADB_PERSIST_DIRECTORY=./chroma_data
COGDB_PATH_PREFIX=./cog_data
```

## Usage

Forma is fully OpenAI-compatible. Point any OpenAI SDK or client to it:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="any-key-works"  # Forma forwards to upstream
)

# Normal chat - Forma extracts and stores automatically
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "My name is Alice and I work at Acme Corp."},
        {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
        {"role": "user", "content": "What's my name?"}  # Forma retrieves context
    ]
)
# Response includes retrieved context about Alice
```

## API Endpoints

### OpenAI-Compatible

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /v1/models` | List available models |
| `POST /v1/chat/completions` | Chat completions (supports streaming) |
| `POST /v1/completions` | Legacy completions |
| `POST /v1/embeddings` | Create embeddings |

### Admin Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /admin/clear` | Clear all stored data (ChromaDB + CogDB) |
| `GET /admin/stats` | Get storage statistics (facts, recipes, entities counts) |

Example:

```bash
# Check storage stats
curl http://localhost:8000/admin/stats

# Clear all stored memory
curl -X POST http://localhost:8000/admin/clear
```

## Extraction

Forma uses a structured extraction prompt to extract:

- **Entities**: Named entities (people, organizations, locations) and conceptual entities (technologies, methodologies, tools, models, datasets)
- **Relationships**: Connections between entities (subject → predicate → object)
- **Facts**: Standalone factual statements (first-person pronouns transformed to "The user")
- **Recipes**: Procedural knowledge (how-to guides, workflows, methods)
- **Queries**: Natural language queries for retrieval

The extraction prompt is located at `src/forma/prompts/extraction.txt` and can be customized.

## Benchmarks

Forma includes a LongMemEval benchmark to measure RAG effectiveness when conversation history exceeds the context window.

```bash
# Run from project root
uv run python benchmarks/longmemeval/run_benchmark.py

# Or with custom question count
MAX_QUESTIONS=50 uv run python benchmarks/longmemeval/run_benchmark.py
```

**Prerequisites:**
- Forma server running at `http://localhost:8000`
- Direct model endpoint available (configured in `.env`)
- Benchmark data file at `benchmarks/longmemeval/data/longmemeval_oracle.json`

The benchmark compares Forma (with RAG retrieval) against the direct upstream model, measuring how much information is retained from overflowed context. Results are saved to `benchmarks/longmemeval/results/`.

## Logs

Forma logs extraction and retrieval operations:

| Log File | Description |
|----------|-------------|
| `logs/extractions.jsonl` | Extraction results (entities, facts, recipes extracted) |
| `logs/retrievals.jsonl` | Retrieval results (context retrieved for augmentation) |

Each entry is JSON with timestamp and full details for debugging.

## Development

```bash
# Install dev dependencies (if not already installed)
uv sync --all-extras

# Run tests
uv run pytest

# Type check
uv run mypy src/forma

# Format
uv run ruff format src/forma

# Lint
uv run ruff check src/forma
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Forma Proxy                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Request ──► Extract ──► Retrieve ──► Augment ──► Forward       │
│                                      │                          │
│                                      ▼                          │
│                               Background Store                  │
│                                                                  │
├───────────────────────┬─────────────────────────────────────────┤
│       CogDB           │              ChromaDB                   │
│   (Graph Storage)     │          (Vector Storage)               │
├───────────────────────┼─────────────────────────────────────────┤
│ - Entities            │ - Facts (semantic search)               │
│ - Relationships       │ - Recipes (semantic search)             │
│ - Graph traversal     │ - Cosine similarity                     │
└───────────────────────┴─────────────────────────────────────────┘
```

## License

MIT