# Forma

Autonomous Cognitive Proxy and Hybrid RAG System

## Overview

Forma is an OpenAI-compatible proxy that augments conversations with retrieved context from a hybrid memory system. It automatically extracts entities, facts, and procedural knowledge from conversations, stores them in GrafitoDB (SQLite-backed graph + vector database), and retrieves relevant context to augment future queries.

### How It Works

For each chat completion request, Forma executes this pipeline:

1. **Extract** - Analyzes the conversation to extract:
   - Entities (people, organizations, concepts, technologies, etc.)
   - Relationships between entities
   - Factual statements
   - Procedural knowledge (recipes/how-to guides)
   - Queries for retrieval

2. **Retrieve** - Uses extracted queries to fetch relevant context from GrafitoDB:
   - Entity relationships (graph traversal)
   - Facts (semantic similarity search)
   - Recipes (semantic similarity search)

3. **Augment** - Injects retrieved context into the prompt, giving the model access to information from previous conversations that may have been lost due to context window limits

4. **Forward** - Sends the augmented request to the upstream model

5. **Store** - Persists newly extracted data in background (async, fire-and-forget)

### Multi-Upstream Support

Forma supports multiple upstream API configurations with model-based routing:

- **Local Model Name** - The name clients use in their requests (routing key)
- **Upstream Model Name** - The model name sent to the upstream API

Example: Client sends `{"model": "gemma-local"}` → Forma routes to configured upstream → sends `{"model": "gemma-4-e4b-it"}` to the upstream API.

Upstreams are configured via the Web UI and stored in the Forma database, not in config files.

### Storage Architecture

Forma uses **GrafitoDB** - a SQLite-backed database that combines graph and vector storage in a single file:

| Data Type | Storage | Why |
|-----------|---------|-----|
| Entities & Relationships | Graph (GrafitoDB) | Efficient traversal of entity connections |
| Facts & Recipes | Vector Index (GrafitoDB) | Semantic similarity search |
| Upstreams & Request History | SQLite (Forma DB) | System configuration and tracking |

**Benefits of GrafitoDB:**
- Single SQLite file - minimal file descriptor usage
- No separate ChromaDB server required
- Built-in SentenceTransformer embeddings
- Local model caching for fast startup

### Retrieval Scoring

Context items are ranked by a composite score:

```
score = confidence × similarity × time_decay
```

- **Similarity**: Vector distance (1 - distance for semantic search)
- **Time decay**: Exponential decay (30-day half-life by default), prioritizing recent information

## Web UI

Forma includes a Vue 3 SPA Web UI for visualizing requests, extractions, retrievals, managing upstreams, and interactive chat:

- **Dashboard**: Overview of system activity
- **Requests List**: Browse recent requests with detailed information
- **Extractions View**: See entities, relationships, facts, and recipes extracted from each request
- **Retrievals View**: See context retrieved for augmentation with confidence and scores
- **Upstreams**: Configure upstream API endpoints with model name mapping
- **Chat**: Interactive chat interface with streaming responses and automatic context compaction

Access the Web UI at: `http://localhost:8000`

### Upstreams Management

Configure upstreams via the Web UI at `http://localhost:8000/upstreams`:

| Field | Description |
|-------|-------------|
| **Local Model Name** | The model name clients send (routing key) |
| **Upstream Model Name** | The model name sent to the upstream API (optional, defaults to local name) |
| **Base URL** | The upstream API endpoint (e.g., `http://192.168.68.10:1234/v1`) |
| **API Key** | Authentication key (optional for local servers) |
| **Timeout** | Request timeout in seconds |
| **Enabled** | Whether this upstream is active |

When a request arrives with a model name, Forma looks up the matching upstream and forwards the request. If no upstream is configured for that model, an error is returned.

### Chat Interface

The Chat page (`http://localhost:8000/chat`) provides an interactive chat experience with:

**Features:**
- **Streaming Responses**: Assistant responses stream in real-time, showing content as it's generated
- **Context Compaction**: When token usage reaches 95% of the configured context size, older messages are automatically summarized
  - Summary appears at the end of the chat and remains visible
  - Keeps the most recent messages (last 4 messages = 2 exchanges)
  - Compaction progress shown with step indicators
- **Token Tracking**: Real-time token count display with visual progress bar
- **Auto-Focus**: Input field automatically focuses after responses complete, allowing immediate typing of the next message
- **Context Size Control**: Adjustable context window size (256-128000 tokens)

**Context Compaction Details:**
When context reaches the threshold, Forma:
1. Analyzes messages to summarize (messages older than the last 2 exchanges)
2. Generates a summary using streaming (visible as it's generated)
3. Displays summary with "📝 Context Summary" label at the end of the chat
4. Removes summarized messages, keeping recent context
5. Resets token count to reflect the compacted state

This approach (similar to OpenCode) ensures conversations can continue indefinitely without losing important context from earlier exchanges.

### Request Detail Sections

Each request shows:
- Original Prompt
- Augmented Prompt (with retrieved context)
- Agent Response
- Extractions (collapsible): Entities, Relationships, Facts, Recipes
- Retrievals (collapsible): Facts and Recipes with Confidence and Score columns

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
# Edit .env with your extractor settings (upstreams configured via Web UI)

# Build the Web UI frontend
cd webui && npm install && npm run build && cd ..

# Run the proxy using the server management script
./server.sh start

# Or run directly
uv run forma
# Or:
uv run python -m forma.main
```

The server runs at `http://localhost:8000` by default.

**First steps after startup:**
1. Open `http://localhost:8000` in your browser
2. Go to **Upstreams** page and add an upstream configuration
3. Set the **Local Model Name** to what your clients will use
4. Set the **Upstream Model Name** to what the upstream API expects

**Note**: On first startup, Forma will download the embedding model (`all-MiniLM-L6-v2`, ~90MB) to `./models/`. Subsequent starts load from cache - no network requests.

## Server Management

Use `server.sh` to manage the Forma server:

```bash
./server.sh start    # Start the server
./server.sh stop     # Stop the server
./server.sh status   # Check server status
./server.sh logs     # View recent server logs
./server.sh restart  # Restart the server
```

The script manages:
- PID file tracking (`/.server.pid`)
- Log file (`server.log`)
- Health checks before reporting success
- Graceful shutdown with force-kill fallback

## Configuration

Edit `.env` to configure. Upstreams are configured via the Web UI, not in `.env`.

### Server

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
| `DEBUG` | `false` | Enable debug mode with auto-reload |

### Extraction LLM

LLM used internally for extracting entities/facts/recipes:

| Variable | Default | Description |
|----------|---------|-------------|
| `EXTRACTOR_BASE_URL` | `""` | Extraction endpoint URL (empty = use upstream from database) |
| `EXTRACTOR_API_KEY` | `""` | API key for extraction endpoint |
| `EXTRACTOR_MODEL_NAME` | `""` | Model for extraction tasks (required) |
| `EXTRACTOR_TIMEOUT` | `120.0` | Extraction timeout (may need time for complex extraction) |
| `EXTRACTOR_SEND_REASONING_PARAMS` | `false` | Send reasoning_effort/enable_thinking params (not supported by all APIs) |

If `EXTRACTOR_BASE_URL` is empty, Forma will use the upstream configured for `EXTRACTOR_MODEL_NAME` in the database.

### GrafitoDB Storage

| Variable | Default | Description |
|----------|---------|-------------|
| `GRAFITODB_PATH` | `./grafito_data/forma.db` | SQLite database file path |
| `GRAFITODB_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model for embeddings |
| `GRAFITODB_VECTOR_DIM` | `384` | Embedding dimension (model-specific) |
| `GRAFITODB_MODEL_CACHE_PATH` | `./models` | Local cache for embedding model |

**Note**: Embedding dimension must match the model:
- `all-MiniLM-L6-v2`: 384 dimensions
- `all-mpnet-base-v2`: 768 dimensions

### Request History (Web UI)

| Variable | Default | Description |
|----------|---------|-------------|
| `HISTORY_ENABLED` | `true` | Enable request tracking for Web UI |
| `FORMA_DB_PATH` | `./data/forma.db` | Forma database path (upstreams + request history) |
| `HISTORY_MAX_RECORDS` | `100` | Maximum request records to keep (older pruned) |

### Example: LM Studio

```env
# Extraction endpoint (LM Studio doesn't need auth)
EXTRACTOR_BASE_URL=http://localhost:1234/v1
EXTRACTOR_API_KEY=
EXTRACTOR_MODEL_NAME=gemma-4-e4b-it

# GrafitoDB defaults work well
GRAFITODB_PATH=./grafito_data/forma.db
GRAFITODB_MODEL_CACHE_PATH=./models
```

Then configure upstreams via Web UI:
- Add upstream with **Local Model Name**: `gemma-local`
- Set **Upstream Model Name**: `gemma-4-e4b-it`
- Set **Base URL**: `http://localhost:1234/v1`

## Usage

Forma is fully OpenAI-compatible. Point any OpenAI SDK or client to it:

```python
from openai import OpenAI

# Configure client to use Forma
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="any-key-works"  # Forma doesn't validate, forwards to upstream
)

# Use the model name you configured in the Upstreams page
response = client.chat.completions.create(
    model="gemma-local",  # This is the Local Model Name from your upstream config
    messages=[
        {"role": "user", "content": "My name is Alice and I work at Acme Corp."},
        {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
        {"role": "user", "content": "What's my name?"}  # Forma retrieves context
    ]
)
# Response includes retrieved context about Alice
```

**Important**: Use the **Local Model Name** from your upstream configuration, not the upstream's actual model name.

## API Endpoints

### OpenAI-Compatible

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /v1/models` | List available models (returns models from configured upstream) |
| `POST /v1/chat/completions` | Chat completions (supports streaming) - **used by Web UI Chat interface** |
| `POST /v1/completions` | Legacy completions |

**Note**: The `/v1/embeddings` endpoint is not provided - embeddings are generated internally using SentenceTransformer.

The Web UI Chat interface uses the `/v1/chat/completions` endpoint with streaming enabled to provide real-time response updates.

### Admin Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /admin/clear` | Clear all stored data from GrafitoDB |
| `GET /admin/stats` | Get storage statistics (facts, recipes, entities counts) |

Example:

```bash
# Check storage stats
curl http://localhost:8000/admin/stats

# Clear all stored memory
curl -X POST http://localhost:8000/admin/clear
```

### Web UI API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /ui/stats` | Get summary statistics for dashboard |
| `GET /ui/requests` | Get list of recent requests |
| `GET /ui/requests/{id}` | Get detailed request information |
| `DELETE /ui/clear` | Clear all request history |
| `GET /ui/upstreams` | Get all upstream configurations |
| `POST /ui/upstreams` | Create new upstream |
| `GET /ui/upstreams/{id}` | Get specific upstream |
| `PUT /ui/upstreams/{id}` | Update upstream |
| `DELETE /ui/upstreams/{id}` | Delete upstream |
| `POST /ui/upstreams/reload` | Reload upstreams from database |

**Chat Feature Types:**
The Chat interface uses specific TypeScript types defined in `webui/src/types/index.ts`:

| Type | Description |
|------|-------------|
| `ChatMessage` | Message with role, content, timestamp, and streaming flags |
| `ChatCompletionChunk` | Streaming response chunk (SSE format) |
| `TokenUsage` | Token count information from API responses |

The `ChatMessage` type includes:
- `role`: "user", "assistant", or "system"
- `content`: Message text
- `timestamp`: Unix timestamp
- `isStreaming`: Flag for messages being streamed
- `isCompacting`: Flag for compaction progress messages

## Extraction

Forma uses a structured extraction prompt to extract:

- **Entities**: Named entities (people, organizations, locations) and conceptual entities (technologies, methodologies, tools, models, datasets)
- **Relationships**: Connections between entities (subject → predicate → object)
- **Facts**: Standalone factual statements (first-person pronouns transformed to "The user")
- **Recipes**: Procedural knowledge (how-to guides, workflows, methods)
- **Queries**: Natural language queries for retrieval

The extraction prompt is located at `src/forma/prompts/extraction.txt` and can be customized.

### Assistant Response Extraction

Forma also extracts facts from assistant responses, allowing the system to learn from model-generated information. This happens automatically for non-streaming responses.

## Model Caching

Forma caches the SentenceTransformer embedding model locally:

- **First startup**: Downloads model to `./models/{model_name}/` (~90MB for `all-MiniLM-L6-v2`)
- **Subsequent starts**: Loads from local cache - no HuggingFace network requests

To use a different embedding model:

```env
GRAFITODB_EMBEDDING_MODEL=all-mpnet-base-v2
GRAFITODB_VECTOR_DIM=768
```

The models directory (`./models/`) is tracked in git but model files are ignored via `.gitignore`.

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
- Upstream configured via Web UI for the benchmark model
- Benchmark data file at `benchmarks/longmemeval/data/longmemeval_oracle.json`

The benchmark compares Forma (with RAG retrieval) against the direct upstream model, measuring how much information is retained from overflowed context. Results are saved to `benchmarks/longmemeval/results/`.

## Logs

Forma logs extraction and retrieval operations:

| Log File | Description |
|----------|-------------|
| `logs/extractions.jsonl` | Extraction results (entities, facts, recipes extracted) |
| `logs/retrievals.jsonl` | Retrieval results (context retrieved for augmentation) |
| `server.log` | Server output (managed by `server.sh`) |

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

# Build Web UI
cd webui && npm run build
```

**Web UI Development Notes:**
- The Chat component (`webui/src/components/Chat.vue`) implements streaming using Server-Sent Events (SSE)
- Streaming updates use Vue's reactivity by modifying array elements by index: `messages.value[index].content += chunk`
- Context compaction triggers at 95% of configured context size
- Auto-focus after responses uses `nextTick()` and `inputTextarea.value?.focus()`

## Architecture

### Request Pipeline

```
Request → Extract → Retrieve → Augment → Forward → Response
                                                      ↓
                                               Background Store
                                               (async, fire-and-forget)
```

### Storage Backend

| Component | Type | Purpose |
|-----------|------|---------|
| **GrafitoDB** | SQLite + Graph + Vector | Entity/relationship/fact/recipe storage |
| **Forma DB** | SQLite | Upstreams configuration + request history |

GrafitoDB provides:
- **Graph storage**: Entities and relationships with Cypher query support
- **Vector indexes**: Facts and recipes with semantic similarity search
- **SentenceTransformer embeddings**: Built-in embedding function with local caching

### Data Flow

| Stage | Input | Output |
|-------|-------|--------|
| **Extract** | Messages | Entities, relationships, facts, recipes, queries |
| **Retrieve** | Queries | Context from GrafitoDB (graph + vector) |
| **Augment** | Context + Prompt | Augmented user message |
| **Forward** | Augmented request | Upstream API response |
| **Store** | Extracted data | Persisted to GrafitoDB (async) |
| **Track** | Request data | Recorded to Forma DB |

## Project Structure

```
forma/
├── server.sh                # Server management CLI
├── src/forma/
│   ├── main.py              # FastAPI application entry point
│   ├── config.py            # Configuration management
│   ├── extractor.py         # Entity/fact/recipe extraction
│   ├── storage.py           # GrafitoDB storage backend
│   ├── forma_db.py          # Forma database (upstreams + request history)
│   ├── upstream_manager.py  # Multi-upstream routing
│   ├── api.py               # Web UI API endpoints
│   ├── proxy.py             # OpenAI API proxy with upstream routing
│   └── prompts/
│       └── extraction.txt   # Extraction prompt template
├── webui/                   # Vue 3 SPA frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── Dashboard.vue
│   │   │   ├── RequestsList.vue
│   │   │   ├── RequestDetail.vue
│   │   │   ├── Upstreams.vue
│   │   │   └── Chat.vue      # Interactive chat with streaming & compaction
│   │   ├── api.ts           # API client (includes streaming chat completion)
│   │   ├── types/           # TypeScript types (includes ChatMessage types)
│   │   └── main.ts
│   ├── package.json
│   └ vite.config.ts
├── webui_dist/              # Built frontend (served by FastAPI)
├── data/                    # Forma database
│   └── forma.db             # Upstreams + request history
├── grafito_data/            # GrafitoDB database
│   └── forma.db             # Entities, relationships, facts, recipes
├── models/                  # Cached embedding models
├── logs/                    # Runtime logs
│   ├── extractions.jsonl
│   ├── retrievals.jsonl
│   └ server.log
└── benchmarks/              # LongMemEval benchmark
```

## License

MIT