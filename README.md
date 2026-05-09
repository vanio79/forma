# Forma

Autonomous Cognitive Proxy and Hybrid RAG System

## Overview

Forma is an OpenAI-compatible proxy that augments conversations with retrieved context from a hybrid memory system. It automatically extracts relationships, facts, and procedural knowledge from conversations, stores them in GrafitoDB (SQLite-backed graph + vector database), and retrieves relevant context to augment future queries.

### How It Works

For each chat completion request, Forma executes this pipeline:

1. **Extract** - Analyzes the conversation to extract:
   - Relationships between entities (entities are auto-created as nodes)
   - Factual statements
   - Procedural knowledge (recipes/how-to guides)
   - Queries for retrieval

2. **Retrieve** - Uses extracted queries to fetch relevant context from GrafitoDB:
   - Entity relationships (graph traversal)
   - Facts (semantic similarity search)
   - Recipes (semantic similarity search)

3. **Augment** - Injects retrieved context into the prompt, giving the model access to information from previous conversations that may have been lost due to context window limits

4. **Execute Tools** - (Optional) If tools are provided in the request, Forma executes them server-side:
   - Model requests tool calls → Forma executes → Appends results → Continues conversation
   - Seamless client experience: receive final response, not intermediate tool calls

5. **Forward** - Sends the augmented request to the upstream model

6. **Store** - Persists newly extracted data in background (async, fire-and-forget)

### Server-Side Tool Calling

Forma supports server-side tool execution, allowing models to use tools without client intervention:

```
Client Request (with tools)
    │
    ▼
Forma executes tools on server with real-time streaming
    │
    ▼
Client receives final response (fully resolved)
```

**Benefits over client-side execution (MCP):**
- Simple client integration - no tool management needed
- Tools work with any OpenAI-compatible client
- **Real-time streaming** - Tool execution events stream to client as they happen, not after completion
- Seamless UX - Users see tool progress immediately, not after waiting for completion

**Current tools:**
- `echo` - Testing tool that echoes input
- `search_web` - Web search using DuckDuckGo (returns titles, URLs, snippets)
- `web_fetch` - Fetch content from URLs (HTML to markdown conversion)
- `get_current_time` - Returns current date/time in UTC or specified timezone
- `query_memory` - Direct query to Forma's GrafitoDB storage (relationships, facts, recipes)

**Real-Time Streaming Architecture:**

Tool execution events are streamed to the client using SSE (Server-Sent Events) with special markers:

```
Tool Start Event → sent immediately when tool call begins
    │ (50ms delay to ensure delivery before execution)
    ▼
Tool Execution (e.g., web search ~1-2 seconds)
    │
    ▼
Tool End Event → sent immediately when execution completes
```

Events use the marker format: `__TOOL_EVENT__{json}__END__` for UI parsing.

**Event Types:**
- `tool_loop_progress` - Iteration count (only when tools are used)
- `tool_calls_received` - List of tools the model wants to call
- `tool_call_start` - Individual tool starting execution
- `tool_call_end` - Individual tool completed (with result preview)
- `tool_loop_complete` - All tool iterations finished (control signal)
- `eval_event` - Meta-agent evaluation notifications (evaluation_start, evaluation_result, retry_attempt)

Enable tools in `.env`:
```env
TOOLS_ENABLED=true
TOOLS_MAX_ITERATIONS=5
TOOLS_TIMEOUT=30.0
```

Tools are automatically injected into requests when enabled - clients don't need to specify them:

```python
response = client.chat.completions.create(
    model="gemma-local",
    messages=[{"role": "user", "content": "Search for Python async tutorials"}],
    stream=True  # Streaming enables real-time tool event display
)
# Forma injects available tools automatically and streams execution events in real-time
```

### Multi-Upstream Support

Forma supports multiple upstream API configurations with model-based routing:

- **Local Model Name** - The name clients use in their requests (routing key)
- **Upstream Model Name** - The model name sent to the upstream API

Example: Client sends `{"model": "gemma-local"}` → Forma routes to configured upstream → sends `{"model": "gemma-4-e4b-it"}` to the upstream API.

Upstreams are configured via the Web UI and stored in the Forma database, not in config files.

### Multi-Agent System

Forma includes a multi-agent architecture where specialized AI agents can discover each other and communicate through mention-based routing:

**Features:**
- **Agent Discovery**: Each request is augmented with information about available agents
- **Mention-Based Routing**: Route messages to specific agents using `@agent_name` syntax
- **Agent-to-Agent Delegation**: Agents can delegate tasks to other agents (max depth = 3)
- **Streaming Support**: Multi-agent responses stream with agent markers
- **Global Shared Memory**: All agents query the same RAG indexes with different retrieval configs
- **Meta-Agent Evaluation**: Automatic quality control with evaluator/summarizer agents

**Default Agents** (configured in `config/agents.json`):
- **@assistant**: General coordinator - delegates to specialists
- **@researcher**: Research specialist - web search, information gathering (tools: search_web, web_fetch)
- **@coder**: Code specialist - code generation, debugging (no tools)
- **@evaluator**: Meta-agent - evaluates subagent task completion (trusted, not evaluated itself)
- **@summarizer**: Meta-agent - compacts subagent context into concise summaries (trusted)

**Example Usage:**

```python
# Direct agent routing
response = client.chat.completions.create(
    model="gemma-local",
    messages=[{"role": "user", "content": "@researcher Find Python async tutorials"}]
)
# Response from @researcher with web search results

# Multi-agent delegation
response = client.chat.completions.create(
    model="gemma-local",
    messages=[{"role": "user", "content": "@researcher find quantum computing papers, @coder explain them"}]
)
# Sequential: researcher → coder
```

**Agent Configuration:**

Each agent has:
- `name`: Agent identifier (used in @mentions)
- `purpose`: Role description shown in discovery context
- `instruction_prompt`: System prompt for this agent
- `upstream`: Model configuration (null = use request's model)
- `tools_enabled`: Whether agent can use tools
- `tool_whitelist`: Specific tools this agent can access
- `max_iterations`: Tool iteration limit
- `rag_config`: RAG retrieval settings (enabled, token_budget, min_confidence, max_distance)

**Important Notes:**
- **NO Broadcast**: The system does NOT support `@all` broadcast - all routing is mention-based
- **Shared Memory**: All agents query the SAME global indexes (facts_index, recipes_index)
- **Agent-Specific RAG**: Each agent has different retrieval thresholds via rag_config
- **Web UI**: Manage agents at `http://localhost:8000/agents`

### Meta-Agent Evaluation System

When an agent delegates a task to a subagent, Forma automatically evaluates the quality of the response using meta-agents:

**Evaluation Flow:**
1. **Delegation**: Calling agent delegates task to subagent (e.g., @assistant → @researcher)
2. **Execution**: Subagent executes task with its tools and configuration
3. **Evaluation**: @evaluator assesses whether task was completed successfully
4. **Retry Loop**: If incomplete, subagent retries with specific guidance (max 50 attempts)
5. **Summarization**: @summarizer compacts subagent context before returning to caller

**Evaluation States:**
- `complete`: Task fully addressed with actionable results
- `incomplete`: Partial progress, needs more depth/tools/verification
- `failed`: Task impossible, wrong approach, should try different strategy

**Automatic Retry with Guidance:**
When evaluation returns `incomplete`, the system automatically retries with:
- Previous response shown to subagent
- Specific instructions from evaluator (e.g., "Use web_fetch tool to get full content")
- Progressive refinement across attempts

**SSE Streaming of Evaluation:**
Evaluation events are streamed to the client in real-time:
```
__EVAL_EVENT__{"type": "evaluation_start", "subagent": "researcher"}__END__
__EVAL_EVENT__{"type": "evaluation_result", "status": "incomplete", "reason": "..."}__END__
__EVAL_EVENT__{"type": "retry_attempt", "attempt": 2, "max_attempts": 10}__END__
```

**Context Compaction:**
When agent-to-agent conversations grow large, automatic compaction triggers at 90% of the context window:
- Threshold: 90% of 38400 tokens (34560 tokens)
- Most recent 4 messages preserved (last 2 exchanges)
- Older messages summarized by @summarizer agent
- For single large messages: iterative summarization reduces context progressively

**Trusted Meta-Agents:**
- @evaluator and @summarizer are marked as trusted in `config/agents.json`
- Trusted agents do not get evaluated themselves (avoid infinite loops)
- They have minimal RAG config (disabled, no context retrieval)

See `docs/multi_agent_design.md` for detailed architecture and implementation documentation.

### Storage Architecture

Forma uses **GrafitoDB** - a SQLite-backed database that combines graph and vector storage in a single file:

| Data Type | Storage | Why |
|-----------|---------|-----|
| Entity Nodes & Relationships | Graph (GrafitoDB) | Efficient traversal of entity connections (nodes created automatically from relationships) |
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

Forma includes a Vue 3 SPA Web UI for visualizing requests, extractions, retrievals, managing upstreams, agents, and interactive chat:

- **Dashboard**: Overview of system activity
- **Requests List**: Browse recent requests with detailed information
- **Extractions View**: See relationships, facts, and recipes extracted from each request
- **Retrievals View**: See context retrieved for augmentation with confidence and scores
- **Upstreams**: Configure upstream API endpoints with model name mapping
- **Agents**: Manage multi-agent configurations (create, edit, delete, view discovery info)
- **Chat**: Interactive chat interface with streaming responses, agent routing, and automatic context compaction

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
- **Agent Routing**: Use `@agent_name` syntax to route messages to specific agents
  - Example: `@researcher Find information about quantum computing`
  - Responses tagged with agent name: `[@researcher] ...`
- **Real-Time Tool Execution**: When tools are used, execution events stream immediately:
  - Tool call starts are shown before execution completes
  - Progress indicators display during tool execution
  - Expandable tool results show full output after completion
- **Context Compaction**: When token usage reaches 95% of the configured context size, older messages are automatically summarized
  - Summary appears at the end of the chat and remains visible
  - Keeps the most recent messages (last 4 messages = 2 exchanges)
  - Compaction progress shown with step indicators
- **Token Tracking**: Real-time token count display with visual progress bar
- **Auto-Focus**: Input field automatically focuses after responses complete, allowing immediate typing of the next message
- **Context Size Control**: Adjustable context window size (256-128000 tokens)

**Agent Routing Display:**
When routing to agents, the UI shows:
- 🤖 **Agent indicator** with agent name and purpose
- **Agent execution** with their specific tools and responses
- **Multi-agent** responses displayed sequentially with separators

**Tool Execution Display:**
When the model uses tools (e.g., searching the web), the UI shows:
- 🔧 **Tool call indicator** with execution count and time
- **Expandable details** showing:
  - Tool name and arguments
  - Execution status (success/failed)
  - Duration in milliseconds
  - Full result preview

This real-time feedback lets users see what's happening during long-running operations like web searches (~1-2 seconds), rather than waiting until completion.

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
- Original Prompt (last user message)
- Augmented Prompt (with retrieved context)
- Agent Response
- Extractions (collapsible): Relationships, Facts, Recipes, and Extraction Full Prompt
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

LLM used internally for extracting relationships/facts/recipes:

| Variable | Default | Description |
|----------|---------|-------------|
| `EXTRACTOR_BASE_URL` | `""` | Extraction endpoint URL (empty = use upstream from database) |
| `EXTRACTOR_API_KEY` | `""` | API key for extraction endpoint |
| `EXTRACTOR_MODEL_NAME` | `""` | Model for extraction tasks (required) |
| `EXTRACTOR_TIMEOUT` | `120.0` | Extraction timeout (may need time for complex extraction) |
| `EXTRACTOR_SEND_REASONING_PARAMS` | `false` | Send reasoning_effort/enable_thinking params (not supported by all APIs) |

If `EXTRACTOR_BASE_URL` is empty, Forma will use the upstream configured for `EXTRACTOR_MODEL_NAME` in the database.

### Tool Execution (Server-Side Tool Calling)

| Variable | Default | Description |
|----------|---------|-------------|
| `TOOLS_ENABLED` | `false` | Enable server-side tool execution |
| `TOOLS_MAX_ITERATIONS` | `5` | Maximum tool call iterations per request |
| `TOOLS_TIMEOUT` | `30.0` | Default timeout for tool execution (seconds) |

When enabled, Forma:
1. Automatically injects available tools into requests
2. Executes tools requested by the model on the server
3. Streams execution events in real-time to the client (SSE)
4. Returns the final response after all tool iterations complete

This differs from MCP where the client executes tools. The real-time streaming ensures users see tool progress immediately, not after waiting for completion.

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
| `GET /admin/stats` | Get storage statistics (nodes, facts, recipes counts) |

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
| `GET /ui/agents` | Get all agent configurations |
| `POST /ui/agents` | Create new agent |
| `GET /ui/agents/{id}` | Get specific agent |
| `PUT /ui/agents/{id}` | Update agent |
| `DELETE /ui/agents/{id}` | Delete agent |
| `POST /ui/agents/reload` | Reload agents from config file |

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

- **Relationships**: Connections between entities (subject → predicate → object). Entity nodes are created automatically when relationships are stored.
- **Facts**: Standalone factual statements (first-person pronouns transformed to "The user")
- **Recipes**: Procedural knowledge (how-to guides, workflows, methods)
- **Queries**: Natural language queries for retrieval

The extraction prompt is located at `src/forma/prompts/extraction.txt` and can be customized.

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
| `logs/extractions.jsonl` | Extraction results (relationships, facts, recipes extracted) |
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
Request → Extract → Retrieve → Augment → Tool Loop (if enabled) → Forward → Response
                                                          ↓
                                                  Real-Time Event Stream
                                                          ↓
                                                   Background Store
                                                   (async, fire-and-forget)
```

When tools are enabled, the tool execution loop:
1. Sends request to upstream with available tools
2. Checks response for tool calls (OpenAI format or Gemma/LM Studio markup)
3. Streams `tool_call_start` event immediately
4. Executes tool (e.g., web search ~1-2 seconds)
5. Streams `tool_call_end` event immediately with result
6. Appends tool results to messages
7. Repeats until no tool calls or max iterations
8. Streams `tool_loop_complete` event (control signal)
9. Proceeds to final response streaming

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
| **Extract** | Messages | Relationships, facts, recipes, queries (entity nodes created automatically) |
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
│   ├── extractor.py         # Relationship/fact/recipe extraction
│   ├── storage.py           # GrafitoDB storage backend (global RAG methods)
│   ├── forma_db.py          # Forma database (upstreams, agents, request history)
│   ├── upstream_manager.py  # Multi-upstream routing
│   ├── api.py               # Web UI API endpoints (upstreams + agents)
│   ├── proxy.py             # OpenAI API proxy with upstream routing
│   ├── agents/               # Multi-agent system
│   │   ├── __init__.py      # Agent exports
│   │   ├── registry.py      # AgentRegistry (CRUD operations)
│   │   ├── discovery.py     # Agent discovery context formatting
│   │   ├── parser.py        # Parse agent mentions (RoutingType: MENTION, EXPLICIT)
│   │   ├── router.py        # AgentRouter (mention-based routing only)
│   │   ├── orchestrator.py  # Multi-agent orchestration (sequential)
│   │   ├── meta_evaluation.py # Meta-agent evaluation/summarization helpers
│   │   └── config_loader.py # Load agents from config/agents.json
│   ├── tools/               # Server-side tool execution
│   │   ├── __init__.py      # Tool exports
│   │   ├── base.py          # Tool base classes (Tool, ToolCall, ToolResult)
│   │   ├── executor.py      # Tool execution loop with real-time events
│   │   ├── registry.py      # Tool registry and registration
│   │   └── builtin/         # Built-in tool implementations
│   │       ├── __init__.py
│   │       ├── echo.py      # Echo test tool
│   │       ├── web_search.py # DuckDuckGo web search
│   │       ├── web_fetch.py  # URL content fetching
│   │       ├── time.py      # Current time retrieval
│   │       └── memory.py    # GrafitoDB query tool
│   └── prompts/
│       └── extraction.txt   # Extraction prompt template
├── config/
│   └── agents.json          # Agent configurations (assistant, researcher, coder, evaluator, summarizer)
├── webui/                   # Vue 3 SPA frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── Dashboard.vue
│   │   │   ├── RequestsList.vue
│   │   │   ├── RequestDetail.vue
│   │   │   ├── Upstreams.vue
│   │   │   ├── Agents.vue     # Agent management UI
│   │   │   └── Chat.vue       # Interactive chat with agent routing & streaming
│   │   ├── api.ts           # API client (includes streaming with agent markers)
│   │   ├── types/           # TypeScript types (Agent, ChatMessage, ToolEvent)
│   │   └── main.ts
│   ├── package.json
│   └ vite.config.ts
├── webui_dist/              # Built frontend (served by FastAPI)
├── data/                    # Forma database
│   └── forma.db             # Upstreams, agents, request history
├── grafito_data/            # GrafitoDB database
│   └── forma.db             # Global indexes: facts_index, recipes_index
├── models/                  # Cached embedding models
├── logs/                    # Runtime logs
│   ├── extractions.jsonl
│   ├── retrievals.jsonl
│   └ server.log
├── docs/                    # Documentation
│   ├── multi_agent_design.md # Multi-agent system architecture
│   └── tool_calling_design.md # Tool execution design
└── benchmarks/              # LongMemEval benchmark
```

## License

MIT