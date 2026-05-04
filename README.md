# Forma

Autonomous Cognitive Proxy and Hybrid RAG System

## Quick Start

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Copy environment file and configure
cp .env.example .env

# Run the proxy
forma
# Or directly:
python -m forma.main
```

## Configuration

Edit `.env` to configure:

### Chat/Completions (Upstream API)
- `UPSTREAM_BASE_URL` - URL of your OpenAI-compatible API (e.g., `https://api.openai.com/v1`)
- `UPSTREAM_API_KEY` - API key (leave empty for local servers)
- `UPSTREAM_TIMEOUT` - Request timeout in seconds (default: 300)
- `MODEL_MAPPING` - Optional model name mapping (format: `local:upstream,local2:upstream2`)

### Embeddings (Separate Endpoint)
You can configure a separate endpoint for embeddings (e.g., LM Studio):

- `EMBEDDING_BASE_URL` - URL of embedding endpoint (e.g., `http://localhost:1234/v1`)
- `EMBEDDING_API_KEY` - API key (leave empty if no auth required)
- `EMBEDDING_MODEL_NAME` - Default embedding model name
- `EMBEDDING_TIMEOUT` - Request timeout in seconds (default: 60)

If `EMBEDDING_BASE_URL` is empty, embeddings will use the upstream endpoint.

### Extraction LLM (Internal Use)
Configure an LLM for extracting entities, relationships, and facts:

- `EXTRACTOR_BASE_URL` - URL of extraction endpoint (e.g., `http://localhost:1234/v1`)
- `EXTRACTOR_API_KEY` - API key (leave empty if no auth required)
- `EXTRACTOR_MODEL_NAME` - Model for extraction tasks
- `EXTRACTOR_TIMEOUT` - Request timeout in seconds (default: 120)

If `EXTRACTOR_BASE_URL` is empty, extraction will use the upstream endpoint.

### Example: LM Studio for Embeddings and Extraction

```env
UPSTREAM_BASE_URL=https://api.openai.com/v1
UPSTREAM_API_KEY=sk-your-key

EMBEDDING_BASE_URL=http://localhost:1234/v1
EMBEDDING_API_KEY=
EMBEDDING_MODEL_NAME=text-embedding-all-minilm-l6-v2-embedding

EXTRACTOR_BASE_URL=http://localhost:1234/v1
EXTRACTOR_API_KEY=
EXTRACTOR_MODEL_NAME=smollm3-3b-128k
```

## Usage

The proxy is fully OpenAI-compatible. Point any OpenAI SDK or client to it:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="any-key-works"  # Proxy forwards to upstream
)

response = client.chat.completions.create(
    model="your-model-name",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Endpoints

- `GET /health` - Health check
- `GET /v1/models` - List models
- `POST /v1/chat/completions` - Chat completions (supports streaming)
- `POST /v1/completions` - Legacy completions
- `POST /v1/embeddings` - Embeddings