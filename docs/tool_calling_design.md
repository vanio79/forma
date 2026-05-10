# Server-Side Tool Calling Design

## Overview

Forma implements server-side tool execution, where the model can request tool calls and Forma executes them automatically before returning the final response to the client. This differs from MCP (client-side execution) and provides a seamless experience where clients receive fully-resolved responses with **real-time streaming** of tool execution events.

## Architecture Comparison

| Aspect | MCP (Client-Side) | Forma (Server-Side) |
|--------|-------------------|---------------------|
| Tool Execution | Client executes | Forma server executes |
| Tool Loop | Client manages loop | Forma manages loop internally |
| Client Experience | Receives tool calls, must execute | Receives final response with real-time progress |
| Streaming | Complex (interleaved tool calls) | Real-time events + final response stream |
| Visibility | Client sees tool call requests | Client sees tool execution as it happens |

## Real-Time Event Streaming

Tool execution events are streamed to the client **as they happen**, not after completion:

```
Client Request (stream: true)
    │
    ▼
┌─────────────────────────────────────┐
│ SSE Stream to Client                │
│                                     │
│  event: tool_loop_progress          │
│  data: {"iteration": 1, ...}        │
│                                     │
│  event: tool_calls_received         │
│  data: {"count": 1, ...}            │
│                                     │
│  event: tool_call_start             │ ← Sent BEFORE execution
│  data: {"name": "search_web", ...}  │
│                                     │
│  [Tool executes (~1-2 seconds)]     │
│                                     │
│  event: tool_call_end               │ ← Sent AFTER execution
│  data: {"result_preview": "..."}    │
│                                     │
│  event: tool_loop_complete          │
│  data: {"iterations": 1, ...}       │
│                                     │
│  event: content                     │
│  data: {"delta": "The search..."}   │
│                                     │
└─────────────────────────────────────┘
```

### Event Marker Format

Tool events are sent as **semantic blocks** in the raw SSE stream:

```
[TOOL_START: search_web]
id: call_abc123
args: {"query": "Python async tutorials"}
[/TOOL_START]
```

After tool execution completes:

```
[TOOL_END: search_web]
id: call_abc123
status: success
duration_ms: 1250
result: Found 5 results...
[/TOOL_END]
```

This human-readable format works with any OpenAI-compatible client (non-parsing clients simply display the raw text), while Forma's UI extracts the blocks for structured display.

### SSE Flush Mechanism

To ensure events arrive immediately (not buffered):

1. **SSE flush comments**: After each event chunk, send `: flush\n\n` (ignored by clients but forces HTTP buffering to flush)
2. **Event loop yield**: `await asyncio.sleep(0)` after yielding to ensure immediate I/O processing
3. **50ms delay after start**: Small delay after `tool_call_start` to ensure delivery before tool execution begins

This prevents the common issue where browser/HTTP buffering causes events to arrive simultaneously despite being sent at different times by the server.

### Control Signal

The `tool_loop_complete` event is always sent (even when no tools are used) because it serves as a **control signal** for the streaming generator to stop waiting:

```python
# Streaming generator waits for this event
while not tool_loop_complete_received:
    event = event_queue.get_nowait()
    if event.event_type == "tool_loop_complete":
        tool_loop_complete_received = True
```

If no tools are used, the event has `{iterations: 0, total_tool_calls: 0}`. The UI checks `total_tool_calls > 0` to determine if the tool execution box should be displayed.

## Execution Flow

```
Client Request (with tools defined)
    │
    ▼
┌─────────────────────────────────────┐
│ Forma Request Handler               │
│                                     │
│  1. Extract from messages           │
│  2. Retrieve context (RAG)          │
│  3. Augment prompt                  │
│                                     │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Tool Execution Loop                 │
│                                     │
│  ┌───────────────────────────────┐  │
│  │ Send to Upstream (with tools) │  │
│  └───────────────────────────────┘  │
│              │                      │
│              ▼                      │
│  ┌───────────────────────────────┐  │
│  │ Check response for tool_calls │  │
│  └───────────────────────────────┘  │
│              │                      │
│         ┌────┴────┐                 │
│         │         │                 │
│    Tool Calls  No Tool Calls        │
│         │         │                 │
│         ▼         ▼                 │
│  ┌───────────┐  ┌───────────────┐   │
│  │ Execute   │  │ Return Final  │───┼──► Client Response
│  │ Tools     │  │ Response      │   │
│  └───────────┘  └───────────────┘   │
│         │                          │
│         ▼                          │
│  ┌───────────────────────────────┐  │
│  │ Append tool results to msgs   │  │
│  └───────────────────────────────┘  │
│         │                          │
│         ▼                          │
│  ┌───────────────────────────────┐  │
│  │ Check max_iterations          │  │
│  └───────────────────────────────┘  │
│         │                          │
│    Continue Loop (if < max)        │
│         │                          │
└─────────┼──────────────────────────┘
          │
          ▼
    Background Store (async)
```

## Tool Definition Format

Tools follow OpenAI's function calling format:

```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {
          "type": "object",
          "properties": {
            "query": {
              "type": "string",
              "description": "The search query"
            },
            "num_results": {
              "type": "integer",
              "description": "Number of results to return",
              "default": 5
            }
          },
          "required": ["query"]
        }
      }
    }
  ]
}
```

## Tool Types

### 1. Built-in Tools (Implemented)

Forma provides built-in tools that execute locally:

| Tool | Description | Use Case |
|------|-------------|----------|
| `search_web` | Web search (DuckDuckGo) | Research, finding information |
| `web_fetch` | Fetch URL content (HTML→Markdown) | Reading web pages, documentation |
| `query_memory` | Query Forma's GrafitoDB directly | Direct access to stored knowledge |
| `get_current_time` | Return current date/time | Time-sensitive queries |
| `echo` | Echo input for testing | Debugging, testing tool execution |

### 2. External Tools (HTTP) - Planned

Tools can be configured as HTTP endpoints:

```json
{
  "name": "get_weather",
  "type": "external",
  "url": "https://api.weather.example.com/current",
  "method": "GET",
  "headers": {
    "Authorization": "Bearer ${WEATHER_API_KEY}"
  },
  "parameter_mapping": {
    "location": "query.city"
  },
  "response_extraction": "$.data.temperature"
}
```

### 3. Python Plugin Tools - Planned

Users can define custom Python tools in a plugins directory.

## Supported Tool Call Formats

Forma supports two tool call formats:

### OpenAI Standard Format

```json
{
  "tool_calls": [{
    "id": "call_abc123",
    "type": "function",
    "function": {
      "name": "search_web",
      "arguments": "{\"query\": \"Python async tutorials\"}"
    }
  }]
}
```

### Gemma/LM Studio Markup Format

Some models (like Gemma 4) use markup instead of structured tool_calls:

```
search_web(query="Python async tutorials", num_results=5)
```

Forma extracts tool calls from this markup and builds the proper tool_calls array internally. This enables tool calling with models that don't support OpenAI's native format.

## Configuration

### Environment Variables

```env
# Tool execution settings
TOOLS_ENABLED=true
TOOLS_MAX_ITERATIONS=5          # Max tool call loops per request
TOOLS_TIMEOUT=30.0              # Timeout per tool execution (seconds)
TOOLS_STREAMING=false           # Stream intermediate tool results to client

# Built-in tool settings
TOOL_WEB_SEARCH_PROVIDER=tavily  # tavily, duckduckgo, or none
TOOL_WEB_SEARCH_API_KEY=         # Required for Tavily
TOOL_WEB_SEARCH_MAX_RESULTS=5

TOOL_FILE_READ_BASE_DIR=/home/user/safe-dir  # Sandbox for file reads
TOOL_CODE_EXEC_TIMEOUT=5.0      # Timeout for code execution

# External tools config file
TOOLS_CONFIG_PATH=./config/tools.json
```

### Tools Configuration File

```json
{
  "tools": [
    {
      "name": "get_weather",
      "type": "external",
      "url": "https://api.openweathermap.org/data/2.5/weather",
      "method": "GET",
      "auth": {
        "type": "api_key",
        "key": "OPENWEATHER_API_KEY",
        "location": "query.appid"
      },
      "parameters": {
        "location": {
          "http_param": "q",
          "required": true
        }
      },
      "response": {
        "format": "json",
        "extraction": {
          "temperature": "$.main.temp",
          "description": "$.weather[0].description"
        }
      },
      "timeout": 10.0
    },
    {
      "name": "query_database",
      "type": "external",
      "url": "http://localhost:8080/api/query",
      "method": "POST",
      "auth": {
        "type": "bearer",
        "key": "DB_API_KEY"
      },
      "parameters": {
        "query": {
          "http_param": "body.query",
          "required": true
        }
      },
      "response": {
        "format": "json",
        "extraction": "$.results"
      }
    }
  ],
  "builtins": {
    "web_search": {
      "enabled": true,
      "provider": "tavily",
      "max_results": 5
    },
    "read_file": {
      "enabled": true,
      "allowed_paths": [
        "/home/user/projects",
        "/home/user/documents"
      ],
      "max_file_size": 1048576  # 1MB
    },
    "execute_code": {
      "enabled": true,
      "allowed_modules": ["math", "json", "datetime", "re"],
      "timeout": 5.0
    },
    "query_memory": {
      "enabled": true,
      "default_limit": 10
    }
  }
}
```

## Tool Execution Loop

### Message Flow

```
Initial Messages:
[
  {"role": "user", "content": "What's the weather in Berlin?"}
]

After First Upstream Call (Model Requests Tool):
[
  {"role": "user", "content": "What's the weather in Berlin?"},
  {"role": "assistant", "tool_calls": [
    {"id": "call_123", "type": "function", "function": {
      "name": "get_weather",
      "arguments": '{"location": "Berlin"}'
    }}
  ]}
]

After Tool Execution:
[
  {"role": "user", "content": "What's the weather in Berlin?"},
  {"role": "assistant", "tool_calls": [
    {"id": "call_123", "type": "function", "function": {
      "name": "get_weather",
      "arguments": '{"location": "Berlin"}'
    }}
  ]},
  {"role": "tool", "tool_call_id": "call_123", "content": '{"temperature": 15, "description": "Cloudy"}'}
]

After Second Upstream Call (Final Response):
[
  {"role": "user", "content": "What's the weather in Berlin?"},
  {"role": "assistant", "tool_calls": [...]},
  {"role": "tool", "tool_call_id": "call_123", "content": '{"temperature": 15, "description": "Cloudy"}'},
  {"role": "assistant", "content": "The current weather in Berlin is 15°C and cloudy."}
]
```

### Loop Implementation

```python
async def execute_with_tools(
    messages: list[dict],
    tools: list[dict],
    max_iterations: int = 5,
) -> dict:
    """
    Execute tool calling loop until final response or max iterations.
    
    Returns final response with accumulated tool call history.
    """
    iteration = 0
    accumulated_messages = messages.copy()
    
    while iteration < max_iterations:
        # Send request with tools
        response = await upstream_client.chat.completions.create(
            model=model,
            messages=accumulated_messages,
            tools=tools,
            tool_choice="auto",
        )
        
        message = response.choices[0].message
        
        # Check for tool calls
        if not message.tool_calls:
            # Final response - no tools needed
            return {
                "response": response,
                "iterations": iteration,
                "tool_calls_executed": count_tool_calls(accumulated_messages),
            }
        
        # Append assistant message with tool calls
        accumulated_messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": message.tool_calls,
        })
        
        # Execute each tool call
        for tool_call in message.tool_calls:
            result = await execute_tool(
                name=tool_call.function.name,
                arguments=json.loads(tool_call.function.arguments),
                tool_call_id=tool_call.id,
            )
            
            # Append tool result
            accumulated_messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result.content,
            })
        
        iteration += 1
    
    # Max iterations reached - return last response
    return {
        "response": response,
        "iterations": iteration,
        "max_iterations_reached": True,
        "tool_calls_executed": count_tool_calls(accumulated_messages),
    }
```

## Security Considerations

### 1. Tool Execution Sandbox

- **Code Execution**: Use restricted Python environment (no file access, network, dangerous builtins)
- **File Read**: Whitelist allowed directories, enforce max file size
- **External HTTP**: Validate URLs, timeout enforcement, rate limiting

### 2. Parameter Validation

- Validate tool parameters against schema before execution
- Reject malformed or potentially dangerous parameters
- Log all tool calls with parameters for audit

### 3. Rate Limiting

```env
TOOLS_RATE_LIMIT_PER_MINUTE=20
TOOLS_RATE_LIMIT_PER_REQUEST=5
```

### 4. Permission Levels

Tools can have permission levels requiring client authorization:

```json
{
  "name": "execute_code",
  "permission": "dangerous",  // Requires client to approve
  "confirmation_message": "The model wants to execute Python code. Allow?"
}
```

Client request includes:

```json
{
  "tools": ["search_web", "get_weather"],
  "dangerous_tools_allowed": false  // Disallow execute_code
}
```

## Streaming with Tools

### Implemented Approach: Real-Time Event Streaming

```
Client Request (stream: true)
    │
    ▼
Forma executes tools with real-time event streaming
    │
    ▼
┌─────────────────────────────────────┐
│ SSE Stream to Client                │
│                                     │
│  [TOOL_START: search_web]           │ ← Raw semantic block
│  id: call_abc123                    │
│  args: {"query": "..."}             │
│  [/TOOL_START]                      │
│  : flush                            │ ← Forces HTTP buffering to flush
│                                     │
│  [Tool executes ~1-2 seconds]       │
│                                     │
│  [TOOL_END: search_web]             │ ← Raw semantic block
│  id: call_abc123                    │
│  status: success                    │
│  duration_ms: 1250                  │
│  result: Found 5 results...         │
│  [/TOOL_END]                        │
│  : flush                            │
│                                     │
│  data: {"choices": [...]}           │ ← JSON SSE content
│  data: {"choices": [...]}           │
│                                     │
└─────────────────────────────────────┘
```

**Key implementation details:**

1. **Semantic Block Format**: Tool/agent/eval/summary events sent as `[TYPE: name]...[/TYPE]` raw text blocks, not embedded in JSON
2. **Two-Pass Frontend Parser**: First pass extracts raw semantic blocks, second pass parses JSON SSE data for content
3. **Async Queue Pattern**: Tool executor puts events into an `asyncio.Queue`, streaming generator reads them with `get_nowait()` + 1ms sleep
4. **SSE Flush Comments**: Send `: flush\n\n` after each event chunk to force HTTP buffering to release
5. **Event Loop Yield**: `await asyncio.sleep(0)` after yielding chunk ensures immediate I/O
6. **Start Event Delay**: 50ms delay after `tool_call_start` ensures delivery before execution begins

**Benefits:**
- Users see tool progress immediately, not after completion
- Works with standard OpenAI clients (events appear as content deltas)
- Forma UI parses events separately for rich display

**Event Types:**

```typescript
interface ToolEvent {
  type: 'tool_call_start' | 'tool_call_end' | 'tool_loop_complete';
  timestamp: number;
  // ...type-specific fields
}
```

- `tool_call_start`: Individual tool starting (sent BEFORE execution)
- `tool_call_end`: Individual tool completed (sent AFTER execution)
- `tool_loop_complete`: All iterations done (control signal, always sent)

**Block format:**
```
[TOOL_START: search_web]
id: call_abc123
args: {"query": "..."}
[/TOOL_START]

[TOOL_END: search_web]
id: call_abc123
status: success
duration_ms: 1250
result: Found 5 results...
[/TOOL_END]
```

### Non-Streaming Requests

For non-streaming requests, tool execution happens silently and the final response includes all tool results in the message history.

## Integration with Existing Pipeline

The tool execution loop integrates into Forma's existing request pipeline:

```python
async def chat_completions(request: Request) -> Response:
    # 1. Extract from messages
    extraction_result = await extractor.extract_from_messages_async(messages)
    
    # 2. Retrieve context
    retrieved_context = await storage.retrieve(extraction_result.queries)
    
    # 3. Augment prompt
    augmented_messages = augment_messages(messages, retrieved_context)
    
    # 4. Execute with tools (if tools provided)
    if request.tools:
        result = await execute_with_tools(
            messages=augmented_messages,
            tools=request.tools,
            max_iterations=settings.tools_max_iterations,
        )
        response = result["response"]
    else:
        # Normal forwarding without tools
        response = await forward_to_upstream(augmented_messages)
    
    # 5. Store extracted data (background)
    asyncio.create_task(store_extraction(extraction_result))
    
    # 6. Return response
    return response
```

## Request/Response Format

### Client Request with Tools

```json
{
  "model": "gemma-local",
  "messages": [
    {"role": "user", "content": "Search for recent AI news and summarize"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "search_web",
        "description": "Search the web",
        "parameters": {
          "type": "object",
          "properties": {
            "query": {"type": "string"}
          },
          "required": ["query"]
        }
      }
    }
  ],
  "tool_choice": "auto",
  
  // Forma-specific options
  "forma_options": {
    "tools_stream_events": true,  // Stream tool events to client
    "tools_max_iterations": 3,     // Override default max iterations
    "dangerous_tools_allowed": ["execute_code"]  // Allow specific dangerous tools
  }
}
```

### Response with Tool History

```json
{
  "id": "chatcmpl-123",
  "model": "gemma-local",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Here's a summary of recent AI news..."
    },
    "finish_reason": "stop"
  }],
  
  // Forma-specific metadata
  "forma_metadata": {
    "tool_iterations": 2,
    "tools_executed": [
      {
        "name": "search_web",
        "arguments": {"query": "AI news 2024"},
        "duration_ms": 1500,
        "result_summary": "Found 5 results"
      }
    ],
    "total_tool_time_ms": 1500
  }
}
```

## Web UI Integration ✅ Implemented

### Tool Call Visualization

The Chat component shows each tool call as a **standalone message box**:

```typescript
interface ChatMessage {
  role: "tool";
  content: "";
  toolName: string;
  toolCallId: string;
  toolStatus: "pending" | "running" | "success" | "failed";
  toolDuration?: number;
  toolArgs?: Record<string, unknown>;
  toolResult?: string;
  agentChain?: string[];  // Inherited from parent agent
}
```

**Display logic:**
- Each `TOOL_START` event creates a new tool message (appended to message array)
- `TOOL_END` updates the existing message with status, duration, and result
- Tool box shows: icon (●/✓/✗), name, duration, args, result preview
- Agent call chain shown above the tool box: `🤖 @assistant → @researcher`
- Failed tools show red styling; successful tools show green

**Message flow in UI:**
```
[User message]
  ↓
[🤖 @assistant → @researcher]  (agent header)
[🔧 search_web  850ms]         (tool box)
  ↓
[🤖 @assistant → @researcher]  (agent header)
[Assistant response content]
```

### SSE Parsing

The API client uses a **two-pass parser** for semantic blocks:

```typescript
// webui/src/api.ts

// First pass: extract raw semantic blocks from buffer
const blockRegex = /\[(\w+)(?:\s*:\s*([^\]]+))?\]([\s\S]*?)\[\/\1\]/g;

// Process tool blocks
const toolEvent = blockToToolEvent(block);
if (toolEvent) onToolEvent(toolEvent);

// Process agent blocks  
const agentEvent = blockToAgentEvent(block);
if (agentEvent) onAgentEvent(agentEvent);

// Second pass: parse JSON SSE lines for content
for (const line of lines) {
  if (line.startsWith("data: ")) {
    const parsed = JSON.parse(line.slice(6));
    // Check for tool events embedded in content
    const content = parsed.choices?.[0]?.delta?.content;
    if (content) {
      const toolBlocks = parseSemanticBlocks(content);
      for (const block of toolBlocks) {
        const toolEvent = blockToToolEvent(block);
        if (toolEvent) onToolEvent(toolEvent);
      }
    }
  }
}
```

**Key difference from old format:**
- Old: `__TOOL_EVENT__{json}__END__` embedded in JSON content
- New: `[TOOL_START: name]...[/TOOL_START]` as raw text blocks outside JSON

### Tool Configuration - Planned

Add a Tools page to configure:
- Enable/disable built-in tools
- Add/remove external HTTP tools
- Configure tool permissions
- Set rate limits

## Implementation Phases

### Phase 1: Core Tool Loop ✅ COMPLETED

- ✅ Basic tool execution loop with multi-iteration support
- ✅ OpenAI-format tool definitions
- ✅ Gemma/LM Studio markup format support
- ✅ Non-streaming tool execution
- ✅ Basic logging

**Files:**
- `src/forma/tools/__init__.py` - Tool exports
- `src/forma/tools/base.py` - Tool base classes (Tool, ToolCall, ToolResult)
- `src/forma/tools/executor.py` - Tool execution loop
- `src/forma/tools/registry.py` - Tool registration
- `src/forma/main.py` - Integration with request handler

### Phase 2: Built-in Tools ✅ COMPLETED

- ✅ `search_web` (DuckDuckGo via `ddgs` package)
- ✅ `web_fetch` (URL content fetching with HTML→Markdown)
- ✅ `query_memory` (GrafitoDB query)
- ✅ `get_current_time` (timezone support)
- ✅ `echo` (testing tool)
- ✅ Parameter validation

**Files:**
- `src/forma/tools/builtin/web_search.py`
- `src/forma/tools/builtin/web_fetch.py`
- `src/forma/tools/builtin/memory.py`
- `src/forma/tools/builtin/time.py`
- `src/forma/tools/builtin/echo.py`

### Phase 3: Real-Time Streaming ✅ COMPLETED

- ✅ Streaming tool events via SSE
- ✅ Event marker format (`__TOOL_EVENT__{json}__END__`)
- ✅ SSE flush mechanism (comments + event loop yield)
- ✅ 50ms delay after start event for delivery guarantee
- ✅ UI parsing and display of tool events
- ✅ Tool execution box only shown when tools are used
- ✅ Control signal (`tool_loop_complete`) always sent

**Files:**
- `src/forma/main.py` - `_stream_with_realtime_events()` generator
- `src/forma/tools/executor.py` - Event emission with timing
- `webui/src/api.ts` - SSE parsing with tool event extraction
- `webui/src/components/Chat.vue` - Real-time tool display
- `webui/src/types/index.ts` - ToolEvent, ToolCallInfo types

### Phase 4: External HTTP Tools - PLANNED

- HTTP tool execution
- JSONPath response extraction
- Authentication (API key, Bearer)
- External tool configuration file

**Files:**
- `src/forma/tools/external/http_tool.py`
- `src/forma/tools/external/config_loader.py`
- `config/tools.json` - External tools configuration

### Phase 5: Advanced Features - PLANNED

- Code execution sandbox
- File read with path whitelisting
- Rate limiting
- Permission levels
- Tool result caching

**Files:**
- `src/forma/tools/builtin/code_exec.py`
- `src/forma/tools/builtin/file_read.py`
- `src/forma/tools/rate_limiter.py`
- `src/forma/tools/permissions.py`

### Phase 6: Web UI Enhancements - PLANNED

- Tool call visualization in RequestsList
- Tool configuration page
- Tool execution metrics dashboard

**Files:**
- `webui/src/components/Tools.vue`
- `webui/src/components/RequestsList.vue` (updates)
- `webui/src/types/tools.ts`

## Testing Strategy

### Unit Tests ✅ Implemented

```python
# tests/tools/test_executor.py

async def test_single_tool_call():
    """Test execution of single tool call."""
    
async def test_multi_iteration_loop():
    """Test tool loop with multiple iterations."""
    
async def test_max_iterations_limit():
    """Test that loop stops at max iterations."""
    
async def test_tool_error_handling():
    """Test graceful handling of tool execution errors."""

async def test_lmstudio_markup_extraction():
    """Test extraction of tool calls from Gemma/LM Studio markup."""

# tests/tools/test_builtin.py

async def test_web_search():
    """Test web search tool."""
    
async def test_web_fetch():
    """Test URL content fetching."""
    
async def test_memory_query():
    """Test query_memory against GrafitoDB."""
    
async def test_get_current_time():
    """Test time tool with timezone support."""

# tests/test_config.py

def test_tools_enabled_default():
    """Test tools disabled by default."""

def test_tools_settings():
    """Test tool configuration loading."""
```

### Integration Tests

```python
# tests/integration/test_tool_pipeline.py

async def test_tool_pipeline_with_rag():
    """Test tools work alongside RAG extraction/retrieval."""
    
async def test_tool_results_stored():
    """Test that tool interactions are logged/stored."""
```

## Performance Considerations

### Timeout Management

```python
# Each tool has independent timeout
tool_timeout = get_tool_timeout(tool_name)  # Default or tool-specific

# Total request timeout accounting for tools
total_timeout = base_timeout + (max_iterations * avg_tool_timeout)
```

### Concurrent Tool Execution

When multiple tools are called in same response:

```python
# Execute tools concurrently (if safe)
results = await asyncio.gather(
    *[execute_tool(call) for call in tool_calls],
    return_exceptions=True
)
```

### Tool Result Caching

Cache results for identical tool calls:

```env
TOOLS_CACHE_ENABLED=true
TOOLS_CACHE_TTL_SECONDS=300  # 5 minutes
TOOLS_CACHE_MAX_SIZE=100
```

## Error Handling

### Tool Execution Errors

```python
{
  "role": "tool",
  "tool_call_id": "call_123",
  "content": json.dumps({
    "error": "Tool execution failed",
    "message": "HTTP request timed out",
    "tool_name": "get_weather"
  })
}
```

Model should handle gracefully and either:
- Retry with different parameters
- Report limitation to user
- Use alternative approach

### Loop Termination Errors

If max iterations reached:

```python
{
  "role": "assistant",
  "content": "I've made multiple tool calls but haven't reached a conclusion. Let me provide a partial response based on what I've learned so far..."
}
```

## Open Questions

1. **Tool result storage**: Should tool results be stored in GrafitoDB for future retrieval?
   - Pros: Knowledge from tools persists, can be retrieved later
   - Cons: Stale data, storage growth
   - Decision: Optional, configurable per tool

2. **Tool calls as extractions**: Should tool calls be treated as extractions (logged, stored)?
   - Likely yes: Useful for debugging and audit

3. **Streaming tool events default**: Should streaming tool events be opt-in or opt-out?
   - Recommendation: Opt-in (non-streaming by default for simplicity)

4. **Tool execution in extraction phase**: Should tools be available during extraction?
   - e.g., "search_web to find entity types"
   - Probably no: Keep extraction simple, tools for main request only

## Appendix: OpenAI Function Calling Reference

### Tool Call Format (Response)

```json
{
  "id": "chatcmpl-123",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"location\": \"Berlin\", \"unit\": \"celsius\"}"
        }
      }]
    }
  }]
}
```

### Tool Result Format (Request)

```json
{
  "role": "tool",
  "tool_call_id": "call_abc123",
  "content": "{\"temperature\": 15, \"unit\": \"celsius\", \"description\": \"Cloudy\"}"
}
```

### Tool Choice Options

```json
"tool_choice": "auto"     // Model decides when to use tools
"tool_choice": "none"     // Model must not use tools
"tool_choice": "required" // Model must use at least one tool
"tool_choice": {"type": "function", "function": {"name": "search_web"}}  // Force specific tool
```