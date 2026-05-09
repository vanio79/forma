# Multi-Agent System Design

## Overview

Forma will support multiple AI agents that can discover each other and communicate. Each agent has its own configuration including upstream model, purpose, and instruction prompt. Requests are augmented with agent discovery information so agents can route messages to specific agents.

## Architecture

### Agent Configuration

Each agent is defined by:

```typescript
interface Agent {
  id: string;                    // Unique identifier (UUID)
  name: string;                  // Human-readable name (e.g., "researcher", "coder")
  purpose: string;               // Brief description of agent's role
  instruction_prompt: string;    // System prompt / instruction for this agent
  upstream_id: string | null;    // Reference to upstream config (null = default)
  tools_enabled: boolean;        // Whether tools are enabled for this agent
  tool_whitelist: string[];      // Allowed tools (empty = all tools)
  max_iterations: number;        // Max tool iterations for this agent
  is_enabled: boolean;           // Whether agent is active
  created_at: number;            // Creation timestamp
  updated_at: number;            // Last update timestamp
}
```

### Agent Registry

The agent registry manages all agent configurations:

```python
class AgentRegistry:
    """Registry for managing agent configurations."""
    
    def __init__(self, db: FormaDatabase):
        self._db = db
    
    def register_agent(self, agent: Agent) -> str:
        """Register a new agent configuration."""
        
    def get_agent(self, agent_id: str) -> Agent | None:
        """Get agent by ID."""
        
    def get_agent_by_name(self, name: str) -> Agent | None:
        """Get agent by name."""
        
    def get_all_agents(self) -> list[Agent]:
        """Get all registered agents."""
        
    def get_enabled_agents(self) -> list[Agent]:
        """Get all enabled agents (for discovery)."""
        
    def update_agent(self, agent_id: str, updates: dict) -> bool:
        """Update agent configuration."""
        
    def delete_agent(self, agent_id: str) -> bool:
        """Delete agent configuration."""
```

### Agent Discovery Augmentation

Before sending a request to an agent, Forma augments the prompt with discovery information:

```python
def format_agent_discovery_context(agents: list[Agent]) -> str:
    """Format agent discovery information for prompt augmentation."""
    
    lines = ["Available agents you can communicate with:"]
    
    for agent in agents:
        lines.append(f"- @{agent.name}: {agent.purpose}")
    
    lines.append("")
    lines.append("To send a message to another agent:")
    lines.append("- Mention the agent by name: '@researcher please search for...'")
    lines.append("- Use explicit routing: '>>> researcher: search for...'")
    lines.append("")
    
    return "\n".join(lines)
```

The augmentation is inserted at the top of the context, before memory context:

```
Available agents you can communicate with:
- @researcher: Web search and information gathering
- @coder: Code generation and debugging
- @analyst: Data analysis and visualization

To send a message to another agent:
- Mention the agent by name: '@researcher please search for...'
- Use explicit routing: '>>> researcher: search for...'
```

### Agent Communication Protocol

Agents communicate using structured message routing:

#### Routing Syntax

1. **Direct mention**: `@agent_name message content`
   - Agent's response routes back to the sender
    
2. **Explicit routing**: `>>> agent_name: message content`
   - More formal, explicitly routes message

**NO BROADCAST MESSAGING**: The design explicitly does NOT support `@all` broadcast syntax. All routing is mention-based to specific agents.

#### Message Flow

```
User: "What's the weather in Berlin? @researcher can you search?"
    │
    ▼
Forma detects @researcher mention
    │
    ▼
Route to researcher agent:
    - Researcher's upstream + tools + instruction prompt
    - Message: "What's the weather in Berlin? Please search."
    │
    ▼
Researcher executes (uses search_web tool)
    │
    ▼
Researcher response: "I found that Berlin is 15°C and cloudy."
    │
    ▼
Response tagged as from @researcher
    │
    ▼
User sees: "[@researcher] I found that Berlin is 15°C and cloudy."
```

#### Multi-Agent Orchestration

For complex requests involving multiple agents:

```
User: "@researcher find Python async tutorials, @coder summarize them"
    │
    ▼
Forma orchestrates:
    1. Route to researcher: "find Python async tutorials"
    2. Wait for researcher response
    3. Route to coder with researcher's result: "Summarize these tutorials: [researcher output]"
    4. Return combined results to user
```

### Message Routing Implementation

```python
class AgentRouter:
    """Routes messages between agents."""
    
    def __init__(
        self,
        registry: AgentRegistry,
        proxy: OpenAIProxy,
        tool_executor: ToolExecutor | None,
    ):
        self._registry = registry
        self._proxy = proxy
        self._tool_executor = tool_executor
    
    def parse_agent_mentions(self, content: str) -> list[str]:
        """Extract agent names mentioned in content.
        
        Patterns:
        - @agent_name
        - >>> agent_name:
        """
        
    def route_to_agent(
        self,
        agent: Agent,
        message: str,
        context: dict | None = None,
    ) -> dict:
        """Route message to specific agent.
        
        1. Load agent's upstream configuration
        2. Load agent's instruction prompt
        3. Augment with agent discovery context (other agents)
        4. Execute with agent's tool settings
        5. Tag response with agent name
        """
        
    async def orchestrate_multi_agent(
        self,
        mentions: list[str],
        original_message: str,
        context: dict | None = None,
    ) -> dict:
        """Orchestrate multi-agent conversation.
        
        - Parse message parts for each agent
        - Route sequentially
        - Combine results
        """
```

### Request Flow with Agents

```
POST /v1/chat/completions
{
  "messages": [
    {"role": "user", "content": "@researcher find Python async tutorials"}
  ],
  "model": "assistant",  // Default agent
  "forma_agent": null    // Optional: explicit agent ID override
}
    │
    ▼
1. Extract entity mentions from message
    │
    ▼
2. Parse agent mentions (@researcher, @coder, etc.)
    │
    ▼
3. If agent mentions found:
   a. Get mentioned agent configurations
   b. Route message to each mentioned agent
   c. Collect responses with agent tags
   d. Return combined response
    │
    ▼
4. If no agent mentions:
   a. Use default agent (or forma_agent if specified)
   b. Augment with agent discovery context
   c. Execute normal flow
   d. Return response
```

### Response Format with Agent Tags

```json
{
  "id": "chatcmpl-123",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "[@researcher] I found several Python async tutorials...",
      "forma_metadata": {
        "agent_name": "researcher",
        "agent_id": "uuid-123",
        "routing_type": "mention"
      }
    }
  }]
}
```

For multi-agent responses:

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "--- [@researcher]\nI found several Python async tutorials...\n\n--- [@coder]\nHere's a summary: ..."
    }
  }]
}
```

### Streaming with Multi-Agent

For streaming requests, each agent's response is streamed with markers:

```
__AGENT_START__{"agent": "researcher", "depth": 0}__END__
[streaming content from researcher]
__AGENT_END__{"agent": "researcher", "depth": 0}__END__

__AGENT_START__{"agent": "coder", "depth": 1}__END__
[streaming content from coder]
__AGENT_END__{"agent": "coder", "depth": 1}__END__
```

### Database Schema

Add agents table to FormaDatabase:

```sql
CREATE TABLE agents (
    id TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    purpose TEXT NOT NULL,
    instruction_prompt TEXT NOT NULL,
    upstream_id TEXT NULL,           -- Reference to upstreams table
    tools_enabled INTEGER DEFAULT 0,
    tool_whitelist TEXT DEFAULT '',  -- JSON array of tool names
    max_iterations INTEGER DEFAULT 5,
    is_enabled INTEGER DEFAULT 1,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    FOREIGN KEY (upstream_id) REFERENCES upstreams(id)
);
```

### Configuration File

Optionally load agents from config file:

```json
{
  "agents": [
    {
      "name": "researcher",
      "purpose": "Web search and information gathering",
      "instruction_prompt": "You are a research assistant. Your job is to search for information and provide comprehensive summaries. Always cite your sources.",
      "upstream": "gpt-4-upstream",
      "tools_enabled": true,
      "tool_whitelist": ["search_web", "web_fetch"],
      "max_iterations": 10
    },
    {
      "name": "coder",
      "purpose": "Code generation and debugging",
      "instruction_prompt": "You are a coding assistant. Help with writing, debugging, and explaining code. Be precise and provide working examples.",
      "upstream": "gpt-4-upstream",
      "tools_enabled": false
    },
    {
      "name": "analyst",
      "purpose": "Data analysis and visualization",
      "instruction_prompt": "You are a data analyst. Analyze data, identify patterns, and provide insights. Suggest visualizations when appropriate.",
      "upstream": "gpt-4-upstream",
      "tools_enabled": true,
      "tool_whitelist": ["query_memory"]
    }
  ]
}
```

### API Endpoints

#### Agent Management API

```python
# Web UI API endpoints

@router.get("/agents")
async def get_agents():
    """Get all agent configurations."""
    
@router.post("/agents")
async def create_agent(
    name: str,
    purpose: str,
    instruction_prompt: str,
    upstream_id: str | None = None,
    tools_enabled: bool = True,
    tool_whitelist: list[str] = [],
    max_iterations: int = 5,
):
    """Create new agent configuration."""
    
@router.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    """Get specific agent configuration."""
    
@router.put("/agents/{agent_id}")
async def update_agent(agent_id: str, **updates):
    """Update agent configuration."""
    
@router.delete("/agents/{agent_id}")
async def delete_agent(agent_id: str):
    """Delete agent configuration."""
```

#### Agent Routing API

```python
@app.post("/v1/agent/{agent_name}/chat/completions")
async def agent_chat_completions(agent_name: str, request: Request):
    """Chat completion routed to specific agent.
    
    Equivalent to setting forma_agent in request payload.
    """
```

### Web UI

Add Agents page to Web UI:

- List all agents with status (enabled/disabled)
- Create/edit/delete agents
- View agent instruction prompt
- Test agent with sample message
- View agent's upstream configuration
- View agent's tool whitelist

### Integration with Existing Features

#### RAG + Agents

**CRITICAL: Memory is shared globally across all agents.**

All agents query the same global indexes in GrafitoDB (`facts_index` and `recipes_index`). There are NO agent-specific indexes or collections.

Each agent has a `rag_config` that controls how much context it retrieves:

```typescript
interface RAGConfig {
  enabled: boolean;           // Whether RAG is enabled for this agent
  token_budget: number;       // Max tokens for retrieved context (e.g., 1500)
  min_confidence: number;     // Minimum confidence threshold (0.0-1.0)
  max_distance: number;       // Maximum vector distance threshold (0.0-1.0)
}
```

**Note:** Earlier design drafts incorrectly included `collection_prefix` and `share_global_context` fields. These were removed - memory is ALWAYS shared globally.

Retrieved context is augmented before agent's instruction prompt:

```
[Agent Discovery Context]
[Retrieved Memory Context (from global facts_index/recipes_index)]
[Agent Instruction Prompt]
[User Message]
```

Example RAG configs from config/agents.json:
- assistant: token_budget=1500, min_confidence=0.5, max_distance=0.7 (more tolerant, retrieves more)
- researcher: token_budget=2000, min_confidence=0.7, max_distance=0.5 (stricter, more focused)
- coder: token_budget=1000, min_confidence=0.7, max_distance=0.5 (minimal context for code)

#### Tools + Agents

Each agent can have different tool settings:
- `tools_enabled`: Whether tools are enabled for this agent
- `tool_whitelist`: Specific tools this agent can use (subset of available tools)
- `max_iterations`: Agent-specific tool iteration limit

#### Upstreams + Agents

Each agent can use a different upstream model:
- Some agents use powerful models (GPT-4)
- Some agents use faster/cheaper models (GPT-3.5)
- Some agents use specialized models (code models)

### Security Considerations

1. **Agent authentication**: Only authorized users can create/modify agents
2. **Agent isolation**: Each agent's instruction prompt is private to that agent
3. **Rate limiting**: Each agent can have its own rate limits
4. **Audit logging**: All agent communications are logged for audit

### Implementation Phases

#### Phase 1: Agent Registry - HIGH PRIORITY ✅ COMPLETE

- ✅ Add agents table to FormaDatabase
- ✅ Implement AgentRegistry class
- ✅ Add agent CRUD operations to forma_db.py
- ✅ Add agent configuration file loading
- ✅ Add rag_config field to agents table

**Files:**
- `src/forma/forma_db.py` - Add agents table, CRUD, and rag_config field
- `src/forma/agents/__init__.py` - Agent exports
- `src/forma/agents/registry.py` - AgentRegistry implementation
- `src/forma/agents/config_loader.py` - Load agents from config file

#### Phase 2: Agent Discovery - HIGH PRIORITY ✅ COMPLETE

- ✅ Implement agent discovery context formatting
- ✅ Add agent discovery to request augmentation
- ✅ Update storage.format_context_for_prompt() to include agents

**Files:**
- `src/forma/agents/discovery.py` - Discovery context formatting
- `src/forma/storage.py` - Update format_context_for_prompt()
- `src/forma/main.py` - Add agent discovery to request flow

#### Phase 3: Agent Routing - HIGH PRIORITY ✅ COMPLETE

- ✅ Implement AgentRouter class
- ✅ Parse agent mentions from messages (mention + explicit routing)
- ✅ Route messages to specific agents
- ✅ Tag responses with agent names
- ✅ Remove all broadcast-related code (NO @all support)

**Files:**
- `src/forma/agents/router.py` - AgentRouter implementation (mention-based only)
- `src/forma/agents/parser.py` - Parse agent mentions (RoutingType: MENTION, EXPLICIT only)
- `src/forma/main.py` - Integrate routing into chat_completions

#### Phase 4: Multi-Agent Orchestration - MEDIUM PRIORITY ✅ COMPLETE

- ✅ Implement sequential orchestration (agents run one after another)
- ✅ Combine multi-agent results
- ✅ Handle agent-to-agent replies (max depth = 3)
- ✅ Remove broadcast logic from orchestrator

**Files:**
- `src/forma/agents/orchestrator.py` - Multi-agent orchestration (sequential)
- `src/forma/main.py` - Orchestration integration

#### Phase 5: Streaming Support - MEDIUM PRIORITY ✅ COMPLETE

- ✅ Add agent markers to SSE stream
- ✅ Stream multi-agent responses with separators
- ✅ UI parsing of agent markers
- ✅ Real-time agent execution display

**Files:**
- `src/forma/main.py` - Agent streaming markers (__AGENT_START__, __AGENT_END__)
- `webui/src/api.ts` - Parse agent markers
- `webui/src/components/Chat.vue` - Display agent-tagged responses

#### Phase 6: Web UI - LOW PRIORITY ✅ COMPLETE

- ✅ Add Agents page component (713 lines)
- ✅ Agent CRUD UI (create, view, edit, delete)
- ✅ Agent discovery information display
- ✅ Remove @all broadcast mention from documentation

**Files:**
- `webui/src/components/Agents.vue` - Agents management page
- `webui/src/api.ts` - Agent API client
- `webui/src/types/index.ts` - Agent TypeScript types

#### Phase 7: Agent-Specific RAG Contexts - LOW PRIORITY ✅ COMPLETE

- ✅ Add rag_config to agents (enabled, token_budget, min_confidence, max_distance)
- ✅ Implement agent RAG retrieval using global indexes
- ✅ Each agent uses retrieve_context() with its own config parameters
- ✅ Memory is SHARED globally (no agent-specific indexes)
- ✅ Removed agent-specific storage methods (store_agent_facts, etc.)
- ✅ Updated docstrings to reflect new rag_config structure

**Implementation Details:**
- Agents query the SAME global `facts_index` and `recipes_index`
- Each agent has different token_budget/min_confidence/max_distance thresholds
- Global methods: store_facts(), store_recipes(), retrieve_context()
- Removed: collection_prefix, share_global_context fields (not needed - always global)
- No broadcast messaging support (explicitly removed per design)

**Files:**
- `src/forma/storage.py` - Removed ~400 lines of agent-specific methods
- `src/forma/main.py` - Agent execution uses retrieve_context() (lines 459-487, 664-690)
- `src/forma/forma_db.py` - rag_config field in agents table
- `src/forma/agents/registry.py` - Updated docstrings
- `config/agents.json` - rag_config for each agent

#### Phase 8: Meta-Agent Evaluation System - HIGH PRIORITY ✅ COMPLETE

- ✅ Add @evaluator and @summarizer meta-agents to config/agents.json
- ✅ Implement evaluation flow after subagent delegation
- ✅ Implement retry loop with evaluator guidance (max 50 attempts)
- ✅ Implement automatic context compaction at 90% threshold
- ✅ SSE streaming of evaluation events (__EVAL_EVENT__ markers)
- ✅ Iterative summarization for large single-message contexts
- ✅ Trusted meta-agents don't get evaluated (avoid infinite loops)

**Architecture:**

```
Calling Agent delegates task
    │
    ▼
Subagent executes (isolated context)
    │
    ▼
@evaluator assesses completion
    │
    ├─── complete ──→ @summarizer compacts context ──→ Return to caller
    │
    ├─── incomplete ──→ Retry with guidance (max 50 attempts)
    │                      │
    │                      ▼
    │                   @evaluator reassesses
    │
    └─── failed ──→ @summarizer compacts context ──→ Return to caller
```

**Evaluation States:**
- `complete`: Task fully addressed with sufficient detail, actionable results provided
- `incomplete`: Partial progress, needs more depth/tools/verification, can be improved with guidance
- `failed`: Task impossible, no relevant results, wrong approach, should try different strategy

**EvaluationResult Data Structure:**
```python
@dataclass
class EvaluationResult:
    status: str          # "complete", "incomplete", "failed"
    reason: str          # Brief explanation
    retry_instructions: str | None  # Specific guidance (if incomplete)
    summary_focus: str | None       # What to highlight (if complete/failed)
    confidence: float   # 0.0-1.0
    is_valid: bool      # Whether JSON was parsed correctly
```

**Retry Loop Mechanics:**
1. Evaluator returns `incomplete` with specific `retry_instructions`
2. System creates retry context with:
   - Original task
   - Previous subagent response
   - Evaluator feedback
3. Subagent executes again with guidance
4. Loop continues until complete/failed or max attempts (50)
5. Progressive refinement improves results each iteration

**Context Compaction Strategy:**
Automatic compaction triggers at 90% of context window (34560 tokens for 38400 window):
- Check: `should_compact_context()` estimates token count
- Trigger: When estimated tokens >= threshold
- Preserve: Most recent 4 messages (last 2 exchanges)
- Summarize: Older messages via @summarizer agent
- Iterative: For single large messages (>1500 chars), chunk and summarize progressively

**SSE Evaluation Events:**
```typescript
interface EvalEvent {
  type: "evaluation_start" | "evaluation_result" | "retry_attempt";
  subagent?: string;
  status?: string;
  reason?: string;
  attempt?: number;
  max_attempts?: number;
}
```

Events streamed as: `__EVAL_EVENT__{json}__END__`

**Example Retry Sequence:**
```
Attempt 1: @researcher finds links → Evaluator: incomplete (95%)
           → Guidance: "Use web_fetch to get full article content"

Attempt 2: @researcher fetches page → Evaluator: incomplete (90%)
           → Guidance: "Extract specific recipe details from content"

Attempt 3: @researcher parses recipe → Evaluator: complete (85%)
           → @summarizer: "Found циганска баница recipe with ingredients..."
```

**Meta-Agent Configuration:**
```json
{
  "name": "evaluator",
  "purpose": "Meta-agent that evaluates subagent task completion",
  "instruction_prompt": "Assess task completion and output JSON...",
  "tools_enabled": false,
  "rag_config": { "enabled": false }
}
```

Key traits:
- No tools enabled (evaluator just assesses, doesn't act)
- No RAG retrieval (isolated assessment)
- Trusted flag prevents evaluation of evaluator itself

**Files:**
- `config/agents.json` - Added @evaluator and @summarizer configurations
- `src/forma/agents/meta_evaluation.py` - EvaluationResult, parsing, context helpers
- `src/forma/main.py` - Evaluation flow integration (lines 1750-2040), SSE events
- `webui/src/api.ts` - Eval event parsing for UI display

### Example Use Cases

#### 1. Research + Summarization (Sequential)

```
User: "@researcher find recent papers on transformers, @coder summarize them"

Response:
--- [@researcher]
I found 5 recent papers:
1. "Attention Is All You Need" (2017)
2. "BERT: Pre-training of Deep Bidirectional Transformers" (2018)
...

--- [@coder]
Summary: Transformers have revolutionized NLP with their attention mechanism...
```

Each agent executes sequentially - researcher finds the papers first, then coder summarizes.

#### 2. Code Review

```
User: "@coder write a Python function to sort a list, @analyst analyze its efficiency"

Response:
--- [@coder]
```python
def sort_list(items):
    return sorted(items)
```

--- [@analyst]
Efficiency analysis: Uses Python's built-in sorted(), which is O(n log n)...
```

#### 3. Agent-to-Agent Delegation (with Meta-Agent Evaluation)

```
User: "@assistant find a recipe for циганска баница"

Response:
--- [@assistant]
Delegating to @researcher: Find a recipe for циганска баница

--- [@researcher] (Attempt 1)
I found several search results mentioning циганска баница:
- https://example.com/recipe1
- https://example.com/recipe2

[Evaluation: incomplete (95%) - need full recipe details]
[Retry guidance: Use web_fetch to get article content]

--- [@researcher] (Attempt 2)
I fetched the article. It mentions traditional Bulgarian pastry...
[Evaluation: incomplete (90%) - missing ingredients/instructions]
[Retry guidance: Parse and extract recipe details]

--- [@researcher] (Attempt 3)
Recipe for циганска баница:
Ingredients: filo dough, eggs, yogurt, cheese, butter...
Instructions: 1. Layer filo sheets, 2. Mix filling...

[Evaluation: complete (85%)]

--- [@summarizer]
Summary: Found traditional Bulgarian циганска баница recipe with filo dough, 
cheese filling, and step-by-step layering instructions.

--- [@assistant]
Here's the recipe for циганска баница that @researcher found:
[recipe details from summary]
```

The meta-agent evaluation system ensures:
- Subagent actually completes the delegated task
- Incomplete responses get specific retry guidance
- Context is summarized before returning to caller
- Quality threshold enforced (no "I searched" without results)

Agents can delegate to other agents, creating delegation chains (max depth = 3). Each delegation is evaluated by @evaluator.

### Testing Strategy

```python
# tests/agents/test_registry.py

def test_register_agent():
    """Test agent registration."""
    
def test_get_agent_by_name():
    """Test agent lookup by name."""
    
def test_update_agent():
    """Test agent configuration update."""

# tests/agents/test_router.py

def test_parse_agent_mentions():
    """Test parsing @agent_name syntax."""
    
def test_route_to_agent():
    """Test routing message to specific agent."""

# tests/agents/test_orchestrator.py

def test_sequential_orchestration():
    """Test sequential multi-agent execution."""
    
def test_agent_reply_chain():
    """Test agent-to-agent communication chain."""

# tests/integration/test_agent_pipeline.py

def test_agent_with_rag():
    """Test agent uses RAG correctly."""
    
def test_agent_with_tools():
    """Test agent uses whitelisted tools."""
    
def test_multi_agent_with_extraction():
    """Test extraction works with multi-agent."""
```

### Configuration Examples

#### Environment Variables

```env
# Agent system configuration
AGENTS_ENABLED=true
AGENTS_CONFIG_PATH=./config/agents.json
AGENTS_DEFAULT_NAME=assistant
AGENTS_DISCOVERY_ENABLED=true
```

#### Agents Config File (Current Implementation)

```json
{
  "agents": [
    {
      "name": "assistant",
      "purpose": "General assistant - coordinates and delegates to specialists",
      "instruction_prompt": "You are a coordinator assistant. Delegate specialized tasks: @researcher for searches, @coder for code.",
      "upstream": null,
      "tools_enabled": true,
      "tool_whitelist": ["echo", "get_current_time", "query_memory"],
      "max_iterations": 5,
      "is_enabled": true,
      "rag_config": {
        "enabled": true,
        "token_budget": 1500,
        "min_confidence": 0.5,
        "max_distance": 0.7
      }
    },
    {
      "name": "researcher",
      "purpose": "Research and information gathering",
      "instruction_prompt": "You are a research specialist. Use search_web and web_fetch tools for information gathering.",
      "upstream": null,
      "tools_enabled": true,
      "tool_whitelist": ["search_web", "web_fetch", "get_current_time"],
      "max_iterations": 10,
      "is_enabled": true,
      "rag_config": {
        "enabled": true,
        "token_budget": 2000,
        "min_confidence": 0.7,
        "max_distance": 0.5
      }
    },
    {
      "name": "coder",
      "purpose": "Code generation and debugging",
      "instruction_prompt": "You are a coding specialist. Write clean, efficient code.",
      "upstream": null,
      "tools_enabled": false,
      "tool_whitelist": [],
      "max_iterations": 3,
      "is_enabled": true,
      "rag_config": {
        "enabled": true,
        "token_budget": 1000,
        "min_confidence": 0.7,
        "max_distance": 0.5
      }
    }
  ]
}
```

**Note:** The `rag_config` field controls agent-specific retrieval parameters from the GLOBAL shared indexes.

#### Meta-Agent Configuration (Phase 8)

Meta-agents are special agents that manage quality control and context optimization:

```json
{
  "name": "evaluator",
  "purpose": "Meta-agent that evaluates subagent task completion",
  "instruction_prompt": "You are an evaluator agent. Assess task completion...",
  "upstream": null,
  "tools_enabled": false,
  "tool_whitelist": [],
  "max_iterations": 1,
  "is_enabled": true,
  "rag_config": {
    "enabled": false,
    "token_budget": 0,
    "min_confidence": 0.0,
    "max_distance": 1.0
  }
},
{
  "name": "summarizer",
  "purpose": "Meta-agent that compacts subagent context into concise summaries",
  "instruction_prompt": "You are a summarizer agent. Create concise summaries...",
  "upstream": null,
  "tools_enabled": false,
  "tool_whitelist": [],
  "max_iterations": 1,
  "is_enabled": true,
  "rag_config": {
    "enabled": false,
    "token_budget": 0,
    "min_confidence": 0.0,
    "max_distance": 1.0
  }
}
```

**Key Meta-Agent Traits:**
- **No tools**: Evaluator/summarizer don't use tools (they assess/summarize only)
- **No RAG**: Disabled RAG retrieval (isolated assessment/summarization)
- **Trusted**: Not evaluated themselves (avoid infinite meta-agent loops)
- **Single iteration**: max_iterations=1 (quick single-pass operations)

## Design Decisions (Resolved)

1. **Agent memory sharing**: ✅ RESOLVED - Shared memory globally
   - **Decision**: All agents query the same global `facts_index` and `recipes_index` in GrafitoDB
   - **Implementation**: `rag_config` controls retrieval parameters (token_budget, min_confidence, max_distance)
   - **No agent-specific indexes**: Removed agent-specific storage methods, collection_prefix field
   - **Reason**: Simpler architecture, agents can share knowledge seamlessly

2. **Agent authentication**: ✅ RESOLVED - No authentication for routing
   - **Decision**: Any user can route to any agent (no auth required)
   - **Future**: Token-based auth for admin operations (create/modify agents)

3. **Agent reply chain**: ✅ RESOLVED - Flat history with depth limit
   - **Decision**: All messages in single conversation, max depth = 3 for delegation chains
   - **Implementation**: Sequential orchestration with agent-to-agent support
   - **Example**: assistant → researcher → coder (depth 2)

4. **Default agent behavior**: ✅ RESOLVED - Use "assistant" as default
   - **Decision**: When no agent mentioned, use "assistant" agent
   - **Implementation**: Created default "assistant" agent in config/agents.json
   - **Role**: Coordinator that delegates to specialist agents

5. **Broadcast messaging**: ✅ RESOLVED - NO broadcast support
    - **Decision**: NO @all broadcast syntax, only mention-based routing (@agent_name)
    - **Implementation**: Removed all broadcast-related code from router, orchestrator, main.py
    - **Reason**: Broadcast creates coordination complexity without clear benefit
    - **Removed**: RoutingType.BROADCAST, get_broadcast_agents(), has_broadcast checks

6. **Subagent quality control**: ✅ RESOLVED - Meta-agent evaluation
    - **Decision**: Automatic evaluation of subagent task completion by @evaluator meta-agent
    - **Implementation**: EvaluationResult with complete/incomplete/failed states, retry loop with guidance
    - **Reason**: Prevents low-quality delegations from polluting calling agent's context
    - **Retry**: Max 50 attempts with progressive refinement based on evaluator feedback
    - **Summarization**: @summarizer compacts context before returning to caller

7. **Context overflow in multi-agent**: ✅ RESOLVED - Automatic compaction at 90%
    - **Decision**: Trigger context compaction when estimated tokens reach 90% of window
    - **Implementation**: `should_compact_context()` with char-based token estimation
    - **Threshold**: 34560 tokens for 38400 window, preserves last 4 messages
    - **Iterative**: For large single messages, chunk and summarize progressively
    - **Reason**: Prevents context overflow errors during long agent-to-agent conversations