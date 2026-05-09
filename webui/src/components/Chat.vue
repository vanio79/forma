<template>
  <div class="chat-page">
    <!-- Settings Panel -->
    <div class="card settings-card">
      <div class="settings-row">
        <div class="form-group">
          <label for="model">Model</label>
          <select id="model" v-model="selectedModel" :disabled="loadingModels || isStreaming || isCompacting">
            <option v-if="loadingModels" value="">Loading models...</option>
            <option v-else-if="enabledUpstreams.length === 0" value="">No models configured</option>
            <option v-else v-for="upstream in enabledUpstreams" :key="upstream.id" :value="upstream.name">
              {{ upstream.name }}
            </option>
          </select>
        </div>

        <div class="form-group">
          <label for="contextSize">Context Size (tokens)</label>
          <input 
            id="contextSize" 
            v-model.number="contextSize" 
            type="number" 
            min="256" 
            max="128000" 
            step="256"
            :disabled="isStreaming || isCompacting"
          />
        </div>

        <div class="token-info">
          <span :class="['token-count', { 'token-warning': tokenUsagePercent >= 95 }]">
            Tokens: {{ actualTokens }} / {{ contextSize }}
            <span v-if="tokenUsagePercent >= 95" class="warning-icon">⚠️</span>
          </span>
          <span :class="['token-bar', tokenLevelClass]">
            <div class="token-bar-fill" :style="{ width: tokenUsagePercent + '%' }"></div>
          </span>
        </div>
      </div>
    </div>

    <!-- Chat Messages -->
    <div class="card messages-card">
      <div class="messages-container" ref="messagesContainer">
        <div v-if="messages.length === 0" class="empty-chat">
          <p>Start a conversation by typing a message below.</p>
          <p class="hint">Select a model and context size first. When context reaches 95% full, earlier messages are summarized to preserve important details.</p>
        </div>

        <div v-for="(msg, idx) in messages" :key="idx" :class="['message', msg.role, { streaming: msg.isStreaming, compacting: msg.isCompacting }]">
          <!-- System messages (summaries) with special format -->
          <template v-if="msg.role === 'system'">
            <!-- Compaction in progress - show progress steps -->
            <template v-if="msg.isCompacting">
              <div class="system-summary-box compacting">
                <div class="system-summary-header">
                  <span class="system-summary-label">🔄 Compacting Context...</span>
                  <span class="system-summary-time">Now</span>
                </div>
                <div class="compaction-progress">
                  <div class="compaction-steps">
                    <div :class="['step', { active: compactionStep >= 1, done: compactionStep > 1 }]">
                      <span class="step-icon">{{ compactionStep > 1 ? '✓' : compactionStep === 1 ? '●' : '○' }}</span>
                      Analyzing conversation...
                    </div>
                    <div :class="['step', { active: compactionStep >= 2, done: compactionStep > 2 }]">
                      <span class="step-icon">{{ compactionStep > 2 ? '✓' : compactionStep === 2 ? '●' : '○' }}</span>
                      Generating summary...
                    </div>
                    <div :class="['step', { active: compactionStep >= 3, done: compactionStep > 3 }]">
                      <span class="step-icon">{{ compactionStep > 3 ? '✓' : compactionStep === 3 ? '●' : '○' }}</span>
                      Applying changes...
                    </div>
                  </div>
                  <div v-if="compactionMessage" class="compaction-detail">{{ compactionMessage }}</div>
                  <!-- Streaming summary during generation -->
                  <div v-if="msg.content" class="streaming-summary">
                    <div class="streaming-summary-label">📝 Summary:</div>
                    <div class="streaming-summary-text">
                      {{ msg.content }}
                      <span class="streaming-indicator"></span>
                    </div>
                  </div>
                </div>
              </div>
            </template>
            <!-- Completed summary - show final content -->
            <template v-else>
              <div class="system-summary-box">
                <div class="system-summary-header">
                  <span class="system-summary-label">📝 Context Summary</span>
                  <span class="system-summary-time">{{ formatTime(msg.timestamp ?? Date.now()) }}</span>
                </div>
                <div class="system-summary-content">{{ msg.content }}</div>
              </div>
            </template>
          </template>
          <!-- Regular user/assistant messages -->
          <template v-else>
            <div class="message-header">
              <span class="message-role">
                <template v-if="msg.role === 'user'">You</template>
                <template v-else-if="msg.agentChain && msg.agentChain.length > 1">
                  🤖 @{{ msg.agentChain.join(' → @') }}
                </template>
                <template v-else-if="msg.agentName">🤖 @{{ msg.agentName }}</template>
                <template v-else-if="currentAgent && msg.isStreaming">🤖 @{{ currentAgent }}</template>
                <template v-else>Assistant</template>
              </span>
              <span class="message-time">{{ formatTime(msg.timestamp ?? Date.now()) }}</span>
            </div>
            
            <!-- Reasoning section (collapsible) -->
            <div v-if="msg.reasoning" class="reasoning-section">
              <button 
                class="reasoning-toggle" 
                @click="toggleReasoning(msg)"
                :disabled="msg.isStreaming"
              >
                <span class="toggle-icon">{{ msg.showReasoning ? '▼' : '▶' }}</span>
                <span class="toggle-label">💭 Reasoning</span>
                <span v-if="msg.isStreaming" class="streaming-indicator-small"></span>
              </button>
              <div v-if="msg.showReasoning" class="reasoning-content">
                {{ msg.reasoning }}
              </div>
            </div>
            
            <!-- Tool execution section (show during tool calling) -->
            <div v-if="msg.toolExecution && !msg.toolExecution.isComplete && msg.toolExecution.toolCalls.length > 0" class="tool-execution-section">
              <div class="tool-execution-header">
                <span class="tool-execution-label">🔧 Tool Execution</span>
                <span class="tool-iteration">Iteration {{ msg.toolExecution.iteration }}/{{ msg.toolExecution.maxIterations }}</span>
              </div>
              <div class="tool-calls-list">
                <div v-for="(call, callIdx) in msg.toolExecution.toolCalls" :key="callIdx" :class="['tool-call-item', call.status]">
                  <div class="tool-call-header">
                    <span class="tool-call-icon">
                      {{ call.status === 'pending' ? '○' : call.status === 'running' ? '●' : call.status === 'success' ? '✓' : '✗' }}
                    </span>
                    <span class="tool-call-name">{{ call.name }}</span>
                    <span v-if="call.status === 'running'" class="tool-spinner"></span>
                    <span v-if="call.duration_ms" class="tool-call-duration">{{ call.duration_ms.toFixed(1) }}ms</span>
                    <button 
                      v-if="call.result && call.status === 'success'" 
                      class="tool-expand-btn"
                      @click="toggleToolResult(call)"
                    >
                      {{ call.expanded ? '▼' : '▶' }}
                    </button>
                  </div>
                  <div v-if="call.arguments && Object.keys(call.arguments).length > 0" class="tool-call-args">
                    {{ JSON.stringify(call.arguments) }}
                  </div>
                  <div v-if="call.expanded && call.result" class="tool-call-result-full">
                    {{ call.result }}
                  </div>
                  <div v-if="!call.expanded && call.result && call.status === 'success'" class="tool-call-result-preview">
                    {{ call.result.length > 100 ? call.result.slice(0, 100) + '...' : call.result }}
                  </div>
                  <div v-if="call.error && call.status === 'failed'" class="tool-call-error">
                    {{ call.error }}
                  </div>
                </div>
              </div>
            </div>
            
            <!-- Tool execution summary (after completion) -->
            <div v-if="msg.toolExecution && msg.toolExecution.isComplete && msg.toolExecution.toolCalls.length > 0" class="tool-execution-summary">
              <div class="tool-summary-header">
                <span class="tool-summary-label">🔧 Tools completed: {{ msg.toolExecution.toolCalls.length }} calls in {{ msg.toolExecution.totalTimeMs.toFixed(1) }}ms</span>
                <button class="tool-expand-btn" @click="toggleToolExecution(msg)">
                  {{ msg.toolExecutionExpanded ? '▼' : '▶' }}
                </button>
              </div>
              <div v-if="msg.toolExecutionExpanded" class="tool-summary-details">
                <div v-for="(call, callIdx) in msg.toolExecution.toolCalls" :key="callIdx" :class="['tool-call-item', call.status]">
                  <div class="tool-call-header">
                    <span class="tool-call-icon">
                      {{ call.status === 'success' ? '✓' : '✗' }}
                    </span>
                    <span class="tool-call-name">{{ call.name }}</span>
                    <span v-if="call.duration_ms" class="tool-call-duration">{{ call.duration_ms.toFixed(1) }}ms</span>
                  </div>
                  <div v-if="call.arguments && Object.keys(call.arguments).length > 0" class="tool-call-args">
                    {{ JSON.stringify(call.arguments) }}
                  </div>
                  <div v-if="call.result && call.status === 'success'" class="tool-call-result-full">
                    {{ call.result }}
                  </div>
                  <div v-if="call.error && call.status === 'failed'" class="tool-call-error">
                    {{ call.error }}
                  </div>
                </div>
              </div>
            </div>
            
            <div class="message-content">
              {{ msg.content }}
              <span v-if="msg.isStreaming && !msg.reasoning" class="streaming-indicator"></span>
            </div>
          </template>
        </div>
      </div>

      <!-- Message Input -->
      <div class="input-area">
        <textarea 
          ref="inputTextarea"
          v-model="inputMessage" 
          placeholder="Type your message..."
          :disabled="!selectedModel || isStreaming || isCompacting"
          @keydown.enter.exact.prevent="sendMessage"
          rows="3"
        ></textarea>
        <button 
          class="btn btn-primary send-btn" 
          @click="sendMessage"
          :disabled="!selectedModel || !inputMessage.trim() || isStreaming || isCompacting"
        >
          {{ isStreaming ? 'Sending...' : isCompacting ? 'Compacting...' : 'Send' }}
        </button>
        <button 
          class="btn btn-secondary clear-btn" 
          @click="clearChat"
          :disabled="messages.length === 0 || isStreaming || isCompacting"
        >
          Clear
        </button>
      </div>

      <div v-if="error" class="error">{{ error }}</div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, nextTick } from "vue";
import type { Upstream, ChatMessage, ToolEvent, ToolExecutionState, ToolCallInfo, AgentEvent } from "../types";
import { getUpstreams, streamChatCompletion } from "../api";

const upstreams = ref<Upstream[]>([]);
const loadingModels = ref(true);
const selectedModel = ref("");
const contextSize = ref(4096);
const messages = ref<ChatMessage[]>([]);
const inputMessage = ref("");
const inputTextarea = ref<HTMLTextAreaElement | null>(null);
const isStreaming = ref(false);
const error = ref("");
const messagesContainer = ref<HTMLElement | null>(null);

// Compaction state (progress steps)
const isCompacting = ref(false);
const compactionStep = ref(0);  // 0: not started, 1: analyzing, 2: generating, 3: applying
const compactionMessage = ref("");

// Tool execution state
const toolExecutionState = ref<ToolExecutionState>({
  iteration: 0,
  maxIterations: 5,
  toolCalls: [],
  isComplete: false,
  totalTimeMs: 0,
});

// Agent tracking state
const currentAgent = ref<string | null>(null);  // Which agent is currently responding
const agentResponses = ref<Record<string, string>>({});  // Track content by agent name
const agentMessageIndices = ref<Record<string, number>>({});  // Track message index for each agent

// Actual token usage tracking (from API responses)
const promptTokens = ref(0);
const completionTokens = ref(0);

// Computed: enabled upstreams
const enabledUpstreams = computed(() => 
  upstreams.value.filter(u => u.is_enabled)
);

// Actual total tokens from API
const actualTokens = computed(() => promptTokens.value + completionTokens.value);

// Token usage percentage
const tokenUsagePercent = computed(() => 
  Math.min(100, (actualTokens.value / contextSize.value) * 100)
);

// Token bar color class
const tokenLevelClass = computed(() => {
  const pct = tokenUsagePercent.value;
  if (pct >= 95) return 'critical';
  if (pct >= 75) return 'warning';
  return 'normal';
});

// Format timestamp
function formatTime(timestamp: number): string {
  const date = new Date(timestamp);
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

// Auto-scroll to bottom
function scrollToBottom() {
  nextTick(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight;
    }
  });
}

// Build summarization messages (like OpenCode's approach)
function buildSummarizationMessages(messagesToSummarize: ChatMessage[]): ChatMessage[] {
  return [
    {
      role: "system",
      content: "You are a helpful AI assistant tasked with summarizing conversations.\n\nWhen asked to summarize, provide a detailed but concise summary of the conversation.\nFocus on information that would be helpful for continuing the conversation, including:\n- What was done\n- What is currently being worked on\n- Which files are being modified\n- What needs to be done next\n\nYour summary should be comprehensive enough to provide context but concise enough to be quickly understood.",
    },
    {
      role: "user",
      content: `Provide a detailed but concise summary of our conversation above. Focus on information that would be helpful for continuing the conversation, including what we did, what we're doing, which files we're working on, and what we're going to do next.\n\nConversation to summarize:\n\n${messagesToSummarize.map(m => `${m.role.toUpperCase()}: ${m.content}`).join('\n\n')}`,
    },
  ];
}

// Tool event handler - updates tool execution state
function handleToolEvent(event: ToolEvent): void {
  const state = toolExecutionState.value;
  
  if (event.type === "tool_loop_progress") {
    state.iteration = event.iteration ?? 0;
    state.maxIterations = event.max_iterations ?? 5;
  } else if (event.type === "tool_calls_received") {
    // Add new tool calls to state
    if (event.tools) {
      for (const tool of event.tools) {
        const existingCall = state.toolCalls.find(c => c.name === tool.name);
        if (!existingCall) {
          state.toolCalls.push({
            id: `call_${tool.name}_${state.toolCalls.length}`,
            name: tool.name,
            arguments: tool.arguments ?? {},
            status: "pending",
            expanded: false,
          });
        }
      }
    }
  } else if (event.type === "tool_call_start") {
    // Mark tool as running
    const call = state.toolCalls.find(c => c.name === event.name);
    if (call) {
      call.status = "running";
    } else if (event.name) {
      // Add if not already tracked
      state.toolCalls.push({
        id: event.id ?? `call_${event.name}`,
        name: event.name,
        arguments: event.arguments ?? {},
        status: "running",
        expanded: false,
      });
    }
  } else if (event.type === "tool_call_end") {
    // Mark tool as complete
    const call = state.toolCalls.find(c => c.name === event.name);
    if (call) {
      call.status = event.success ? "success" : "failed";
      call.duration_ms = event.duration_ms;
      call.result = event.result_preview;
      call.error = event.success ? undefined : event.result_preview;
    }
  } else if (event.type === "tool_loop_complete") {
    state.isComplete = true;
    state.totalTimeMs = event.total_tool_time_ms ?? 0;
  }
  
  scrollToBottom();
}

// Agent event handler - creates separate messages for each agent
function handleAgentEvent(event: AgentEvent): void {
  if (event.type === "agent_start") {
    // Agent is starting - create a NEW message for this agent
    currentAgent.value = event.agent;
    agentResponses.value[event.agent] = "";
    
    // Create a new message for this agent
    const agentMsg: ChatMessage = {
      role: "assistant",
      content: "",
      reasoning: "",
      timestamp: Date.now(),
      isStreaming: true,
      showReasoning: true,
      agentName: event.agent,
      agentChain: event.chain,  // Store the delegation chain
      toolExecution: event.depth === 0 ? toolExecutionState.value : undefined,  // Only track tools for top-level agent
      toolExecutionExpanded: false,
    };
    messages.value.push(agentMsg);
    agentMessageIndices.value[event.agent] = messages.value.length - 1;
    
  } else if (event.type === "agent_end") {
    // Agent finished - mark its message as complete
    const msgIndex = agentMessageIndices.value[event.agent];
    if (msgIndex !== undefined && messages.value[msgIndex]) {
      messages.value[msgIndex].isStreaming = false;
      messages.value[msgIndex].timestamp = Date.now();
      
      // If this agent had tool calls, mark tool execution as complete
      if (toolExecutionState.value.toolCalls.length > 0 && event.depth === 0) {
        toolExecutionState.value.isComplete = true;
        messages.value[msgIndex].toolExecution = toolExecutionState.value;
      }
      
      // Hide reasoning section if empty
      if (!messages.value[msgIndex].reasoning) {
        messages.value[msgIndex].showReasoning = false;
      }
    }
    currentAgent.value = null;
  }
  
  scrollToBottom();
}

// Compaction: summarize and replace old messages when context is 95% full (like OpenCode)
// Uses STREAMING for summary generation (like OpenCode does)
async function compactMessages(): Promise<boolean> {
  // Check if we're at 95% threshold (OpenCode's approach)
  if (actualTokens.value >= contextSize.value * 0.95 && messages.value.length > 4) {
    // Keep the last few messages (2 exchanges = 4 messages)
    const keepCount = 4;
    const messagesToSummarize = messages.value.slice(0, -keepCount);
    
    if (messagesToSummarize.length === 0) {
      return false;
    }

    // Create compaction message that will be streamed into
    // This will be added to the END of the messages array after streaming
    const compactionMsg: ChatMessage = {
      role: "system",
      content: "",
      timestamp: Date.now(),
      isCompacting: true,
      isStreaming: true,
    };
    
    // Get the kept messages first (the last keepCount messages)
    const keptMessages = messages.value.slice(-keepCount);
    
    // Temporarily set messages to just the kept messages
    messages.value = [...keptMessages];
    
    // Add compaction message at the END so it's visible during streaming
    messages.value.push(compactionMsg);
    const compactionMsgIndex = messages.value.length - 1;
    
    // Start compaction progress
    isCompacting.value = true;
    compactionStep.value = 1;
    compactionMessage.value = `Found ${messagesToSummarize.length} messages to summarize`;
    scrollToBottom();
    
    try {
      // Step 1: Analyze (already done above)
      await new Promise(r => setTimeout(r, 500));  // Small delay for visual feedback
      compactionStep.value = 2;
      compactionMessage.value = "Generating summary...";
      scrollToBottom();
      
      // Step 2: Generate summary using STREAMING - stream directly into the message
      const summarizationMessages = buildSummarizationMessages(messagesToSummarize);
      
      await streamChatCompletion(
        selectedModel.value,
        summarizationMessages,
        (chunk: string) => {
          // Stream directly into the compaction message at the specific index
          // This ensures Vue's reactivity detects the change
          messages.value[compactionMsgIndex].content += chunk;
          scrollToBottom();
        },
        () => {
          // Stream complete - summary now fully generated
        },
        (err: string) => {
          throw new Error(err);
        }
      );
      
      // Step 3: Apply changes
      compactionStep.value = 3;
      compactionMessage.value = `Replacing ${messagesToSummarize.length} messages with summary`;
      scrollToBottom();
      
      // Small delay before finalizing
      await new Promise(r => setTimeout(r, 500));
      
      // Mark compaction as complete (stops showing progress steps)
      messages.value[compactionMsgIndex].isCompacting = false;
      messages.value[compactionMsgIndex].isStreaming = false;
      messages.value[compactionMsgIndex].content = messages.value[compactionMsgIndex].content.trim();
      messages.value[compactionMsgIndex].timestamp = Date.now();
      
      // The summary is already at the end of the array, no need to rebuild
      
      // Reset token counts after compaction
      promptTokens.value = 0;
      completionTokens.value = 0;
      
      // Finish compaction - clear progress state
      isCompacting.value = false;
      compactionStep.value = 0;
      compactionMessage.value = "";
      
      // Scroll to show the new summary message
      scrollToBottom();
      
      // Focus the input textarea for next message
      nextTick(() => {
        inputTextarea.value?.focus();
      });
      return true;
    } catch (e) {
      // Remove the failed compaction message from the end
      messages.value.pop();
      isCompacting.value = false;
      compactionStep.value = 0;
      compactionMessage.value = "";
      error.value = `Compaction failed: ${e instanceof Error ? e.message : 'Unknown error'}`;
      return false;
    }
  }
  
  return false;
}

// Send message
async function sendMessage(): Promise<void> {
  if (!selectedModel.value || !inputMessage.value.trim() || isStreaming.value || isCompacting.value) {
    return;
  }

  error.value = "";

  // Reset tool execution state for new message
  toolExecutionState.value = {
    iteration: 0,
    maxIterations: 5,
    toolCalls: [],
    isComplete: false,
    totalTimeMs: 0,
  };

  // Reset agent state for new message
  currentAgent.value = null;
  agentResponses.value = {};
  agentMessageIndices.value = {};

  // Add user message
  const userMsg: ChatMessage = {
    role: "user",
    content: inputMessage.value.trim(),
    timestamp: Date.now(),
  };
  messages.value.push(userMsg);
  inputMessage.value = "";
  scrollToBottom();

  // Prepare messages for API (format for OpenAI)
  const apiMessages = messages.value.map(m => ({
    role: m.role,
    content: m.content,
  }));

  // DON'T create assistant message placeholder upfront
  // Messages will be created by handleAgentEvent when AGENT_START events come
  // If no AGENT_START events come (non-agent request), we'll create a fallback message

  // Start streaming
  isStreaming.value = true;
  let hasReceivedAgentStart = false;

  try {
    await streamChatCompletion(
      selectedModel.value,
      apiMessages,
      (chunk: string) => {
        // If no agent started yet, create a fallback message
        if (!hasReceivedAgentStart && !currentAgent.value) {
          hasReceivedAgentStart = true;
          const fallbackMsg: ChatMessage = {
            role: "assistant",
            content: "",
            reasoning: "",
            timestamp: Date.now(),
            isStreaming: true,
            showReasoning: true,
            toolExecution: toolExecutionState.value,
            toolExecutionExpanded: false,
          };
          messages.value.push(fallbackMsg);
          agentMessageIndices.value["fallback"] = messages.value.length - 1;
        }
        
        // Stream content into the current agent's message
        const currentAgentName = currentAgent.value || "fallback";
        const msgIndex = agentMessageIndices.value[currentAgentName];
        if (msgIndex !== undefined && messages.value[msgIndex]) {
          messages.value[msgIndex].content += chunk;
          agentResponses.value[currentAgentName] += chunk;
        }
        scrollToBottom();
      },
      () => {
        // Stream complete - handle all agent messages
        // Mark any still-streaming messages as complete
        messages.value.forEach(msg => {
          if (msg.isStreaming) {
            msg.isStreaming = false;
            msg.timestamp = Date.now();
          }
          
          // Hide reasoning section if empty
          if (!msg.reasoning) {
            msg.showReasoning = false;
          }
          
          // Mark tool execution as complete if there were tool calls
          if (msg.toolExecution && msg.toolExecution.toolCalls.length > 0) {
            msg.toolExecution.isComplete = true;
          }
        });
        
        // Clear agent state
        currentAgent.value = null;
        
        isStreaming.value = false;
        scrollToBottom();
        
        // Estimate tokens for the new messages
        const userTokens = Math.ceil(userMsg.content.length / 4);
        let totalAssistantTokens = 0;
        let totalReasoningTokens = 0;
        
        // Sum tokens from all agent messages added in this response
        Object.values(agentMessageIndices.value).forEach(idx => {
          if (messages.value[idx]) {
            totalAssistantTokens += Math.ceil(messages.value[idx].content.length / 4);
            totalReasoningTokens += Math.ceil((messages.value[idx].reasoning?.length || 0) / 4);
          }
        });
        
        promptTokens.value += userTokens;
        completionTokens.value += totalAssistantTokens + totalReasoningTokens;
        
        // Trigger compaction AFTER response completes (like OpenCode)
        compactMessages();
        
        // Focus the input textarea for next message
        nextTick(() => {
          inputTextarea.value?.focus();
        });
      },
      (err: string) => {
        // On error, remove all messages added for this request
        // Remove fallback message if it exists
        if (agentMessageIndices.value["fallback"] !== undefined) {
          messages.value.splice(agentMessageIndices.value["fallback"], 1);
        }
        // Remove all agent messages
        Object.values(agentMessageIndices.value).forEach(idx => {
          if (idx !== agentMessageIndices.value["fallback"]) {
            messages.value.splice(idx, 1);
          }
        });
        error.value = err;
        isStreaming.value = false;
      },
      (reasoningChunk: string) => {
        // Stream reasoning content to current agent's message
        const currentAgentName = currentAgent.value || "fallback";
        const msgIndex = agentMessageIndices.value[currentAgentName];
        if (msgIndex !== undefined && messages.value[msgIndex]) {
          messages.value[msgIndex].reasoning += reasoningChunk;
        }
        scrollToBottom();
      },
      (toolEvent: ToolEvent) => {
        // Handle tool execution events
        handleToolEvent(toolEvent);
        // Update the tool execution state on the current agent's message
        const currentAgentName = currentAgent.value || "fallback";
        const msgIndex = agentMessageIndices.value[currentAgentName];
        if (msgIndex !== undefined && messages.value[msgIndex]) {
          messages.value[msgIndex].toolExecution = toolExecutionState.value;
        }
      },
      (agentEvent: AgentEvent) => {
        // Handle agent start/end events
        hasReceivedAgentStart = true;
        handleAgentEvent(agentEvent);
      }
    );
  } catch (e) {
    // On error, remove the failed assistant message
    messages.value.pop();
    error.value = e instanceof Error ? e.message : "Failed to send message";
    isStreaming.value = false;
  }
}

// Toggle reasoning visibility
function toggleReasoning(msg: ChatMessage): void {
  msg.showReasoning = !msg.showReasoning;
}

// Toggle tool result expansion
function toggleToolResult(call: ToolCallInfo): void {
  call.expanded = !call.expanded;
}

// Toggle full tool execution details
function toggleToolExecution(msg: ChatMessage): void {
  msg.toolExecutionExpanded = !msg.toolExecutionExpanded;
}

// Clear chat
function clearChat(): void {
  messages.value = [];
  error.value = "";
  promptTokens.value = 0;
  completionTokens.value = 0;
}

// Load upstreams on mount
async function loadUpstreams() {
  loadingModels.value = true;
  try {
    const result = await getUpstreams();
    upstreams.value = result.upstreams;
    
    // Auto-select first enabled model if available
    if (enabledUpstreams.value.length > 0 && !selectedModel.value) {
      selectedModel.value = enabledUpstreams.value[0].name;
    }
  } catch (e) {
    error.value = e instanceof Error ? e.message : "Failed to load models";
  } finally {
    loadingModels.value = false;
  }
}

onMounted(loadUpstreams);
</script>

<style scoped>
.chat-page {
  max-width: 800px;
}

.settings-card {
  margin-bottom: 1rem;
}

.settings-row {
  display: flex;
  gap: 1rem;
  align-items: flex-end;
  flex-wrap: wrap;
}

.settings-row .form-group {
  flex: 1;
  min-width: 200px;
}

.token-info {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  min-width: 150px;
}

.token-count {
  font-size: 0.85rem;
  color: #718096;
}

.token-count.token-warning {
  color: #e53e3e;
  font-weight: 600;
}

.warning-icon {
  margin-left: 0.25rem;
}

.token-bar {
  height: 8px;
  background-color: #e2e8f0;
  border-radius: 4px;
  overflow: hidden;
}

.token-bar-fill {
  height: 100%;
  transition: width 0.3s ease;
}

.token-bar.normal .token-bar-fill {
  background-color: #48bb78;
}

.token-bar.warning .token-bar-fill {
  background-color: #ed8936;
}

.token-bar.critical .token-bar-fill {
  background-color: #e53e3e;
}

.messages-card {
  display: flex;
  flex-direction: column;
  min-height: 400px;
}

.messages-container {
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
  background-color: #f7fafc;
  border-radius: 4px;
  margin-bottom: 1rem;
  max-height: 500px;
}

.empty-chat {
  text-align: center;
  color: #718096;
  padding: 2rem;
}

.empty-chat .hint {
  font-size: 0.85rem;
  color: #a0aec0;
  margin-top: 0.5rem;
}

.message {
  margin-bottom: 1rem;
  max-width: 85%;
}

.message.user {
  margin-left: auto;
}

.message.user .message-content {
  background-color: #667eea;
  color: white;
}

.message.assistant .message-content {
  background-color: white;
  color: #2c3e50;
  border: 1px solid #e2e8f0;
}

.message.system {
  max-width: 100%;
}

/* System summary box - matches streaming summary style */
.system-summary-box {
  background-color: #fff7e6;
  border: 2px solid #ffc53d;
  border-radius: 8px;
  padding: 1rem;
  margin-bottom: 1.5rem;
}

.system-summary-box.compacting {
  border-color: #ffc53d;
}

.system-summary-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.75rem;
}

.system-summary-label {
  font-weight: bold;
  color: #ad6800;
  font-size: 0.95rem;
}

.system-summary-time {
  color: #d69e2e;
  font-size: 0.8rem;
}

.system-summary-content {
  color: #744210;
  font-size: 0.9rem;
  line-height: 1.5;
  white-space: pre-wrap;
}

.compaction-progress {
  margin-top: 0.5rem;
}

.message.compaction {
  max-width: 100%;
}

.message.compaction .message-content {
  background-color: #fff7e6;
  color: #ad6800;
  border: 2px solid #ffc53d;
}

.compaction-icon {
  font-size: 1rem;
}

.summary-icon {
  font-size: 0.85rem;
}

.compaction-progress {
  padding: 1rem;
}

.compaction-steps {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.step {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: #a0aec0;
  font-size: 0.85rem;
}

.step.active {
  color: #ad6800;
  font-weight: 500;
}

.step.done {
  color: #38a169;
}

.step-icon {
  width: 1.5rem;
  text-align: center;
}

.compaction-detail {
  margin-top: 0.75rem;
  padding-top: 0.75rem;
  border-top: 1px dashed #ffc53d;
  font-size: 0.85rem;
  color: #718096;
}

/* Streaming summary during compaction - matches final system summary style */
.streaming-summary {
  margin-top: 1rem;
  padding: 1rem;
  background-color: #fff7e6;
  border-radius: 8px;
  border: 2px solid #ffc53d;
}

.streaming-summary-label {
  font-size: 0.95rem;
  font-weight: bold;
  color: #ad6800;
  margin-bottom: 0.75rem;
}

.streaming-summary-text {
  font-size: 0.9rem;
  color: #744210;
  line-height: 1.5;
  white-space: pre-wrap;
  word-wrap: break-word;
}

.message-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 0.25rem;
  font-size: 0.75rem;
  color: #718096;
}

.message.user .message-header {
  justify-content: flex-end;
}

.message.compaction .message-header {
  color: #ad6800;
}

.message-role {
  font-weight: 500;
}

.message-time {
  opacity: 0.7;
}

.message-content {
  padding: 0.75rem 1rem;
  border-radius: 8px;
  white-space: pre-wrap;
  word-wrap: break-word;
}

/* Reasoning section styles */
.reasoning-section {
  margin-bottom: 0.5rem;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  overflow: hidden;
}

.reasoning-toggle {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  width: 100%;
  padding: 0.5rem 0.75rem;
  background-color: #f7fafc;
  border: none;
  cursor: pointer;
  font-size: 0.85rem;
  color: #4a5568;
  transition: background-color 0.2s;
}

.reasoning-toggle:hover:not(:disabled) {
  background-color: #edf2f7;
}

.reasoning-toggle:disabled {
  cursor: default;
}

.toggle-icon {
  font-size: 0.75rem;
  color: #718096;
}

.toggle-label {
  font-weight: 500;
}

.reasoning-content {
  padding: 0.75rem;
  background-color: #f7fafc;
  border-top: 1px solid #e2e8f0;
  font-size: 0.85rem;
  color: #4a5568;
  white-space: pre-wrap;
  word-wrap: break-word;
  line-height: 1.5;
}

.streaming-indicator {
  display: inline-block;
  width: 8px;
  height: 8px;
  background-color: #667eea;
  border-radius: 50%;
  animation: pulse 1s infinite;
}

.streaming-indicator-small {
  display: inline-block;
  width: 6px;
  height: 6px;
  background-color: #718096;
  border-radius: 50%;
  animation: pulse 1s infinite;
}

/* Tool execution section styles */
.tool-execution-section {
  margin-bottom: 0.5rem;
  background-color: #e6f7ff;
  border: 1px solid #91d5ff;
  border-radius: 6px;
  padding: 0.5rem;
}

.tool-execution-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
}

.tool-execution-label {
  font-weight: 600;
  color: #0050b3;
  font-size: 0.85rem;
}

.tool-iteration {
  color: #69c0ff;
  font-size: 0.75rem;
}

.tool-calls-list {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.tool-call-item {
  padding: 0.5rem;
  background-color: white;
  border-radius: 4px;
  border-left: 3px solid #91d5ff;
}

.tool-call-item.pending {
  border-left-color: #d9d9d9;
}

.tool-call-item.running {
  border-left-color: #69c0ff;
  background-color: #f0faff;
}

.tool-call-item.success {
  border-left-color: #52c41a;
  background-color: #f6ffed;
}

.tool-call-item.failed {
  border-left-color: #ff4d4f;
  background-color: #fff2f0;
}

.tool-call-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.8rem;
}

.tool-call-icon {
  width: 1rem;
  text-align: center;
  font-size: 0.7rem;
}

.tool-call-name {
  font-weight: 500;
  color: #262626;
}

.tool-call-duration {
  color: #8c8c8c;
  font-size: 0.7rem;
}

.tool-call-args {
  margin-top: 0.25rem;
  font-size: 0.75rem;
  color: #595959;
  background-color: #fafafa;
  padding: 0.25rem 0.5rem;
  border-radius: 2px;
}

.tool-call-result {
  margin-top: 0.25rem;
  font-size: 0.75rem;
  color: #52c41a;
  background-color: #f6ffed;
  padding: 0.25rem 0.5rem;
  border-radius: 2px;
}

.tool-call-error {
  margin-top: 0.25rem;
  font-size: 0.75rem;
  color: #ff4d4f;
  background-color: #fff2f0;
  padding: 0.25rem 0.5rem;
  border-radius: 2px;
}

.tool-expand-btn {
  background: none;
  border: none;
  cursor: pointer;
  font-size: 0.7rem;
  color: #8c8c8c;
  padding: 0;
  margin-left: auto;
}

.tool-expand-btn:hover {
  color: #0050b3;
}

.tool-spinner {
  display: inline-block;
  width: 8px;
  height: 8px;
  border: 2px solid #69c0ff;
  border-top-color: transparent;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

.tool-execution-summary {
  margin-bottom: 0.25rem;
  padding: 0.25rem 0.5rem;
  background-color: #f6ffed;
  border-radius: 4px;
  font-size: 0.75rem;
  color: #52c41a;
}

.tool-summary-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.tool-summary-label {
  font-weight: 500;
}

.tool-summary-details {
  margin-top: 0.5rem;
  padding: 0.5rem;
  background-color: #f6ffed;
  border-radius: 4px;
}

.tool-call-result-full {
  margin-top: 0.25rem;
  font-size: 0.75rem;
  color: #52c41a;
  background-color: #f6ffed;
  padding: 0.25rem 0.5rem;
  border-radius: 2px;
  white-space: pre-wrap;
  word-wrap: break-word;
}

.tool-call-result-preview {
  margin-top: 0.25rem;
  font-size: 0.75rem;
  color: #52c41a;
  background-color: #f6ffed;
  padding: 0.25rem 0.5rem;
  border-radius: 2px;
}

/* Agent badge styles */
.message-role {
  font-weight: 500;
}

/* Agent emoji indicator */
.message-role::first-line {
  line-height: 1.2;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.input-area {
  display: flex;
  gap: 0.5rem;
  align-items: flex-end;
}

.input-area textarea {
  flex: 1;
  padding: 0.75rem;
  border: 1px solid #e2e8f0;
  border-radius: 4px;
  font-size: 1rem;
  resize: none;
  min-height: 60px;
}

.input-area textarea:focus {
  outline: none;
  border-color: #667eea;
  box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
}

.input-area textarea:disabled {
  background-color: #f7fafc;
  color: #a0aec0;
}

.send-btn, .clear-btn {
  padding: 0.75rem 1.5rem;
  height: 60px;
}

.form-group label {
  display: block;
  margin-bottom: 0.25rem;
  font-weight: 500;
  color: #4a5568;
}

.form-group select,
.form-group input[type="number"] {
  width: 100%;
  padding: 0.5rem;
  border: 1px solid #e2e8f0;
  border-radius: 4px;
  font-size: 1rem;
}

.form-group select:focus,
.form-group input[type="number"]:focus {
  outline: none;
  border-color: #667eea;
  box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
}

.form-group select:disabled,
.form-group input[type="number"]:disabled {
  background-color: #f7fafc;
  color: #a0aec0;
}

.error {
  padding: 0.75rem 1rem;
  background-color: #fed7d7;
  color: #c53030;
  border-radius: 4px;
  margin-top: 0.5rem;
  font-size: 0.85rem;
}
</style>