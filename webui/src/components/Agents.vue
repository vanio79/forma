<template>
  <div class="agents-page">
    <div class="page-header">
      <h2>Agents Management</h2>
      <button class="btn btn-primary" @click="showCreateModal = true" :disabled="loading">
        + New Agent
      </button>
    </div>

    <div v-if="loading" class="loading">Loading agents...</div>
    
    <div v-else-if="agents.length === 0" class="empty-state">
      <p>No agents configured.</p>
      <p class="hint">Agents can route requests to specific AI instances with custom prompts and tool permissions.</p>
    </div>

    <div v-else class="agents-list">
      <div v-for="agent in agents" :key="agent.id" class="agent-card">
        <div class="agent-header">
          <div class="agent-title-row">
            <h3 class="agent-name">{{ agent.name }}</h3>
            <span :class="['status-badge', agent.is_enabled ? 'enabled' : 'disabled']">
              {{ agent.is_enabled ? 'Enabled' : 'Disabled' }}
            </span>
          </div>
          <div class="agent-purpose">{{ agent.purpose }}</div>
        </div>

        <div class="agent-details">
          <div class="detail-row">
            <span class="detail-label">Upstream:</span>
            <span class="detail-value">{{ getUpstreamName(agent.upstream_id) || 'Default' }}</span>
          </div>
          
          <div class="detail-row">
            <span class="detail-label">Tools:</span>
            <span class="detail-value">
              {{ agent.tools_enabled ? 'Enabled' : 'Disabled' }}
              <span v-if="agent.tools_enabled && agent.tool_whitelist.length > 0" class="tool-count">
                ({{ agent.tool_whitelist.length }} whitelisted)
              </span>
            </span>
          </div>
          
          <div class="detail-row">
            <span class="detail-label">Max Iterations:</span>
            <span class="detail-value">{{ agent.max_iterations }}</span>
          </div>
        </div>

        <div class="agent-instruction">
          <div class="instruction-label">Instruction Prompt:</div>
          <div class="instruction-preview">
            {{ agent.instruction_prompt.length > 200 ? agent.instruction_prompt.slice(0, 200) + '...' : agent.instruction_prompt }}
          </div>
        </div>

        <div class="agent-actions">
          <button class="btn btn-secondary btn-sm" @click="editAgent(agent)">
            Edit
          </button>
          <button class="btn btn-danger btn-sm" @click="confirmDelete(agent)">
            Delete
          </button>
        </div>
      </div>
    </div>

    <!-- Create/Edit Modal -->
    <div v-if="showCreateModal || showEditModal" class="modal-overlay" @click.self="closeModal">
      <div class="modal">
        <div class="modal-header">
          <h3>{{ showEditModal ? 'Edit Agent' : 'Create New Agent' }}</h3>
          <button class="modal-close" @click="closeModal">×</button>
        </div>

        <div class="modal-body">
          <div class="form-group">
            <label for="agentName">Name *</label>
            <input 
              id="agentName"
              v-model="formData.name"
              type="text"
              placeholder="e.g., researcher, coder, assistant"
              required
            />
          </div>

          <div class="form-group">
            <label for="agentPurpose">Purpose *</label>
            <input 
              id="agentPurpose"
              v-model="formData.purpose"
              type="text"
              placeholder="e.g., Search for information and synthesize findings"
              required
            />
          </div>

          <div class="form-group">
            <label for="agentInstruction">Instruction Prompt *</label>
            <textarea 
              id="agentInstruction"
              v-model="formData.instruction_prompt"
              rows="5"
              placeholder="Detailed instructions for the agent..."
              required
            ></textarea>
          </div>

          <div class="form-group">
            <label for="agentUpstream">Upstream</label>
            <select id="agentUpstream" v-model="formData.upstream_id">
              <option value="">Default (use request model)</option>
              <option v-for="upstream in upstreams" :key="upstream.id" :value="upstream.id">
                {{ upstream.name }} ({{ upstream.upstream_model || upstream.name }})
              </option>
            </select>
            <div class="hint">Select a specific upstream for this agent, or leave empty to use the request's model.</div>
          </div>

          <div class="form-group">
            <label for="agentToolsEnabled">Enable Tools</label>
            <select id="agentToolsEnabled" v-model="formData.tools_enabled">
              <option :value="true">Enabled</option>
              <option :value="false">Disabled</option>
            </select>
          </div>

          <div v-if="formData.tools_enabled" class="form-group">
            <label for="agentToolWhitelist">Tool Whitelist (JSON array)</label>
            <textarea 
              id="agentToolWhitelist"
              v-model="toolWhitelistJson"
              rows="3"
              placeholder='["read_file", "bash", "grep"]'
            ></textarea>
            <div class="hint">List of tool names this agent can use. Empty array = all tools allowed.</div>
          </div>

          <div class="form-group">
            <label for="agentMaxIterations">Max Tool Iterations</label>
            <input 
              id="agentMaxIterations"
              v-model.number="formData.max_iterations"
              type="number"
              min="1"
              max="20"
            />
          </div>

          <div class="form-group">
            <label for="agentEnabled">Status</label>
            <select id="agentEnabled" v-model="formData.is_enabled">
              <option :value="true">Enabled</option>
              <option :value="false">Disabled</option>
            </select>
          </div>
        </div>

        <div class="modal-footer">
          <button class="btn btn-secondary" @click="closeModal">Cancel</button>
          <button 
            class="btn btn-primary" 
            @click="submitForm"
            :disabled="!formData.name || !formData.purpose || !formData.instruction_prompt"
          >
            {{ showEditModal ? 'Save Changes' : 'Create Agent' }}
          </button>
        </div>

        <div v-if="error" class="error">{{ error }}</div>
      </div>
    </div>

    <!-- Delete Confirmation Modal -->
    <div v-if="showDeleteModal" class="modal-overlay" @click.self="showDeleteModal = false">
      <div class="modal modal-sm">
        <div class="modal-header">
          <h3>Delete Agent</h3>
          <button class="modal-close" @click="showDeleteModal = false">×</button>
        </div>

        <div class="modal-body">
          <p>Are you sure you want to delete agent <strong>{{ deleteTarget?.name }}</strong>?</p>
          <p class="warning">This action cannot be undone.</p>
        </div>

        <div class="modal-footer">
          <button class="btn btn-secondary" @click="showDeleteModal = false">Cancel</button>
          <button class="btn btn-danger" @click="deleteAgentConfirmed">Delete</button>
        </div>

        <div v-if="error" class="error">{{ error }}</div>
      </div>
    </div>

    <!-- Discovery Info Card -->
    <div class="card discovery-card">
      <div class="discovery-header">
        <h4>Agent Discovery</h4>
        <span class="info-badge">ℹ️</span>
      </div>
      <p class="discovery-info">
        When agents are enabled, requests are augmented with agent discovery information.
        Users can route messages to specific agents using <code>@agent_name</code> syntax.
      </p>
      <div class="discovery-example">
        <div class="example-label">Example:</div>
        <div class="example-text">@researcher Find information about quantum computing</div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from "vue";
import type { Agent, Upstream } from "../types";
import { getAgents, createAgent, updateAgent, deleteAgent, getUpstreams } from "../api";

const agents = ref<Agent[]>([]);
const upstreams = ref<Upstream[]>([]);
const loading = ref(true);
const error = ref("");

// Modal states
const showCreateModal = ref(false);
const showEditModal = ref(false);
const showDeleteModal = ref(false);
const deleteTarget = ref<Agent | null>(null);
const editTargetId = ref<string | null>(null);

// Form data
const formData = ref({
  name: "",
  purpose: "",
  instruction_prompt: "",
  upstream_id: "",
  tools_enabled: false,
  tool_whitelist: [] as string[],
  max_iterations: 5,
  is_enabled: true,
});

// Tool whitelist as JSON string for textarea
const toolWhitelistJson = computed({
  get: () => JSON.stringify(formData.value.tool_whitelist),
  set: (value: string) => {
    try {
      formData.value.tool_whitelist = JSON.parse(value);
    } catch {
      // Keep previous value if parse fails
    }
  },
});

// Load agents and upstreams
async function loadData() {
  loading.value = true;
  error.value = "";
  
  try {
    const [agentsResult, upstreamsResult] = await Promise.all([
      getAgents(),
      getUpstreams(),
    ]);
    
    agents.value = agentsResult.agents;
    upstreams.value = upstreamsResult.upstreams;
  } catch (e) {
    error.value = e instanceof Error ? e.message : "Failed to load data";
  } finally {
    loading.value = false;
  }
}

// Get upstream name by ID
function getUpstreamName(upstreamId: string | null): string | null {
  if (!upstreamId) return null;
  const upstream = upstreams.value.find(u => u.id === upstreamId);
  return upstream?.name || null;
}

// Edit agent
function editAgent(agent: Agent) {
  editTargetId.value = agent.id;
  formData.value = {
    name: agent.name,
    purpose: agent.purpose,
    instruction_prompt: agent.instruction_prompt,
    upstream_id: agent.upstream_id || "",
    tools_enabled: agent.tools_enabled,
    tool_whitelist: agent.tool_whitelist,
    max_iterations: agent.max_iterations,
    is_enabled: agent.is_enabled,
  };
  showEditModal.value = true;
}

// Confirm delete
function confirmDelete(agent: Agent) {
  deleteTarget.value = agent;
  showDeleteModal.value = true;
}

// Close modal
function closeModal() {
  showCreateModal.value = false;
  showEditModal.value = false;
  editTargetId.value = null;
  error.value = "";
  resetForm();
}

// Reset form
function resetForm() {
  formData.value = {
    name: "",
    purpose: "",
    instruction_prompt: "",
    upstream_id: "",
    tools_enabled: false,
    tool_whitelist: [],
    max_iterations: 5,
    is_enabled: true,
  };
}

// Submit form
async function submitForm() {
  error.value = "";
  
  if (!formData.value.name || !formData.value.purpose || !formData.value.instruction_prompt) {
    error.value = "Name, purpose, and instruction prompt are required";
    return;
  }
  
  try {
    const params = {
      name: formData.value.name,
      purpose: formData.value.purpose,
      instruction_prompt: formData.value.instruction_prompt,
      upstream_id: formData.value.upstream_id || undefined,
      tools_enabled: formData.value.tools_enabled,
      tool_whitelist: formData.value.tool_whitelist,
      max_iterations: formData.value.max_iterations,
      is_enabled: formData.value.is_enabled,
    };
    
    if (showEditModal.value && editTargetId.value) {
      await updateAgent(editTargetId.value, params);
    } else {
      await createAgent(params);
    }
    
    closeModal();
    await loadData();
  } catch (e) {
    error.value = e instanceof Error ? e.message : "Failed to save agent";
  }
}

// Delete agent confirmed
async function deleteAgentConfirmed() {
  if (!deleteTarget.value) return;
  
  error.value = "";
  
  try {
    await deleteAgent(deleteTarget.value.id);
    showDeleteModal.value = false;
    deleteTarget.value = null;
    await loadData();
  } catch (e) {
    error.value = e instanceof Error ? e.message : "Failed to delete agent";
  }
}

onMounted(loadData);
</script>

<style scoped>
.agents-page {
  max-width: 1000px;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
}

.page-header h2 {
  margin: 0;
}

.loading,
.empty-state {
  text-align: center;
  color: #718096;
  padding: 2rem;
}

.empty-state .hint {
  font-size: 0.85rem;
  color: #a0aec0;
  margin-top: 0.5rem;
}

.agents-list {
  display: grid;
  gap: 1rem;
}

.agent-card {
  background: white;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 1.25rem;
}

.agent-header {
  margin-bottom: 1rem;
}

.agent-title-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
}

.agent-name {
  font-size: 1.2rem;
  font-weight: 600;
  margin: 0;
  color: #2d3748;
}

.status-badge {
  padding: 0.25rem 0.75rem;
  border-radius: 12px;
  font-size: 0.75rem;
  font-weight: 500;
}

.status-badge.enabled {
  background-color: #c6f6d5;
  color: #276749;
}

.status-badge.disabled {
  background-color: #e2e8f0;
  color: #718096;
}

.agent-purpose {
  color: #4a5568;
  font-size: 0.9rem;
}

.agent-details {
  display: grid;
  gap: 0.5rem;
  margin-bottom: 1rem;
  padding: 0.75rem;
  background-color: #f7fafc;
  border-radius: 4px;
}

.detail-row {
  display: flex;
  gap: 0.5rem;
  font-size: 0.85rem;
}

.detail-label {
  color: #718096;
  font-weight: 500;
  min-width: 100px;
}

.detail-value {
  color: #2d3748;
}

.tool-count {
  color: #718096;
  font-size: 0.8rem;
}

.agent-instruction {
  margin-bottom: 1rem;
}

.instruction-label {
  font-size: 0.85rem;
  color: #718096;
  font-weight: 500;
  margin-bottom: 0.25rem;
}

.instruction-preview {
  color: #4a5568;
  font-size: 0.85rem;
  padding: 0.5rem;
  background-color: #f7fafc;
  border-radius: 4px;
  white-space: pre-wrap;
  word-wrap: break-word;
}

.agent-actions {
  display: flex;
  gap: 0.5rem;
}

.btn-sm {
  padding: 0.4rem 1rem;
  font-size: 0.85rem;
}

.btn-danger {
  background-color: #e53e3e;
  color: white;
}

.btn-danger:hover {
  background-color: #c53030;
}

/* Modal styles */
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.modal {
  background: white;
  border-radius: 8px;
  max-width: 600px;
  width: 90%;
  max-height: 90vh;
  overflow-y: auto;
}

.modal-sm {
  max-width: 400px;
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 1.5rem;
  border-bottom: 1px solid #e2e8f0;
}

.modal-header h3 {
  margin: 0;
  font-size: 1.1rem;
}

.modal-close {
  background: none;
  border: none;
  font-size: 1.5rem;
  color: #718096;
  cursor: pointer;
  padding: 0;
  line-height: 1;
}

.modal-close:hover {
  color: #2d3748;
}

.modal-body {
  padding: 1.5rem;
}

.modal-footer {
  display: flex;
  gap: 0.75rem;
  justify-content: flex-end;
  padding: 1rem 1.5rem;
  border-top: 1px solid #e2e8f0;
}

.warning {
  color: #c53030;
  font-size: 0.85rem;
  margin-top: 0.5rem;
}

/* Form styles */
.form-group {
  margin-bottom: 1rem;
}

.form-group label {
  display: block;
  margin-bottom: 0.25rem;
  font-weight: 500;
  color: #4a5568;
}

.form-group input[type="text"],
.form-group input[type="number"],
.form-group select,
.form-group textarea {
  width: 100%;
  padding: 0.5rem;
  border: 1px solid #e2e8f0;
  border-radius: 4px;
  font-size: 0.95rem;
}

.form-group input[type="text"]:focus,
.form-group input[type="number"]:focus,
.form-group select:focus,
.form-group textarea:focus {
  outline: none;
  border-color: #667eea;
  box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
}

.form-group textarea {
  resize: vertical;
  min-height: 80px;
}

.hint {
  font-size: 0.8rem;
  color: #a0aec0;
  margin-top: 0.25rem;
}

.error {
  padding: 0.75rem 1rem;
  background-color: #fed7d7;
  color: #c53030;
  border-radius: 4px;
  margin-top: 0.5rem;
  font-size: 0.85rem;
}

/* Discovery info card */
.discovery-card {
  margin-top: 2rem;
  padding: 1rem 1.5rem;
}

.discovery-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.75rem;
}

.discovery-header h4 {
  margin: 0;
  font-size: 1rem;
}

.info-badge {
  font-size: 1rem;
  color: #667eea;
}

.discovery-info {
  color: #4a5568;
  font-size: 0.9rem;
  line-height: 1.5;
}

.discovery-info code {
  background-color: #edf2f7;
  padding: 0.15rem 0.4rem;
  border-radius: 3px;
  font-family: monospace;
  font-size: 0.85rem;
  color: #667eea;
}

.discovery-example {
  margin-top: 1rem;
  padding: 0.75rem;
  background-color: #f7fafc;
  border-radius: 4px;
}

.example-label {
  font-size: 0.8rem;
  color: #718096;
  font-weight: 500;
  margin-bottom: 0.25rem;
}

.example-text {
  color: #2d3748;
  font-family: monospace;
  font-size: 0.9rem;
}
</style>