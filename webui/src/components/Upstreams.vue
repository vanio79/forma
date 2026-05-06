<template>
  <div class="upstreams-page">
    <div class="card">
      <div class="card-header">
        <h2 class="card-title">Upstreams</h2>
        <button class="btn btn-primary" @click="showCreateModal = true">Add Upstream</button>
      </div>

      <div v-if="loading" class="loading">Loading upstreams...</div>
      <div v-if="error" class="error">{{ error }}</div>

      <table v-if="!loading && !error && upstreams.length > 0">
        <thead>
          <tr>
            <th>Local Name</th>
            <th>Upstream Model</th>
            <th>Base URL</th>
            <th>Timeout</th>
            <th>Status</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="upstream in upstreams" :key="upstream.id">
            <td>
              <span class="upstream-name">{{ upstream.name }}</span>
            </td>
            <td>{{ upstream.upstream_model || upstream.name }}</td>
            <td>{{ upstream.base_url }}</td>
            <td>{{ upstream.timeout }}s</td>
            <td>
              <span :class="upstream.is_enabled ? 'badge badge-success' : 'badge badge-danger'">
                {{ upstream.is_enabled ? 'Enabled' : 'Disabled' }}
              </span>
            </td>
            <td>
              <button class="btn btn-secondary btn-sm" @click="editUpstream(upstream)">Edit</button>
              <button class="btn btn-danger btn-sm" @click="confirmDelete(upstream)">Delete</button>
            </td>
          </tr>
        </tbody>
      </table>

      <div v-if="!loading && !error && upstreams.length === 0" class="empty-state">
        <p>No upstreams configured. Add one to get started.</p>
        <p class="hint">The upstream "name" is the local model name used for routing. "upstream_model" is sent to the upstream API.</p>
      </div>
    </div>

    <!-- Create/Edit Modal -->
    <div v-if="showCreateModal || showEditModal" class="modal-overlay" @click.self="closeModal">
      <div class="modal">
        <h3>{{ showEditModal ? 'Edit Upstream' : 'Add Upstream' }}</h3>
        
        <div class="form-group">
          <label for="name">Local Model Name *</label>
          <input id="name" v-model="form.name" type="text" required placeholder="e.g., gemma-local, gpt-4" />
          <p class="field-hint">Clients send requests with this model name</p>
        </div>

        <div class="form-group">
          <label for="upstream_model">Upstream Model Name</label>
          <input id="upstream_model" v-model="form.upstream_model" type="text" placeholder="e.g., gemma-4-e4b-it, gpt-4o" />
          <p class="field-hint">Model name sent to upstream API (leave empty to use local name)</p>
        </div>

        <div class="form-group">
          <label for="base_url">Base URL *</label>
          <input id="base_url" v-model="form.base_url" type="text" required placeholder="e.g., http://192.168.68.10:1234/v1" />
        </div>

        <div class="form-group">
          <label for="api_key">API Key</label>
          <input id="api_key" v-model="form.api_key" type="password" placeholder="Enter API key (optional)" />
        </div>

        <div class="form-group">
          <label for="timeout">Timeout (seconds)</label>
          <input id="timeout" v-model.number="form.timeout" type="number" min="1" max="600" />
        </div>

        <div class="form-group checkbox-group">
          <label>
            <input type="checkbox" v-model="form.is_enabled" />
            Enabled
          </label>
        </div>

        <div v-if="modalError" class="error">{{ modalError }}</div>

        <div class="modal-actions">
          <button class="btn btn-secondary" @click="closeModal">Cancel</button>
          <button class="btn btn-primary" @click="saveUpstream">
            {{ showEditModal ? 'Save Changes' : 'Create' }}
          </button>
        </div>
      </div>
    </div>

    <!-- Delete Confirmation Modal -->
    <div v-if="showDeleteModal" class="modal-overlay" @click.self="showDeleteModal = false">
      <div class="modal modal-sm">
        <h3>Delete Upstream</h3>
        <p>Are you sure you want to delete "{{ deleteTarget?.name }}"?</p>
        <div class="modal-actions">
          <button class="btn btn-secondary" @click="showDeleteModal = false">Cancel</button>
          <button class="btn btn-danger" @click="deleteUpstreamConfirmed">Delete</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from "vue";
import type { Upstream } from "../types";
import { getUpstreams, createUpstream, updateUpstream, deleteUpstream } from "../api";

const upstreams = ref<Upstream[]>([]);
const loading = ref(true);
const error = ref("");

const showCreateModal = ref(false);
const showEditModal = ref(false);
const showDeleteModal = ref(false);
const modalError = ref("");
const editTarget = ref<Upstream | null>(null);
const deleteTarget = ref<Upstream | null>(null);

const form = ref({
  name: "",
  upstream_model: "",
  base_url: "",
  api_key: "",
  timeout: 300,
  is_enabled: true,
});

async function loadUpstreams() {
  loading.value = true;
  error.value = "";
  try {
    const result = await getUpstreams();
    upstreams.value = result.upstreams;
  } catch (e) {
    error.value = e instanceof Error ? e.message : "Failed to load upstreams";
  } finally {
    loading.value = false;
  }
}

function editUpstream(upstream: Upstream) {
  editTarget.value = upstream;
  form.value = {
    name: upstream.name,
    upstream_model: upstream.upstream_model || "",
    base_url: upstream.base_url,
    api_key: upstream.api_key,
    timeout: upstream.timeout,
    is_enabled: upstream.is_enabled,
  };
  showEditModal.value = true;
}

function closeModal() {
  showCreateModal.value = false;
  showEditModal.value = false;
  modalError.value = "";
  editTarget.value = null;
  form.value = {
    name: "",
    upstream_model: "",
    base_url: "",
    api_key: "",
    timeout: 300,
    is_enabled: true,
  };
}

async function saveUpstream() {
  modalError.value = "";
  
  if (!form.value.name || !form.value.base_url) {
    modalError.value = "Local Model Name and Base URL are required";
    return;
  }

  try {
    if (showEditModal.value && editTarget.value) {
      await updateUpstream(editTarget.value.id, {
        name: form.value.name,
        upstream_model: form.value.upstream_model,
        base_url: form.value.base_url,
        api_key: form.value.api_key,
        timeout: form.value.timeout,
        is_enabled: form.value.is_enabled,
      });
    } else {
      await createUpstream({
        name: form.value.name,
        upstream_model: form.value.upstream_model,
        base_url: form.value.base_url,
        api_key: form.value.api_key,
        timeout: form.value.timeout,
        is_enabled: form.value.is_enabled,
      });
    }
    closeModal();
    await loadUpstreams();
  } catch (e) {
    modalError.value = e instanceof Error ? e.message : "Failed to save upstream";
  }
}

function confirmDelete(upstream: Upstream) {
  deleteTarget.value = upstream;
  showDeleteModal.value = true;
}

async function deleteUpstreamConfirmed() {
  if (!deleteTarget.value) return;
  
  try {
    await deleteUpstream(deleteTarget.value.id);
    showDeleteModal.value = false;
    deleteTarget.value = null;
    await loadUpstreams();
  } catch (e) {
    error.value = e instanceof Error ? e.message : "Failed to delete upstream";
    showDeleteModal.value = false;
  }
}

onMounted(loadUpstreams);
</script>

<style scoped>
.upstreams-page {
  max-width: 1000px;
}

.upstream-name {
  font-weight: 500;
}

.btn-sm {
  padding: 0.25rem 0.5rem;
  font-size: 0.85rem;
  margin-right: 0.25rem;
}

.empty-state {
  padding: 2rem;
  text-align: center;
  color: #718096;
}

.empty-state .hint {
  font-size: 0.85rem;
  color: #a0aec0;
  margin-top: 0.5rem;
}

.field-hint {
  font-size: 0.85rem;
  color: #a0aec0;
  margin-top: 0.25rem;
}

.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}

.modal {
  background: white;
  border-radius: 8px;
  padding: 1.5rem;
  max-width: 500px;
  width: 90%;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.modal-sm {
  max-width: 350px;
}

.modal h3 {
  margin-bottom: 1rem;
  color: #2c3e50;
}

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
.form-group input[type="password"],
.form-group input[type="number"] {
  width: 100%;
  padding: 0.5rem;
  border: 1px solid #e2e8f0;
  border-radius: 4px;
  font-size: 1rem;
}

.form-group input:focus {
  outline: none;
  border-color: #667eea;
  box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
}

.checkbox-group label {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.checkbox-group input[type="checkbox"] {
  width: 1rem;
  height: 1rem;
}

.modal-actions {
  display: flex;
  justify-content: flex-end;
  gap: 0.5rem;
  margin-top: 1.5rem;
}
</style>