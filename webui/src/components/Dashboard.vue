<template>
  <div class="dashboard">
    <div v-if="loading" class="loading">Loading statistics...</div>
    <div v-else-if="error" class="error">{{ error }}</div>
    <div v-else>
      <div class="card">
        <div class="card-header">
          <h2 class="card-title">Overview</h2>
          <button class="btn btn-danger" @click="handleClear" :disabled="clearing">
            {{ clearing ? "Clearing..." : "Clear Data" }}
          </button>
        </div>
        <div class="grid">
          <div class="card">
            <div class="metric-value">{{ stats?.total_requests ?? 0 }}</div>
            <div class="metric-label">Total Requests</div>
          </div>
          <div class="card">
            <div class="metric-value">{{ stats?.total_extractions ?? 0 }}</div>
            <div class="metric-label">Extractions</div>
          </div>
          <div class="card">
            <div class="metric-value">{{ stats?.total_retrievals ?? 0 }}</div>
            <div class="metric-label">Retrievals</div>
          </div>
          <div class="card">
            <div class="metric-value">{{ stats?.upstream_count ?? 0 }}</div>
            <div class="metric-label">Upstreams</div>
          </div>
        </div>
      </div>

      <div class="card">
        <h2 class="card-title">Performance</h2>
        <div class="grid">
          <div class="card">
            <div class="metric-value">{{ formatMs(stats?.avg_extraction_ms ?? 0) }}</div>
            <div class="metric-label">Avg Extraction Latency</div>
          </div>
        </div>
      </div>

      <div class="card">
        <h2 class="card-title">Extractions by Type</h2>
        <div class="grid">
          <div class="card">
            <div class="metric-value">{{ stats?.extractions_by_type?.entities ?? 0 }}</div>
            <div class="metric-label">Entities</div>
          </div>
          <div class="card">
            <div class="metric-value">{{ stats?.extractions_by_type?.relationships ?? 0 }}</div>
            <div class="metric-label">Relationships</div>
          </div>
          <div class="card">
            <div class="metric-value">{{ stats?.extractions_by_type?.facts ?? 0 }}</div>
            <div class="metric-label">Facts</div>
          </div>
          <div class="card">
            <div class="metric-value">{{ stats?.extractions_by_type?.recipes ?? 0 }}</div>
            <div class="metric-label">Recipes</div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from "vue";
import type { Stats } from "../types";
import { getStats, clearData } from "../api";

const stats = ref<Stats | null>(null);
const loading = ref(true);
const error = ref("");
const clearing = ref(false);

const formatMs = (ms: number): string => {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
};

const loadStats = async () => {
  loading.value = true;
  error.value = "";
  try {
    stats.value = await getStats();
  } catch (e) {
    error.value = `Failed to load statistics: ${e}`;
  } finally {
    loading.value = false;
  }
};

const handleClear = async () => {
  if (!confirm("Are you sure you want to clear all request history? This cannot be undone.")) {
    return;
  }
  clearing.value = true;
  try {
    await clearData();
    await loadStats();
  } catch (e) {
    error.value = `Failed to clear data: ${e}`;
  } finally {
    clearing.value = false;
  }
};

onMounted(loadStats);
</script>

<style scoped>
.dashboard {
  width: 100%;
}
</style>