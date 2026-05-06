<template>
  <div class="requests-list">
    <div class="card">
      <div class="card-header">
        <h2 class="card-title">Recent Requests</h2>
        <button class="btn btn-primary" @click="loadRequests" :disabled="loading">
          {{ loading ? "Refreshing..." : "Refresh" }}
        </button>
      </div>

      <div v-if="loading && requests.length === 0" class="loading">Loading requests...</div>
      <div v-else-if="error" class="error">{{ error }}</div>
      <div v-else>
        <div v-for="request in requests" :key="request.id" class="request-row">
          <!-- Request summary row -->
          <div class="request-summary clickable" @click="toggleExpand(request.id)">
            <div class="request-row-content">
              <div class="request-col timestamp">
                {{ formatTimestamp(request.timestamp_formatted) }}
              </div>
              <div class="request-col model">
                {{ request.model }}
              </div>
              <div class="request-col prompt-preview">
                {{ truncatePrompt(request.user_prompt) }}
              </div>
              <div class="request-col latency">
                {{ formatMs(request.extraction_ms) }}
              </div>
              <div class="request-col flags">
                <span v-if="request.has_extraction" class="badge badge-success">Extraction</span>
                <span v-if="request.has_augmentation" class="badge badge-warning">Augmented</span>
              </div>
              <div class="request-col expand">
                <button class="expand-btn">
                  {{ expanded[request.id] ? "▼" : "▶" }}
                </button>
              </div>
            </div>
          </div>

          <!-- Expanded details -->
          <div v-if="expanded[request.id]" class="request-details">
            <div v-if="detailLoading[request.id]" class="loading">Loading details...</div>
            <div v-else-if="details[request.id]">
              <!-- 1. Original Prompt -->
              <div class="detail-section" v-if="details[request.id].request.user_prompt">
                <h4>📝 Original Prompt</h4>
                <div class="prompt-box">{{ details[request.id].request.user_prompt }}</div>
              </div>

              <!-- 2. Augmented Prompt -->
              <div
                class="detail-section"
                v-if="details[request.id].request.augmented_prompt"
              >
                <h4>✨ Augmented Prompt</h4>
                <div class="prompt-box">{{ details[request.id].request.augmented_prompt }}</div>
              </div>

              <!-- 3. Agent Response -->
              <div
                class="detail-section"
                v-if="details[request.id].request.agent_response"
              >
                <h4>🤖 Agent Response</h4>
                <div class="prompt-box">{{ details[request.id].request.agent_response }}</div>
              </div>

              <!-- 4. Extractions (Collapsible) -->
              <div
                class="detail-section collapsible"
                v-if="hasAnyExtractions(details[request.id].extractions) || details[request.id].request.extraction_response"
              >
                <div class="collapsible-header clickable" @click="toggleSection(request.id, 'extractions')">
                  <h4>
                    <span class="collapse-icon">{{ sectionsExpanded[request.id]?.extractions ? "▼" : "▶" }}</span>
                    📊 Extractions
                  </h4>
                </div>
                <div v-if="sectionsExpanded[request.id]?.extractions" class="collapsible-content">
                  <!-- Entities -->
                  <div
                    v-if="details[request.id].extractions.entity?.length"
                    class="sub-section"
                  >
                    <strong>Entities</strong>
                    <table class="detail-table">
                      <thead>
                        <tr>
                          <th>Data</th>
                          <th>Confidence</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr
                          v-for="item in details[request.id].extractions.entity"
                          :key="item.id"
                        >
                          <td>{{ item.data }}</td>
                          <td>
                            <span
                              class="confidence-badge"
                              :class="getConfidenceClass(item.confidence)"
                            >
                              {{ Math.round(item.confidence * 100) }}%
                            </span>
                          </td>
                        </tr>
                      </tbody>
                    </table>
                  </div>

                  <!-- Relationships -->
                  <div
                    v-if="details[request.id].extractions.relationship?.length"
                    class="sub-section"
                  >
                    <strong>Relationships</strong>
                    <table class="detail-table">
                      <thead>
                        <tr>
                          <th>Data</th>
                          <th>Confidence</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr
                          v-for="item in details[request.id].extractions.relationship"
                          :key="item.id"
                        >
                          <td>{{ item.data }}</td>
                          <td>
                            <span
                              class="confidence-badge"
                              :class="getConfidenceClass(item.confidence)"
                            >
                              {{ Math.round(item.confidence * 100) }}%
                            </span>
                          </td>
                        </tr>
                      </tbody>
                    </table>
                  </div>

                  <!-- Facts -->
                  <div v-if="details[request.id].extractions.fact?.length" class="sub-section">
                    <strong>Facts</strong>
                    <table class="detail-table">
                      <thead>
                        <tr>
                          <th>Statement</th>
                          <th>Confidence</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr
                          v-for="item in details[request.id].extractions.fact"
                          :key="item.id"
                        >
                          <td>{{ item.data }}</td>
                          <td>
                            <span
                              class="confidence-badge"
                              :class="getConfidenceClass(item.confidence)"
                            >
                              {{ Math.round(item.confidence * 100) }}%
                            </span>
                          </td>
                        </tr>
                      </tbody>
                    </table>
                  </div>

                  <!-- Recipes -->
                  <div
                    v-if="details[request.id].extractions.recipe?.length"
                    class="sub-section"
                  >
                    <strong>Recipes</strong>
                    <table class="detail-table">
                      <thead>
                        <tr>
                          <th>Description</th>
                          <th>Confidence</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr
                          v-for="item in details[request.id].extractions.recipe"
                          :key="item.id"
                        >
                          <td>{{ item.data }}</td>
                          <td>
                            <span
                              class="confidence-badge"
                              :class="getConfidenceClass(item.confidence)"
                            >
                              {{ Math.round(item.confidence * 100) }}%
                            </span>
                          </td>
                        </tr>
                      </tbody>
                    </table>
                  </div>

                  <!-- Raw Extraction Output (Collapsible within Extractions) -->
                  <div
                    class="sub-section collapsible"
                    v-if="details[request.id].request.extraction_response"
                  >
                    <div class="collapsible-header-sub clickable" @click="toggleSection(request.id, 'rawExtraction')">
                      <strong>
                        <span class="collapse-icon">{{ sectionsExpanded[request.id]?.rawExtraction ? "▼" : "▶" }}</span>
                        Raw Extraction Output
                        <span class="extraction-latency">({{ formatMs(details[request.id].request.extraction_ms) }})</span>
                      </strong>
                    </div>
                    <div v-if="sectionsExpanded[request.id]?.rawExtraction" class="collapsible-content">
                      <div class="prompt-box">{{ details[request.id].request.extraction_response }}</div>
                    </div>
                  </div>
                </div>
              </div>

              <!-- 4. Retrievals (Collapsible) -->
              <div
                class="detail-section collapsible"
                v-if="hasAnyRetrievals(details[request.id].retrievals)"
              >
                <div class="collapsible-header clickable" @click="toggleSection(request.id, 'retrievals')">
                  <h4>
                    <span class="collapse-icon">{{ sectionsExpanded[request.id]?.retrievals ? "▼" : "▶" }}</span>
                    🔎 Retrievals
                  </h4>
                </div>
                <div v-if="sectionsExpanded[request.id]?.retrievals" class="collapsible-content">
                  <!-- Relationships -->
                  <div
                    v-if="details[request.id].retrievals.relationship?.length"
                    class="sub-section"
                  >
                    <strong>Relationships</strong>
                    <table class="detail-table">
                      <thead>
                        <tr>
                          <th>Data</th>
                          <th>Confidence</th>
                          <th>Score</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr
                          v-for="item in details[request.id].retrievals.relationship"
                          :key="item.id"
                        >
                          <td>{{ item.data }}</td>
                          <td>
                            <span
                              class="confidence-badge"
                              :class="getConfidenceClass(item.confidence)"
                            >
                              {{ Math.round(item.confidence * 100) }}%
                            </span>
                          </td>
                          <td>{{ Math.round(item.score * 100) }}%</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>

                  <!-- Facts -->
                  <div v-if="details[request.id].retrievals.fact?.length" class="sub-section">
                    <strong>Facts</strong>
                    <table class="detail-table">
                      <thead>
                        <tr>
                          <th>Data</th>
                          <th>Confidence</th>
                          <th>Score</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr
                          v-for="item in details[request.id].retrievals.fact"
                          :key="item.id"
                        >
                          <td>{{ item.data }}</td>
                          <td>
                            <span
                              class="confidence-badge"
                              :class="getConfidenceClass(item.confidence)"
                            >
                              {{ Math.round(item.confidence * 100) }}%
                            </span>
                          </td>
                          <td>{{ Math.round(item.score * 100) }}%</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>

                  <!-- Recipes -->
                  <div
                    v-if="details[request.id].retrievals.recipe?.length"
                    class="sub-section"
                  >
                    <strong>Recipes</strong>
                    <table class="detail-table">
                      <thead>
                        <tr>
                          <th>Data</th>
                          <th>Confidence</th>
                          <th>Score</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr
                          v-for="item in details[request.id].retrievals.recipe"
                          :key="item.id"
                        >
                          <td>{{ item.data }}</td>
                          <td>
                            <span
                              class="confidence-badge"
                              :class="getConfidenceClass(item.confidence)"
                            >
                              {{ Math.round(item.confidence * 100) }}%
                            </span>
                          </td>
                          <td>{{ Math.round(item.score * 100) }}%</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>

              <!-- No details message -->
              <div
                v-if="!hasAnyDetails(details[request.id])"
                class="detail-section"
              >
                <p class="no-details">No extraction, retrieval, or augmentation data for this request.</p>
              </div>
            </div>
          </div>
        </div>

        <div v-if="requests.length === 0" class="loading">No requests recorded yet.</div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, reactive } from "vue";
import type { RequestListItem, RequestFullDetail, ExtractionsByType, RetrievalsByType } from "../types";
import { getRequests, getRequestDetail } from "../api";

const requests = ref<RequestListItem[]>([]);
const loading = ref(true);
const error = ref("");
const expanded = reactive<Record<string, boolean>>({});
const details = reactive<Record<string, RequestFullDetail>>({});
const detailLoading = reactive<Record<string, boolean>>({});
const sectionsExpanded = reactive<Record<string, Record<string, boolean>>>({});

const formatTimestamp = (ts: string): string => {
  const date = new Date(ts);
  return date.toLocaleString();
};

const formatMs = (ms: number): string => {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
};

const truncatePrompt = (prompt: string): string => {
  if (!prompt) return "-";
  if (prompt.length <= 50) return prompt;
  return prompt.slice(0, 50) + "...";
};

const getConfidenceClass = (confidence: number): string => {
  if (confidence >= 0.9) return "confidence-high";
  if (confidence >= 0.7) return "confidence-medium";
  return "confidence-low";
};

const hasAnyExtractions = (extractions: ExtractionsByType): boolean => {
  return (
    (extractions.entity?.length ?? 0) > 0 ||
    (extractions.relationship?.length ?? 0) > 0 ||
    (extractions.fact?.length ?? 0) > 0 ||
    (extractions.recipe?.length ?? 0) > 0
  );
};

const hasAnyRetrievals = (retrievals: RetrievalsByType): boolean => {
  return (
    (retrievals.relationship?.length ?? 0) > 0 ||
    (retrievals.fact?.length ?? 0) > 0 ||
    (retrievals.recipe?.length ?? 0) > 0
  );
};

const hasAnyDetails = (detail: RequestFullDetail): boolean => {
  return (
    hasAnyExtractions(detail.extractions) ||
    hasAnyRetrievals(detail.retrievals) ||
    (detail.request.extraction_response?.length ?? 0) > 0 ||
    (detail.request.augmented_prompt?.length ?? 0) > 0 ||
    (detail.request.user_prompt?.length ?? 0) > 0
  );
};

const toggleSection = (requestId: string, section: string) => {
  if (!sectionsExpanded[requestId]) {
    sectionsExpanded[requestId] = {};
  }
  sectionsExpanded[requestId][section] = !sectionsExpanded[requestId][section];
};

const toggleExpand = async (requestId: string) => {
  expanded[requestId] = !expanded[requestId];

  if (expanded[requestId] && !details[requestId]) {
    detailLoading[requestId] = true;
    try {
      details[requestId] = await getRequestDetail(requestId);
      // Initialize sections as collapsed by default
      sectionsExpanded[requestId] = {
        extractions: false,
        rawExtraction: false,
        retrievals: false,
      };
    } catch (e) {
      console.error("Failed to load request details:", e);
    } finally {
      detailLoading[requestId] = false;
    }
  }
};

const loadRequests = async () => {
  loading.value = true;
  error.value = "";
  try {
    const response = await getRequests(100);
    requests.value = response.requests;
  } catch (e) {
    error.value = `Failed to load requests: ${e}`;
  } finally {
    loading.value = false;
  }
};

onMounted(loadRequests);
</script>

<style scoped>
.request-row {
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  margin-bottom: 0.5rem;
  overflow: hidden;
}

.request-summary {
  background: white;
  padding: 1rem;
}

.clickable {
  cursor: pointer;
}

.request-row-content {
  display: grid;
  grid-template-columns: 1fr 0.8fr 1.5fr 0.6fr 1fr 0.4fr;
  gap: 1rem;
  align-items: center;
}

.request-col {
  font-size: 0.9rem;
}

.timestamp {
  color: #718096;
}

.model {
  font-weight: 500;
}

.prompt-preview {
  color: #4a5568;
  max-width: 200px;
  overflow: hidden;
  text-overflow: ellipsis;
}

.latency {
  font-weight: 500;
  color: #2c3e50;
}

.flags {
  display: flex;
  gap: 0.25rem;
  flex-wrap: wrap;
}

.expand {
  text-align: center;
}

.expand-btn {
  background: none;
  border: none;
  font-size: 1rem;
  cursor: pointer;
}

.request-details {
  background-color: #f7fafc;
  padding: 1rem;
  border-top: 1px solid #e2e8f0;
}

.detail-section {
  margin-bottom: 1.5rem;
}

.detail-section h4 {
  color: #4a5568;
  margin-bottom: 0.75rem;
}

.collapsible {
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  background: white;
}

.collapsible-header {
  padding: 0.75rem 1rem;
  background: #f7fafc;
  border-bottom: 1px solid #e2e8f0;
  display: flex;
  align-items: center;
}

.collapsible-header h4 {
  margin: 0;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.collapsible-header-sub {
  padding: 0.5rem;
  background: #f7fafc;
  border: 1px solid #e2e8f0;
  border-radius: 4px;
  margin-top: 0.5rem;
}

.collapsible-header-sub strong {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.collapse-icon {
  font-size: 0.8rem;
  color: #718096;
}

.collapsible-content {
  padding: 1rem;
}

.extraction-latency {
  color: #718096;
  font-weight: normal;
  font-size: 0.85rem;
}

.sub-section {
  margin-top: 0.75rem;
}

.sub-section > strong {
  color: #2d3748;
  font-size: 1rem;
  margin-bottom: 0.5rem;
  display: block;
}

.detail-table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 0.5rem;
  font-size: 0.9rem;
}

.detail-table th {
  background-color: #edf2f7;
  padding: 0.5rem 0.75rem;
  text-align: left;
  font-weight: 600;
  color: #4a5568;
  border-bottom: 2px solid #e2e8f0;
}

.detail-table td {
  padding: 0.5rem 0.75rem;
  border-bottom: 1px solid #e2e8f0;
}

.detail-table tbody tr:hover {
  background-color: #f7fafc;
}

.prompt-box {
  background-color: #2d3748;
  color: #e2e8f0;
  padding: 1rem;
  border-radius: 4px;
  font-size: 0.9rem;
  white-space: pre-wrap;
  word-wrap: break-word;
  margin-top: 0.5rem;
}

.no-details {
  color: #718096;
}

.confidence-badge {
  display: inline-block;
  padding: 0.125rem 0.5rem;
  border-radius: 9999px;
  font-size: 0.85rem;
  font-weight: 500;
}

.confidence-high {
  background-color: #c6f6d5;
  color: #22543d;
}

.confidence-medium {
  background-color: #fef3c7;
  color: #92400e;
}

.confidence-low {
  background-color: #fed7d7;
  color: #c53030;
}

@media (max-width: 800px) {
  .request-row-content {
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
  }

  .request-col.expand {
    grid-column: span 2;
    text-align: left;
  }

  .detail-table {
    font-size: 0.8rem;
  }
}
</style>