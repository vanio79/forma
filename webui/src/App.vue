<template>
  <div class="app-container">
    <header class="header">
      <div class="header-content">
        <h1 class="title">
          <router-link to="/">Forma</router-link>
        </h1>
        <nav class="nav">
          <router-link to="/" class="nav-link">Dashboard</router-link>
          <router-link to="/requests" class="nav-link">Requests</router-link>
        </nav>
      </div>
    </header>
    <main class="main-content">
      <router-view />
    </main>
    <footer class="footer">
      <p>Forma Request Tracker &copy; 2026</p>
    </footer>
  </div>
</template>

<script setup lang="ts">
</script>

<style>
* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell,
    "Fira Sans", "Droid Sans", "Helvetica Neue", sans-serif;
  background-color: #f5f7fa;
  color: #2c3e50;
  line-height: 1.6;
}

.app-container {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 1rem 2rem;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.header-content {
  max-width: 1200px;
  margin: 0 auto;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.title {
  font-size: 1.5rem;
  font-weight: 700;
}

.title a {
  color: white;
  text-decoration: none;
}

.nav {
  display: flex;
  gap: 1rem;
}

.nav-link {
  color: white;
  text-decoration: none;
  padding: 0.5rem 1rem;
  border-radius: 4px;
  transition: background-color 0.2s;
}

.nav-link:hover {
  background-color: rgba(255, 255, 255, 0.2);
}

.nav-link.router-link-active {
  background-color: rgba(255, 255, 255, 0.3);
}

.main-content {
  flex: 1;
  padding: 2rem;
  max-width: 1200px;
  margin: 0 auto;
  width: 100%;
}

.footer {
  background-color: #2c3e50;
  color: white;
  padding: 1rem 2rem;
  text-align: center;
}

.card {
  background: white;
  border-radius: 8px;
  padding: 1.5rem;
  margin-bottom: 1rem;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.card-title {
  font-size: 1.25rem;
  font-weight: 600;
  color: #2c3e50;
}

.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 0.5rem 1rem;
  border-radius: 4px;
  border: none;
  cursor: pointer;
  font-size: 0.9rem;
  transition: all 0.2s;
}

.btn-primary {
  background-color: #667eea;
  color: white;
}

.btn-primary:hover {
  background-color: #5a67d8;
}

.btn-secondary {
  background-color: #e2e8f0;
  color: #4a5568;
}

.btn-secondary:hover {
  background-color: #cbd5e0;
}

.btn-danger {
  background-color: #e53e3e;
  color: white;
}

.btn-danger:hover {
  background-color: #c53030;
}

.loading {
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 2rem;
  color: #718096;
}

.error {
  color: #e53e3e;
  padding: 1rem;
  background-color: #fed7d7;
  border-radius: 4px;
  margin-bottom: 1rem;
}

table {
  width: 100%;
  border-collapse: collapse;
}

th,
td {
  padding: 0.75rem 1rem;
  text-align: left;
  border-bottom: 1px solid #e2e8f0;
}

th {
  background-color: #f7fafc;
  font-weight: 600;
  color: #4a5568;
}

tr:hover {
  background-color: #f7fafc;
}

.badge {
  display: inline-block;
  padding: 0.25rem 0.5rem;
  border-radius: 9999px;
  font-size: 0.75rem;
  font-weight: 500;
}

.badge-success {
  background-color: #c6f6d5;
  color: #22543d;
}

.badge-warning {
  background-color: #fef3c7;
  color: #92400e;
}

.badge-danger {
  background-color: #fed7d7;
  color: #c53030;
}

.badge-info {
  background-color: #bee3f8;
  color: #2b6cb0;
}

.metric-value {
  font-size: 2rem;
  font-weight: 700;
  color: #2c3e50;
}

.metric-label {
  font-size: 0.9rem;
  color: #718096;
}

.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
}

a {
  color: #667eea;
}

a:hover {
  color: #5a67d8;
}

.clickable {
  cursor: pointer;
}

.clickable:hover {
  background-color: #edf2f7;
}

.expand-btn {
  background: none;
  border: none;
  cursor: pointer;
  padding: 0.25rem 0.5rem;
  color: #667eea;
  font-size: 1rem;
}

.expand-btn:hover {
  color: #5a67d8;
}

.detail-section {
  background-color: #f7fafc;
  padding: 1rem;
  margin-top: 0.5rem;
  border-radius: 4px;
}

.detail-section h4 {
  margin-bottom: 0.5rem;
  color: #4a5568;
}

.entity-list {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.entity-item {
  background-color: #bee3f8;
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  font-size: 0.9rem;
}

.relationship-item {
  background-color: #e9d8fd;
  padding: 0.5rem;
  border-radius: 4px;
  margin-bottom: 0.25rem;
}

.fact-item {
  background-color: #c6f6d5;
  padding: 0.5rem;
  border-radius: 4px;
  margin-bottom: 0.25rem;
}

.prompt-box {
  background-color: #2d3748;
  color: #e2e8f0;
  padding: 1rem;
  border-radius: 4px;
  font-size: 0.9rem;
  white-space: pre-wrap;
  word-wrap: break-word;
}

.retrieval-item {
  background-color: #fef3c7;
  padding: 0.5rem;
  border-radius: 4px;
  margin-bottom: 0.25rem;
}
</style>