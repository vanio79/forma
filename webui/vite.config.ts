import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import { resolve } from "path";

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      "@": resolve(__dirname, "src"),
    },
  },
  build: {
    outDir: "../webui_dist",
    emptyOutDir: true,
    rollupOptions: {
      output: {
        manualChunks: undefined,
      },
    },
  },
  server: {
    proxy: {
      "/ui": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/v1": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});