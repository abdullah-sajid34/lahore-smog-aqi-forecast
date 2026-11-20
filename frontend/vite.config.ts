import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  resolve: { alias: { "@": path.resolve(__dirname, "./src") } },
  server: { port: 5173 },
  build: {
    chunkSizeWarningLimit: 2500,
    rollupOptions: {
      output: {
        manualChunks: {
          "vendor-react":   ["react","react-dom","react-router-dom"],
          "vendor-query":   ["@tanstack/react-query"],
          "vendor-motion":  ["framer-motion"],
          "vendor-echarts": ["echarts","echarts-for-react"],
          "vendor-three":   ["three","@react-three/fiber","@react-three/drei"],
        },
      },
    },
  },
});
