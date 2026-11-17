// Builds the app into journal/webgpu/ so the journal can link to it (relative base: any host path works).
import { defineConfig } from 'vite';

export default defineConfig({
  base: './',
  build: { outDir: '../journal/webgpu', emptyOutDir: true, target: 'es2022', chunkSizeWarningLimit: 4096 },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
});
