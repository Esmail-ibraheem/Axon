import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';

// https://vitejs.dev/config/
export default defineConfig({
  base: "Axon",
  plugins: [vue()],
  resolve: {
    alias: {
      '@': '/src'
      // Add more aliases if needed
    }
  }
});
