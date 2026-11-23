import { defineConfig } from "vite";

export default defineConfig({
  // Relative asset URLs, so `dist/` works from any path on a static host.
  base: "./",
  build: { target: "es2023" },
  worker: { format: "es" },
});
