// Axis U: the app's build. Two entries in the end, the page (index.html)
// and the engine's worker (slice 3, step 4), bundled as ES modules. The
// site's base path is slice 6's. React's plugin compiles JSX and gives the
// dev server fast refresh.
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  worker: { format: "es" },
  build: { outDir: "dist", target: "es2023" },
});
