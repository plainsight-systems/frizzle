// Axis U: the app's build. Two entries, the page (index.html) and the
// engine's worker (src/worker/worker.ts, which the page starts), bundled as
// ES modules, with a relative base ("./"), so the built app works wherever
// it is served: the site serves it at app/ (web/hugo.toml), under any base
// path. React's plugin compiles JSX and gives the dev server fast refresh.
//
// "@frizzle/engine" is the engine's browser module, built by CMake's
// wasm-release preset into build/wasm-release/web/ (frizzle.mjs and its
// .wasm), outside app/: a missing build fails the app's build by name. The
// worker also reads the pin, config/model.json, at the repository's root;
// the dev server may serve both.
import { fileURLToPath } from "node:url";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));

export default defineConfig({
  base: "./",
  plugins: [react()],
  resolve: {
    alias: {
      "@frizzle/engine": `${root}build/wasm-release/web/frizzle.mjs`,
    },
  },
  server: { fs: { allow: [root] } },
  worker: { format: "es" },
  build: { outDir: "dist", target: "es2023" },
});
