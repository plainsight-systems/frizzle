// Axis R: starting the engine's worker, the page's one way to reach it.
//
// Contract
//   - startEngine(onDevice): starts the worker (worker/worker.ts) as an ES
//     module worker and returns the EngineClient over it; onDevice receives
//     its "device" notice. Each call starts another worker: the page calls
//     it once (main.tsx).
//   - A worker that fails to start or throws outside a request (its
//     "error" event) is reported to onDevice as a device failure, in the
//     browser's words, so it is never silent.
//
// Design
//   - The worker is named by URL, which Vite bundles as the app's second
//     entry: no page module imports worker code (file-mapping.md), and
//     this file is the one place the URL is written.

import { EngineClient } from "./client.ts";
import type { DeviceNotice } from "./messages.ts";

export function startEngine(
  onDevice: (notice: DeviceNotice) => void,
): EngineClient {
  const worker = new Worker(new URL("../worker/worker.ts", import.meta.url), {
    type: "module",
  });
  worker.addEventListener("error", (event) => {
    onDevice({
      kind: "device",
      ok: false,
      error: `the engine's worker stopped: ${event.message || "no message"}`,
    });
  });
  return new EngineClient(worker, onDevice);
}
