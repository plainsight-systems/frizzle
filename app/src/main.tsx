// Axis H (frame): the page's entry. Starts the engine's worker and the
// session over it, once per page, and mounts the app into #root; a page
// without #root is a broken build, so its absence is thrown, not tolerated.
//
// Contract
//   - One Session (session/session.ts) per page, made here with
//     startEngine (protocol/start.ts) before React renders, and handed to
//     App inside StrictMode: the frame owns it, and nothing else makes one.

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { App } from "./App.tsx";
import { startEngine } from "./protocol/start.ts";
import { Session } from "./session/session.ts";

const root = document.getElementById("root");
if (root === null) {
  throw new Error("index.html has no #root to mount the app in");
}
const session = new Session(startEngine);
createRoot(root).render(
  <StrictMode>
    <App session={session} />
  </StrictMode>,
);
