// Axis H (frame): the page's entry. Mounts the app into #root; a page
// without it is a broken build, so its absence is thrown, not tolerated.
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App.tsx";

const root = document.getElementById("root");
if (root === null) {
  throw new Error("index.html has no #root to mount the app in");
}
createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
