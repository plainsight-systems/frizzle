// Axis H (frame): the app's frame. Shows the prompt view over the session
// it is given.
//
// Contract
//   - App({session}): renders the plain page (views/prompt/) for that
//     session. The session, and the worker behind it, are made once by
//     main.tsx, not here: React may mount a component twice (StrictMode),
//     and a page must not start two workers.
//   - Nothing loads until asked: the model is 722 MB, so the download
//     begins at the person's Load, never when the page opens.

import type { Session } from "./session/session.ts";
import { PromptView } from "./views/prompt/PromptView.tsx";

export function App({ session }: { readonly session: Session }) {
  return <PromptView session={session} />;
}
