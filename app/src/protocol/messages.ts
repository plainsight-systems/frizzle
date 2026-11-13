// Axis R: the messages between the page and the engine's worker. Both sides
// import this file, so every kind is spelled once.
//
// Contract
//   - The page sends requests; each carries an id, a number the page has not
//     used before. The worker answers every request exactly once, finally:
//     with "done" and its value, or with "failed" and a named WorkerFailure.
//     Before that it may send "progress" (a load) or "text" (a generation)
//     with the same id. A request the worker does not know is answered
//     "failed" at the "protocol" stage.
//   - Notices carry no id: the worker sends "device" once, unprompted, when
//     the GPU device is acquired or cannot be.
//   - Requests:
//       load       Download the pinned model if it is not kept, verify it,
//                  and load it into the engine. Progress: {phase:
//                  "download" | "load", done, total} in bytes. Done:
//                  {contextTokens}.
//       generate   {messages, kept, edits, seed, maxTokens}: render the
//                  conversation through the model's chat template, then
//                  read `kept`, reply token identifiers from an earlier
//                  answer (a branch that keeps them; [] for a run from the
//                  reply's start), run it with the ranged edits
//                  (engine/edits.ts), and stream the reply: "text" once an
//                  emitted token, {text, token}, the text possibly "".
//                  maxTokens 0 inspects: no reply, the captures alone.
//                  Done: {end, tokens, promptTokens, reusedTokens, reply,
//                  tail, captures}: reply, the emitted identifiers in
//                  order; tail, text the last token ends with that its
//                  "text" could not carry (U+FFFD for a character the run
//                  stopped inside), else "";
//                  captures (Captures below), the whole capture store at
//                  the input's last position, step kept.length, with the
//                  edits in force there, as the engine read it back
//                  (runtime/engine.h, capture/capture.h): every layer's
//                  neurons, residual coming in, what attention and the MLP
//                  added, Q as attention read it, attention weights and
//                  their statistics, and the layer's attention window and
//                  kind; with them the input those captures read, a token
//                  a position up to the captures', each its identifier,
//                  type and text. They come once, with the answer, never
//                  on their own.
//       cancel     {target}: end the request `target` early: a load while
//                  it downloads (once it streams into the engine it runs to
//                  its end), a generation running, or either not yet
//                  started. Done: whether there was one to end. The target
//                  is still answered, once: a load "failed" at stage
//                  "cancelled", a generation "done" with end "cancelled".
//
// Design
//   - Plain data only, so a message is structured-cloned unchanged. The one
//     typed array, a run's store (865,744 words, 3.3 MiB, for Gemma 3 1B),
//     is transferred, not copied: the worker made it for this message and
//     keeps no hold on it (worker.ts).
//   - The page reads every value by the slots the module sent, never by a
//     layout of its own: the store's layout is the engine's
//     (capture/capture.h), and these types only name what crosses.
//   - The page renders nothing: it sends whole messages, and the worker
//     renders the template (logical-overview.md, the parts).

import type { RangedEdit } from "../engine/edits.ts";

export type Id = number;

// One turn of a conversation, as the page holds it and the chat template
// reads it (worker/template/template.ts).
export interface Message {
  readonly role: "system" | "user" | "assistant";
  readonly content: string;
}

export interface LoadRequest {
  readonly kind: "load";
  readonly id: Id;
}

export interface GenerateRequest {
  readonly kind: "generate";
  readonly id: Id;
  readonly messages: readonly Message[];
  readonly kept: readonly number[]; // reply token identifiers, 32-bit
  readonly edits: readonly RangedEdit[];
  readonly seed: number; // an integer below 2^53
  readonly maxTokens: number;
}

export interface CancelRequest {
  readonly kind: "cancel";
  readonly id: Id;
  readonly target: Id;
}

export type Request = LoadRequest | GenerateRequest | CancelRequest;

export interface Progress {
  readonly phase: "download" | "load";
  readonly done: number;
  readonly total: number;
}

export interface Loaded {
  readonly contextTokens: number;
}

// A range of the store: its first word and its width, in 32-bit words.
export interface Slot {
  readonly first: number;
  readonly width: number;
}

// One layer's slots, as capture/capture.h's LayerSlots.
export interface LayerSlots {
  readonly neurons: Slot; // feed-forward width
  readonly residual: Slot; // embedding width, coming into the layer
  readonly attentionAdded: Slot; // embedding width
  readonly mlpAdded: Slot; // embedding width
  readonly query: Slot; // queryHeads × head dimension; head h at h × d
  readonly weights: Slot; // queryHeads × keys; head h, key i at h × keys + i
  readonly stats: Slot; // queryHeads × 2; head h's maximum at 2h, sum 2h + 1
  readonly keys: number;
  readonly queryHeads: number;
  readonly window: number; // the layer's attention window, in tokens
  readonly full: boolean; // full attention: the window spans the trained context
}

// A token's type, as the vocabulary names it (tokenizer/vocabulary.h).
export type TokenType =
  | "normal"
  | "unknown"
  | "control"
  | "user-defined"
  | "unused"
  | "byte"
  | "undefined";

// One token of the input the captures read (runtime/engine.h, Captured):
// its text the characters it completes, "" for a token that only begins a
// character; a control token's text is the template's spelling of it,
// which the page never shows as text (its type says so).
export interface InputToken {
  readonly id: number;
  readonly type: TokenType;
  readonly text: string;
}

// A run's captures, as the engine gave them (runtime/engine.h, Captured).
// For "taken", every slot lies inside `words` (worker/engine/engine.ts
// refuses an answer where one does not).
export type Captures =
  | {
      readonly kind: "taken";
      readonly layers: readonly LayerSlots[]; // one a layer, in layer order
      readonly words: Float32Array; // the whole store, bit for bit
      // The input's last position, which each layer's weights count their
      // keys back from: key i is at position − min(position + 1, keys) + 1
      // + i (capture/weights.h).
      readonly position: number;
      // The input up to `position`, position + 1 tokens, in order; the
      // token at a key's position is input[position].
      readonly input: readonly InputToken[];
    }
  | { readonly kind: "not-taken" } // the turn was cancelled
  | { readonly kind: "unread"; readonly reason: string }; // the read back failed

export interface Generated {
  readonly end: "stop" | "limit" | "context" | "cancelled";
  readonly tokens: number;
  readonly promptTokens: number;
  readonly reusedTokens: number;
  readonly reply: readonly number[]; // the emitted identifiers, in order
  readonly tail: string; // the last token's text past its "text", maybe ""
  readonly captures: Captures;
}

// Why a request did not complete: where (`stage`), and the engine's or the
// module's own words (`message`); `code` names what the page may act on.
export interface WorkerFailure {
  readonly stage: string;
  readonly message: string;
  readonly code?: string;
}

export type Reply =
  | { readonly kind: "progress"; readonly id: Id; readonly progress: Progress }
  | {
      readonly kind: "text";
      readonly id: Id;
      readonly text: string; // the characters this token completes, maybe ""
      readonly token: number;
    }
  | { readonly kind: "done"; readonly id: Id; readonly value: unknown }
  | {
      readonly kind: "failed";
      readonly id: Id;
      readonly failure: WorkerFailure;
    };

export type DeviceNotice =
  | { readonly kind: "device"; readonly ok: true; readonly adapter: string }
  | { readonly kind: "device"; readonly ok: false; readonly error: string };

export type FromWorker = Reply | DeviceNotice;
