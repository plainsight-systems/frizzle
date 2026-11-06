// Axis T: Gemma 3's chat template, rendered. Changes with the template's
// language or what the template is given; never with what it says, which
// is the model file's.
//
// Renders the chat template the model file carries
// (tokenizer.chat_template, Jinja) over a conversation, with the variables
// transformers gives every chat template. It renders; it does not
// interpret: the text it returns is what the model reads, and the engine
// reads it as given (src/core/runtime/engine.h). Wrapping a message in
// <bos> and the turn markers is this module's job, not the engine's.
//
// Contract
//   - compileTemplate(chat) parses chat.template once, and returns
//     render(messages): the template rendered over the whole conversation,
//     with add_generation_prompt true (the text ends where the model's
//     reply begins), bos_token and eos_token as chat gives them, and the
//     globals below. A pure function of its arguments: the same messages
//     always give the same text.
//   - For Gemma 3, the template gives, byte for byte as Python's jinja2
//     configured as transformers configures it:
//       one user message  <bos><start_of_turn>user\n{message, trimmed}
//                         <end_of_turn>\n<start_of_turn>model\n
//       an assistant turn <start_of_turn>model\n{content, trimmed}
//                         <end_of_turn>\n
//       a system message  its text and a blank line begin the first user
//                         turn's content
//   - Refused, as a TemplateError naming why, never as empty or partial
//     text: a template that does not parse (at compile); one that raises,
//     with its own message, such as Gemma's for turns that do not
//     alternate; one that fails while rendering, such as calling a function
//     it is not given.
//   - The whole conversation is rendered every turn: a template may rewrite
//     earlier turns, so the text is not an append to the last, and the
//     engine's prefix diff decides what its cache reuses.
//
// Algorithm
//   1. compileTemplate: parse chat.template with @huggingface/jinja's
//      Template, which applies the whitespace rules transformers renders
//      with (trim_blocks, lstrip_blocks); keep only its parse.
//   2. render: a fresh environment of the globals, then the context
//      (messages, add_generation_prompt, bos_token, eos_token); run the
//      parse in an Interpreter; return its text.
//   Globals, as transformers defines them: true, false, none and their
//   Python spellings; range(); raise_exception(message), which throws a
//   TemplateError with the message. Not strftime_now: Gemma 3's template
//   does not call it, and a template that did would be refused by name
//   rather than read the clock.
//
// Design and cost
//   - The Jinja interpreter is @huggingface/jinja 0.5.10, an exact-pinned
//     npm dependency: no dependencies of its own, its own types, and the
//     lockfile's integrity hash in place of a vendored copy's. A
//     dependency decision.
//   - Only what Gemma 3's template uses: no thinking switch, no template
//     variables, no clock.
//   - Cost: a parse at load; a render a turn, linear in the conversation's
//     text, far below a turn's prefill.

import { Environment, Interpreter, Template } from "@huggingface/jinja";
import type { Message } from "../../protocol/messages.ts";

export type { Message };

export interface ChatTemplate {
  readonly template: string; // tokenizer.chat_template
  readonly bosToken: string; // the BOS token's text, "<bos>" for Gemma 3
  readonly eosToken: string; // the EOS token's text
}

export type Render = (messages: readonly Message[]) => string;

export class TemplateError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "TemplateError";
  }
}

export function compileTemplate(chat: ChatTemplate): Render {
  let parsed: Template["parsed"];
  try {
    // Template applies transformers' whitespace rules; only its parse is
    // kept, so rendering gets this module's globals, not Template's own.
    parsed = new Template(chat.template).parsed;
  } catch (error) {
    throw new TemplateError(
      `the model's chat template does not parse: ${messageOf(error)}`,
    );
  }
  return (messages) => {
    const environment = globals();
    const context: Record<string, unknown> = {
      messages,
      add_generation_prompt: true,
      bos_token: chat.bosToken,
      eos_token: chat.eosToken,
    };
    try {
      for (const [name, value] of Object.entries(context)) {
        environment.set(name, value);
      }
      const text: unknown = new Interpreter(environment).run(parsed).value;
      if (typeof text !== "string") {
        throw new TemplateError("the model's chat template rendered no text");
      }
      return text;
    } catch (error) {
      if (error instanceof TemplateError) throw error;
      throw new TemplateError(
        `the model's chat template failed: ${messageOf(error)}`,
      );
    }
  };
}

// The names every chat template may use, as transformers defines them, but
// for strftime_now (Algorithm, above).
function globals(): Environment {
  const environment = new Environment();
  const constants: Record<string, boolean | null> = {
    true: true,
    false: false,
    none: null,
    True: true,
    False: false,
    None: null,
  };
  for (const [name, value] of Object.entries(constants)) {
    environment.set(name, value);
  }
  environment.set("range", range);
  environment.set("raise_exception", (message: string) => {
    throw new TemplateError(message);
  });
  return environment;
}

// Python's range().
function range(start: number, stop?: number, step = 1): number[] {
  const [from, to] = stop === undefined ? [0, start] : [start, stop];
  if (step === 0) throw new TemplateError("range() step must not be zero");
  const values: number[] = [];
  for (let i = from; step > 0 ? i < to : i > to; i += step) values.push(i);
  return values;
}

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
