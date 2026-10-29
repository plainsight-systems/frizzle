// Axis R: the engine's browser module as the worker sees it: the build of
// src/wasm/bindings.cpp (its header is the contract), an ES module whose
// default export instantiates it. Vite resolves "@frizzle/engine" to the
// build's output (vite.config.ts); this declares its shape for tsc.

declare module "@frizzle/engine" {
  export interface FrizzleModule {
    readonly HEAPU8: Uint8Array;
    _malloc(bytes: number): number;
    _free(pointer: number): void;
    _frizzle_start(): void;
    _frizzle_load_begin(
      request: number,
      front: number,
      frontLength: number,
      fileSize: number,
    ): void;
    _frizzle_load_chunk(request: number, offset: number, length: number): void;
    _frizzle_load_finish(request: number): void;
    _frizzle_generate(
      request: number,
      text: number,
      textLength: number,
      edits: number,
      editCount: number,
      seed: number,
      maxTokens: number,
    ): void;
    _frizzle_cancel(): void;
  }
  const createModule: () => Promise<FrizzleModule>;
  export default createModule;
}
