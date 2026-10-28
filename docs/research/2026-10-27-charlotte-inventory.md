# What Frizzle takes from Charlotte

*Traced 2026-10-27 from Charlotte commit `e003048` (Apache-2.0, same author). Method: follow `#include` from Gemma 3's architecture, the graph, and the text-generation runtime; add the files CMake compiles directly and the files the WebAssembly boundary calls; check every left-behind file for real `#include` use (several mentions were only comments).*

Charlotte's core is about 30,400 lines of C++. Frizzle's Gemma 3 1B path needs 111 files, about 20,500 lines, copied whole. One registry file is trimmed and the WebAssembly boundary is rewritten.

## The model file decides the formats

Charlotte serves `unsloth/gemma-3-1b-it-GGUF`, `gemma-3-1b-it-Q4_0.gguf` (721,918,496 bytes, SHA-256 `27ee88e0…0276e`). Its 340 tensors, read from the file:

| Format | Tensors | What |
|---|---|---|
| Q4_0 | 179 | attention Q, K, V, output; gate and up |
| F32 | 157 | norm gains |
| Q4_1 | 3 | `ffn_down` in three layers |
| Q8_0 | 1 | `token_embd`, about 321 MB, also the output head |

The KV cache is F16 by default (`policy/policy.h`), so F16's pack shader is needed although no weight is F16.

## C++: copied whole (111 files)

| Area | Files |
|---|---|
| Architecture | `arch/architecture.h`, `arch/describe.*`, `arch/gemma3/*`, `model/model_description.h`, `policy/policy.h` |
| Model file | `gguf/arrays.*`, `byte_source.h`, `checked.h`, `index.*`, `reader.*`, `types.h` |
| Formats | `formats/format.h`, `device_layout.h`, `f32/`, `f16/`, `q4_0/`, `q4_1/`, `q8_0/`; `capability/capability.h` |
| GPU | `gpu/device.*`, `device_requirements.h`, `wgpu_handles.h`, `unique_handle.h`, `userdata.h`, `callback_mode.h`, `self_check.*`, `dispatch_math.*` |
| Weights on the GPU | `residency/file_ranges.*`, `mapping.*`, `piece_writer.*`, `plan.*`, `routes.*`, `upload.*`, `upload_check.*`, `weight_view.h`, `witness.cpp` |
| Graph and kernels | `graph/graph.*`; `kernels/interface.h`, `order.h`, `program.*`, `schedule.*`, `attention/`, `gather/`, `matmul/`, `norm/`, `rope/`, `topk/` |
| Cache | `cache/kv.*`, `cache/prefix.*` |
| Sampling and the turn | `sampler/*`, `runtime/runtime.*`, `turn.*`, `generator.*`, `stops.*` |
| Tokenizer (SentencePiece BPE) | `tokenizer/tokenizer.h`, `load.*`, `vocabulary.*`, `special.*`, `admission.*`, `detokenizer.*`, `unicode.*`, `unicode_tables.*`, `utf8_stream.cpp`; `bpe/sentencepiece_bpe.*`, `sentencepiece_merges.*`, `merge.*`, `merge_table.*` |
| Other | `diagnostics.h`, `run_guard.h` |

Shaders: `formats/{f32,f16,f16_pack,q4_0,q4_1,q8_0}.wgsl`, `kernels/{attention,gather,matmul,norm,topk}.wgsl`, `rope/rope.wgsl` and `rope_cache.wgsl`, `kernels/step.wgsl`, `sampler/draw.wgsl`, and `shaders/vector_add.wgsl` (the GPU self-check uses it). Build support: `cmake/EmbedShaders.cmake`, `cmake/embed_shaders_script.cmake`, `cmake/dawn.json`.

The shared kernels and graph blocks still carry branches for the other families (Qwen3's head size, Llama's layout). They come over intact; removing those branches is a later, separate change.

## C++: trimmed or rewritten

| File | Change |
|---|---|
| `capability/capability.cpp` (107 lines) | The registry of architectures, formats and tokenizers. Keep Gemma 3, F32, F16, Q4_0, Q4_1, Q8_0 and SentencePiece BPE. |
| `wasm/bindings.cpp` (2,082 lines) | The only browser-aware C++. Rewritten for Frizzle's worker protocol: captures, edits, the edited and original replies. Charlotte's is a reference, not a starting file. |
| `preflight/preflight.*` (≈500 lines) | Judges arbitrary model files from their header. Frizzle pins one file by SHA-256, so this goes; the part that sizes the context to the device's limits is still needed and has to be found in it and kept. |

## C++: left behind

- **Other architectures:** Qwen3, Llama, Gemma 4 (audio, vision), EmbeddingGemma 2, `arch/media_encoder.h`.
- **Encoders and media:** `graph/encoder`, `vision`, `audio`; `media/*`; `runtime/embedder`, `vision`, `audio`, `video`, `ends`, `media_stage.h`.
- **Kernels only encoders use:** `block_attention`, `local_attention`, `light_conv`, `pool`, `subsample`, `rope_encode.wgsl`.
- **Formats Gemma 3 1B does not use:** BF16, Q6_K (and its repack); `quant/q4_0.h` (only tests use it).
- **Byte-level BPE** (Qwen3, Llama), Gemma 4's tokenizer, `nfc`, `pretokenize`, `piece_cache`.
- `gpu/readback_bench.*` (a measurement tool).

## Web: port to TypeScript, or leave

The React front end replaces every Charlotte interface module. The engine-side modules are ported to TypeScript:

| Port | Why |
|---|---|
| `worker.js`, `worker_client.js`, `protocol.js` | The worker that owns the engine, and its message contract |
| `wasm_runtime.js` (trim) | Loads the module and calls the boundary; drop its image, audio and video paths |
| `load.js`, `opfs.js`, `download.js`, `fetch.js`, `sha256.js` | Download once, verify by SHA-256, cache in the browser, stream into the engine |
| `template.js`, `vendor/jinja.js`, `conversation.js` | Render Gemma's chat template from the file |
| `platform.js`, `platform_policy.js` | Which browsers it runs in; the model's load policy |

Left behind: the chat, picker, catalog, drawer, model, cache and status views; search, embedding and media inputs (`search*`, `passages`, `photos`, `recordings`, `videos`, `*_input`, `companion`, `task_prompt`); `thinking_stream.js` (Qwen3's thinking); `duplicates.js` (several models); `preflight.js`; diagnostics, to be reconsidered later.

## Tests and tools

- **Tests that come with the kept code:** the GGUF, tokenizer (SentencePiece, merges, vocabulary, detokenizer, admission, unicode, UTF-8 stream), kernel, cache, residency, runtime, sampler, graph, description and shader-embedding tests, trimmed to Gemma 3. GPU tests for the kept kernels, upload and device.
- **The correctness oracle:** `tests/gpu/reference_test.cpp` with `tools/reference_logits/` and `make_reference_logits.sh`, which compare the model's logits with llama.cpp. Frizzle's "no edits reproduces the original exactly" requirement builds on this.
- **Tools:** `fetch_dawn.py`, `fetch_test_data.py` (Gemma 3 only), `gguf_header.py`, `make_tokenizer_fixtures.py`, `make_unicode_tables.py`, `tests/fixtures/tokenizer/gemma-3-1b-it-q4_0.inc`, and `tests/web/template.test.js` with its Gemma template fixture.
- **Left behind:** media, encoder, embedder, Q6_K, Gemma 4 and other-model tests and fixtures; the site, boundary and review shell checks, which are Charlotte's own.

## Addendum, 2026-10-28: found while porting

- **`kernels/rope/rope_encode.wgsl` comes over too.** `rope.cpp` builds the encoder form of the rope kernel as well as the decoder's, so it does not compile without that shader. It is ported unused, like the other families' branches in the shared kernels, and leaves with them.
- **Preflight goes entirely.** Its only caller is Charlotte's `wasm/bindings.cpp`, which Frizzle rewrites, and the context is already sized to the device in `residency/plan`, which comes over whole; nothing needs to be found in preflight and kept.
