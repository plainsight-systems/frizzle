#pragma once

#include "core/arch/architecture.h"

namespace frizzle::arch {

// Axis A: changes with Gemma 3's order of work in a layer, or how its file
// names and configures its layers.
//
// Gemma 3's architecture: its describe, which names its 13 weights a layer,
// slides the window over five layers of every six, and its graph, the order it
// runs the shared blocks in (graph/graph.h) with the embedding scale the
// description does not carry. The capability table lists it under
// general.architecture "gemma3" (capability/capability.cpp).
//
// Contract
//   describe(index, out): Ok with `out` as model/model_description.h states,
//   or the first refusal, naming the key or tensor, with `out` partly
//   written. Every refusal of read_hyperparameters and describe_layers
//   (arch/describe.h), with "gemma3" as <arch>, and these of its own:
//     - The window, the pattern or the window's base (below) declared with
//       another type than the UINT32, UINT32 and FLOAT32 read: WrongKeyType;
//       the pattern and the base are read only when the window is nonzero.
//     - A sliding-window pattern stored as an array, a flag a layer: valid,
//       and UnsupportedValue, as the index locates an array without decoding
//       it. Checked only when the window is nonzero.
//     - A pattern of 0: InvalidValue.
//   Role names, each required in every layer as blk.<i>.<suffix>, each shape
//   and gain type checked as describe.h states:
//     AttentionNorm        attn_norm.weight
//     Query                attn_q.weight
//     Key                  attn_k.weight
//     Value                attn_v.weight
//     QueryNorm            attn_q_norm.weight
//     KeyNorm              attn_k_norm.weight
//     AttentionOutput      attn_output.weight
//     PostAttentionNorm    post_attention_norm.weight
//     FeedForwardNorm      ffn_norm.weight
//     Gate                 ffn_gate.weight
//     Up                   ffn_up.weight
//     Down                 ffn_down.weight
//     PostFeedForwardNorm  post_ffw_norm.weight
//   The sliding window, with W = gemma3.attention.sliding_window (0 where
//   the file omits it):
//     - W = 0: no layer slides; every layer attends over the trained context
//       with gemma3.rope.freq_base, as describe_layers left it.
//     - W > 0: P = gemma3.attention.sliding_window_pattern, 6 where the file
//       omits it, and B = gemma3.rope.freq_base_swa, 10,000 where it omits
//       it. Layer l with l mod P = P − 1 is global and keeps the trained
//       context and rope.freq_base; every other layer attends over W tokens
//       with base B. With P = 6, five layers of every six slide.
//   The attention scale: 1 / sqrt(head dimension), as describe_layers sets
//   it, unchanged.
//   graph(model, plan, cache_format, out): GraphFn's preconditions
//   (arch/architecture.h). Appends the launches below, or returns the first
//   GraphResult a block refuses with, naming the layer.
//
// Algorithm
//   describe:
//     1. read_hyperparameters(index, "gemma3").
//     2. describe_layers with the 13 role names above.
//     3. The sliding window above, over the layer records describe_layers
//        made.
//   graph, over a Builder of (model, plan, cache_format, out):
//     1. embed: the token embedding's rows times sqrt(embedding width),
//        computed in f32: sqrt(1,152) ≈ 33.94 for Gemma 3 1B.
//     2. For each layer in order: attention(layer), then
//        gated_feed_forward(layer). Each block's QK-norm, post-norm and
//        residual add are graph.h's.
//     3. output: the final norm, the head, selection and the draw.
//
// For Gemma 3 1B's pinned file: W = 512, no pattern and no
// rope.freq_base_swa, so P = 6 and B = 10,000; 26 layers, of which 5, 11, 17
// and 23 are global, with the trained context of 32,768 and base 1,000,000,
// and the other 22 slide over 512 tokens with base 10,000. Its attention
// scale is 1 / sqrt(256) = 1 / 16.
//
// Design and cost
//   - The pattern's default and the window's base apply only where the file
//     omits them, and only when the file states a window. The pinned file
//     omits both, so the defaults are what runs.
//   - Counted for Gemma 3 1B, once a load: four metadata lookups beyond
//     describe.h's (the window, the pattern as an array, the pattern as a
//     number, the window's base), and one pass over the 26 layer records.
//     The graph's launches are counted in graph/graph.h.
//
// Verification
//   The pinned file reaches describe's success path, with the window's
//   defaults, and this graph, through every load; the oracle checks the
//   logits that follow (tests/README.md, integration). Untested: every
//   refusal above, a file stating the pattern or the window's base, and a
//   file with no window.
//
// Corpus guidance
//   C++ Core Guidelines
//     ES.45  Symbolic constants: the default pattern and the window's
//            default base are named constants in gemma3.cpp.
//     Con.1  Immutable by default: the role names are one constexpr table,
//            and kGemma3 a const entry.
//     E.27   Error codes systematically: each refusal here is a
//            DescribeError naming its key.

// The gemma3 architecture, as the capability table lists it.
extern const Architecture kGemma3;

}  // namespace frizzle::arch
