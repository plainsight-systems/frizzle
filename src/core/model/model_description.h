#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "core/gguf/index.h"

namespace frizzle::model {

// Axis A: changes with Gemma 3's graph: a weight or number its order of work
// in a layer needs.
//
// Every number the planner, the cache and the graph need, filled once by the
// architecture's describe (arch/architecture.h) from the tensor index.
// Nothing downstream reads the file's metadata again: the residency plan, the
// cache, the graph, the capture layout and the edit table all take the
// model's widths, and the graph its weights, from here.
//
// Contract
//   - Filled by describe alone, which takes it by non-const reference; every
//     other reader takes it by const reference. A describe that fails leaves
//     it partly written, and the engine discards it.
//   - Weights are named by role and TensorId, never by the tensor's name in
//     the file: the graph and the kernels address a weight as "layer 3, Key",
//     and only the architecture knows that its file calls it
//     blk.3.attn_k.weight. Each TensorId came from the index describe read,
//     and is valid with that index alone.
//   - token_embedding and output_norm are always set by a describe that
//     succeeds. output_head is optional: absent, the head reads the token
//     embedding, as it does for Gemma 3 1B's pinned file, which has no
//     output.weight.
//   - Values that differ between layers are per layer, in `layers`, one
//     record a layer in the file's order (blk.0 first). Gemma 3's window and
//     rotary base differ between its layers; its head counts, head dimension
//     and feed-forward width do not, and are carried in each record beside
//     them, so a layer's launches read one record.
//   - Role: the 13 roles a weight plays in a layer. A layer's `tensors` holds
//     the tensor filling each role, or nothing where the layer has no such
//     weight. Gemma 3's describe requires all 13 in every layer, so in its
//     description every slot is set; the graph still detects the post-norms
//     from their slots rather than assuming them (graph/graph.h). Adding a
//     role is a change to this contract.
//   - attention_scale: what attention multiplies q · k by. Describe sets
//     1 / sqrt(head dimension).
//   - Numbers as the file states them, with no unit conversion: widths and
//     counts in elements, the context and windows in tokens, norm_epsilon
//     and rope_base as the file's f32 values, or where the file omits a
//     window layer's base, Gemma 3's default (arch/gemma3/gemma3.h).
//
// Algorithm
//   None: a plain aggregate. Describe writes it in this order: vocabulary,
//   widths, context and epsilon; the three global weights; the attention
//   scale; then one layer record at a time, each with its tensors by role;
//   then the architecture's per-layer window.
//
// Gemma 3 1B's pinned file, as describe fills it from that file's metadata
// and tensors: vocabulary 262,144; embedding width 1,152; trained context
// 32,768; norm epsilon 1e-6 (as f32); token embedding Q8_0, 1,152 × 262,144;
// output norm F32; no output head; attention scale 1 / 16. 26 layers, each 4
// query heads, 1 key-value head, head dimension 256, feed-forward width
// 6,912. Layers 5, 11, 17 and 23 attend over the whole 32,768-token context
// with rotary base 1,000,000; the other 22 over a window of 512 with base
// 10,000.
//
// Design and cost
//   - One allocation, the vector of layer records, made once a load by
//     describe. A record is six 4-byte numbers and 13 optional TensorIds;
//     counted from the declaration, 128 bytes on the usual ABIs, so 3,328
//     bytes for Gemma 3 1B's 26 layers.
//   - Per-layer records rather than global fields with exceptions: the
//     cache and the graph read a layer's window and base from its record,
//     with no case for which layers slide.
//   - No test locks describe's output field by field (arch/describe.h). The
//     pinned file's description is reached by every load, the oracle's
//     among them (tests/README.md, integration), whose logits depend on each
//     field above.
//
// Corpus guidance
//   C++ Core Guidelines
//     I.4    Precisely and strongly typed interfaces: TensorId and LayerIndex
//            are their own types, and a weight is addressed by Role, so a
//            tensor's position cannot be passed where a layer is meant.
//     Enum.3 Prefer class enums: Role converts to an index only through an
//            explicit cast, as `tensors` is indexed.
//     Enum.7 Specify an enumeration's underlying type only when necessary:
//            Role takes one byte, the smallest that holds it.
//     ES.45  Symbolic constants: kRoleCount is Role::Count, so the array of
//            slots grows with the enumeration, never a literal 13.
//     Con.1  Immutable by default: written by describe, then read through
//            const references only.

enum class LayerIndex : std::uint32_t {};

// The roles a weight plays in a layer. Adding one is a change to this
// contract.
enum class Role : std::uint8_t {
  AttentionNorm,
  Query,
  Key,
  Value,
  AttentionOutput,
  QueryNorm,
  KeyNorm,
  PostAttentionNorm,
  FeedForwardNorm,
  Gate,
  Up,
  Down,
  PostFeedForwardNorm,
  Count,
};

inline constexpr std::size_t kRoleCount = static_cast<std::size_t>(Role::Count);

struct LayerDescription {
  std::uint32_t query_heads = 0;
  std::uint32_t key_value_heads = 0;
  std::uint32_t head_dimension = 0;
  std::uint32_t feed_forward_width = 0;
  // Tokens this layer attends over: the trained context for a
  // full-attention layer, the window for a sliding-window layer. A query at
  // position p reads keys at p - attention_window + 1 .. p.
  std::uint32_t attention_window = 0;
  float rope_base = 0.0f;
  // The tensor filling each role. Empty where this layer has no such weight.
  std::array<std::optional<gguf::TensorId>, kRoleCount> tensors;
};

struct ModelDescription {
  std::uint32_t vocabulary_size;
  std::uint32_t embedding_width;
  std::uint32_t trained_context;
  float norm_epsilon;
  gguf::TensorId token_embedding;
  gguf::TensorId output_norm;
  // Empty when the output head reads the token embedding.
  std::optional<gguf::TensorId> output_head;
  // What attention multiplies q · k by (above). Set by describe.
  float attention_scale;
  std::vector<LayerDescription> layers;
};

}  // namespace frizzle::model
