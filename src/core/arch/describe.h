#pragma once

#include <cstdint>
#include <span>
#include <string_view>

#include "core/arch/architecture.h"
#include "core/gguf/index.h"
#include "core/model/model_description.h"

namespace frizzle::arch {

// Axis C: changes with the GGUF keys and tensor names describe reads, or
// what it refuses of them.
//
// What describing does that is not particular to Gemma 3: reading the
// hyperparameters a GGUF file states as "<arch>.<key>", refusing the values
// and forms the kernels do not implement, and finding each layer's weights,
// which GGUF files name blk.<layer>.<suffix>, each checked against the shape
// its role has in terms of the hyperparameters. Gemma 3's describe
// (arch/gemma3/gemma3.h) supplies the architecture name, its role names and
// what its layers do differently from one another.
//
// Contract
//   read_hyperparameters(index, arch, out): Ok with `out` filled, or the
//   first refusal, naming the key, with `out` partly written. In order:
//     1. block_count, context_length, embedding_length, feed_forward_length
//        and attention.head_count: required UINT32 values; zero is
//        InvalidValue.
//     2. attention.head_count_kv: a required UINT32, one count for every
//        layer. Zero is InvalidValue; an array, like any other type, is
//        WrongKeyType from the scalar read.
//     3. attention.layer_norm_rms_epsilon and rope.freq_base: required
//        FLOAT32 values, not range-checked.
//     4. Query heads not a multiple of key-value heads: InvalidValue, naming
//        attention.head_count_kv.
//     5. attention.key_length, the head dimension: optional, by default
//        embedding_length / head_count in integers. Zero is InvalidValue; odd
//        is UnsupportedValue, as rope pairs a head's dimensions and the last
//        would have no pair.
//     6. rope.dimension_count: optional, by default the head dimension; any
//        other value, part of each head rotated, is UnsupportedValue.
//     7. attn_logit_softcapping and final_logit_softcapping: optional, by
//        default 0; nonzero, a tanh cap no kernel applies, is
//        UnsupportedValue.
//     8. rope.scaling.type: optional; a string other than "none", positions
//        scaled, is UnsupportedValue.
//     9. attention.value_length: optional, by default the head dimension; any
//        other value is UnsupportedValue, as the description has one head
//        dimension for keys and values.
//    10. block_count larger than the file's tensor count: InvalidValue, as
//        every layer needs tensors of its own; checked before the count
//        sizes anything.
//     A required key the file lacks is MissingKey; any key declared with
//     another type than the one read is WrongKeyType. A file in any form
//     refused in 5 to 9 would otherwise load and run with wrong results.
//   describe_layers(index, hp, roles, out): Ok with `out` filled as
//   model/model_description.h states, or the first refusal, naming the key
//   or tensor, with `out` partly written. The vocabulary is the element
//   count of tokenizer.ggml.tokens, which must be an array. Shapes are as
//   GGUF lists them, a matrix's input width first, with e the embedding
//   width, q = query heads × head dimension, kv = key-value heads × head
//   dimension, f the feed-forward width and V the vocabulary:
//     token_embd.weight    required, [e, V]
//     output_norm.weight   required, [e], F32
//     output.weight        optional, [e, V] where present
//     blk.<i>.<suffix>     required for every role in `roles`, every layer:
//       Query [e, q]; Key and Value [e, kv]; AttentionOutput [q, e];
//       Gate and Up [e, f]; Down [f, e]; QueryNorm and KeyNorm [head
//       dimension]; AttentionNorm, PostAttentionNorm, FeedForwardNorm and
//       PostFeedForwardNorm [e].
//   A tensor absent is MissingTensor, a shape that differs ShapeMismatch. A
//   norm gain, the output norm's or a layer's, stored as anything but F32 is
//   UnsupportedValue: the norm and rope kernels bind gains as f32 and read
//   no format. Other weights' types are not checked here; the capability
//   table's formats decide whether they run (residency/routes.h). A file
//   with rope_freqs.weight is UnsupportedValue: rope applies no frequency
//   factors. Every layer comes out attending over the full context with the
//   file's rotary base, and attention_scale 1 / sqrt(head dimension); Gemma
//   3's describe adjusts both afterwards.
//   read_u32_or and read_f32_or: an optional "<arch>.<key>": `fallback` where
//   the file omits it, WrongKeyType where it declares another type.
//
// Algorithm, describe_layers
//   1. The vocabulary from tokenizer.ggml.tokens; the embedding width,
//      trained context and epsilon from `hp`.
//   2. token_embd.weight, then output_norm.weight and its type, then
//      output.weight if present, each by name and shape.
//   3. attention_scale = 1 / sqrt(head dimension), in f32.
//   4. Refuse rope_freqs.weight.
//   5. For layer 0 to block_count − 1: append a record of the query heads,
//      key-value heads, head dimension, feed-forward width, the trained
//      context as its window and the file's rotary base; then, for each
//      role in `roles`, find blk.<layer>.<suffix>, check its shape, and a
//      gain's type.
//
// For Gemma 3 1B's pinned file: block_count 26, context 32,768, embedding
// 1,152, feed-forward 6,912, 4 query heads, 1 key-value head, key_length and
// value_length 256 (stated, not the default 288), rope.freq_base 1,000,000,
// no rope.dimension_count, logit caps or rope.scaling.type; 340 tensors.
// Query [1152, 1024], Key and Value [1152, 256], AttentionOutput [1024,
// 1152], Gate and Up [1152, 6912], Down [6912, 1152].
//
// Design and cost
//   - Out parameters, filled as describe goes and left partly written on a
//     refusal, which the engine discards with the description.
//   - The vocabulary's count is narrowed to 32 bits with a cast and not
//     checked: a count past 2^32 would wrap, and the token embedding's shape
//     would then be checked against the wrapped value. 262,144 fits.
//   - Counted for Gemma 3 1B, once a load: read_hyperparameters makes 14
//     metadata lookups, each a scan of at most the file's 38 keys and one
//     std::string key built; describe_layers makes one more, and 342 tensor
//     finds (13 roles × 26 layers, and four global names), each a binary
//     search of about nine comparisons and one std::string name built. The
//     layer records are appended one at a time, not reserved.
//
// Verification
//   No test exercises describe directly. Gemma 3 1B's pinned file reaches
//   its success path through every load, the oracle's among them
//   (tests/README.md, integration), with key_length and value_length stated.
//   Untested: every refusal above, the key_length default, each optional
//   key present with an accepted value (rope.dimension_count, the logit caps
//   at 0, rope.scaling.type "none"), an output.weight present and its shape
//   check, and the vocabulary's narrowing.
//
// Corpus guidance
//   C++ Core Guidelines
//     E.27   Error codes systematically: every refusal is a DescribeError
//            with the key or tensor as its subject.
//     I.4    Precisely and strongly typed interfaces: a role is a
//            model::Role and a tensor a gguf::TensorId, not an index.
//     F.20   Prefer returning values to out parameters: not followed;
//            describe fills the description in place, partly on a refusal.
//     ES.46  Avoid lossy narrowing: not followed for the vocabulary's count,
//            as above.
//   C++ performance guidelines
//     Searched for the cost of key and name lookups and for validating an
//     untrusted file's counts: the rules found are for hot paths and byte
//     ranges, and none applies to a parse made once a load.

// The hyperparameters read_hyperparameters reads, checked as above.
struct Hyperparameters {
  std::uint32_t block_count;
  std::uint32_t context_length;
  std::uint32_t embedding_length;
  std::uint32_t feed_forward_length;
  std::uint32_t head_count;
  std::uint32_t head_count_kv;
  std::uint32_t head_dimension;
  float norm_epsilon;
  float rope_base;
};

// What an architecture's file calls a layer's weight in a role: the suffix
// of blk.<i>.<suffix>.
struct RoleName {
  model::Role role;
  std::string_view suffix;
};

// The hyperparameters, read and refused in the order the header lists.
[[nodiscard]] DescribeResult read_hyperparameters(
    const gguf::TensorIndex& index, std::string_view arch,
    Hyperparameters& out);

// The vocabulary, the global weights and every layer's weights by role,
// each checked as the header lists; every layer full context with the file's
// rotary base, and attention_scale 1 / sqrt(head dimension).
[[nodiscard]] DescribeResult describe_layers(const gguf::TensorIndex& index,
                                             const Hyperparameters& hp,
                                             std::span<const RoleName> roles,
                                             model::ModelDescription& out);

// Reads an optional "<arch>.<key>": `fallback` when the file omits it, an
// error when it declares it with another type.
[[nodiscard]] DescribeResult read_u32_or(const gguf::TensorIndex& index,
                                         std::string_view arch,
                                         std::string_view key,
                                         std::uint32_t fallback,
                                         std::uint32_t& out);
[[nodiscard]] DescribeResult read_f32_or(const gguf::TensorIndex& index,
                                         std::string_view arch,
                                         std::string_view key, float fallback,
                                         float& out);

}  // namespace frizzle::arch
