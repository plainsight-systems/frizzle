#pragma once

#include <cstdint>
#include <vector>

#include "core/model/model_description.h"
#include "core/policy/policy.h"
#include "core/residency/plan.h"

namespace frizzle::cache {

// Axis G: changes with a new cache or context mechanism.
//
// The KV cache's bookkeeping: how many tokens it holds, and whether a
// truncation keeps them. The storage is planned by the residency plan
// (residency/plan.h) and created by upload; the cache names its buffers by
// plan index and holds no GPU object.
//
// Contract
//   - A full-attention layer holds every token of the context offered. A
//     sliding-window layer holds a ring of the slots the residency plan
//     gives it: its window, a prefill block and the policy's rollback
//     reserve. Position p lives in slot p mod slots, in every layer.
//   - A query at position p reads keys at p − window + 1 .. p, its own
//     position included.
//   - Truncation resets a counter and moves no data. A ring has overwritten
//     what lies further back than its slots, and recomputing those entries
//     needs the ones before them, back to the first token. So a rollback
//     within the reserve keeps the cache, and a deeper one empties it.
//   - What a ring holds is set by how far it has been written, not by the
//     length: a rollback leaves the entries past it in their slots until new
//     tokens overwrite them, and they had already overwritten older ones. So
//     the cache keeps a high-water mark, the most tokens written since it
//     was last emptied, and a layer holds positions mark − slots .. mark − 1,
//     or all of them while mark <= slots.
//   - truncate(tokens) keeps the first `tokens` tokens and returns `tokens`
//     when, in every layer, every earlier position the next query at
//     `tokens` reads, tokens − window + 1 .. tokens − 1, is still held; the
//     entries from `tokens` on are overwritten in place as new tokens are
//     written. Otherwise it empties the cache and returns 0, and the caller
//     prefills from the first token. A full-attention layer's slots are the
//     capacity, so it never overwrites and never decides; one rule serves
//     both kinds of layer.
//   - The rope kernel writes the step's new entries (kernels/rope/rope.h),
//     and attention reads the layer's window; the runtime then advances the
//     cache by the step's token count.
//   - Capacity is the context offered. Each layer's window comes from the
//     model description, storage precision from load policy, and packing
//     from the format for that precision.
//   - Invariant: length() <= capacity(). Preconditions, stated on each
//     member, not checked: a broken one leaves the counts wrong and nothing
//     reports it. The constructor's: `model` describes at least as many
//     layers as `plan.cache` holds, in the same order.
//
// Algorithm
//   truncate(t):
//     1. For each layer, when t > 0 and its window is above 1: the oldest
//        position held is written − slots when written > slots, else 0; the
//        oldest the next query reads is t − window + 1 when t >= window,
//        else 0. When the oldest held is past the oldest read, length and
//        written become 0 and truncate returns 0.
//     2. Otherwise length becomes t, written becomes 0 when t is 0, and
//        truncate returns t.
//   advance(n): length grows by n, and written becomes the larger of written
//   and length.
//
// Design and cost
//   - Counted for Gemma 3 1B (26 layers, 1 key-value head, head dimension
//     256, the cache F16, the one precision policy offers): truncate, once a
//     turn, visits each of the 26 layer records at most once, stopping at the
//     first that fails, 64 bytes each, at most 1,664 bytes, reading two u32
//     counts from each; advance and layer are constant time, once a step.
//     Construction copies the plan's 26 layer records once at load; nothing
//     is allocated after it, so the per-token path allocates nothing
//     (MEM.9). No key or value moves on the CPU.
//   - What a truncation keeps, counted from Algorithm 1 for Gemma 3 1B at
//     the default reserve of 4,096 and a context offered of at least 5,120:
//     its 22 sliding layers ring 5,120 slots (512 window, 512 prefill block,
//     4,096 reserve; residency/plan.h, which gives a smaller context its
//     whole length), so with a window of 512 a truncation to t keeps the
//     cache when the high-water mark is at most 4,609 positions past t
//     (5,120 − 511), or no more than 5,120. Its 4 full-attention layers
//     never decide.
//   - Kept apart from the storage, so its bookkeeping runs without a device.
//     No unit test covers it (tests/README.md); the replay scenario's cache
//     reuse cases run it through the engine.
//
// Corpus guidance
//   C++ Core Guidelines
//     C.2    Use class if the class has an invariant: length() <=
//            capacity(), kept by the stated preconditions.
//     I.5    State preconditions: on the constructor, truncate, advance and
//            layer; stated, not checked, as elsewhere in the core.
//     E.12   noexcept where a throw is impossible: the accessors, truncate
//            and advance only read and write counts; layer reads an element
//            its precondition keeps in bounds.
//   C++ performance guidelines
//     MEM.9  Allocate at init, not in steady state: the layers, at load.
//     GPU.1  Keep data on the device: keys and values never cross to the
//            CPU; the cache keeps only counts.
//   Searched, and no rule applies: the bookkeeping of a ring that backs a
//   sliding window (the rules found are for rings between threads,
//   block-scaled codes and memory tiers).

class KvCache {
 public:
  KvCache(const model::ModelDescription& model,
          const residency::ResidencyPlan& plan,
          policy::CachePrecision precision, std::uint32_t capacity);

  // Tokens whose keys and values are held.
  [[nodiscard]] std::uint32_t length() const noexcept;
  [[nodiscard]] std::uint32_t capacity() const noexcept;
  [[nodiscard]] policy::CachePrecision precision() const noexcept;

  // Keeps the first `tokens` tokens and returns how many were kept: `tokens`
  // when every sliding-window layer still holds the window before that
  // position, otherwise 0, and the caller prefills from the first token.
  // Precondition: tokens <= length().
  [[nodiscard]] std::uint32_t truncate(std::uint32_t tokens) noexcept;

  // Precondition: length() + tokens <= capacity().
  void advance(std::uint32_t tokens) noexcept;

  // Precondition: `layer` is a layer of the model.
  [[nodiscard]] const residency::PlannedCacheLayer& layer(
      model::LayerIndex layer) const noexcept;

 private:
  struct Layer {
    residency::PlannedCacheLayer planned;
    std::uint32_t window;  // the model's attention window for the layer
  };

  std::vector<Layer> layers_;
  // The most tokens written since the cache was last emptied: what bounds
  // what each ring still holds, which length alone does not after a
  // rollback.
  std::uint32_t written_ = 0;
  std::uint32_t length_ = 0;
  std::uint32_t capacity_;
  policy::CachePrecision precision_;
};

}  // namespace frizzle::cache
