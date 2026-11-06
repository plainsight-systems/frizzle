#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "core/gguf/index.h"
#include "core/model/model_description.h"
#include "core/policy/policy.h"
#include "core/residency/weight_view.h"

namespace frizzle::residency {

// Axis D: changes with the limits a device grants, or how the model is placed
// within them.
//
// The residency plan: where every weight, the KV cache and every working
// buffer lies on the device, and the context offered. A pure function of the
// tensor index, the model description, the granted limits, the load policy
// and the working set the graph declares: no GPU, no I/O. The engine runs it
// in begin_load from the file's index alone, before any weight byte is
// written (runtime/engine.h); upload carries it out (upload.h); every kernel
// finds its weights, cache and working buffers through it.
//
// Contract
//   - plan_residency(index, model, limits, policy, working, out): Ok with the
//     whole plan in `out`, or a PlanError and its subject. `out` is reset
//     first and out.cache_type set before anything else, so a failed plan
//     carries it too. On ExceedsBudget, out.total_bytes is what the shortest
//     context worth offering would need, so the refusal states needed
//     against available (runtime/engine.cpp, fit_failure). After any other
//     failure the rest of `out` is partial and is not used.
//   - Preconditions, checked by the engine before the call: every limit is
//     nonzero and storage_offset_alignment is a power of two; `model` was
//     described from `index`.
//   - Three pools, never sharing a buffer: the weights; the KV cache; the
//     working buffers. Every tensor in the index has one PlannedTensor, in
//     index order, whether or not the graph binds it.
//   - Every range starts at a multiple of storage_offset_alignment, is its
//     bytes rounded up to 4 long (a storage binding's size is a multiple of
//     4), and is at most min(max_storage_binding_size, max_buffer_size); no
//     buffer is larger than max_buffer_size.
//   - A weight is cut into pieces of whole rows, a row being every dimension
//     but the last (formats/format.h), so no block is split. A row wider than
//     one range is RowExceedsBinding, naming the tensor.
//   - Each working buffer is a buffer of its own (Design and cost), its rows
//     × width f32 values. One wider than a range is ScratchExceedsBinding,
//     naming it.
//   - A layer's keys and values each hold `slots` entries of key_value_heads
//     × head_dimension values at cache_type; position p lives in slot
//     p mod slots. A full-attention layer's slots are the context offered,
//     so no slot is reused; a sliding-window layer's are never more than its
//     window + kPrefillBlock + the policy's rollback reserve.
//   - The context offered is the largest from min(kPrefillBlock, trained
//     context) to the trained context whose cache fits beside the weights
//     and working buffers within the memory budget, each layer's keys within
//     one range. If the shortest does not fit, ExceedsBudget.
//   - A layer's Q, K and V, and its gate and up, become a PlannedGroup, its
//     members in output order and `span` the range from the first member's
//     first byte to the last member's last, when the members share a format
//     and an input width, each is one piece, and the span lies in one buffer
//     within one range. Members stay where file order puts them, with
//     whatever the file holds between them; upload is the same either way.
//     A pair or triple that does not qualify is no group, and its products
//     run apart (kernels/matmul/matmul.h).
//   - A size past 64 bits is Overflow: a file no device could hold.
//
// Algorithm
//   1. Weights. Find the candidate groups and keep those whose members agree
//      in type and dimensions[0] and whose span, each member and everything
//      between padded and aligned as the packer will place it, fits one
//      range. Then, for each tensor in file order:
//        a. if a kept group's span starts here, close the open buffer unless
//           the span fits after its next aligned offset;
//        b. rows_per_piece = floor(range / row bytes); place each run of up
//           to that many rows: align the open buffer's end, open a new buffer
//           when the run does not fit, pad the length to 4.
//      Last, record each kept group whose members landed in one buffer with
//      a span of at most one range.
//   2. Working buffers, in the order the working set lists them, each rows ×
//      width × 4 bytes, each in a new buffer.
//   3. Cache. For each layer: slot bytes = key_value_heads × (head_dimension
//      / block elements) × block bytes of cache_type; its most slots, window
//      + kPrefillBlock + rollback reserve (a full-attention layer's window is
//      the trained context, so that never binds). fixed = weight bytes +
//      working bytes + 2 × layers × (alignment + 4), the most padding the
//      cache's ranges can add. Bisect for the largest context c with fixed +
//      Σ 2 × min(c, most slots) × slot bytes within the budget and each
//      layer's keys within a range: the cost never falls as c grows. Place
//      each layer's keys, then its values.
//   4. Sum each pool's buffers, padding included, and the three.
//
// Design and cost
//   - Three pools because their lifetimes and uses differ: weights written
//     once at load and only read; the cache written every step and kept
//     across turns; working buffers rewritten inside every step. Weights and
//     cache are packed into as few large buffers as the limits allow,
//     suballocated, rather than one allocation a tensor.
//   - Each working buffer has a buffer of its own. WebGPU tracks a
//     dispatch's use of a buffer across the whole buffer, not by range, and
//     refuses one bound both writable and read-only in one dispatch; a
//     kernel reads one working buffer and writes another (the norm reads
//     `output` and writes `hidden`), so two in one buffer could not be bound
//     together. Weights are only read, and no dispatch both reads and writes
//     the cache, so those pools stay suballocated. Gemma 3 1B's graph
//     declares 19 working buffers (graph/graph.cpp, decoder_working_set): 19
//     allocations at load, nothing a step.
//   - Limits are the ones the device granted, which are WebGPU's defaults
//     (gpu/device_requirements.h): 128 MiB a binding, 256 MiB a buffer. The
//     packing works at those.
//   - Gemma 3 1B, counted from the pinned file: 340 tensors, 715,385,344
//     bytes (682 MiB) before padding; 342 pieces, the Q8_0 token embedding's
//     320,864,256 bytes cut into three of at most 109,655 rows. 52 groups: a
//     layer's QKV span runs from attn_k to attn_v and holds attn_output and
//     three norms between, 1,665,536 bytes before padding; its gate and up
//     span holds ffn_norm between, 8,962,560 bytes.
//   - Working values are f32 everywhere: f16 arithmetic needs WebGPU's
//     optional shader-f16 feature, which the device is not requested with
//     (gpu/device_requirements.h), and f16 overflows at 65,504, which
//     activations can reach. Only the cache's storage is narrower, F16, so
//     its rounding is the only storage error the plan introduces.
//   - The working buffers hold a prefill block of kPrefillBlock tokens, and
//     one row of logits: only a step's last token is scored. 512 is
//     Frizzle's choice, not measured: a larger block reads each weight once
//     for more tokens; a smaller one shrinks the working buffers, which grow
//     with it, returns to the page sooner and leaves more of the budget to
//     the cache.
//   - Attention never stores a block-by-context matrix of scores: for Gemma
//     3 1B, 512 tokens × 4 heads × 32,768 keys × 4 bytes is 256 MiB, counted,
//     more than one binding. Kernels work within these buffers, attention in
//     chunks of kChunkKeys keys.
//   - Optimization: a sliding-window layer's cache is a ring of window +
//     kPrefillBlock + rollback reserve. A step writes up to kPrefillBlock new
//     tokens while its first query still reads a window behind them, so
//     window + kPrefillBlock is the least a step needs; the reserve is how
//     far a turn can roll back. Gemma 3 1B at its trained 32,768 tokens, F16,
//     the default reserve of 4,096, counted: 22 layers of 5,120 slots and 4
//     of 32,768, 1 KiB a slot for keys and values, 238 MiB, against 832 MiB
//     with every layer holding the whole context.
//   - Cost: a pure function, run once a load. The bisection is about 15
//     cache sums for a 32,768-token trained context, each a loop over the
//     layers.
//
// Corpus guidance
//   C++ performance guidelines
//     GPU.9   Suballocate from large heaps, pools apart by lifetime: weights
//             and cache suballocated, three pools; working buffers dedicated,
//             as its guidance allows where an API requirement justifies it.
//     WASM.10 Plan from the granted limits, never the adapter's, and work at
//             the defaults: DeviceLimits is what the device granted.
//     CACHE.5 Field order decides padding: PlannedBuffer's one-byte Pool is
//             padded to its size's alignment, so its size is unchanged.
//     MEM.11  The schedule is known before execution, so every offset is
//             planned once and replayed; invalid input fails explicitly.
//             Working buffers are not aliased by lifetime: each binds alone.
//   C++ Core Guidelines
//     E.27    Error codes systematically: PlanError, with the subject named.
//     ES.103  Don't overflow: sizes are summed and multiplied with checked
//             arithmetic, and overflow is a PlanError.
//     F.20    Prefer return values: not followed. `out` is filled in place so
//             that a failed plan still carries cache_type and, on
//             ExceedsBudget, total_bytes for the refusal.
//     Enum.7  Specify an enumeration's underlying type only when necessary:
//             Pool and PlanError take one byte, the smallest that holds them;
//             its reason, space, is real only where no padding follows.
//     C.12    No reference data members in a copyable type: the packer holds
//             its pool's buffers by pointer.
//     SL.con.3 Avoid bounds errors: a group's members are walked as a span of
//             its count, a layer's roles read with at().

// The block of tokens one prefill step processes, and the shortest context
// worth offering.
inline constexpr std::uint32_t kPrefillBlock = 512;

// The keys an attention chunk holds (kernels/attention/attention.h), by
// which the partial buffers are sized.
inline constexpr std::uint32_t kChunkKeys = 64;

// The query rows a split step's partial buffers hold for a model of
// `trained_context`: a prefill block's, or every chunk of the context where
// that is more, so a decode step splits at every position any context the
// plan offers reaches.
[[nodiscard]] constexpr std::uint32_t partial_rows_for(
    std::uint32_t trained_context) noexcept {
  const std::uint32_t chunks = trained_context / kChunkKeys +
                               (trained_context % kChunkKeys != 0 ? 1 : 0);
  return chunks > kPrefillBlock ? chunks : kPrefillBlock;
}

// The candidates top-k selection keeps for the draw, and the entries one of
// its workgroups reduces to them (kernels/topk/topk.h): what its working
// buffers are sized by.
inline constexpr std::uint32_t kCandidates = 64;
inline constexpr std::uint32_t kSelectionTile = 1024;

// A working buffer a graph binds: its purpose, which the graph finds it by,
// and its rows of `width` f32 values.
struct ScratchNeed {
  std::string_view purpose;
  std::uint64_t rows;
  std::uint64_t width;
};

// What a graph asks the planner to place beside the weights: its working
// buffers, in order; and, for one that splits attention by key chunks, the
// partial buffers' query rows (kernels/attention/attention.h), or 0.
struct WorkingSet {
  std::vector<ScratchNeed> needs;
  std::uint32_t partial_rows = 0;
};

// The limits the device was granted. How many bytes the plan may place on the
// device is not among them — WebGPU does not report device memory — and comes
// from load policy instead.
struct DeviceLimits {
  std::uint64_t max_buffer_size;
  std::uint64_t max_storage_binding_size;
  std::uint32_t storage_offset_alignment;
};

// One byte, all its values need. A host-side value only, never stored in a
// GPU buffer, a file or a message. In PlannedBuffer the size beside it pads
// it back to 8 bytes, so nothing is saved there (CACHE.5).
enum class Pool : std::uint8_t {
  Weights,
  Cache,
  Scratch,
};

struct PlannedBuffer {
  Pool pool;
  std::uint64_t size;
};

// A bound range of one planned buffer. Its length is what is bound: the bytes
// it holds rounded up to 4, since a storage binding's size must be a multiple
// of 4 (WebGPU). A weight piece's bytes in the file are its rows times the
// row's bytes.
struct BufferRange {
  BufferIndex buffer;
  std::uint64_t offset;
  std::uint64_t length;
};

struct PlannedTensor {
  gguf::TensorId tensor{};
  WeightView view;
};

// One layer's cache storage. Position p lives in slot p mod slots; for a
// full-attention layer slots is the context offered, so no slot is reused.
struct PlannedCacheLayer {
  BufferRange keys;
  BufferRange values;
  std::uint32_t slots;
};

// A working buffer, named for what it holds.
struct PlannedScratch {
  std::string_view purpose;
  BufferRange range;
};

// A fused group: its members in output order — Q, K, V, or gate, up — and
// the range from the first member's first byte to the last member's last.
struct PlannedGroup {
  std::array<gguf::TensorId, 3> members;
  std::uint32_t count;
  BufferRange span;
};

struct ResidencyPlan {
  std::vector<PlannedBuffer> buffers;
  std::vector<PlannedTensor> tensors;
  std::vector<PlannedGroup> groups;
  std::vector<PlannedCacheLayer> cache;
  // The type the cache is stored in, from the policy's precision; set by
  // plan_residency before anything else, so a failed plan has it too.
  gguf::TensorType cache_type{};
  std::vector<PlannedScratch> scratch;
  // The partial buffers' query rows, as the working set gave them.
  std::uint32_t partial_rows = 0;
  std::uint32_t context_offered = 0;
  // Bytes in each pool, alignment padding included, and their sum. On
  // ExceedsBudget, total_bytes is what the shortest context worth offering
  // would need, so a rejection can state needed against available.
  std::uint64_t weight_bytes = 0;
  std::uint64_t cache_bytes = 0;
  std::uint64_t scratch_bytes = 0;
  std::uint64_t total_bytes = 0;
};

// One byte, as Pool: returned to the engine, never stored or sent as a number.
enum class PlanError : std::uint8_t {
  Ok,
  // Weights and working buffers leave too little of the budget for a
  // kPrefillBlock-token cache.
  ExceedsBudget,
  // One row of a weight is wider than a storage binding, so no split by
  // rows can place it. The subject names the tensor.
  RowExceedsBinding,
  // A working buffer is wider than a binding. The subject names it.
  ScratchExceedsBinding,
  // A size that overflows 64 bits: a file no device could hold.
  Overflow,
};

struct PlanResult {
  PlanError error = PlanError::Ok;
  std::string subject;

  [[nodiscard]] bool ok() const noexcept { return error == PlanError::Ok; }
};

// Plans `model` for `limits` and `policy`, placing the working set its
// graph declared.
[[nodiscard]] PlanResult plan_residency(const gguf::TensorIndex& index,
                                        const model::ModelDescription& model,
                                        const DeviceLimits& limits,
                                        const policy::LoadPolicy& policy,
                                        const WorkingSet& working,
                                        ResidencyPlan& out);

}  // namespace frizzle::residency
