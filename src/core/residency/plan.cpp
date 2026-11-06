// Axis D: changes with the limits a device grants, or how the model is placed
// within them.
//
// Contract, algorithm, cost and corpus guidance: plan.h. This file defines
// what that header states.

#include "core/residency/plan.h"

#include <algorithm>
#include <array>
#include <optional>
#include <span>

#include "core/formats/format.h"
#include "core/gguf/checked.h"

namespace frizzle::residency {
namespace {

using gguf::checked_add;
using gguf::checked_mul;

constexpr std::uint64_t kBindingGranule =
    4;  // a storage binding's size is a multiple of 4
constexpr std::uint64_t kF32Bytes = 4;

// v rounded up to a multiple of `to`, a power of two. False on overflow.
bool round_up(std::uint64_t v, std::uint64_t to, std::uint64_t& out) {
  std::uint64_t sum = 0;
  if (!checked_add(v, to - 1, sum)) return false;
  out = sum & ~(to - 1);
  return true;
}

PlanResult failure(PlanError error, std::string subject = {}) {
  return PlanResult{error, std::move(subject)};
}

// Places ranges into the buffers of one pool, filling a buffer before opening
// the next. Every range starts at an aligned offset and is padded to the
// binding granule.
class Packer {
 public:
  // `buffers` outlives the packer, which appends to it.
  Packer(std::vector<PlannedBuffer>& buffers, Pool pool,
         const DeviceLimits& limits)
      : buffers_(&buffers), pool_(pool), limits_(limits) {}

  // Closes the open buffer if `length` more bytes, at the next aligned
  // offset, would not fit it, so a group placed next starts a new one.
  bool reserve(std::uint64_t length) {
    if (!open_) return true;
    std::uint64_t offset = 0;
    if (!round_up(buffers_->back().size, limits_.storage_offset_alignment,
                  offset)) {
      return false;
    }
    if (offset > limits_.max_buffer_size ||
        length > limits_.max_buffer_size - offset) {
      open_ = false;
    }
    return true;
  }

  // Places `length` bytes. `alone` gives the range a buffer nobody else
  // shares. Precondition: the padded length fits in one buffer.
  bool place(std::uint64_t length, bool alone, BufferRange& out) {
    std::uint64_t padded = 0;
    if (!round_up(length, kBindingGranule, padded)) return false;
    std::uint64_t offset = 0;
    if (open_ && !alone) {
      if (!round_up(buffers_->back().size, limits_.storage_offset_alignment,
                    offset)) {
        return false;
      }
      if (offset > limits_.max_buffer_size ||
          padded > limits_.max_buffer_size - offset) {
        open_ = false;
      }
    }
    if (!open_ || alone) {
      buffers_->push_back({pool_, 0});
      offset = 0;
    }
    buffers_->back().size = offset + padded;
    out = {static_cast<BufferIndex>(buffers_->size() - 1), offset, padded};
    open_ = !alone;
    return true;
  }

 private:
  // A pointer, not a reference, so the packer stays assignable (C.12).
  std::vector<PlannedBuffer>* buffers_;
  Pool pool_;
  DeviceLimits limits_;
  bool open_ = false;
};

// The most one range may hold: it must fit both a binding and a buffer.
std::uint64_t range_limit(const DeviceLimits& limits) {
  return std::min(limits.max_storage_binding_size, limits.max_buffer_size) &
         ~(kBindingGranule - 1);
}

// A group's members, in output order: the first `count` of them.
std::span<const gguf::TensorId> members_of(const PlannedGroup& g) {
  return std::span(g.members).first(g.count);
}

// A layer's candidate groups, members in output order: Q, K, V and gate,
// up, where the layer has them.
std::vector<PlannedGroup> candidate_groups(
    const model::ModelDescription& model) {
  std::vector<PlannedGroup> groups;
  for (const model::LayerDescription& l : model.layers) {
    const auto at = [&](model::Role r) {
      return l.tensors.at(static_cast<std::size_t>(r));
    };
    using model::Role;
    // Each role read once into a local, so every access below is checked
    // where it is made.
    const std::optional<gguf::TensorId> query = at(Role::Query);
    const std::optional<gguf::TensorId> key = at(Role::Key);
    const std::optional<gguf::TensorId> value = at(Role::Value);
    if (query && key && value) {
      groups.push_back({{*query, *key, *value}, 3, {}});
    }
    const std::optional<gguf::TensorId> gate = at(Role::Gate);
    const std::optional<gguf::TensorId> up = at(Role::Up);
    if (gate && up) {
      groups.push_back({{*gate, *up, gguf::TensorId{}}, 2, {}});
    }
  }
  return groups;
}

// The bytes from tensor `first` to tensor `last`, in file order, as the
// packer would place them in one buffer: each padded and aligned. False
// when one of them is more than one piece, so the group cannot be one span.
bool span_bytes(std::span<const gguf::TensorEntry> tensors, std::size_t first,
                std::size_t last, const DeviceLimits& limits,
                std::uint64_t& out) {
  out = 0;
  for (std::size_t k = first; k <= last; ++k) {
    std::uint64_t padded = 0, aligned = 0;
    if (tensors[k].data_length > range_limit(limits) ||
        !round_up(tensors[k].data_length, kBindingGranule, padded) ||
        !round_up(padded, limits.storage_offset_alignment, aligned) ||
        !checked_add(out, aligned, out)) {
      return false;
    }
  }
  return true;
}

PlanResult place_weights(const gguf::TensorIndex& index,
                         const model::ModelDescription& model,
                         const DeviceLimits& limits, ResidencyPlan& out) {
  Packer packer{out.buffers, Pool::Weights, limits};
  const std::uint64_t limit = range_limit(limits);
  const auto tensors = index.tensors();

  // Groups whose members agree in format and input width, by the file
  // index of their first member, and the bytes of their span.
  std::vector<PlannedGroup> groups = candidate_groups(model);
  std::vector<std::uint64_t> reserve_before(tensors.size(), 0);
  std::erase_if(groups, [&](const PlannedGroup& g) {
    const gguf::TensorEntry& head = index.tensor(g.members[0]);
    std::size_t first = tensors.size();
    std::size_t last = 0;
    for (const gguf::TensorId member : members_of(g)) {
      const gguf::TensorEntry& t = index.tensor(member);
      if (t.type != head.type || t.dimensions[0] != head.dimensions[0]) {
        return true;
      }
      first = std::min(first, static_cast<std::size_t>(member));
      last = std::max(last, static_cast<std::size_t>(member));
    }
    std::uint64_t bytes = 0;
    if (!span_bytes(tensors, first, last, limits, bytes) || bytes > limit) {
      return true;
    }
    reserve_before[first] = std::max(reserve_before[first], bytes);
    return false;
  });
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    if (reserve_before[i] != 0 && !packer.reserve(reserve_before[i])) {
      return failure(PlanError::Overflow);
    }
    const gguf::TensorEntry& t = tensors[i];
    const gguf::TensorShape shape{
        t.dimension_count,
        {t.dimensions[0], t.dimensions[1], t.dimensions[2], t.dimensions[3]},
        t.element_count};
    std::vector<WeightPiece> pieces;

    // A row is every dimension but the last (formats/format.h): whole
    // blocks, never split.
    const std::uint64_t row = formats::row_elements(t);
    const std::uint64_t rows = row == 0 ? 0 : t.element_count / row;
    if (rows != 0) {
      const std::uint64_t row_bytes = t.data_length / rows;
      if (row_bytes > limit) {
        return failure(PlanError::RowExceedsBinding, t.name);
      }
      const std::uint64_t rows_per_piece = limit / row_bytes;
      for (std::uint64_t first = 0; first < rows; first += rows_per_piece) {
        const std::uint64_t count = std::min(rows_per_piece, rows - first);
        BufferRange range{};
        if (!packer.place(count * row_bytes, false, range)) {
          return failure(PlanError::Overflow, t.name);
        }
        pieces.push_back(
            {range.buffer, range.offset, range.length, first, count});
      }
    }
    out.tensors.push_back({static_cast<gguf::TensorId>(i),
                           WeightView{t.type, shape, std::move(pieces)}});
  }

  // Each group's span, now its members are placed: one buffer, from the
  // first member's first byte to the last member's last.
  for (PlannedGroup& g : groups) {
    const WeightPiece& head =
        out.tensors[static_cast<std::size_t>(g.members[0])]
            .view.pieces()
            .front();
    std::uint64_t begin = head.offset, end = head.offset + head.length;
    bool one_buffer = true;
    for (const gguf::TensorId member : members_of(g)) {
      const WeightPiece& p =
          out.tensors[static_cast<std::size_t>(member)].view.pieces().front();
      one_buffer = one_buffer && p.buffer == head.buffer;
      begin = std::min(begin, p.offset);
      end = std::max(end, p.offset + p.length);
    }
    if (one_buffer && end - begin <= limit) {
      g.span = {head.buffer, begin, end - begin};
      out.groups.push_back(g);
    }
  }
  return {};
}

// The working buffers the graph declared, shared by every layer, in order.
PlanResult place_scratch(const WorkingSet& working, const DeviceLimits& limits,
                         ResidencyPlan& out) {
  out.partial_rows = working.partial_rows;
  Packer packer{out.buffers, Pool::Scratch, limits};
  for (const ScratchNeed& need : working.needs) {
    std::uint64_t bytes = 0;
    std::uint64_t values = 0;
    if (!checked_mul(need.rows, need.width, values) ||
        !checked_mul(values, kF32Bytes, bytes)) {
      return failure(PlanError::Overflow, std::string(need.purpose));
    }
    if (bytes > range_limit(limits)) {
      return failure(PlanError::ScratchExceedsBinding,
                     std::string(need.purpose));
    }
    BufferRange range{};
    // Alone: a kernel binds one working buffer read-only and another
    // writable, which WebGPU refuses within one buffer (plan.h).
    if (!packer.place(bytes, true, range)) {
      return failure(PlanError::Overflow, std::string(need.purpose));
    }
    out.scratch.push_back({need.purpose, range});
  }
  return {};
}

std::uint64_t pool_bytes(const std::vector<PlannedBuffer>& buffers, Pool pool) {
  std::uint64_t total = 0;
  for (const PlannedBuffer& b : buffers) {
    if (b.pool == pool) total += b.size;
  }
  return total;
}

gguf::TensorType storage_type(policy::CachePrecision precision) {
  switch (precision) {
    case policy::CachePrecision::F16:
      return gguf::TensorType::F16;
  }
  return gguf::TensorType::F16;
}

// What one layer's cache costs: the bytes of one slot of keys (and as many
// again of values), and the most slots the layer ever holds.
struct CacheCost {
  std::uint64_t slot_bytes;
  std::uint64_t ring;
};

// Slots a layer holds at a context of `context` tokens.
std::uint64_t slots_at(const CacheCost& layer, std::uint64_t context) {
  return std::min(context, layer.ring);
}

// The cache's bytes at a context of `context` tokens, before alignment
// padding. False when a size overflows or a layer's keys outgrow a binding.
bool cache_bytes_at(std::span<const CacheCost> layers, std::uint64_t context,
                    std::uint64_t binding, std::uint64_t& out) {
  out = 0;
  for (const CacheCost& layer : layers) {
    std::uint64_t length = 0;
    if (!checked_mul(slots_at(layer, context), layer.slot_bytes, length) ||
        length > binding) {
      return false;
    }
    // length <= binding, so 2 * length cannot wrap.
    if (!checked_add(out, 2 * length, out)) {
      return false;
    }
  }
  return true;
}

// Sizes the cache to the largest context the budget and the binding limit
// allow, and places it.
PlanResult place_cache(const model::ModelDescription& model,
                       const DeviceLimits& limits,
                       const policy::LoadPolicy& policy, ResidencyPlan& out) {
  const gguf::FormatLayout& layout =
      *gguf::format_layout(storage_type(policy.cache_precision));
  const std::uint64_t limit = range_limit(limits);

  // A full-attention layer's window is the trained context, so its ring
  // never binds; a sliding-window layer's ring is its window, a prefill
  // block and the rollback reserve.
  std::vector<CacheCost> layers;
  layers.reserve(model.layers.size());
  for (std::size_t i = 0; i < model.layers.size(); ++i) {
    const model::LayerDescription& l = model.layers[i];
    CacheCost cost{0, std::uint64_t{l.attention_window} + kPrefillBlock +
                          policy.rollback_reserve};
    const std::uint64_t blocks = std::uint64_t{l.key_value_heads} *
                                 (l.head_dimension / layout.block_elements);
    if (!checked_mul(blocks, layout.block_bytes, cost.slot_bytes)) {
      return failure(PlanError::Overflow, "layer " + std::to_string(i));
    }
    layers.push_back(cost);
  }
  // The binding never limits the cache below a prefill block: the "key"
  // working buffer holds kPrefillBlock tokens of the same width at 4 bytes a
  // value and already fits one binding, and every cache precision stores
  // fewer bytes a value than that.
  const std::uint64_t shortest =
      std::min<std::uint64_t>(kPrefillBlock, model.trained_context);

  // Alignment padding the cache may add: at most one granule per range.
  const std::uint64_t slack =
      2 * model.layers.size() *
      (limits.storage_offset_alignment + kBindingGranule);
  const std::uint64_t fixed = out.weight_bytes + out.scratch_bytes + slack;
  const auto fits = [&](std::uint64_t context) {
    std::uint64_t bytes = 0;
    std::uint64_t total = 0;
    return cache_bytes_at(layers, context, limit, bytes) &&
           checked_add(fixed, bytes, total) && total <= policy.memory_budget;
  };
  if (!fits(shortest)) {
    std::uint64_t bytes = 0;
    if (!cache_bytes_at(layers, shortest, limit, bytes) ||
        !checked_add(fixed, bytes, out.total_bytes)) {
      return failure(PlanError::Overflow);
    }
    return failure(PlanError::ExceedsBudget);
  }

  // The cost never falls as the context grows, so the largest context that
  // fits is found by bisection between the shortest and the trained context.
  std::uint64_t low = shortest;
  std::uint64_t high = model.trained_context;
  while (low < high) {
    const std::uint64_t mid = low + (high - low + 1) / 2;
    if (fits(mid)) {
      low = mid;
    } else {
      high = mid - 1;
    }
  }
  out.context_offered = static_cast<std::uint32_t>(low);

  Packer packer{out.buffers, Pool::Cache, limits};
  out.cache.reserve(layers.size());
  for (const CacheCost& layer : layers) {
    PlannedCacheLayer planned{};
    planned.slots =
        static_cast<std::uint32_t>(slots_at(layer, out.context_offered));
    const std::uint64_t length =
        std::uint64_t{planned.slots} * layer.slot_bytes;  // checked by fits()
    if (!packer.place(length, false, planned.keys) ||
        !packer.place(length, false, planned.values)) {
      return failure(PlanError::Overflow);
    }
    out.cache.push_back(planned);
  }
  return {};
}

}  // namespace

PlanResult plan_residency(const gguf::TensorIndex& index,
                          const model::ModelDescription& model,
                          const DeviceLimits& limits,
                          const policy::LoadPolicy& policy,
                          const WorkingSet& working, ResidencyPlan& out) {
  out = ResidencyPlan{};
  out.cache_type = storage_type(policy.cache_precision);
  if (auto r = place_weights(index, model, limits, out); !r.ok()) return r;
  out.weight_bytes = pool_bytes(out.buffers, Pool::Weights);
  if (auto r = place_scratch(working, limits, out); !r.ok()) return r;
  out.scratch_bytes = pool_bytes(out.buffers, Pool::Scratch);
  if (auto r = place_cache(model, limits, policy, out); !r.ok()) return r;
  out.cache_bytes = pool_bytes(out.buffers, Pool::Cache);
  out.total_bytes = out.weight_bytes + out.scratch_bytes + out.cache_bytes;
  return {};
}

}  // namespace frizzle::residency
