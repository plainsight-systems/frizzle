// Axis Q: changes with a kind of edit, or what one does.
//
// Contract, algorithm, cost and corpus guidance: edit.h. This file defines
// what that header states.

#include "core/edit/edit.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <iterator>
#include <map>
#include <string>
#include <tuple>

namespace frizzle::edit {
namespace {

constexpr std::size_t kTableHeadWords = 2;  // [first, count] a layer

// Edit indices by (layer, neuron, from): the table's order.
std::vector<std::size_t> by_start(std::span<const Edit> edits) {
  std::vector<std::size_t> order(edits.size());
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::ranges::sort(order, [&](std::size_t a, std::size_t b) {
    const Edit& x = edits[a];
    const Edit& y = edits[b];
    return std::tie(x.layer, x.neuron, x.from, a) <
           std::tie(y.layer, y.neuron, y.from, b);
  });
  return order;
}

// Step s at position origin + s, saturating at kToTheEnd.
std::uint32_t place(std::uint32_t step, std::uint32_t origin) noexcept {
  if (step == kToTheEnd || step > kToTheEnd - origin) return kToTheEnd;
  return step + origin;
}

EditCheck refuse(EditError error, std::size_t index, const std::string& why) {
  return {error, index, "edit " + std::to_string(index) + ": " + why};
}

std::string named(const Edit& e) {
  return "layer " + std::to_string(e.layer) + " neuron " +
         std::to_string(e.neuron);
}

std::string range(const Edit& e) {
  return "[" + std::to_string(e.from) + ", " +
         (e.until == kToTheEnd ? std::string("end") : std::to_string(e.until)) +
         ")";
}

// The fault in one edit on its own, or Ok.
EditCheck check_one(const Edit& e, std::size_t index,
                    const model::ModelDescription& model) {
  if (e.kind != EditKind::Set && e.kind != EditKind::Scale &&
      e.kind != EditKind::Zero) {
    return refuse(EditError::Kind, index,
                  "kind " + std::to_string(static_cast<unsigned>(e.kind)) +
                      " is not Set, Scale or Zero");
  }
  if (e.layer >= model.layers.size()) {
    return refuse(EditError::Layer, index,
                  "layer " + std::to_string(e.layer) + " is past the model's " +
                      std::to_string(model.layers.size()) + " layers");
  }
  const std::uint32_t width = model.layers.at(e.layer).feed_forward_width;
  if (e.neuron >= width) {
    return refuse(EditError::Neuron, index,
                  "neuron " + std::to_string(e.neuron) + " is past layer " +
                      std::to_string(e.layer) + "'s " + std::to_string(width) +
                      " neurons");
  }
  if (!std::isfinite(e.value)) {
    return refuse(EditError::NotFinite, index, "the value is not finite");
  }
  if (e.kind == EditKind::Zero && e.value != 0.0F) {
    return refuse(
        EditError::ZeroValue, index,
        "a Zero edit's value is " + std::to_string(e.value) + ", not 0");
  }
  if (e.from >= e.until) {
    return refuse(EditError::Range, index,
                  named(e) + "'s steps " + range(e) + " hold no step");
  }
  return {};
}

}  // namespace

float apply(EditKind kind, float value, float neuron) noexcept {
  switch (kind) {
    case EditKind::Set:
      return value;
    case EditKind::Scale:
      return neuron * value;
    case EditKind::Zero:
      return 0.0F;
  }
  return neuron;  // not reached for a checked set (check refuses the kind)
}

EditCheck check(std::span<const Edit> edits,
                const model::ModelDescription& model) {
  if (edits.size() > kMaxEdits) {
    return refuse(EditError::TooMany, kMaxEdits,
                  std::to_string(edits.size()) + " edits; at most " +
                      std::to_string(kMaxEdits));
  }
  for (std::size_t i = 0; i < edits.size(); ++i) {
    if (EditCheck one = check_one(edits[i], i, model); !one.ok()) return one;
  }
  // The ranges held so far, which never overlap one another: an edit
  // overlaps one of them exactly when it overlaps the range just before its
  // key or just after it.
  using Key = std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>;
  std::map<Key, std::size_t> held;
  for (std::size_t i = 0; i < edits.size(); ++i) {
    const Edit& e = edits[i];
    const Key key{e.layer, e.neuron, e.from};
    const auto after = held.lower_bound(key);
    std::size_t other = edits.size();
    if (after != held.end()) {
      const Edit& o = edits[after->second];
      if (o.layer == e.layer && o.neuron == e.neuron && o.from < e.until) {
        other = after->second;
      }
    }
    if (other == edits.size() && after != held.begin()) {
      const Edit& o = edits[std::prev(after)->second];
      if (o.layer == e.layer && o.neuron == e.neuron && e.from < o.until) {
        other = std::prev(after)->second;
      }
    }
    if (other < edits.size()) {
      return refuse(EditError::Overlap, i,
                    named(e) + "'s steps " + range(e) + " overlap edit " +
                        std::to_string(other) + "'s " + range(edits[other]));
    }
    held.emplace(key, i);
  }
  return {};
}

std::vector<std::uint32_t> table(std::span<const Edit> edits,
                                 const model::ModelDescription& model,
                                 std::uint32_t origin) {
  const std::size_t layers = model.layers.size();
  std::vector<std::uint32_t> words(kTableHeadWords * layers +
                                   kEditWords * edits.size());
  std::size_t at = kTableHeadWords * layers;
  for (const std::size_t i : by_start(edits)) {
    const Edit& e = edits[i];
    const std::size_t head = kTableHeadWords * e.layer;
    // At most 52 + 5 × kMaxEdits words, so every offset fits a u32.
    if (words.at(head + 1) == 0) {
      words.at(head) = static_cast<std::uint32_t>(at);
    }
    ++words.at(head + 1);
    words.at(at) = e.neuron;
    words.at(at + 1) = static_cast<std::uint32_t>(e.kind);
    words.at(at + 2) = std::bit_cast<std::uint32_t>(e.value);
    words.at(at + 3) = place(e.from, origin);
    words.at(at + 4) = place(e.until, origin);
    at += kEditWords;
  }
  return words;
}

std::vector<Edit> placed(std::span<const Edit> edits, std::uint32_t origin) {
  std::vector<Edit> out(edits.begin(), edits.end());
  for (Edit& e : out) {
    e.from = place(e.from, origin);
    e.until = place(e.until, origin);
  }
  return out;
}

}  // namespace frizzle::edit
