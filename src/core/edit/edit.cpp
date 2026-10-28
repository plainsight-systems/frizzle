// Axis Q: changes with a kind of edit, or what one does.
//
// Contract, algorithm, cost and corpus guidance: edit.h. This file defines
// what that header states.

#include "core/edit/edit.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <string>
#include <tuple>

namespace frizzle::edit {
namespace {

constexpr std::size_t kTableHeadWords = 2;  // [first, count] a layer
constexpr std::size_t kEditWords = 3;       // [neuron, kind, value bits]

struct Keyed {
  std::uint32_t layer;
  std::uint32_t neuron;
  std::size_t index;
};

bool by_neuron(const Keyed& a, const Keyed& b) {
  return std::tie(a.layer, a.neuron, a.index) <
         std::tie(b.layer, b.neuron, b.index);
}

std::vector<Keyed> sorted(std::span<const Edit> edits) {
  std::vector<Keyed> keyed;
  keyed.reserve(edits.size());
  for (std::size_t i = 0; i < edits.size(); ++i) {
    keyed.push_back({edits[i].layer, edits[i].neuron, i});
  }
  std::ranges::sort(keyed, by_neuron);
  return keyed;
}

EditCheck refuse(EditError error, std::size_t index, const std::string& why) {
  return {error, index, "edit " + std::to_string(index) + ": " + why};
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
  // After sorting, a repeat follows the edit it repeats; within one neuron
  // the indices ascend, so each repeat is not its neuron's first edit.
  const std::vector<Keyed> keyed = sorted(edits);
  std::size_t repeat = edits.size();
  for (std::size_t i = 1; i < keyed.size(); ++i) {
    const Keyed& a = keyed[i - 1];
    const Keyed& b = keyed[i];
    if (a.layer == b.layer && a.neuron == b.neuron) {
      repeat = std::min(repeat, b.index);
    }
  }
  if (repeat < edits.size()) {
    const Edit& e = edits[repeat];
    return refuse(EditError::Duplicate, repeat,
                  "layer " + std::to_string(e.layer) + " neuron " +
                      std::to_string(e.neuron) + " is edited twice");
  }
  return {};
}

std::vector<std::uint32_t> table(std::span<const Edit> edits,
                                 const model::ModelDescription& model) {
  const std::size_t layers = model.layers.size();
  std::vector<std::uint32_t> words(kTableHeadWords * layers +
                                   kEditWords * edits.size());
  std::size_t at = kTableHeadWords * layers;
  for (const Keyed& k : sorted(edits)) {
    const Edit& e = edits[k.index];
    const std::size_t head = kTableHeadWords * e.layer;
    // At most 52 + 3 × kMaxEdits words, so every offset fits a u32.
    if (words.at(head + 1) == 0) {
      words.at(head) = static_cast<std::uint32_t>(at);
    }
    ++words.at(head + 1);
    words.at(at) = e.neuron;
    words.at(at + 1) = static_cast<std::uint32_t>(e.kind);
    words.at(at + 2) = std::bit_cast<std::uint32_t>(e.value);
    at += kEditWords;
  }
  return words;
}

}  // namespace frizzle::edit
