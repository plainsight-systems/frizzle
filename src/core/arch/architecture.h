#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "core/formats/format.h"
#include "core/gguf/index.h"
#include "core/graph/graph.h"
#include "core/kernels/interface.h"
#include "core/model/model_description.h"
#include "core/residency/plan.h"

namespace frizzle::arch {

// Axis A: changes with what Gemma 3's describe and graph take and return.
//
// The contract every architecture supplies: a describe and a graph, behind
// one Architecture entry. The capability table lists the one entry there is,
// kGemma3, under general.architecture "gemma3" (capability/capability.cpp).
// An architecture is the only code that knows what its file calls its
// hyperparameters and weights, and the order its model runs the shared
// blocks in.
//
// Contract
//   - describe(index, out): Ok with `out` filled (model/model_description.h),
//     or a DescribeResult whose `subject` names the key or tensor at fault.
//     On failure `out` is left partly written; the caller discards it.
//   - graph(model, plan, cache_format, out): appends every launch of a step
//     to `out`, in order, composed from the blocks in graph/graph.h; Ok, or a
//     GraphResult naming the layer at fault. Its preconditions are on GraphFn
//     below. One graph kind: a decoder, which runs a turn over the KV cache
//     and ends in the draw.
//   - An architecture holds no state. An entry is constant static data, its
//     name and two function pointers.
//   - It rewrites no weights. The file stores each norm gain as the factor
//     the norm multiplies by (for Gemma 3, 1 + w), and no kernel adds 1, so
//     upload writes what the file holds.
//   - DescribeError, by kind: MissingKey, a key the file lacks; WrongKeyType,
//     a key declared with another type, an array among them where one number
//     is read; MissingTensor and ShapeMismatch, a weight absent or not the
//     shape its role requires; InvalidValue, a value no model could run
//     with; UnsupportedValue, a valid value or form the kernels do not
//     implement. Each refusal is listed where it is made (arch/describe.h,
//     arch/gemma3/gemma3.h).
//
// Algorithm, as the engine uses an entry (runtime/engine.h)
//   1. begin_load reads general.architecture, finds the entry in the
//      capability table, and calls describe once.
//   2. The residency plan is made from the description, and upload carries
//      it out.
//   3. finish_load, once the upload is complete, calls graph once, over the
//      plan as upload carried it out (Upload::plan()); the program is built
//      from the launches.
//   No step of a turn calls through the entry.
//
// Design and cost
//   - Function pointers, not a pure abstract class: an architecture holds no
//     state, and the capability table lists its entry as constant data, so
//     there is no object to construct. The entry has the property C.121 asks
//     of an interface, no data of its own beyond what names it.
//   - Two indirect calls a load, describe's and graph's, and none a token
//     (WASM.4). The reference-logits test makes the same two calls on the
//     pinned file (tests/integration/reference-logits/logits_test.cpp).
//   - No test exercises describe's refusals (arch/describe.h, verification).
//
// Corpus guidance
//   C++ Core Guidelines
//     C.121  A base class used as an interface is pure abstract: there is no
//            base class; the interface is a struct of function pointers with
//            no state, as stable for the same reason.
//     E.27   Error codes systematically: DescribeError and GraphError, each
//            with a subject naming where.
//     I.5    State preconditions: GraphFn's, below.
//     Enum.7 Specify an enumeration's underlying type only when necessary:
//            DescribeError takes one byte, the smallest that holds it.
//   C++ performance guidelines
//     WASM.4 Reduce indirect dispatch in hot paths: each entry's function
//            pointers are called once a load, never per token.

enum class DescribeError : std::uint8_t {
  Ok,
  MissingKey,
  WrongKeyType,
  MissingTensor,
  ShapeMismatch,
  // A value no model could run with: a zero count or width, key-value heads
  // that do not divide the query heads, a layer count larger than the
  // file's tensors could fill.
  InvalidValue,
  // A valid value or form the kernels do not implement, such as a norm gain
  // not stored as F32 or a rotary form rope does not apply.
  UnsupportedValue,
};

// The outcome of describing a file. `subject` names the key or tensor at
// fault, so a failure says where it is, not only what kind it is.
struct DescribeResult {
  DescribeError error = DescribeError::Ok;
  std::string subject;

  [[nodiscard]] bool ok() const noexcept { return error == DescribeError::Ok; }
};

using DescribeFn = DescribeResult (*)(const gguf::TensorIndex& index,
                                      model::ModelDescription& out);

// Fills `out` with every launch of a step. Preconditions (I.5): `model` is
// this architecture's describe's; `plan` was made from it, as upload carried it
// out (Upload::plan()); `cache_format` has a pack and is the format the plan
// sized the cache in.
using GraphFn = graph::GraphResult (*)(const model::ModelDescription& model,
                                       const residency::ResidencyPlan& plan,
                                       const formats::Format& cache_format,
                                       std::vector<kernels::Launch>& out);

struct Architecture {
  std::string_view name;
  DescribeFn describe;
  GraphFn graph;
};

}  // namespace frizzle::arch
