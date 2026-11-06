#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace frizzle::policy {

// Contract 12: policy.
//
// A model's measured configuration, from its entry in web/models.json. The
// values change when a model is measured (axis J); this header changes only
// when what can be configured changes.
//
// Two parts cross the boundary at different times. Cache precision, the
// memory budget and the stop tokens are fixed at load. Sampling settings
// depend on the mode a turn runs in, such as thinking or not, so the
// JavaScript side resolves them for the turn and passes them with each
// generate, together with the seed and the reply's token limit.
//
// An unmeasured model runs on the defaults below and is shown as unmeasured.
// The context offered is not policy: it is derived from the file, the memory
// budget and the cache precision.

enum class CachePrecision : std::uint8_t {
  F16,
};

// The defaults below, named.
inline constexpr std::uint64_t kDefaultMemoryBudget = 2ULL << 30U;  // 2 GiB
inline constexpr std::uint32_t kDefaultRollbackReserve = 4096;
inline constexpr float kDefaultTemperature = 0.8F;
inline constexpr std::uint32_t kDefaultTopK = 40;
inline constexpr float kDefaultTopP = 0.95F;
inline constexpr float kDefaultMinP = 0.05F;

struct LoadPolicy {
  CachePrecision cache_precision = CachePrecision::F16;
  // Bytes the harness may place on the device: weights, cache and
  // activations. WebGPU does not report device memory, so this is measured
  // per model on the devices it is listed for. The default covers a model of
  // about 1 GB with a working cache; a larger unmeasured model is rejected
  // at preflight's Fit stage, by name, rather than attempted.
  std::uint64_t memory_budget = kDefaultMemoryBudget;
  // Tokens a sliding-window layer's cache keeps beyond what one step needs,
  // so a turn can roll the cache back that far without restarting it:
  // enough to regenerate a long reply. A deeper rollback restarts the cache
  // from the first token.
  std::uint32_t rollback_reserve = kDefaultRollbackReserve;
  // Stop tokens, by their text, beyond the end-of-generation tokens the
  // file names, where the model's generation config lists more than its
  // file carries (runtime/stops.h). Each must be one control token of the
  // vocabulary. None by default: an unmeasured model stops on its file's.
  std::vector<std::string> stop;
};

// Defaults are the most widely exercised sampling settings, for a model
// without a card of its own.
struct SamplingSettings {
  float temperature = kDefaultTemperature;
  std::uint32_t top_k = kDefaultTopK;
  float top_p = kDefaultTopP;
  float min_p = kDefaultMinP;
};

// The randomness of a run. Its own type, so a seed cannot be passed where a
// count or a position is meant.
enum class Seed : std::uint64_t {};

struct TurnPolicy {
  SamplingSettings sampling;
  Seed seed{};
  // The most tokens the reply may draw; the context offered bounds it
  // first, so the largest value asks for no other limit. 0 inspects: the
  // input is run to its last position and nothing is drawn
  // (runtime/turn.h).
  std::uint32_t max_tokens = UINT32_MAX;
};

}  // namespace frizzle::policy
