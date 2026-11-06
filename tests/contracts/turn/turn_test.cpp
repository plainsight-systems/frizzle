// Axis M: the turn suite (tests/README.md).
//
// Contract locked (runtime/turn.h), the inspection: a turn with a limit of
// 0 runs its prompt's prefill, the last block asking for logits so the
// scoring launches run there (the edits, the capture); its draw is
// discarded, whatever it is, stop token or not; no decode step is planned;
// the turn ends Limit having emitted nothing, and the cache keeps the whole
// prompt. A turn with a limit of 1 still emits its one draw, as before.
//
// Not locked: every other way a turn ends (runtime.h, Verification). Plain
// values, no device.

#include "core/runtime/turn.h"

#include <doctest/doctest.h>

#include <cstdint>
#include <optional>

#include "core/residency/plan.h"
#include "core/sampler/sampler.h"

namespace {

using frizzle::runtime::Draw;
using frizzle::runtime::Planned;
using frizzle::runtime::Turn;
using frizzle::runtime::TurnEnd;
using frizzle::runtime::TurnFailure;
using frizzle::runtime::TurnSize;
using frizzle::sampler::SampledRecord;

constexpr std::uint32_t kCapacity = 4096;

// Runs the turn's prefill: reports every block, the last with a draw of
// `token`; returns what became of that draw and how many blocks ran.
struct Ran {
  Draw last = Draw::None;
  std::uint32_t blocks = 0;
  bool asked_logits_last = false;
};

Ran prefill(Turn& turn, std::uint32_t token, bool stop) {
  Ran ran;
  while (auto planned = turn.next()) {
    REQUIRE_FALSE(planned->fed);
    ++ran.blocks;
    ran.asked_logits_last = planned->logits;
    ran.last = turn.report(
        planned->logits ? std::optional<SampledRecord>{{token, 0, 0.5F, 0}}
                        : std::nullopt,
        stop && planned->logits);
    if (planned->logits) break;
  }
  return ran;
}

}  // namespace

TEST_CASE("a limit of 0 inspects: the prefill runs, nothing is drawn") {
  constexpr std::uint32_t kPrompt = frizzle::residency::kPrefillBlock + 37;
  for (const bool stop : {false, true}) {
    Turn turn(TurnSize{.cached = 0,
                       .prompt = kPrompt,
                       .capacity = kCapacity,
                       .max_tokens = 0});
    const Ran ran = prefill(turn, 123, stop);
    CHECK(ran.blocks == 2);
    CHECK(ran.asked_logits_last);
    CHECK(ran.last == Draw::Discard);
    CHECK_FALSE(turn.next().has_value());
    REQUIRE(turn.finished());
    CHECK(turn.failure() == TurnFailure::None);
    CHECK(turn.end() == TurnEnd::Limit);
    CHECK(turn.emitted() == 0);
    CHECK(turn.written() == kPrompt);
    CHECK(turn.kept() == kPrompt);
  }
}

TEST_CASE("an inspection of a mostly cached prompt runs its last position") {
  Turn turn(TurnSize{
      .cached = 99, .prompt = 100, .capacity = kCapacity, .max_tokens = 0});
  const Ran ran = prefill(turn, 7, false);
  CHECK(ran.blocks == 1);
  CHECK(ran.last == Draw::Discard);
  REQUIRE(turn.finished());
  CHECK(turn.end() == TurnEnd::Limit);
  CHECK(turn.kept() == 100);
}

TEST_CASE("a limit of 1 still emits its one draw") {
  Turn turn(TurnSize{
      .cached = 0, .prompt = 10, .capacity = kCapacity, .max_tokens = 1});
  const Ran ran = prefill(turn, 7, false);
  CHECK(ran.last == Draw::Emit);
  CHECK_FALSE(turn.next().has_value());
  REQUIRE(turn.finished());
  CHECK(turn.end() == TurnEnd::Limit);
  CHECK(turn.emitted() == 1);
  CHECK(turn.kept() == 10);
}
