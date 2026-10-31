// Axis M: changes with a new stage in the generation loop.
//
// Contract, algorithm, cost and corpus guidance: generator.h. This file defines
// what that header states.

#include "core/runtime/generator.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <utility>
#include <vector>

#include "core/tokenizer/detokenizer.h"

namespace frizzle::runtime {

struct Generator::State {
  std::unique_ptr<tokenizer::Tokenizer> tokenizer;
  std::unique_ptr<Runtime> runtime;  // null once the generator is destroyed
  std::uint32_t capacity = 0;
  std::optional<tokenizer::Detokenizer>
      text;  // over `tokenizer`, made once it is set
  std::vector<tokenizer::TokenId> tokens;
  bool prepared = false;  // a turn prepared since the last start

  // The running turn.
  ReplyTokenCallback on_token = nullptr;
  EndCallback on_end = nullptr;
  void* userdata = nullptr;
  // This state, held while a turn runs, so its callbacks outlive the generator.
  std::shared_ptr<State> in_flight;
};

namespace {

// NFC's greatest shrinking of UTF-8, 7 bytes to 2: a character NFC writes
// stands for at most its longest canonical equivalent, which is at most 7/2
// times its bytes — U+0390 from U+1FBE U+0308 U+0301 — in Unicode 16.0, the
// tables' version (tokenizer/nfc.h), as tokenizer_nfc_test computes over
// them.
constexpr std::uint64_t kNfcShrinkFrom = 7;
constexpr std::uint64_t kNfcShrinkTo = 2;

// The reply's UTF-8 stream, made at construction; none is a broken
// precondition.
tokenizer::Detokenizer& stream(Generator::State& s) {
  if (!s.text.has_value()) std::abort();
  return *s.text;
}

void on_token(tokenizer::TokenId token, void* userdata) {
  Generator::State& s = *static_cast<Generator::State*>(userdata);
  s.on_token(token, stream(s).push(token), s.userdata);
}

void on_turn(const TurnResult& result, void* userdata) {
  Generator::State& s = *static_cast<Generator::State*>(userdata);
  // This turn's callbacks, taken before the generator is idle: the last
  // token callback may start the next turn, which replaces them.
  const EndCallback on_end = s.on_end;
  void* const caller = s.userdata;
  // Bytes still pending, the turn stopped inside a character: U+FFFD, owned
  // by the last token emitted, the result's tail.
  TurnResult ended = result;
  ended.tail = stream(s).finish();
  const std::shared_ptr<Generator::State> keep = std::move(s.in_flight);
  on_end(ended, caller);
}

}  // namespace

Generator::Generator(std::unique_ptr<tokenizer::Tokenizer> tokenizer,
                     std::unique_ptr<Runtime> runtime)
    : state_(std::make_shared<State>()) {
  State& s = *state_;
  s.tokenizer = std::move(tokenizer);
  s.runtime = std::move(runtime);
  s.capacity = s.runtime->capacity();
  s.text.emplace(*s.tokenizer);
  s.tokens.reserve(std::size_t{s.capacity} + 1);
}

Generator::~Generator() {
  // A turn running finishes Cancelled: the runtime, destroyed, ends it, and
  // this state lives until its end callback has run.
  state_->runtime.reset();
}

GenerateResult Generator::prepare(std::string_view text,
                                  std::span<const tokenizer::TokenId> kept,
                                  Prepared& made) {
  State& s = *state_;
  if (s.runtime == nullptr || s.in_flight != nullptr) {
    return {{}, {StartError::Busy, "a turn is running"}, {}};
  }
  s.prepared = false;
  // A token covers at most `longest` bytes of normalized text, and
  // normalizing shrinks text at most 7/2-fold, so text longer than 7/2 ×
  // the context × that makes more tokens than the context holds: refused
  // unencoded.
  const std::uint64_t longest = s.tokenizer->longest_cover();
  const std::uint64_t covered2 =
      kNfcShrinkFrom * longest;  // twice the bytes a token covers, unnormalized
  const std::uint64_t bytes2 = kNfcShrinkTo * text.size();
  if (bytes2 > std::uint64_t{s.capacity} * covered2) {
    const std::uint64_t least = (bytes2 + covered2 - 1) / covered2;
    return {
        {},
        {StartError::PromptTooLong,
         "the prompt is " + std::to_string(text.size()) + " bytes, at least " +
             std::to_string(least) + " tokens, and the context offered " +
             std::to_string(s.capacity),
         least, s.capacity},
        {}};
  }
  const std::size_t vocabulary = s.tokenizer->vocabulary().size();
  for (std::size_t i = 0; i < kept.size(); ++i) {
    if (static_cast<std::size_t>(kept[i]) >= vocabulary) {
      return {{}, {}, i};
    }
  }
  s.tokens.clear();
  // Encoded with the context as its limit: text that makes more tokens at
  // least is refused before it is merged, as the runtime refuses one.
  const tokenizer::EncodeResult encoded =
      s.tokenizer->encode(text, s.capacity, s.tokens);
  if (encoded.error == tokenizer::EncodeError::TooManyTokens) {
    return {
        {},
        {StartError::PromptTooLong,
         "the prompt is at least " + std::to_string(encoded.least_tokens) +
             " tokens, and the context offered " + std::to_string(s.capacity),
         encoded.least_tokens, s.capacity},
        {}};
  }
  if (encoded.error != tokenizer::EncodeError::Ok) return {encoded, {}, {}};
  const std::size_t text_tokens = s.tokens.size();
  // Past the context the runtime refuses the turn, naming both counts; the
  // reserve holds the context plus one, so growing past it is that case's
  // one allocation.
  s.tokens.insert(s.tokens.end(), kept.begin(), kept.end());
  made.text_tokens = static_cast<std::uint32_t>(
      std::min<std::size_t>(text_tokens, UINT32_MAX));
  made.input_tokens = static_cast<std::uint32_t>(
      std::min<std::size_t>(s.tokens.size(), UINT32_MAX));
  s.prepared = true;
  return {};
}

GenerateResult Generator::start(const policy::TurnPolicy& policy,
                                ReplyTokenCallback on_token, EndCallback on_end,
                                void* userdata) {
  State& s = *state_;
  if (s.runtime == nullptr || s.in_flight != nullptr) {
    return {{}, {StartError::Busy, "a turn is running"}, {}};
  }
  if (!s.prepared) {
    return {{}, {StartError::EmptyPrompt, "no turn is prepared"}, {}};
  }
  s.prepared = false;
  s.on_token = on_token;
  s.on_end = on_end;
  s.userdata = userdata;
  s.in_flight = state_;
  StartResult started = s.runtime->start(s.tokens, policy, runtime::on_token,
                                         runtime::on_turn, &s);
  if (started.error != StartError::Ok) s.in_flight.reset();
  return {{}, std::move(started), {}};
}

void Generator::cancel() noexcept {
  if (state_->runtime) state_->runtime->cancel();
}

bool Generator::rewind(std::uint32_t tokens) noexcept {
  const State& s = *state_;
  if (s.runtime == nullptr || s.in_flight != nullptr) return false;
  return s.runtime->rewind(tokens);
}

std::uint32_t Generator::cached() const noexcept {
  return state_->runtime ? state_->runtime->cached() : 0;
}

#if FRIZZLE_DIAGNOSTICS_ENABLED
bool Generator::observe_steps(StepObserver observer, void* userdata) noexcept {
  return state_->runtime && state_->runtime->observe_steps(observer, userdata);
}
#endif

}  // namespace frizzle::runtime
