// Axis V: the replay and edits scenario, an integration test
// (tests/README.md). Run when edits, the graph or the turn loop change, not
// with the routine suite.
//
// Contract locked, through the engine (runtime/engine.h) on the pinned
// model, for one message and one seed:
//   1. Scale × 1 edits in several layers give the reply without edits, byte
//      for byte, with the same token count and ending.
//   2. Without edits, then with a strong edit from step 0, then without
//      again: the third run equals the first, and reuses all the prompt's
//      cache but its last token, the first position whose edits differ.
//   3. One edit set across layers, given in two orders, gives one reply,
//      different from the reply without edits.
//   4. After an edited run, a conversation that contains its reply reuses
//      only the prompt before the edits, where the edits in force first
//      differ, and its reply equals a fresh engine's.
//   5. A branch that keeps the first k tokens of the original, with no
//      edits, continues it bit for bit: its tokens are the original's from
//      k on, with its ending, for k at the start, inside and at the end.
//      A run's limit counts the tokens it emits (runtime/turn.h), so each
//      branch is given what the original's limit leaves after k, and ends
//      where the original did, whether by a stop or the limit; at the end,
//      that is 0, an inspection, when the original ended on its limit.
//      The kept tokens' entries, written by decode steps in the original,
//      are written by a prefill block here (edit/edit_kernel.h, kernels
//      batch-invariant, graph/graph.h).
//   6. Accept, then branch again: a branch from b1 with a strong edit is
//      accepted (the original's first b1 tokens, then the branch's, its
//      edit from b1 its history); a branch elsewhere cuts the cache below
//      b1; then a branch from b2 > b1 of the accepted reply, sending its
//      history and no new pins, continues it bit for bit, given what the
//      steered run's limit leaves after b2. Only here are an
//      accepted reply's edits recomputed inside prefill rows; a fresh
//      engine gives the same.
//   7. An inspection (max_tokens 0) at k emits nothing, ends Limit, and
//      captures every layer's neurons equal, bit for bit, to the captures
//      of the branch from k with the same edits.
//   8. A kept identifier past the vocabulary is refused at once, Kept,
//      naming its index; nothing runs.
//
// Inputs: the pinned model file (scripts/fetch_model.py); a missing file
// fails the test. The message is wrapped as Gemma 3's chat template renders
// it, the caller's job (engine.h). The strong edits are ones seen to change
// this reply at this seed; if the model or its settings change, they are
// chosen again, and the test says so rather than passing.
//
// Cost: the model loads three times, about 722 MB each time; about twenty
// replies of at most 40 tokens, and one inspection.
//
// Corpus guidance
//   C++ Core Guidelines
//     I.7    State postconditions: the rewind and the untouched cache are
//            each checked by the count of reused tokens, not by the text
//            alone, which would also match if the rewind did nothing.

#include <doctest/doctest.h>
#include <webgpu/webgpu.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/edit/edit.h"
#include "core/gguf/byte_source.h"
#include "core/gpu/device.h"
#include "core/gpu/wgpu_handles.h"
#include "core/policy/policy.h"
#include "core/runtime/engine.h"
#include "core/runtime/runtime.h"
#include "support/files.h"
#include "support/gpu.h"

namespace {

namespace frz = frizzle;
using frz::edit::Edit;
using frz::edit::EditKind;
using frz::edit::kToTheEnd;
using frz::tokenizer::TokenId;

constexpr std::size_t kChunk = std::size_t{16} << 20U;
constexpr std::uint64_t kSeed = 7;
constexpr std::uint32_t kMaxTokens = 40;

// One user turn and the model's turn opened, as the template renders them;
// a conversation begins with <bos>.
std::string user_turn(std::string_view message) {
  return "<start_of_turn>user\n" + std::string(message) +
         "<end_of_turn>\n<start_of_turn>model\n";
}

// An edit over steps [from, until), every step by default.
Edit E(std::uint32_t layer, std::uint32_t neuron, EditKind kind, float value,
       std::uint32_t from = 0, std::uint32_t until = kToTheEnd) {
  return {layer, neuron, kind, value, from, until};
}

// A reply and how it ended: everything the engine reports of it.
struct Reply {
  std::string text;
  std::vector<TokenId> tokens;  // as emitted, one callback each
  std::vector<float> neurons;   // the captures, copied, when taken
  frz::runtime::CaptureStatus captured{};
  frz::runtime::TurnFailure failure{};
  frz::runtime::TurnEnd end{};
  std::optional<frz::tokenizer::TokenId> stop;
  std::uint32_t prompt = 0;
  std::uint32_t reused = 0;
  std::uint32_t emitted = 0;
  bool done = false;
};

bool same_reply(const Reply& a, const Reply& b) {
  return a.text == b.text && a.end == b.end && a.stop == b.stop &&
         a.emitted == b.emitted;
}

// A branch's tokens and ending are the reply's from k on: the reply
// continued, bit for bit.
bool continues(const Reply& branch, std::span<const TokenId> reply,
               std::size_t k, const Reply& ended) {
  return std::vector<TokenId>(branch.tokens) ==
             std::vector<TokenId>(
                 reply.begin() + static_cast<std::ptrdiff_t>(k), reply.end()) &&
         branch.end == ended.end && branch.stop == ended.stop;
}

std::span<const TokenId> first(const std::vector<TokenId>& tokens,
                               std::size_t k) {
  return std::span(tokens).first(k);
}

// An engine with the pinned model loaded, and the device it borrows.
class Loaded {
 public:
  Loaded()
      : instance_(wgpuCreateInstance(nullptr)),
        device_(frz::testing::acquire(instance_.get())),
        engine_(std::make_unique<frz::runtime::Engine>(*device_)) {
    const frz::testing::Front front =
        frz::testing::read_front(FRIZZLE_MODEL_FILE);
    frz::gguf::MemoryByteSource source{front.bytes, front.file_size};
    struct Stage {
      frz::runtime::LoadResult result;
      bool done = false;
    };
    const auto on_stage = [](const frz::runtime::LoadResult& result,
                             void* userdata) {
      auto& s = *static_cast<Stage*>(userdata);
      s.result = result;
      s.done = true;
    };
    Stage stage;
    const frz::runtime::LoadResult begun = engine_->begin_load(
        source, kChunk, frz::policy::LoadPolicy{}, on_stage, &stage);
    REQUIRE_MESSAGE(begun.ok(), begun.subject);
    frz::testing::pump_until(instance_.get(), stage.done, "the buffers");
    REQUIRE_MESSAGE(stage.result.ok(), stage.result.subject);

    std::ifstream file(FRIZZLE_MODEL_FILE, std::ios::binary);
    REQUIRE(file.is_open());
    std::vector<char> raw(kChunk);
    std::vector<std::byte> chunk(kChunk);
    for (std::uint64_t at = 0; at < front.file_size;) {
      const std::uint64_t left = front.file_size - at;
      const std::size_t length =
          left < kChunk ? static_cast<std::size_t>(left) : kChunk;
      file.read(raw.data(), static_cast<std::streamsize>(length));
      REQUIRE(static_cast<std::size_t>(file.gcount()) == length);
      for (std::size_t i = 0; i < length; ++i) {
        chunk[i] = static_cast<std::byte>(static_cast<unsigned char>(raw[i]));
      }
      stage = {};
      const frz::runtime::LoadResult written =
          engine_->write(at, std::span(chunk).first(length), on_stage, &stage);
      REQUIRE_MESSAGE(written.ok(), written.subject);
      frz::testing::pump_until(instance_.get(), stage.done, "a chunk");
      REQUIRE_MESSAGE(stage.result.ok(), stage.result.subject);
      at += length;
    }
    stage = {};
    const frz::runtime::LoadResult finished =
        engine_->finish_load(on_stage, &stage);
    REQUIRE_MESSAGE(finished.ok(), finished.subject);
    frz::testing::pump_until(instance_.get(), stage.done, "the program");
    REQUIRE_MESSAGE(stage.result.ok(), stage.result.subject);
  }

  // The reply to `text` then `kept` under `edits`, at the scenario's seed
  // and `max_tokens` (0 inspects).
  Reply generate(const std::string& text, std::span<const Edit> edits,
                 std::span<const TokenId> kept = {},
                 std::uint32_t max_tokens = kMaxTokens) {
    frz::policy::TurnPolicy policy;
    policy.seed = frz::policy::Seed{kSeed};
    policy.max_tokens = max_tokens;
    Reply reply;
    const frz::runtime::OperationStart started = engine_->generate(
        text, kept, edits, policy,
        [](TokenId token, std::string_view piece, void* userdata) {
          auto& r = *static_cast<Reply*>(userdata);
          r.tokens.push_back(token);
          r.text += piece;
        },
        [](const frz::runtime::TurnResult& result,
           const frz::runtime::Captured& captured, void* userdata) {
          auto& r = *static_cast<Reply*>(userdata);
          r.captured = captured.status;
          r.neurons.assign(captured.neurons.begin(), captured.neurons.end());
          r.text += result.tail;
          r.failure = result.failure;
          r.end = result.end;
          r.stop = result.stop;
          r.prompt = result.prompt;
          r.reused = result.reused;
          r.emitted = result.emitted;
          r.done = true;
        },
        &reply);
    REQUIRE(started.ok());
    frz::testing::pump_until(instance_.get(), reply.done, "a reply");
    REQUIRE(reply.failure == frz::runtime::TurnFailure::None);
    REQUIRE(reply.tokens.size() == reply.emitted);
    return reply;
  }

  // What generate refuses at once, with nothing run.
  frz::runtime::OperationStart refused(const std::string& text,
                                       std::span<const TokenId> kept) {
    frz::policy::TurnPolicy policy;
    policy.seed = frz::policy::Seed{kSeed};
    return engine_->generate(
        text, kept, {}, policy,
        [](TokenId, std::string_view, void*) { FAIL("a token was emitted"); },
        [](const frz::runtime::TurnResult&, const frz::runtime::Captured&,
           void*) { FAIL("a refused run ended"); },
        nullptr);
  }

 private:
  frz::gpu::Instance instance_;
  std::unique_ptr<frz::gpu::Device> device_;
  std::unique_ptr<frz::runtime::Engine> engine_;  // destroyed first: it
                                                  // borrows the device
};

}  // namespace

TEST_CASE("replay and edits") {
  const std::string prompt =
      "<bos>" + user_turn("Name three colours of the rainbow.");
  const std::vector<Edit> identity{E(12, 4031, EditKind::Scale, 1.0F),
                                   E(0, 0, EditKind::Scale, 1.0F),
                                   E(25, 6911, EditKind::Scale, 1.0F)};
  const std::vector<Edit> strong{E(12, 4031, EditKind::Set, 100.0F)};
  const std::vector<Edit> across{E(12, 4031, EditKind::Set, 100.0F),
                                 E(6, 100, EditKind::Set, -200.0F),
                                 E(20, 50, EditKind::Scale, 30.0F)};
  const std::vector<Edit> reversed(across.rbegin(), across.rend());

  std::optional<Loaded> engine;
  engine.emplace();
  const Reply original = engine->generate(prompt, {});
  const std::uint32_t before_edits = original.prompt - 1;
  REQUIRE_MESSAGE(original.emitted >= 12,
                  "the original reply is too short to branch inside");

  // 1. Scale × 1 is the reply without edits.
  const Reply scaled_by_one = engine->generate(prompt, identity);
  CHECK(same_reply(scaled_by_one, original));

  // 2. An edited run's earlier positions are reused where edits agree.
  const Reply edited = engine->generate(prompt, strong);
  REQUIRE_MESSAGE(!same_reply(edited, original),
                  "the strong edit no longer changes this reply; choose "
                  "another that does");
  const Reply again = engine->generate(prompt, {});
  CHECK(same_reply(again, original));
  CHECK(again.reused == before_edits);

  // 3. A set, not a sequence.
  const Reply in_order = engine->generate(prompt, across);
  const Reply in_reverse = engine->generate(prompt, reversed);
  CHECK(same_reply(in_order, in_reverse));
  CHECK_FALSE(same_reply(in_order, original));

  // 4. A conversation after an edited reply: reused to where the edits in
  //    force first differ, then what a fresh engine gives.
  const Reply replied = engine->generate(prompt, strong);
  const std::string conversation =
      prompt + replied.text + "<end_of_turn>\n" + user_turn("And three more?");
  const Reply continued = engine->generate(conversation, {});
  CHECK(continued.reused == before_edits);

  // 5. A branch with no edits continues the original, from anywhere.
  for (const std::size_t k :
       {std::size_t{0}, std::size_t{1}, std::size_t{original.emitted / 2},
        std::size_t{original.emitted}}) {
    CAPTURE(k);
    const Reply branch =
        engine->generate(prompt, {}, first(original.tokens, k),
                         kMaxTokens - static_cast<std::uint32_t>(k));
    CHECK(continues(branch, original.tokens, k, original));
  }

  // 6. Accept a steered branch, cut the cache below it, branch again.
  constexpr std::uint32_t kB1 = 3;
  const std::vector<Edit> steer{E(12, 4031, EditKind::Set, 100.0F, kB1)};
  const Reply steered =
      engine->generate(prompt, steer, first(original.tokens, kB1));
  std::vector<TokenId> accepted(original.tokens.begin(),
                                original.tokens.begin() + kB1);
  accepted.insert(accepted.end(), steered.tokens.begin(), steered.tokens.end());
  REQUIRE_MESSAGE(accepted != original.tokens,
                  "the steer no longer changes the reply after token 3; "
                  "choose another");
  REQUIRE_MESSAGE(accepted.size() >= kB1 + 6,
                  "the accepted reply is too short to branch inside");
  (void)engine->generate(prompt, {}, first(original.tokens, 1));
  const std::size_t b2 = kB1 + 4;
  // The steered run emitted from b1; what its limit leaves after b2.
  const std::uint32_t left = kMaxTokens - static_cast<std::uint32_t>(b2 - kB1);
  const Reply onward =
      engine->generate(prompt, steer, first(accepted, b2), left);
  // Reused to where the steer's edits begin, position prompt − 1 + b1: the
  // accepted reply's edited positions b1 .. b2 − 1 were recomputed.
  CHECK(onward.reused == original.prompt - 1 + kB1);
  CHECK(continues(onward, accepted, b2, steered));

  // 7. An inspection is the branch's step k, captured, nothing drawn.
  constexpr std::size_t kInspect = 5;
  const std::vector<Edit> at_k{E(6, 100, EditKind::Set, -200.0F, kInspect)};
  const Reply inspected =
      engine->generate(prompt, at_k, first(original.tokens, kInspect), 0);
  CHECK(inspected.emitted == 0);
  CHECK(inspected.end == frz::runtime::TurnEnd::Limit);
  REQUIRE(inspected.captured == frz::runtime::CaptureStatus::Taken);
  const Reply branched =
      engine->generate(prompt, at_k, first(original.tokens, kInspect));
  REQUIRE(branched.captured == frz::runtime::CaptureStatus::Taken);
  CHECK(inspected.neurons == branched.neurons);  // operator== on floats:
                                                 // no NaN is captured here

  // 8. A kept identifier that names no token is refused, by index.
  const std::vector<TokenId> bad{original.tokens.at(0), TokenId{0xFFFFFFF0U}};
  const frz::runtime::OperationStart no = engine->refused(prompt, bad);
  CHECK(no.error == frz::runtime::OperationError::Kept);
  CHECK(no.start.kept_out_of_vocabulary == std::optional<std::size_t>{1});

  // 4 and 6 again, on fresh engines: nothing held from before changes them.
  engine.reset();
  engine.emplace();
  const Reply fresh = engine->generate(conversation, {});
  CHECK(fresh.reused == 0);
  CHECK(same_reply(continued, fresh));
  engine.reset();
  engine.emplace();
  const Reply fresh_onward =
      engine->generate(prompt, steer, first(accepted, b2), left);
  CHECK(fresh_onward.reused == 0);
  CHECK(continues(fresh_onward, accepted, b2, steered));
}
