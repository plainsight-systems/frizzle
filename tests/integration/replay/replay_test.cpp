// Axis V: the replay and edits scenario, an integration test
// (tests/README.md). Run when edits, the graph or the turn loop change, not
// with the routine suite.
//
// Contract locked, through the engine (runtime/engine.h) on the pinned
// model, for one message and one seed:
//   1. Scale × 1 edits in several layers give the reply without edits, byte
//      for byte, with the same token count and ending.
//   2. Without edits, then with a strong edit, then without again: the third
//      run equals the first, and reuses all the prompt's cache but its last
//      token, so the edited run left the earlier positions' entries as they
//      were.
//   3. One edit set across layers, given in two orders, gives one reply,
//      different from the reply without edits.
//   4. After an edited run, a conversation that contains its reply reuses
//      only the prompt before the edits (the rewind), and its reply equals a
//      fresh engine's.
//
// Inputs: the pinned model file (scripts/fetch_model.py); a missing file
// fails the test. The message is wrapped as Gemma 3's chat template renders
// it, the caller's job (engine.h). The strong edits are ones seen to change
// this reply at this seed; if the model or its settings change, they are
// chosen again, and the test says so rather than passing.
//
// Cost: the model loads twice, about 722 MB each time; nine replies of at
// most 40 tokens.
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

constexpr std::size_t kChunk = std::size_t{16} << 20U;
constexpr std::uint64_t kSeed = 7;
constexpr std::uint32_t kMaxTokens = 40;

// One user turn and the model's turn opened, as the template renders them;
// a conversation begins with <bos>.
std::string user_turn(std::string_view message) {
  return "<start_of_turn>user\n" + std::string(message) +
         "<end_of_turn>\n<start_of_turn>model\n";
}

// A reply and how it ended: everything the engine reports of it.
struct Reply {
  std::string text;
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

  // The reply to `text` under `edits`, at the scenario's seed and limit.
  Reply generate(const std::string& text, std::span<const Edit> edits) {
    frz::policy::TurnPolicy policy;
    policy.seed = frz::policy::Seed{kSeed};
    policy.max_tokens = kMaxTokens;
    Reply reply;
    const frz::runtime::OperationStart started = engine_->generate(
        text, edits, policy,
        [](std::string_view piece, void* userdata) {
          static_cast<Reply*>(userdata)->text += piece;
        },
        // The scenario guards replies and the cache, not captures
        // (tests/README.md).
        [](const frz::runtime::TurnResult& result,
           const frz::runtime::Captured& /*captured*/, void* userdata) {
          auto& r = *static_cast<Reply*>(userdata);
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
    return reply;
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
  const std::vector<Edit> identity{{12, 4031, EditKind::Scale, 1.0F},
                                   {0, 0, EditKind::Scale, 1.0F},
                                   {25, 6911, EditKind::Scale, 1.0F}};
  const std::vector<Edit> strong{{12, 4031, EditKind::Set, 100.0F}};
  const std::vector<Edit> across{{12, 4031, EditKind::Set, 100.0F},
                                 {6, 100, EditKind::Set, -200.0F},
                                 {20, 50, EditKind::Scale, 30.0F}};
  const std::vector<Edit> reversed(across.rbegin(), across.rend());

  std::optional<Loaded> engine;
  engine.emplace();
  const Reply original = engine->generate(prompt, {});
  const std::uint32_t before_edits = original.prompt - 1;

  // 1. Scale × 1 is the reply without edits.
  const Reply scaled_by_one = engine->generate(prompt, identity);
  CHECK(same_reply(scaled_by_one, original));

  // 2. An edited run leaves the earlier positions' cache as it was.
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

  // 4. A conversation after an edited reply: the rewind, then what a fresh
  //    engine gives.
  const Reply replied = engine->generate(prompt, strong);
  const std::string conversation =
      prompt + replied.text + "<end_of_turn>\n" + user_turn("And three more?");
  const Reply continued = engine->generate(conversation, {});
  CHECK(continued.reused == before_edits);
  engine.reset();
  engine.emplace();
  const Reply fresh = engine->generate(conversation, {});
  CHECK(fresh.reused == 0);
  CHECK(same_reply(continued, fresh));
}
