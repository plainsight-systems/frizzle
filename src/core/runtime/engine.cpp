// Axis M: changes with an operation the engine offers, or the order of the
// stages inside one.
//
// Contract, algorithm, cost and corpus guidance: engine.h. This file defines
// the Engine as that header states it: load's three stages and generate,
// each as a chain of callbacks that carry the engine's state. The
// refusals' wording, and the checks begin_load makes before the upload,
// follow Charlotte e003048's preflight.cpp (plan_load, load_tokenizer),
// which Frizzle did not port; Frizzle's engine names them itself.

#include "core/runtime/engine.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/arch/architecture.h"
#include "core/cache/kv.h"
#include "core/cache/prefix.h"
#include "core/capability/capability.h"
#include "core/capture/capture.h"
#include "core/capture/reader.h"
#include "core/edit/edit.h"
#include "core/gguf/index.h"
#include "core/gguf/reader.h"
#include "core/gpu/userdata.h"
#include "core/graph/graph.h"
#include "core/kernels/interface.h"
#include "core/kernels/program.h"
#include "core/model/model_description.h"
#include "core/residency/plan.h"
#include "core/residency/upload.h"
#include "core/runtime/runtime.h"
#include "core/runtime/stops.h"
#include "core/sampler/sampler.h"
#include "core/tokenizer/vocabulary.h"

namespace frizzle::runtime {

struct Engine::State {
  enum class Phase : std::uint8_t {
    Unloaded,   // nothing loaded, or the last load failed
    Beginning,  // the upload's buffers are being made
    Writing,    // the file's bytes are being written
    Finishing,  // the upload is finishing, then the program building
    Loaded,
  };

  const gpu::Device* device = nullptr;  // borrowed; outlives the engine
  bool cancelled = false;               // the engine was destroyed
  Phase phase = Phase::Unloaded;
  bool call_pending = false;  // a load stage's callback has not yet run

  // What begin_load made, for finish_load.
  model::ModelDescription description{};
  const arch::Architecture* architecture = nullptr;
  policy::CachePrecision precision = policy::CachePrecision::F16;
  std::unique_ptr<tokenizer::Tokenizer> tokenizer;  // until the generator
  StopSet stops;
  std::optional<ChatTemplate> chat;  // the file's, for the host (engine.h)

  // The loaded model. Declared so the generator, whose program borrows the
  // upload's buffers, is released before the upload.
  std::unique_ptr<residency::Upload> upload;
  std::unique_ptr<Generator> generator;
  // Reads the plan's capture store, which it borrows from the upload, so it
  // is released before the upload too.
  std::unique_ptr<capture::Reader> reader;
  residency::BufferRange edit_table{};  // the plan's, written each run
  residency::BufferRange captures{};    // the plan's, the capture store
  capture::Layout layout{};             // each layer's slot in it

  bool running = false;  // a turn is in flight
  // The placed edits (edit::placed) of the run that wrote the cache; empty
  // with it (engine.h, the invariant between runs).
  std::vector<edit::Edit> held;
};

namespace {

using State = Engine::State;
using Phase = State::Phase;

// A load stage, from its call to its callback.
struct LoadCall {
  std::shared_ptr<State> state;
  LoadCallback done;
  void* userdata;
};

struct GenerateCall {
  std::shared_ptr<State> state;
  ReplyTokenCallback on_token;
  RunEndCallback on_end;
  void* userdata;
  // The turn's result while the store is read, its message copied to
  // outlive the turn's callback.
  TurnResult result{};
  std::string message;
  std::string tail;
};

// --- Names, for refusals and failures ------------------------------------

std::string_view name(arch::DescribeError error) noexcept {
  switch (error) {
    case arch::DescribeError::Ok:
      return "ok";
    case arch::DescribeError::MissingKey:
      return "a required key is missing";
    case arch::DescribeError::WrongKeyType:
      return "a key has the wrong type";
    case arch::DescribeError::MissingTensor:
      return "a required tensor is missing";
    case arch::DescribeError::ShapeMismatch:
      return "a tensor has the wrong shape";
    case arch::DescribeError::InvalidValue:
      return "a value no model could have";
    case arch::DescribeError::UnsupportedValue:
      return "a value in a form this build does not read";
  }
  return "unknown";
}

std::string_view name(graph::GraphError error) noexcept {
  switch (error) {
    case graph::GraphError::Ok:
      return "ok";
    case graph::GraphError::UngroupedFeedForward:
      return "a layer's gate and up are not one group";
    case graph::GraphError::UnsupportedShape:
      return "a layer's shape is outside the kernels";
  }
  return "unknown";
}

// Bytes in mebibytes, rounded to the nearest.
std::string mebibytes(std::uint64_t bytes) {
  constexpr unsigned kShift = 20;
  constexpr std::uint64_t kHalf = std::uint64_t{1} << (kShift - 1);
  return std::to_string((bytes + kHalf) >> kShift) + " MiB";
}

std::string fit_failure(const residency::PlanResult& r,
                        const residency::ResidencyPlan& plan,
                        const policy::LoadPolicy& policy) {
  using residency::PlanError;
  switch (r.error) {
    case PlanError::Ok:
      return "fits";
    case PlanError::ExceedsBudget:
      return "needs " + mebibytes(plan.total_bytes) +
             " for the shortest context worth offering (" +
             std::to_string(residency::kPrefillBlock) +
             " tokens); the memory budget is " +
             mebibytes(policy.memory_budget);
    case PlanError::RowExceedsBinding:
      return "a row of " + r.subject + " is wider than one storage binding";
    case PlanError::ScratchExceedsBinding:
      return "the " + r.subject +
             " working buffer is wider than one storage binding";
    case PlanError::UnsupportedCachePrecision:
      return "the cache precision cannot store the head dimension of " +
             r.subject;
    case PlanError::Overflow:
      return "sizes overflow (" + r.subject + ")";
    case PlanError::MissingTensorBytes:
      return r.subject + " lies outside the bytes the load reads";
    case PlanError::UnboundSpan:
      return r.subject +
             " and the bounds its product reads cannot share one storage "
             "binding";
  }
  return "unknown";
}

std::string quoted(std::string_view text) {
  return "\"" + std::string(text) + "\"";
}

std::string unreadable_key(gguf::MetadataError error, std::string_view key) {
  return error == gguf::MetadataError::MissingKey
             ? "the file does not declare " + std::string(key)
             : std::string(key) + " is not a string";
}

// The file's chat template and its BOS and EOS tokens' text, into `out`;
// none when the file carries no template. A template key of another type,
// or ends that are unreadable or past the vocabulary, refuses the load as
// Tokenizer: the file is malformed, not merely without a template.
LoadResult chat_of(const gguf::TensorIndex& index,
                   const tokenizer::Vocabulary& vocabulary,
                   std::optional<ChatTemplate>& out) {
  constexpr std::string_view kTemplate = "tokenizer.chat_template";
  std::string_view text;
  if (const auto e = index.read_string(kTemplate, text);
      e == gguf::MetadataError::MissingKey) {
    out.reset();
    return {};
  } else if (e != gguf::MetadataError::Ok) {
    return {LoadError::Tokenizer, unreadable_key(e, kTemplate)};
  }
  const auto end = [&](std::string_view key, std::string& into) -> LoadResult {
    std::uint32_t id = 0;
    if (index.read_u32(key, id) != gguf::MetadataError::Ok ||
        id >= vocabulary.size()) {
      return {LoadError::Tokenizer,
              std::string(key) +
                  " is missing, not a u32, or past the "
                  "vocabulary, so the chat template's ends are "
                  "unknown"};
    }
    into = vocabulary.text(static_cast<tokenizer::TokenId>(id));
    return {};
  };
  ChatTemplate chat{std::string(text), {}, {}};
  if (LoadResult r = end("tokenizer.ggml.bos_token_id", chat.bos); !r.ok()) {
    return r;
  }
  if (LoadResult r = end("tokenizer.ggml.eos_token_id", chat.eos); !r.ok()) {
    return r;
  }
  out = std::move(chat);
  return {};
}

std::optional<residency::BufferRange> scratch(
    const residency::ResidencyPlan& plan, std::string_view purpose) {
  for (const residency::PlannedScratch& s : plan.scratch) {
    if (s.purpose == purpose) return s.range;
  }
  return std::nullopt;
}

// --- Load ------------------------------------------------------------------

// Ends a load stage: the stage's state, then its caller's callback.
void settle(std::unique_ptr<LoadCall> call, LoadError error,
            std::string subject) {
  State& s = *call->state;
  s.call_pending = false;
  if (s.cancelled) {
    error = LoadError::Cancelled;
    subject = "the engine was destroyed";
  }
  if (error != LoadError::Ok) {
    s.phase = Phase::Unloaded;
    s.generator.reset();
    s.held.clear();
    s.reader.reset();
    s.upload.reset();
    s.tokenizer.reset();
    s.chat.reset();
  }
  const LoadResult result{error, std::move(subject)};
  call->done(result, call->userdata);
}

void on_ready(std::unique_ptr<residency::Upload> upload,
              residency::UploadError error, const char* subject,
              void* userdata) {
  auto call = gpu::take_back<LoadCall>(userdata);
  State& s = *call->state;
  if (upload == nullptr || s.cancelled) {
    std::string named(residency::to_string(error));
    if (subject != nullptr && *subject != '\0') {
      named += " (" + std::string(subject) + ")";
    }
    settle(std::move(call), LoadError::Upload, std::move(named));
    return;
  }
  s.upload = std::move(upload);
  s.phase = Phase::Writing;
  settle(std::move(call), LoadError::Ok, {});
}

void on_accepted(residency::UploadError error, void* userdata) {
  auto call = gpu::take_back<LoadCall>(userdata);
  if (error != residency::UploadError::Ok) {
    settle(std::move(call), LoadError::Upload,
           std::string(residency::to_string(error)));
    return;
  }
  settle(std::move(call), LoadError::Ok, {});
}

void on_reader_made(std::unique_ptr<capture::Reader> reader,
                    std::string_view message, void* userdata) {
  auto call = gpu::take_back<LoadCall>(userdata);
  State& s = *call->state;
  if (s.cancelled) {
    settle(std::move(call), LoadError::Cancelled, {});
    return;
  }
  if (reader == nullptr) {
    settle(std::move(call), LoadError::Capture,
           "the capture store's readback was refused: " + std::string(message));
    return;
  }
  s.reader = std::move(reader);
  s.phase = Phase::Loaded;
  settle(std::move(call), LoadError::Ok, {});
}

void on_built(std::unique_ptr<kernels::Program> program,
              kernels::ProgramError error, std::string_view message,
              void* userdata) {
  auto call = gpu::take_back<LoadCall>(userdata);
  State& s = *call->state;
  if (s.cancelled) {
    settle(std::move(call), LoadError::Cancelled, {});
    return;
  }
  if (error != kernels::ProgramError::Ok || program == nullptr) {
    settle(std::move(call), LoadError::Program,
           "the model's kernels did not build: " + std::string(message));
    return;
  }
  const residency::ResidencyPlan& plan = s.upload->plan();
  cache::KvCache cache{s.description, plan, s.precision, plan.context_offered};
  auto turns =
      std::make_unique<Runtime>(std::move(program), std::move(cache), s.stops);
  s.generator =
      std::make_unique<Generator>(std::move(s.tokenizer), std::move(turns));
  s.held.clear();  // a new, empty cache
  const residency::Upload& upload = *s.upload;
  capture::Reader::make(*s.device, upload.buffer(s.captures.buffer), s.captures,
                        on_reader_made, gpu::hand_off(std::move(call)));
}

void on_finished(residency::UploadError error, void* userdata) {
  auto call = gpu::take_back<LoadCall>(userdata);
  State& s = *call->state;
  if (s.cancelled || error != residency::UploadError::Ok) {
    settle(std::move(call), LoadError::Upload,
           std::string(residency::to_string(error)));
    return;
  }
  // The graph over the plan, reading back the draw's record.
  const residency::ResidencyPlan& plan = s.upload->plan();
  const formats::Format* cache_format =
      capability::find_format(plan.cache_type);
  if (cache_format == nullptr) {
    settle(std::move(call), LoadError::Graph,
           "the cache's format is not implemented");
    return;
  }
  std::vector<kernels::Launch> launches;
  const graph::GraphResult built = arch::decoder_graph(*s.architecture)(
      s.description, plan, *cache_format, launches);
  if (!built.ok()) {
    settle(std::move(call), LoadError::Graph,
           std::string(name(built.error)) + " (" + built.subject + ")");
    return;
  }
  const std::optional<residency::BufferRange> sampled =
      scratch(plan, "sampled");
  if (!sampled) {
    settle(std::move(call), LoadError::Graph,
           "the plan has no buffer for the sampler's record");
    return;
  }
  const std::optional<residency::BufferRange> edit_table =
      scratch(plan, "edit_table");
  if (!edit_table) {
    settle(std::move(call), LoadError::Graph,
           "the plan has no buffer for the edit table");
    return;
  }
  s.edit_table = *edit_table;
  const std::optional<residency::BufferRange> captures =
      scratch(plan, "captures");
  if (!captures) {
    settle(std::move(call), LoadError::Graph,
           "the plan has no buffer for the capture store");
    return;
  }
  s.captures = *captures;
  s.layout = capture::layout(s.description);
  const kernels::Binding draw_record{sampled->buffer, sampled->offset,
                                     sizeof(sampler::SampledRecord)};
  const residency::Upload& upload = *s.upload;
  kernels::Program::build(upload, std::move(launches), draw_record, on_built,
                          gpu::hand_off(std::move(call)));
}

// --- Generate --------------------------------------------------------------

void on_generate_token(tokenizer::TokenId token, std::string_view text,
                       void* userdata) {
  const auto& call = *static_cast<const GenerateCall*>(userdata);
  call.on_token(token, text, call.userdata);
}

// Whether the turn's step that ends the prompt ran: a turn that ended Stop,
// Limit or Context drew at least once, and its first draw is that step's
// (runtime/turn.h).
bool prompt_step_ran(const TurnResult& result) noexcept {
  return result.failure == TurnFailure::None &&
         result.end != TurnEnd::Cancelled;
}

void on_captures_read(residency::Mapped outcome, std::span<const float> words,
                      void* userdata) {
  auto call = gpu::take_back<GenerateCall>(userdata);
  State& s = *call->state;
  Captured captured;
  if (s.cancelled) {
    captured.status = CaptureStatus::Unread;
    captured.mapped = residency::Mapped::Cancelled;
  } else if (outcome != residency::Mapped::Ok) {
    captured.status = CaptureStatus::Unread;
    captured.mapped = outcome;
  } else {
    captured.status = CaptureStatus::Taken;
    captured.layout = &s.layout;
    captured.words = words;
    // The step that ends the prompt's last row: the prompt's last position
    // (a turn's prompt is never empty, runtime.h).
    captured.position = call->result.prompt - 1;
  }
  call->result.message = call->message;
  call->result.tail = call->tail;
  s.running = false;
  call->on_end(call->result, captured, call->userdata);
}

void on_generate_end(const TurnResult& result, void* userdata) {
  auto call = gpu::take_back<GenerateCall>(userdata);
  State& s = *call->state;
  // A turn whose step failed empties the cache (runtime.h): the held edits
  // go with it.
  if (s.generator == nullptr || s.generator->cached() == 0) s.held.clear();
  // A loaded engine has its reader (on_reader_made); a turn runs only on a
  // loaded one, and no load begins while it runs.
  if (s.reader == nullptr) std::abort();
  // Only a turn whose prompt step ran has this run's captures in the store;
  // any other may hold an earlier run's, which are never reported.
  if (!prompt_step_ran(result) || s.cancelled) {
    s.running = false;
    call->on_end(result, Captured{}, call->userdata);
    return;
  }
  // Still running: no turn starts while the store is read. The queue orders
  // the read's copy after every step the turn submitted.
  call->result = result;
  call->message = std::string(result.message);
  call->tail = std::string(result.tail);
  capture::Reader& reader = *s.reader;
  reader.read(on_captures_read, gpu::hand_off(std::move(call)));
}

}  // namespace

// --- Engine ----------------------------------------------------------------

Engine::Engine(const gpu::Device& device) : state_(std::make_shared<State>()) {
  state_->device = &device;
}

Engine::~Engine() {
  state_->cancelled = true;
  if (state_->generator != nullptr) state_->generator->cancel();
}

LoadResult Engine::begin_load(gguf::ByteSource& front, std::size_t max_chunk,
                              const policy::LoadPolicy& policy,
                              LoadCallback ready, void* userdata) {
  State& s = *state_;
  if (s.call_pending || s.running ||
      (s.phase != Phase::Unloaded && s.phase != Phase::Loaded)) {
    return {LoadError::Busy, "a load or a turn is in progress"};
  }
  // The model loaded before goes first: its generator and reader, then its
  // buffers.
  s.generator.reset();
  s.reader.reset();
  s.upload.reset();
  s.tokenizer.reset();
  s.chat.reset();
  s.phase = Phase::Unloaded;

  gguf::TensorIndex index;
  if (const gguf::ReadResult read = gguf::read_index(front, index);
      read.error != gguf::ReadError::Ok) {
    std::string subject(gguf::to_string(read.error));
    if (read.error == gguf::ReadError::NeedMoreBytes) {
      subject += " (the first " + std::to_string(read.bytes_needed) +
                 " bytes are needed)";
      return {LoadError::Index, std::move(subject), read.bytes_needed};
    }
    return {LoadError::Index, std::move(subject)};
  }

  // The architecture, described.
  constexpr std::string_view kArchitectureKey = "general.architecture";
  std::string_view architecture_name;
  if (const auto e = index.read_string(kArchitectureKey, architecture_name);
      e != gguf::MetadataError::Ok) {
    return {LoadError::Architecture, unreadable_key(e, kArchitectureKey)};
  }
  const arch::Architecture* architecture =
      capability::find_architecture(architecture_name);
  if (architecture == nullptr) {
    return {
        LoadError::Architecture,
        "architecture " + quoted(architecture_name) + " is not implemented"};
  }
  if (arch::decoder_graph(*architecture) == nullptr) {
    return {LoadError::Architecture, "architecture " +
                                         quoted(architecture_name) +
                                         " does not generate text"};
  }
  model::ModelDescription description{};
  if (const arch::DescribeResult r = architecture->describe(index, description);
      !r.ok()) {
    return {LoadError::Architecture,
            "architecture " + quoted(architecture_name) +
                " cannot read this file: " + std::string(name(r.error)) + " (" +
                r.subject + ")"};
  }

  // The plan, within the device's limits and the policy.
  const gpu::DeviceLimits& granted = s.device->limits();
  const residency::DeviceLimits limits{
      granted.max_buffer_size, granted.max_storage_buffer_binding_size,
      granted.min_storage_buffer_offset_alignment};
  if (limits.max_buffer_size == 0 || limits.max_storage_binding_size == 0 ||
      limits.storage_offset_alignment == 0 ||
      (limits.storage_offset_alignment &
       (limits.storage_offset_alignment - 1)) != 0) {
    return {LoadError::Fit, "the device's limits cannot hold a plan"};
  }
  residency::ResidencyPlan plan;
  if (const residency::PlanResult r = residency::plan_residency(
          index, description, limits, policy,
          graph::decoder_working_set(description), plan);
      !r.ok()) {
    return {LoadError::Fit, fit_failure(r, plan, policy)};
  }

  // The tokenizer, from the front's metadata, and the stops over it.
  constexpr std::string_view kTokenizerKey = "tokenizer.ggml.model";
  std::string_view tokenizer_name;
  if (const auto e = index.read_string(kTokenizerKey, tokenizer_name);
      e != gguf::MetadataError::Ok) {
    return {LoadError::Tokenizer, unreadable_key(e, kTokenizerKey)};
  }
  const tokenizer::Algorithm* algorithm =
      capability::find_tokenizer(tokenizer_name);
  if (algorithm == nullptr) {
    return {LoadError::Tokenizer,
            "tokenizer " + quoted(tokenizer_name) + " is not implemented"};
  }
  if (algorithm->requires_pretokenizer) {
    return {LoadError::Tokenizer, "tokenizer " + quoted(tokenizer_name) +
                                      " needs a pre-tokenizer, and Frizzle "
                                      "implements none"};
  }
  std::unique_ptr<tokenizer::Tokenizer> loaded;
  if (const tokenizer::LoadResult r =
          algorithm->load(front, index, nullptr, loaded);
      !r.ok()) {
    return {LoadError::Tokenizer,
            "tokenizer " + quoted(tokenizer_name) +
                " does not load from this file: " + r.subject};
  }
  StopSet stops;
  if (const StopsResult r =
          resolve_stops(index, loaded->vocabulary(), policy.stop, stops);
      r.error != StopsError::Ok) {
    return {LoadError::Stops,
            "the stop tokens cannot be resolved: " + r.subject};
  }
  std::optional<ChatTemplate> chat;
  if (const LoadResult r = chat_of(index, loaded->vocabulary(), chat);
      !r.ok()) {
    return r;
  }

  // Begin the upload. Its callback may run before this returns.
  s.description = description;
  s.architecture = architecture;
  s.precision = policy.cache_precision;
  s.tokenizer = std::move(loaded);
  s.stops = stops;
  s.chat = std::move(chat);
  s.phase = Phase::Beginning;
  s.call_pending = true;
  residency::Upload::begin(*s.device, index, plan, front.size(),
                           capability::find_format, {}, max_chunk, on_ready,
                           gpu::hand_off(std::make_unique<LoadCall>(
                               LoadCall{state_, ready, userdata})));
  return {};
}

LoadResult Engine::write(std::uint64_t offset, std::span<const std::byte> chunk,
                         LoadCallback accepted, void* userdata) {
  State& s = *state_;
  if (s.phase == Phase::Unloaded || s.phase == Phase::Loaded) {
    return {LoadError::NotLoading, "no load is in progress"};
  }
  if (s.phase != Phase::Writing || s.call_pending) {
    return {LoadError::Busy, "the previous load call has not been answered"};
  }
  s.call_pending = true;
  s.upload->write(offset, chunk, on_accepted,
                  gpu::hand_off(std::make_unique<LoadCall>(
                      LoadCall{state_, accepted, userdata})));
  return {};
}

LoadResult Engine::finish_load(LoadCallback done, void* userdata) {
  State& s = *state_;
  if (s.phase == Phase::Unloaded || s.phase == Phase::Loaded) {
    return {LoadError::NotLoading, "no load is in progress"};
  }
  if (s.phase != Phase::Writing || s.call_pending) {
    return {LoadError::Busy, "the previous load call has not been answered"};
  }
  s.phase = Phase::Finishing;
  s.call_pending = true;
  s.upload->finish(on_finished, gpu::hand_off(std::make_unique<LoadCall>(
                                    LoadCall{state_, done, userdata})));
  return {};
}

OperationStart Engine::generate(std::string_view text,
                                std::span<const tokenizer::TokenId> kept,
                                std::span<const edit::Edit> edits,
                                const policy::TurnPolicy& policy,
                                ReplyTokenCallback on_token,
                                RunEndCallback on_end, void* userdata) {
  State& s = *state_;
  if (s.phase != Phase::Loaded) return {OperationError::NotLoaded, {}, {}};
  if (s.running) return {OperationError::Busy, {}, {}};
  if (edit::EditCheck checked = edit::check(edits, s.description);
      !checked.ok()) {
    return {OperationError::Edits, std::move(checked), {}};
  }
  Prepared made;
  if (GenerateResult prepared = s.generator->prepare(text, kept, made);
      !prepared.ok()) {
    const OperationError error = prepared.kept_out_of_vocabulary
                                     ? OperationError::Kept
                                     : OperationError::Start;
    return {error, {}, std::move(prepared)};
  }
  // The steps count from the text's last token, so text that makes none
  // has no step 0.
  if (made.text_tokens == 0) {
    return {OperationError::Start,
            {},
            {{}, {StartError::EmptyPrompt, "the text makes no tokens"}, {}}};
  }
  const std::uint32_t origin = made.text_tokens - 1;
  // The cache is reused only where this run's edits in force equal those
  // that wrote it: back to the first position they differ. Rewind never
  // lengthens the cache, and no turn runs, so it is not refused.
  std::vector<edit::Edit> placed = edit::placed(edits, origin);
  const std::uint32_t differ = cache::first_edit_difference(s.held, placed);
  if (differ != edit::kToTheEnd && !s.generator->rewind(differ)) std::abort();
  // Every entry the cache now holds is before `differ`, where the two sets
  // agree, so the placed edits describe it already, whether or not the
  // turn starts.
  s.held = std::move(placed);
  if (s.generator->cached() == 0) s.held.clear();
  // The whole table, every run: no turn is in flight, and the queue orders
  // the write before the turn's first step.
  const std::vector<std::uint32_t> table =
      edit::table(edits, s.description, origin);
  wgpuQueueWriteBuffer(s.device->queue(), s.upload->buffer(s.edit_table.buffer),
                       s.edit_table.offset, table.data(),
                       table.size() * sizeof(std::uint32_t));
  // Set before the turn starts, since its end may be reported before start
  // returns.
  s.running = true;
  void* call = gpu::hand_off(std::make_unique<GenerateCall>(
      GenerateCall{state_, on_token, on_end, userdata, {}, {}}));
  const GenerateResult started =
      s.generator->start(policy, on_generate_token, on_generate_end, call);
  if (!started.ok()) {
    (void)gpu::take_back<GenerateCall>(call);
    s.running = false;
    return {OperationError::Start, {}, started};
  }
  return {};
}

void Engine::cancel() noexcept {
  if (state_->generator != nullptr) state_->generator->cancel();
}

std::optional<ChatTemplate> Engine::chat_template() const {
  const State& s = *state_;
  if (s.phase == Phase::Unloaded) return std::nullopt;
  return s.chat;
}

std::uint32_t Engine::context() const noexcept {
  const State& s = *state_;
  if (s.phase != Phase::Loaded) return 0;
  return s.upload->plan().context_offered;
}

}  // namespace frizzle::runtime
