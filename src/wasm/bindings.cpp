// Axis R: the boundary between the engine and the worker. Changes with an
// operation the worker calls, or how one crosses.
//
// The only Emscripten-aware C++ (docs/architecture/file-mapping.md). It
// translates: each exported function below calls the engine
// (core/runtime/engine.h) and sends its answer to the worker. Every
// decision about the model, the load or a turn is the engine's.
//
// Contract
//   The worker (app/src/worker/engine/engine.ts) calls these, one at a
//   time, each with a request number its answer carries. An answer crosses
//   once, as JSON, to globalThis.frizzleOnAnswer(request, json): {"ok":
//   true, ...} with what the call returns, or {"ok": false, "stage",
//   "error", "subject"} naming the refusal or failure (the engine's
//   LoadError or OperationError, or the turn's failure), never an empty
//   success. A refusal may be answered before the call returns.
//
//   frizzle_start()
//     Requests the WebGPU device, as the native tool does (gpu/device.h),
//     and makes the engine over it. Answers request 0 with {"ok": true,
//     "adapter", "backend"}, or why there is no device. Called once, first.
//   frizzle_load_begin(request, front, front_length, file_size)
//     `front` is the file's first front_length bytes in the module's
//     memory, which the worker frees once this returns: engine.begin_load
//     reads them during the call. Allocates the chunk buffer, kChunkBytes,
//     once. Answers when the engine's buffers exist: {"ok": true, "chunk":
//     its address, "chunkBytes", "chat": {"template", "bos", "eos"} or
//     null}; or the refusal, with "bytesNeeded" when the front was short.
//   frizzle_load_chunk(request, offset, length)
//     The worker has put `length` bytes of the file at `offset` into the
//     chunk buffer. Answers when the engine has accepted them and the
//     buffer may be reused.
//   frizzle_load_finish(request)
//     Answers once the program is built: {"ok": true, "context"}.
//   frizzle_generate(request, text, text_length, kept, kept_count, edits,
//                    edit_count, seed, max_tokens)
//     `text` is the rendered conversation, UTF-8; `kept` kept_count reply
//     token identifiers, 32-bit words, the branch's kept tokens
//     (runtime/engine.h); `edits` edit_count records of six 32-bit words,
//     [layer, neuron, kind, value's bits, from, until], kind Set 0, Scale 1,
//     Zero 2, from and until in steps, until 0xFFFFFFFF for to the end
//     (edit/edit.h). All three are in the module's memory and freed by the
//     worker once this returns: the engine reads them during the call. A
//     kind word past 255 crosses as 255, which the engine's check refuses
//     as Kind, so no word wraps into a valid kind. The seed is a double
//     holding an integer below 2^53; max_tokens 0 inspects (no reply, the
//     captures at the input's last position). Each emitted token crosses
//     as it is emitted, to globalThis.frizzleOnText(request, text, token):
//     its identifier, and the text it completes, possibly "" (generator.h);
//     then the answer: {"ok": true, "end": "stop" | "limit" | "context" |
//     "cancelled", "tokens", "promptTokens", "reusedTokens", "reply",
//     "tail", "captures"}, where "reply" is the emitted identifiers in
//     order, as a JSON array, the same the token crossings carried; "tail"
//     the text the last token ends with that its crossing could not carry,
//     U+FFFD for a character the turn stopped inside, else "" (generator.h);
//     and "captures" is
//     the engine's Captured (runtime/engine.h), as it gave it:
//       {"status": "taken", "slots": [[first, width], ...]}, one pair a
//         layer, its neurons slot, with the neurons themselves crossing in
//         the same call as the answer: frizzleOnAnswer(request, answer,
//         neurons), `neurons` a Float32Array of the store's neuron region,
//         every layer's neurons slot, bit for bit (capture/capture.h,
//         region 1). The store's other captures do not cross: the page
//         has nothing to show them in, and sending 2.5 MiB more a run for
//         nothing to read is not worth it;
//       {"status": "not-taken"}: the turn was cancelled, so there are none;
//       {"status": "unread", "reason"}: the read back failed, named by
//         residency::Mapped ("cancelled", "device-lost", "unexplained",
//         "internal").
//     A turn that failed is refused as before, and carries no captures.
//   frizzle_cancel()
//     engine.cancel(): the running turn ends Cancelled and is answered so.
//
// Algorithm, the state between crossings
//   One module, one page, one device: the device, the engine over it, and
//   the chunk buffer live here, the boundary's state, made by
//   frizzle_start and frizzle_load_begin and held for the module's life.
//   Each engine callback carries the request number and sends the answer.
//
// Design and cost
//   - A phase a crossing (WASM.2): a load is 2 + ceil(size / 16 MiB)
//     crossings, 46 for Gemma 3's 722 MB; a turn one in, then one out for
//     each emitted token, and one answer. The kept identifiers and the
//     reply's cross once a run, a few kilobytes, in the run's own calls.
//   - Bytes cross by address and length into the module's memory, which
//     the worker writes through a view made afresh for each call: growing
//     the heap detaches any view kept (WASM.1). The heap's initial size and
//     its cap are set at link (CMakeLists.txt), growth a fallback, and
//     measured in slice 3's last step.
//   - The chunk buffer is the only large allocation, made once (WASM.9): a
//     chunk is read from the browser's storage straight into it.
//   - Text crosses as UTF-8 bytes in the module's memory, read by the
//     worker at once, before the next call can grow the heap.
//   - JSON for answers: one small object a call, escaped here; nothing a
//     token but its text and identifier. The reply's identifiers in the
//     answer, about six bytes a token in JSON: a 4,000-token reply is about
//     24 KB, once.
//   - The neurons, 702 KiB for Gemma 3 1B, cross once a run, in the
//     answer's own call (WASM.2): the engine's span is valid only during
//     its callback, so they are copied out of the heap there, by a slice of
//     a view made at that moment and not kept (WASM.1). One copy, bits
//     intact: a typed array's slice copies bytes, never values. Slots are
//     26 pairs in the JSON, so the worker never reckons a layer's place.
//   - Session state at namespace scope, which I.1 cautions against: the
//     module has one engine, as a page has one, and every function names
//     what it reads; nothing outside this file sees it.
//
// Corpus guidance
//   C++ Core Guidelines
//     CPL.3  If you must use C for interfaces, use C++ in the calling code:
//            the exports are C, each a thin call into the engine's C++.
//     I.1    Make interfaces explicit: each crossing's bytes, ownership and
//            answer are stated above; the namespace-scope state is named.
//   C++ performance guidelines
//     WASM.1 Size linear memory, and never hold a view across growth: the
//            neurons are sliced out of a view made in the call that uses
//            it.
//     WASM.2 Batch work across the JS boundary: a phase a crossing; the
//            neurons ride with the answer, one crossing a run.
//     WASM.9 Stream in bounded chunks: one chunk buffer, reused.

#include <emscripten.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/capture/capture.h"
#include "core/edit/edit.h"
#include "core/gguf/byte_source.h"
#include "core/gpu/device.h"
#include "core/policy/policy.h"
#include "core/residency/mapping.h"
#include "core/runtime/engine.h"
#include "core/runtime/runtime.h"

// Answers and text, to the worker (app/src/worker/engine/engine.ts).
EM_JS_DEPS(frizzle_bindings, "$UTF8ToString");
EM_JS(void, frizzle_answer, (std::uint32_t request, const char* json),
      { globalThis.frizzleOnAnswer(request, JSON.parse(UTF8ToString(json))); });
// An answer with a run's neurons: sliced out of a view made now, so the
// copy is the worker's and survives the heap growing (the header, WASM.1).
EM_JS(void, frizzle_answer_neurons,
      (std::uint32_t request, const char* json, const float* neurons,
       std::uint32_t count),
      {
        const first = neurons >> 2;
        globalThis.frizzleOnAnswer(request, JSON.parse(UTF8ToString(json)),
                                   HEAPF32.slice(first, first + count));
      });
EM_JS(void, frizzle_text,
      (std::uint32_t request, const char* text, std::uint32_t length,
       std::uint32_t token),
      {
        globalThis.frizzleOnText(request, UTF8ToString(text, length), token);
      });

namespace {

namespace runtime = frizzle::runtime;
namespace edit = frizzle::edit;

constexpr std::size_t kChunkBytes = std::size_t{16} << 20U;
constexpr std::uint32_t kStartRequest = 0;
constexpr std::uint32_t kKindCeiling = 255;

// The boundary's state (the header's Algorithm).
std::unique_ptr<frizzle::gpu::Device> device;
std::unique_ptr<runtime::Engine> engine;
std::vector<std::byte> chunk;
// The running turn's emitted identifiers, for its answer's "reply":
// emptied once each turn is answered, never when a run is refused, which
// may be refused because a turn still runs; reserved at load for the
// context, so a token appends without allocating.
std::vector<std::uint32_t> reply;

// A JSON string literal of `text`, escaped.
std::string quoted(std::string_view text) {
  std::string out = "\"";
  for (const char c : text) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char escaped[8];
          std::snprintf(escaped, sizeof escaped, "\\u%04x",
                        static_cast<unsigned>(static_cast<unsigned char>(c)));
          out += escaped;
        } else {
          out += c;
        }
    }
  }
  return out + "\"";
}

void answer(std::uint32_t request, const std::string& json) {
  frizzle_answer(request, json.c_str());
}

void refuse(std::uint32_t request, std::string_view stage,
            std::string_view error, std::string_view subject,
            const std::string& extra = {}) {
  answer(request, "{\"ok\":false,\"stage\":" + quoted(stage) +
                      ",\"error\":" + quoted(error) +
                      ",\"subject\":" + quoted(subject) + extra + "}");
}

std::string_view name(runtime::LoadError error) {
  switch (error) {
    case runtime::LoadError::Ok:
      return "ok";
    case runtime::LoadError::Busy:
      return "busy";
    case runtime::LoadError::NotLoading:
      return "not-loading";
    case runtime::LoadError::Index:
      return "index";
    case runtime::LoadError::Architecture:
      return "architecture";
    case runtime::LoadError::Fit:
      return "fit";
    case runtime::LoadError::Tokenizer:
      return "tokenizer";
    case runtime::LoadError::Stops:
      return "stops";
    case runtime::LoadError::Upload:
      return "upload";
    case runtime::LoadError::Graph:
      return "graph";
    case runtime::LoadError::Program:
      return "program";
    case runtime::LoadError::Capture:
      return "capture";
    case runtime::LoadError::Cancelled:
      return "cancelled";
  }
  return "unknown";
}

std::string_view name(runtime::TurnEnd end) {
  switch (end) {
    case runtime::TurnEnd::Stop:
      return "stop";
    case runtime::TurnEnd::Limit:
      return "limit";
    case runtime::TurnEnd::Context:
      return "context";
    case runtime::TurnEnd::Cancelled:
      return "cancelled";
  }
  return "unknown";
}

std::string_view name(runtime::TurnFailure failure) {
  switch (failure) {
    case runtime::TurnFailure::None:
      return "none";
    case runtime::TurnFailure::NonFinite:
      return "non-finite";
    case runtime::TurnFailure::Step:
      return "step";
    case runtime::TurnFailure::DeviceLost:
      return "device-lost";
  }
  return "unknown";
}

// A load stage's answer: ok with `ok_fields`, or its refusal.
void answer_load(std::uint32_t request, const runtime::LoadResult& result,
                 const std::string& ok_fields) {
  if (!result.ok()) {
    refuse(request, "load", name(result.error), result.subject,
           result.bytes_needed == 0
               ? std::string{}
               : ",\"bytesNeeded\":" + std::to_string(result.bytes_needed));
    return;
  }
  answer(request, "{\"ok\":true" + ok_fields + "}");
}

std::uint32_t request_of(void* userdata) {
  return static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(userdata));
}

void* userdata_of(std::uint32_t request) {
  return reinterpret_cast<void*>(static_cast<std::uintptr_t>(request));
}

void on_begun(const runtime::LoadResult& result, void* userdata) {
  std::string chat = ",\"chat\":null";
  if (result.ok()) {
    if (const std::optional<runtime::ChatTemplate> c =
            engine->chat_template()) {
      chat = ",\"chat\":{\"template\":" + quoted(c->text) +
             ",\"bos\":" + quoted(c->bos) + ",\"eos\":" + quoted(c->eos) + "}";
    }
  }
  answer_load(
      request_of(userdata), result,
      ",\"chunk\":" +
          std::to_string(reinterpret_cast<std::uintptr_t>(chunk.data())) +
          ",\"chunkBytes\":" + std::to_string(chunk.size()) + chat);
}

void on_written(const runtime::LoadResult& result, void* userdata) {
  answer_load(request_of(userdata), result, {});
}

void on_finished(const runtime::LoadResult& result, void* userdata) {
  if (result.ok()) reply.reserve(engine->context());
  answer_load(request_of(userdata), result,
              ",\"context\":" + std::to_string(engine->context()));
}

void on_token(frizzle::tokenizer::TokenId token, std::string_view text,
              void* userdata) {
  const auto id = static_cast<std::uint32_t>(token);
  // One callback an emitted token (generator.h).
  reply.push_back(id);
  frizzle_text(request_of(userdata), text.data(),
               static_cast<std::uint32_t>(text.size()), id);
}

std::string_view name(frizzle::residency::Mapped mapped) {
  switch (mapped) {
    case frizzle::residency::Mapped::Ok:
      return "ok";
    case frizzle::residency::Mapped::Cancelled:
      return "cancelled";
    case frizzle::residency::Mapped::Internal:
      return "internal";
    case frizzle::residency::Mapped::DeviceLost:
      return "device-lost";
    case frizzle::residency::Mapped::Unexplained:
      return "unexplained";
  }
  return "unknown";
}

// The answer's "captures" (the header's frizzle_generate).
std::string captures_json(const runtime::Captured& captured) {
  switch (captured.status) {
    case runtime::CaptureStatus::Taken: {
      std::string slots;
      // Each layer's neurons slot: the region that crosses (the header).
      for (const frizzle::capture::LayerSlots& l : captured.layout->layers) {
        if (!slots.empty()) slots += ',';
        slots += "[" + std::to_string(l.neurons.first) + "," +
                 std::to_string(l.neurons.width) + "]";
      }
      return "{\"status\":\"taken\",\"slots\":[" + slots + "]}";
    }
    case runtime::CaptureStatus::NotTaken:
      return "{\"status\":\"not-taken\"}";
    case runtime::CaptureStatus::Unread:
      return "{\"status\":\"unread\",\"reason\":" +
             quoted(name(captured.mapped)) + "}";
  }
  return "{\"status\":\"unread\",\"reason\":\"unknown\"}";
}

// The answer's "reply": the emitted identifiers, as a JSON array.
std::string reply_json() {
  std::string out = "[";
  for (std::size_t i = 0; i < reply.size(); ++i) {
    if (i > 0) out += ',';
    out += std::to_string(reply[i]);
  }
  return out + "]";
}

void on_end(const runtime::TurnResult& result,
            const runtime::Captured& captured, void* userdata) {
  const std::uint32_t request = request_of(userdata);
  if (result.failure != runtime::TurnFailure::None) {
    reply.clear();
    refuse(request, "generate", name(result.failure), result.message);
    return;
  }
  const std::string json =
      "{\"ok\":true,\"end\":" + quoted(name(result.end)) +
      ",\"tokens\":" + std::to_string(result.emitted) +
      ",\"promptTokens\":" + std::to_string(result.prompt) +
      ",\"reusedTokens\":" + std::to_string(result.reused) +
      ",\"reply\":" + reply_json() + ",\"tail\":" + quoted(result.tail) +
      ",\"captures\":" + captures_json(captured) + "}";
  reply.clear();
  if (captured.status == runtime::CaptureStatus::Taken) {
    // The store's first region, every layer's neurons; the rest crosses in
    // slice 4, step 6 (the header).
    const std::span<const float> neurons =
        captured.words.first(captured.layout->neuron_words);
    frizzle_answer_neurons(request, json.c_str(), neurons.data(),
                           static_cast<std::uint32_t>(neurons.size()));
    return;
  }
  answer(request, json);
}

// A generate refused at once, named as the engine names it.
std::string refusal(const runtime::OperationStart& start) {
  switch (start.error) {
    case runtime::OperationError::Ok:
      return "ok";
    case runtime::OperationError::NotLoaded:
      return "no model is loaded";
    case runtime::OperationError::Busy:
      return "a turn is running";
    case runtime::OperationError::Edits:
      return start.edits.subject;
    case runtime::OperationError::Kept:
      return "kept token " +
             std::to_string(start.start.kept_out_of_vocabulary.value_or(0)) +
             " names no token";
    case runtime::OperationError::Start:
      break;
  }
  if (!start.start.start.subject.empty()) return start.start.start.subject;
  return "the text could not be encoded";
}

}  // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE void frizzle_start() {
  frizzle::gpu::Device::request(
      [](std::unique_ptr<frizzle::gpu::Device> granted, const char* error,
         void* /*userdata*/) {
        if (granted == nullptr) {
          refuse(kStartRequest, "device", "device",
                 error != nullptr ? error : "no device");
          return;
        }
        device = std::move(granted);
        engine = std::make_unique<runtime::Engine>(*device);
        const frizzle::gpu::AdapterInfo& info = device->adapter_info();
        answer(kStartRequest,
               "{\"ok\":true,\"adapter\":" + quoted(info.description) +
                   ",\"backend\":" + quoted(info.backend) + "}");
      },
      nullptr);
}

EMSCRIPTEN_KEEPALIVE void frizzle_load_begin(std::uint32_t request,
                                             const std::byte* front,
                                             std::uint32_t front_length,
                                             double file_size) {
  if (engine == nullptr) {
    refuse(request, "load", "no-device", "the device was never acquired");
    return;
  }
  if (chunk.empty()) chunk.resize(kChunkBytes);
  frizzle::gguf::MemoryByteSource source(
      std::span<const std::byte>(front, front_length),
      static_cast<std::uint64_t>(file_size));
  const runtime::LoadResult begun =
      engine->begin_load(source, kChunkBytes, frizzle::policy::LoadPolicy{},
                         on_begun, userdata_of(request));
  if (!begun.ok()) answer_load(request, begun, {});
}

EMSCRIPTEN_KEEPALIVE void frizzle_load_chunk(std::uint32_t request,
                                             double offset,
                                             std::uint32_t length) {
  if (engine == nullptr || length > chunk.size()) {
    refuse(request, "load", "chunk", "no load, or a chunk past the buffer");
    return;
  }
  const runtime::LoadResult written =
      engine->write(static_cast<std::uint64_t>(offset),
                    std::span<const std::byte>(chunk.data(), length),
                    on_written, userdata_of(request));
  if (!written.ok()) answer_load(request, written, {});
}

EMSCRIPTEN_KEEPALIVE void frizzle_load_finish(std::uint32_t request) {
  if (engine == nullptr) {
    refuse(request, "load", "no-device", "the device was never acquired");
    return;
  }
  const runtime::LoadResult finished =
      engine->finish_load(on_finished, userdata_of(request));
  if (!finished.ok()) answer_load(request, finished, {});
}

EMSCRIPTEN_KEEPALIVE void frizzle_generate(
    std::uint32_t request, const char* text, std::uint32_t text_length,
    const std::uint32_t* kept, std::uint32_t kept_count,
    const std::uint32_t* edits, std::uint32_t edit_count, double seed,
    std::uint32_t max_tokens) {
  if (engine == nullptr) {
    refuse(request, "generate", "no-device", "the device was never acquired");
    return;
  }
  // TokenId is an enum over u32, so the words are read one by one rather
  // than reinterpreted.
  std::vector<frizzle::tokenizer::TokenId> branch;
  branch.reserve(kept_count);
  for (const std::uint32_t id : std::span(kept, kept_count)) {
    branch.push_back(frizzle::tokenizer::TokenId{id});
  }
  constexpr std::size_t kWords = 6;
  const std::span<const std::uint32_t> words(edits,
                                             std::size_t{edit_count} * kWords);
  std::vector<edit::Edit> set;
  set.reserve(edit_count);
  for (std::size_t i = 0; i + kWords <= words.size(); i += kWords) {
    const std::uint32_t kind =
        words[i + 2] > kKindCeiling ? kKindCeiling : words[i + 2];
    set.push_back({words[i], words[i + 1],
                   static_cast<edit::EditKind>(static_cast<std::uint8_t>(kind)),
                   std::bit_cast<float>(words[i + 3]), words[i + 4],
                   words[i + 5]});
  }
  frizzle::policy::TurnPolicy policy;
  policy.seed = frizzle::policy::Seed{static_cast<std::uint64_t>(seed)};
  policy.max_tokens = max_tokens;
  const runtime::OperationStart started =
      engine->generate(std::string_view(text, text_length), branch, set, policy,
                       on_token, on_end, userdata_of(request));
  if (!started.ok()) refuse(request, "generate", "refused", refusal(started));
}

EMSCRIPTEN_KEEPALIVE void frizzle_cancel() {
  if (engine != nullptr) engine->cancel();
}

}  // extern "C"
