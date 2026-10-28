// Axis V: a native tool for analysis, never part of the product.
//
// frizzle: for one prompt, prints the reply the model generates, natively on
// Dawn, through the same engine operations the browser calls
// (src/core/runtime/engine.h).
//
// Contract
//   frizzle [--model PATH] [--seed N] [--max-tokens N] TEXT
//   frizzle [options] --prompt-file FILE
//   - TEXT, or FILE's contents, is exactly what the model reads: Gemma's
//     chat template already applied. The tool renders no template; its
//     usage shows Gemma 3's rendered form for one user message. (A shell's
//     $(...) drops trailing newlines, the one after "model" among them;
//     --prompt-file keeps every byte.)
//   - PATH defaults to .cache/models/gemma-3-1b-it-q4_0.gguf, relative to
//     the working directory, as scripts/fetch_model.py leaves it from the
//     repository's root. The tool does not verify the file's hash; the
//     fetch script did.
//   - --seed defaults to 0, --max-tokens to 256.
//   - Prints, in order: the device; the load and turn settings it ran with;
//     the reply, streamed as the model writes it; how the reply ended, its
//     token count and the prompt's.
//   - Exit status 0 when the reply completed; 1 with
//     the engine's named refusal or failure on stderr otherwise; 2 for bad
//     arguments. Nothing is printed as a result that was not computed.
//
// Algorithm
//   1. Parse the arguments; read the prompt text.
//   2. Create a WebGPU instance, request a device (gpu/device.h) and pump the
//      instance's events until it arrives.
//   3. Open the model file. begin_load with a byte source that reads the
//      file at an offset on demand, so nothing holds the whole file in
//      memory; write the file in 16 MiB chunks, each accepted before the
//      next; finish_load. Pump events throughout.
//   4. generate(text, policy), printing text as it streams; print how it
//      ended.
//
// Design and cost
//   - The file-backed byte source lives here, not in the core: the browser
//     supplies its bytes another way, and the core reads only through the
//     ByteSource interface (gguf/byte_source.h). ByteSource::read has no
//     status for an I/O error; the source records one, reports the range
//     NotResident (true: it was not supplied), and the tool prints the
//     recorded error beside whatever refusal follows, so it is never hidden.
//   - The file is read with the C library's stdio (fopen, fseek, fread), not
//     POSIX calls, so the tool has no platform-specific code of its own;
//     fseek takes a long, which holds the file's offsets on the 64-bit hosts
//     the native build supports, and the tool refuses a file past it.
//   - Output goes through std::format and one fwrite helper, never a
//     variadic C function.
//   - Single-threaded: every callback runs inside wgpuInstanceProcessEvents,
//     called by the tool's own loop (gpu/callback_mode.h). The loop sleeps
//     100 µs between pumps while it waits, which bounds the CPU it burns and
//     adds at most that to each wait. There is no timeout: a wait ends with
//     its callback, and a lost device ends every wait with a failure.
//   - Memory: one 16 MiB chunk buffer, reused; the model's weights live on
//     the GPU. Peak residency for the file is the chunk, whatever its size.
//
// Corpus guidance
//   C++ Core Guidelines
//     F.46   int is the return type for main(): with the exit statuses above;
//            main catches every exception, reporting it as a failure.
//     R.1    Manage resources with RAII: the file is closed by its handle's
//            destructor, on every path.
//     E.27   If you can't throw, use error codes systematically: each stage's
//            named result decides the exit status.
//     I.11   Never transfer ownership by a raw pointer or reference: the
//            engine borrows the device, which main holds and outlives it; the
//            byte source borrows the file.
//   C++ performance guidelines
//     WASM.9  Stream in bounded chunks and release each one: 16 MiB at a time
//             through one reused buffer; offsets checked against the file's
//             size without an addition that could wrap.

#include <webgpu/webgpu.h>

#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <format>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "core/gguf/byte_source.h"
#include "core/gpu/device.h"
#include "core/gpu/wgpu_handles.h"
#include "core/policy/policy.h"
#include "core/runtime/engine.h"
#include "core/runtime/generator.h"
#include "core/runtime/runtime.h"
#include "core/runtime/turn.h"
#include "core/tokenizer/tokenizer.h"

namespace {

using frizzle::gguf::ByteSource;
using frizzle::gguf::ReadStatus;
namespace runtime = frizzle::runtime;
namespace policy = frizzle::policy;

constexpr int kOk = 0;
constexpr int kFailed = 1;
constexpr int kBadArguments = 2;
constexpr std::size_t kChunkBytes = std::size_t{16} << 20U;
constexpr std::uint32_t kDefaultMaxTokens = 256;
constexpr unsigned kMebibyteShift = 20;
constexpr auto kPumpInterval = std::chrono::microseconds{100};
constexpr std::string_view kDefaultModel =
    ".cache/models/gemma-3-1b-it-q4_0.gguf";

constexpr std::string_view kUsage =
    "usage: frizzle [--model PATH] [--seed N] [--max-tokens N] TEXT\n"
    "       frizzle [options] --prompt-file FILE\n"
    "\n"
    "TEXT is exactly what the model reads: Gemma's chat template already\n"
    "applied. For one user message, Gemma 3's rendered form is\n"
    "  <bos><start_of_turn>user\\n{message}<end_of_turn>\\n"
    "<start_of_turn>model\\n\n"
    "with each \\n a newline. A shell's $(...) drops the last newline;\n"
    "--prompt-file keeps every byte.\n"
    "\n"
    "  --model PATH      default .cache/models/gemma-3-1b-it-q4_0.gguf\n"
    "  --seed N          the reply's sampling seed, default 0\n"
    "  --max-tokens N    the reply's most tokens, default 256\n";

// --- Output ------------------------------------------------------------------

void put(std::FILE* to, std::string_view text) {
  std::fwrite(text.data(), 1, text.size(), to);
}

// --- Arguments ---------------------------------------------------------------

struct Options {
  std::string model{kDefaultModel};
  std::uint64_t seed = 0;
  std::uint32_t max_tokens = kDefaultMaxTokens;
  std::string prompt;  // the text, or the file that holds it
  bool prompt_is_file = false;
  bool has_prompt = false;
};

template <typename T>
bool parse_number(std::string_view text, T& out) {
  const char* begin = std::to_address(text.begin());
  const char* end = std::to_address(text.end());
  const auto [at, error] = std::from_chars(begin, end, out);
  return error == std::errc{} && at == end;
}

// The options, or the reason they are refused.
std::optional<Options> parse(std::span<char*> args, std::string& refused) {
  Options o;
  for (std::size_t i = 1; i < args.size(); ++i) {
    const std::string_view arg = args[i];
    const bool takes_value = arg == "--model" || arg == "--prompt-file" ||
                             arg == "--seed" || arg == "--max-tokens";
    if (takes_value) {
      if (i + 1 >= args.size()) {
        refused = std::format("{} needs a value", arg);
        return std::nullopt;
      }
      const std::string_view value = args[++i];
      if (arg == "--model") {
        o.model = value;
      } else if (arg == "--prompt-file") {
        if (o.has_prompt) {
          refused = "give exactly one of TEXT and --prompt-file";
          return std::nullopt;
        }
        o.prompt = value;
        o.prompt_is_file = true;
        o.has_prompt = true;
      } else if (arg == "--seed") {
        if (!parse_number(value, o.seed)) {
          refused = "--seed takes an unsigned 64-bit integer";
          return std::nullopt;
        }
      } else if (!parse_number(value, o.max_tokens) || o.max_tokens == 0) {
        refused = "--max-tokens takes an integer from 1 to 4294967295";
        return std::nullopt;
      }
    } else if (arg.starts_with("--")) {
      refused = std::format("unknown option {}", arg);
      return std::nullopt;
    } else if (o.has_prompt) {
      refused = "give exactly one of TEXT and --prompt-file";
      return std::nullopt;
    } else {
      o.prompt = arg;
      o.has_prompt = true;
    }
  }
  if (!o.has_prompt) {
    refused = "give exactly one of TEXT and --prompt-file";
    return std::nullopt;
  }
  return o;
}

// --- Files -------------------------------------------------------------------

struct CloseFile {
  void operator()(std::FILE* file) const noexcept {
    // The unique_ptr below owns the FILE; the check wants GSL's owner<>
    // marker, and GSL is not a dependency of Frizzle.
    std::fclose(file);  // NOLINT(cppcoreguidelines-owning-memory)
  }
};
using File = std::unique_ptr<std::FILE, CloseFile>;  // R.1

// fseek's own offset type, which the C library spells long.
using SeekOffset = long;  // NOLINT(google-runtime-int)

// Reads [offset, offset + out.size()) whole: 0, or the errno that stopped it
// (EIO when the C library set none). Precondition: the range is inside the
// file and offset fits a SeekOffset.
int read_at(std::FILE* file, std::uint64_t offset, std::span<std::byte> out) {
  errno = 0;
  if (std::fseek(file, static_cast<SeekOffset>(offset), SEEK_SET) != 0) {
    return errno != 0 ? errno : EIO;
  }
  if (std::fread(out.data(), 1, out.size(), file) != out.size()) {
    return errno != 0 ? errno : EIO;
  }
  return 0;
}

std::optional<std::string> read_text(const std::string& path) {
  const File file(std::fopen(path.c_str(), "rb"));
  if (!file) return std::nullopt;
  std::string text;
  constexpr std::size_t kReadBytes = 4096;
  std::array<char, kReadBytes> buffer{};
  for (;;) {
    const std::size_t got =
        std::fread(buffer.data(), 1, buffer.size(), file.get());
    text.append(buffer.data(), got);
    if (got < buffer.size()) break;
  }
  if (std::ferror(file.get()) != 0) return std::nullopt;
  return text;
}

// The model file as a ByteSource, read at an offset on demand. An I/O error
// is recorded and the range reported NotResident (Design and cost, above).
class FileByteSource final : public ByteSource {
 public:
  struct Opened {
    std::FILE* file;
    std::uint64_t size;
  };
  explicit FileByteSource(Opened opened) noexcept
      : file_(opened.file), size_(opened.size) {}
  [[nodiscard]] std::uint64_t size() const noexcept override { return size_; }
  [[nodiscard]] ReadStatus read(std::uint64_t offset,
                                std::span<std::byte> out) noexcept override {
    const auto want = static_cast<std::uint64_t>(out.size());
    if (want > size_ || offset > size_ - want) return ReadStatus::OutsideFile;
    if (const int error = read_at(file_, offset, out); error != 0) {
      if (io_error_ == 0) io_error_ = error;
      return ReadStatus::NotResident;
    }
    return ReadStatus::Ok;
  }
  // The first I/O error a read met, as errno; 0 for none.
  [[nodiscard]] int io_error() const noexcept { return io_error_; }

 private:
  std::FILE* file_;  // borrowed
  std::uint64_t size_;
  int io_error_ = 0;
};

// --- Waiting on callbacks ----------------------------------------------------

void pump_until(WGPUInstance instance, const bool& done) {
  while (!done) {
    wgpuInstanceProcessEvents(instance);
    if (done) break;
    std::this_thread::sleep_for(kPumpInterval);
  }
}

struct Acquired {
  std::unique_ptr<frizzle::gpu::Device> device;
  std::string error;
  bool done = false;
};

struct Load {
  runtime::LoadResult result;
  bool done = false;
};

void on_load(const runtime::LoadResult& result, void* userdata) {
  auto& load = *static_cast<Load*>(userdata);
  load.result = result;
  load.done = true;
}

struct Reply {
  runtime::TurnResult result{};
  std::string message;
  bool done = false;
};

// --- Naming ------------------------------------------------------------------

std::string_view name(runtime::StartError error) noexcept {
  switch (error) {
    case runtime::StartError::Ok:
      return "ok";
    case runtime::StartError::Busy:
      return "a turn is running";
    case runtime::StartError::DeviceLost:
      return "the GPU device was lost";
    case runtime::StartError::EmptyPrompt:
      return "the prompt is empty";
    case runtime::StartError::PromptTooLong:
      return "the prompt is longer than the context";
    case runtime::StartError::Settings:
      return "the turn's settings are refused";
  }
  return "unknown";
}

std::string_view name(frizzle::tokenizer::EncodeError error) noexcept {
  using frizzle::tokenizer::EncodeError;
  switch (error) {
    case EncodeError::Ok:
      return "ok";
    case EncodeError::InvalidUtf8:
      return "the text is not valid UTF-8";
    case EncodeError::TooManyTokens:
      return "the text makes more tokens than the context holds";
    case EncodeError::TooLong:
      return "the text is too long to encode";
  }
  return "unknown";
}

std::string refusal(const runtime::OperationStart& start) {
  switch (start.error) {
    case runtime::OperationError::Ok:
      return "ok";
    case runtime::OperationError::NotLoaded:
      return "no model is loaded";
    case runtime::OperationError::Busy:
      return "a turn is running";
    case runtime::OperationError::Start:
      break;
  }
  const runtime::GenerateResult& g = start.start;
  if (g.encode.error != frizzle::tokenizer::EncodeError::Ok) {
    return std::string(name(g.encode.error));
  }
  // The runtime's subject names the refusal and its numbers where it has
  // them; otherwise the refusal's name.
  if (!g.start.subject.empty()) return g.start.subject;
  return std::string(name(g.start.error));
}

std::string_view name(runtime::TurnEnd end) noexcept {
  switch (end) {
    case runtime::TurnEnd::Stop:
      return "a stop token";
    case runtime::TurnEnd::Limit:
      return "the token limit";
    case runtime::TurnEnd::Context:
      return "the end of the context";
    case runtime::TurnEnd::Cancelled:
      return "cancelled";
  }
  return "unknown";
}

std::string_view name(runtime::TurnFailure failure) noexcept {
  switch (failure) {
    case runtime::TurnFailure::None:
      return "none";
    case runtime::TurnFailure::NonFinite:
      return "the top candidate's logit was not finite";
    case runtime::TurnFailure::Step:
      return "a step failed";
    case runtime::TurnFailure::DeviceLost:
      return "the GPU device was lost";
  }
  return "unknown";
}

std::string_view name(policy::CachePrecision precision) noexcept {
  switch (precision) {
    case policy::CachePrecision::F16:
      return "F16";
    case policy::CachePrecision::BF16:
      return "BF16";
    case policy::CachePrecision::Q8_0:
      return "Q8_0";
  }
  return "unknown";
}

int fail(std::string_view stage, std::string_view why) {
  put(stderr, std::format("frizzle: {}: {}\n", stage, why));
  return kFailed;
}

// The load's refusal or failure, with the file's I/O error when there was
// one, so a read that failed is never hidden behind what it caused.
int fail_load(const runtime::LoadResult& result, const FileByteSource& source) {
  std::string why = result.subject;
  if (source.io_error() != 0) {
    why += std::format(" [reading the model file failed: {}]",
                       std::strerror(source.io_error()));
  }
  return fail("load", why);
}

// --- The run -----------------------------------------------------------------

int run(const Options& options, const std::string& text) {
  // The device.
  const frizzle::gpu::Instance instance(wgpuCreateInstance(nullptr));
  if (!instance) return fail("device", "no WebGPU instance could be created");
  Acquired acquired;
  frizzle::gpu::Device::request(
      instance.get(),
      [](std::unique_ptr<frizzle::gpu::Device> device, const char* error,
         void* userdata) {
        auto& a = *static_cast<Acquired*>(userdata);
        a.device = std::move(device);
        if (error != nullptr) a.error = error;
        a.done = true;
      },
      &acquired);
  pump_until(instance.get(), acquired.done);
  if (acquired.device == nullptr) return fail("device", acquired.error);
  const frizzle::gpu::AdapterInfo& adapter = acquired.device->adapter_info();
  put(stdout,
      std::format("device: {} ({})\n", adapter.description, adapter.backend));

  // The file.
  std::error_code size_error;
  const std::uintmax_t file_size =
      std::filesystem::file_size(options.model, size_error);
  if (size_error) {
    return fail("model",
                std::format("{}: {}", options.model, size_error.message()));
  }
  if (file_size >
      static_cast<std::uintmax_t>(std::numeric_limits<SeekOffset>::max())) {
    return fail("model", std::format("{}: larger than this tool can seek",
                                     options.model));
  }
  const File file(std::fopen(options.model.c_str(), "rb"));
  if (!file) {
    return fail("model",
                std::format("{}: {}", options.model, std::strerror(errno)));
  }
  FileByteSource source({file.get(), file_size});

  // The load.
  const policy::LoadPolicy load_policy;
  runtime::Engine engine(*acquired.device);
  Load load;
  if (const runtime::LoadResult begun =
          engine.begin_load(source, kChunkBytes, load_policy, on_load, &load);
      !begun.ok()) {
    return fail_load(begun, source);
  }
  pump_until(instance.get(), load.done);
  if (!load.result.ok()) return fail_load(load.result, source);

  std::vector<std::byte> chunk(kChunkBytes);
  for (std::uint64_t offset = 0; offset < file_size;) {
    const std::uint64_t left = file_size - offset;
    const std::size_t length =
        left < kChunkBytes ? static_cast<std::size_t>(left) : kChunkBytes;
    const std::span<std::byte> piece(chunk.data(), length);
    if (const int error = read_at(file.get(), offset, piece); error != 0) {
      return fail("model",
                  std::format("{}: {}", options.model, std::strerror(error)));
    }
    load = {};
    if (const runtime::LoadResult written =
            engine.write(offset, piece, on_load, &load);
        !written.ok()) {
      return fail_load(written, source);
    }
    pump_until(instance.get(), load.done);
    if (!load.result.ok()) return fail_load(load.result, source);
    offset += length;
  }
  load = {};
  if (const runtime::LoadResult finished = engine.finish_load(on_load, &load);
      !finished.ok()) {
    return fail_load(finished, source);
  }
  pump_until(instance.get(), load.done);
  if (!load.result.ok()) return fail_load(load.result, source);

  policy::TurnPolicy turn;
  turn.seed = policy::Seed{options.seed};
  turn.max_tokens = options.max_tokens;
  put(stdout,
      std::format("load: cache {}, memory budget {} MiB, rollback reserve {}; "
                  "context offered {} tokens\n",
                  name(load_policy.cache_precision),
                  load_policy.memory_budget >> kMebibyteShift,
                  load_policy.rollback_reserve, engine.context()));
  put(stdout,
      std::format("turn: temperature {}, top_k {}, top_p {}, min_p {}, seed "
                  "{}, max tokens {} (policy.h's defaults for an unmeasured "
                  "model)\n",
                  turn.sampling.temperature, turn.sampling.top_k,
                  turn.sampling.top_p, turn.sampling.min_p, options.seed,
                  turn.max_tokens));

  // The reply.
  put(stdout, "reply:\n");
  Reply reply;
  const runtime::OperationStart generating = engine.generate(
      text, turn,
      [](std::string_view piece, void* /*userdata*/) {
        put(stdout, piece);
        std::fflush(stdout);
      },
      [](const runtime::TurnResult& result, void* userdata) {
        auto& r = *static_cast<Reply*>(userdata);
        r.result = result;
        r.message = result.message;  // valid only during the call
        r.done = true;
      },
      &reply);
  if (!generating.ok()) return fail("generate", refusal(generating));
  pump_until(instance.get(), reply.done);
  put(stdout, "\n");
  const runtime::TurnResult& r = reply.result;
  if (r.failure != runtime::TurnFailure::None) {
    std::string why(name(r.failure));
    if (!reply.message.empty()) why += ": " + reply.message;
    return fail("generate", why);
  }
  if (r.end == runtime::TurnEnd::Cancelled) {
    return fail("generate", "cancelled");
  }
  std::string ended(name(r.end));
  if (r.stop) {
    ended += std::format(" (token {})", static_cast<std::uint32_t>(*r.stop));
  }
  put(stdout,
      std::format("end: {}, {} tokens; prompt {} tokens, {} reused from the "
                  "cache\n",
                  ended, r.emitted, r.prompt, r.reused));
  return kOk;
}

int tool(std::span<char*> args) {
  std::string refused;
  const std::optional<Options> options = parse(args, refused);
  if (!options) {
    put(stderr, std::format("frizzle: {}\n\n{}", refused, kUsage));
    return kBadArguments;
  }
  if (!options->prompt_is_file) return run(*options, options->prompt);
  const std::optional<std::string> text = read_text(options->prompt);
  if (!text) {
    put(stderr, std::format("frizzle: cannot read {}\n", options->prompt));
    return kBadArguments;
  }
  return run(*options, *text);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return tool(std::span<char*>(argv, static_cast<std::size_t>(argc)));
  } catch (const std::exception& e) {
    put(stderr, "frizzle: failed: ");
    put(stderr, e.what());
    put(stderr, "\n");
  } catch (...) {
    put(stderr, "frizzle: failed: an unknown exception\n");
  }
  return kFailed;
}
