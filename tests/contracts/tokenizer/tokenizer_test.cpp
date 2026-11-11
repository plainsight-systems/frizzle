// Axis V: the tokenizer suite (tests/README.md).
//
// Contract locked: text tokenizes to the reference token IDs for Gemma 3's
// vocabulary, and the IDs decode back to the same text, whole and a token at
// a time through the stream the generator uses.
//
// Inputs, both committed, so the suite needs no model:
// tests/fixtures/tokenizer/gemma-3-1b-it-q4_0.gguf, the pinned model's
// tokenizer entries alone (scripts/make_tokenizer_fixture.py), read through
// the capability table as the engine reads them; and
// tests/fixtures/tokenizer/gemma-3-1b-it-q4_0.inc, Charlotte e003048's
// frozen cases, each written only where Hugging Face tokenizers and
// llama.cpp encode it alike (its tools/make_tokenizer_fixtures.py). The
// cases include rendered chat prompts, whose control tokens are written as
// text and encode to their single IDs.
//
// Corpus guidance
//   C++ Core Guidelines
//     P.6    What cannot be checked at compile time should be checkable at
//            run time: a vocabulary that will not load fails, by name.

#include "core/tokenizer/tokenizer.h"

#include <doctest/doctest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "core/capability/capability.h"
#include "core/gguf/byte_source.h"
#include "core/gguf/index.h"
#include "core/gguf/reader.h"
#include "core/tokenizer/unicode.h"
#include "core/tokenizer/vocabulary.h"
#include "support/files.h"

namespace {

using frizzle::tokenizer::TokenId;
using frizzle::tokenizer::Tokenizer;

struct EncodeCase {
  std::string_view name;
  std::string_view text;
  std::vector<std::uint32_t> ids;
};

const auto kCases = std::to_array<EncodeCase>({
#include "fixtures/tokenizer/gemma-3-1b-it-q4_0.inc"
});

constexpr std::string_view kSpace = "\xE2\x96\x81";  // "▁"

std::unique_ptr<Tokenizer> gemma() {
  const std::vector<std::byte> bytes = frizzle::testing::read_file(
      std::string(FRIZZLE_FIXTURE_DIR) + "/tokenizer/gemma-3-1b-it-q4_0.gguf");
  frizzle::gguf::MemoryByteSource source{bytes};
  frizzle::gguf::TensorIndex index;
  REQUIRE(frizzle::gguf::read_index(source, index).error ==
          frizzle::gguf::ReadError::Ok);
  std::string_view name;
  REQUIRE(index.read_string("tokenizer.ggml.model", name) ==
          frizzle::gguf::MetadataError::Ok);
  const frizzle::tokenizer::Algorithm* algorithm =
      frizzle::capability::find_tokenizer(name);
  REQUIRE(algorithm != nullptr);
  std::unique_ptr<Tokenizer> out;
  const frizzle::tokenizer::LoadResult loaded =
      algorithm->load(source, index, out);
  REQUIRE_MESSAGE(loaded.ok(), loaded.subject);
  return out;
}

std::vector<std::uint32_t> encode(Tokenizer& tokenizer, std::string_view text) {
  std::vector<TokenId> tokens;
  REQUIRE(
      tokenizer.encode(text, frizzle::tokenizer::kNoTokenLimit, tokens).error ==
      frizzle::tokenizer::EncodeError::Ok);
  std::vector<std::uint32_t> ids;
  ids.reserve(tokens.size());
  for (const TokenId t : tokens) ids.push_back(static_cast<std::uint32_t>(t));
  return ids;
}

bool well_formed(std::string_view s) {
  for (std::size_t at = 0; at < s.size();) {
    frizzle::tokenizer::Utf8Char c{};
    if (!frizzle::tokenizer::decode_utf8(s, at, c)) return false;
    at += c.length;
  }
  return true;
}

// The text a case's IDs decode to: its own, with a typed ▁ read as a space,
// since the vocabulary spells both alike.
std::string decoded_text(std::string_view text) {
  std::string out;
  for (std::size_t at = 0; at < text.size();) {
    if (text.substr(at, kSpace.size()) == kSpace) {
      out.push_back(' ');
      at += kSpace.size();
    } else {
      out.push_back(text[at++]);
    }
  }
  return out;
}

}  // namespace

TEST_CASE("Gemma 3 encodes every reference text to the reference token IDs") {
  const std::unique_ptr<Tokenizer> tokenizer = gemma();
  for (const EncodeCase& c : kCases) {
    CAPTURE(c.name);
    CHECK(encode(*tokenizer, c.text) == c.ids);
  }
}

TEST_CASE("Gemma 3 decodes every reference's token IDs back to its text") {
  // Whole, and a token at a time through a stream that passes only whole
  // characters: some tokens are single bytes of a character, which the
  // stream holds until the character is complete.
  const std::unique_ptr<Tokenizer> tokenizer = gemma();
  for (const EncodeCase& c : kCases) {
    CAPTURE(c.name);
    std::string whole;
    std::string streamed;
    frizzle::tokenizer::Utf8Stream stream;
    for (const std::uint32_t id : c.ids) {
      std::string bytes;
      tokenizer->decode(static_cast<TokenId>(id), bytes);
      whole += bytes;
      std::string passed;
      stream.push(bytes, passed);
      CHECK(well_formed(passed));
      streamed += passed;
    }
    CHECK(stream.finish(streamed));
    const std::string expected = decoded_text(c.text);
    CHECK(whole == expected);
    CHECK(streamed == expected);
  }
}
