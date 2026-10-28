#!/bin/sh
# Writes tests/fixtures/reference/gemma-3-1b-it-q4_0.inc: llama.cpp's logits
# for a pinned prompt on the pinned model, on its CPU backend and on Metal,
# for the oracle (tests/integration/reference-logits/). 600 tokens, past
# Gemma 3's 512-key window. Run on a Mac with Metal.
#
# The committed fixture is Charlotte e003048's, made by its
# tools/make_reference_logits.sh with the same llama.cpp commit, text and
# file; this script makes it again. Regenerate only to change the reference
# deliberately: a new llama.cpp commit or a new model file.
#
# llama.cpp is fetched at the commit below into .cache/llama.cpp; the model
# is the one scripts/fetch_model.py pins.
set -eu
cd "$(dirname "$0")/../.."

LLAMA_COMMIT=988190680d5a89fce97de3c20df2c2813731fd61
MODEL=.cache/models/gemma-3-1b-it-q4_0.gguf
MODEL_SHA256=27ee88e03be02e9ba73def9a819d570d8ad73716e50769e87f374ae394b0276e
TEXT="The lighthouse keeper climbed the spiral stairs each evening at dusk, counting the steps as he went. \
There were one hundred and twelve of them, worn smooth in the middle by a century of boots. At the top he \
trimmed the wick, polished the great lens until it shone, and watched the first ships of the night turn \
toward the harbour, their lanterns small and steady against the dark water."
# The paragraph eight times over, each time a chapter, to run past the window.
LONG_TEXT=""
for chapter in one two three four five six seven eight; do
    LONG_TEXT="${LONG_TEXT}Chapter ${chapter}. ${TEXT} "
done

python3 scripts/fetch_model.py
SRC=.cache/llama.cpp
if [ ! -d "$SRC/.git" ]; then
    git init -q "$SRC"
    git -C "$SRC" remote add origin https://github.com/ggml-org/llama.cpp
fi
git -C "$SRC" fetch -q --depth 1 origin "$LLAMA_COMMIT"
git -C "$SRC" checkout -q FETCH_HEAD

cmake -S tools/reference_logits -B .cache/reference_logits -DLLAMA_SOURCE="$PWD/$SRC" -DCMAKE_BUILD_TYPE=Release
cmake --build .cache/reference_logits --target reference_logits
.cache/reference_logits/reference_logits "$MODEL" "$LLAMA_COMMIT" "$MODEL_SHA256" 600 "$LONG_TEXT" \
    > tests/fixtures/reference/gemma-3-1b-it-q4_0.inc
