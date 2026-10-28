# Test inventory

This list is the whole test suite. Each entry locks the contract it names.
Follow the [test rules](../docs/process/workflow.md#tests): refactors never
change tests, and adding or removing an entry is a recorded decision.

An entry is "built" when the test exists and passes.

**Running them.** The contract suites are one executable, `frizzle_tests`,
which `python3 scripts/check.py test` (and `all`) builds and runs. The oracle
is `frizzle_oracle`, run by hand when the inference path changes:

```sh
cmake --build build/native-debug --target frizzle_oracle
build/native-debug/tests/frizzle_oracle
```

The contract suites read only committed inputs. The oracle reads the pinned
model file (`python3 scripts/fetch_model.py`); a missing file fails it, never
skips it.

## Integration scenarios (four of five)

| Scenario | Contract it guards | Status |
| --- | --- | --- |
| Logits vs llama.cpp | The native engine, on Dawn, computes Gemma 3 1B's logits for fixed token sequences within a frozen tolerance of llama.cpp's for the same file. Run when the inference path changes, not with the routine suite. `tests/integration/reference-logits/`, against the frozen reference in `tests/fixtures/reference/`. | Built |
| Replay and edits | With no edits, or Scale × 1, the edited reply is the original reply bit for bit, including after other edits were made and removed. Several edits across layers apply together. Edits change the last prompt position and the generated tokens and leave earlier positions' cache entries untouched. Edits are held at every generated token. | Planned |
| Browser, end to end | In a real browser: download and verify the pinned file, generate a reply, pin neurons in two layers, and get the edited reply beside the original. | Planned |
| Site build | The Hugo site builds and publishes with the app inside it. | Planned |

One slot is unassigned.

## Contract suites

| Suite | Contract it guards | Status |
| --- | --- | --- |
| Tokenizer | Prompts tokenize to the reference token IDs for Gemma 3's vocabulary, and token IDs decode back to the same text. `tests/contracts/tokenizer/`, over the model's vocabulary alone in `tests/fixtures/tokenizer/` (`scripts/make_tokenizer_fixture.py`). | Built |
| Model file | A malformed or hostile GGUF is refused by name, never read past its bounds or misinterpreted. `tests/contracts/gguf/`, over the files `scripts/make_gguf_fixtures.py` writes. | Built |
| Kernels | Each kernel family (gather, norm, matmul, rope, attention, top-k), and the new capture and edit outputs, matches a CPU reference on small inputs. From slice 2, when the capture and edit work changes kernels; until then the oracle covers the ported kernels end to end. | Planned |
| Edit specs | An invalid edit (unknown layer, neuron out of range, non-finite value) is refused by name; Scale × 1 is an exact no-op. | Planned |
