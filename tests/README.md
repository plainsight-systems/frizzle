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
| Replay and edits | Through the engine, on the pinned model, one message and one seed. (1) Scale × 1 edits in several layers give the reply without edits, byte for byte, with the same token count and ending. (2) Without edits, then with a strong edit, then without again: the third run equals the first, and reuses all the prompt's cache but its last token, so the edited run left the earlier positions' entries as they were. (3) One edit set across layers, given in two orders, gives one reply, different from the reply without edits. (4) After an edited run, a conversation that contains its reply reuses only the prompt before the edits (the rewind), and its reply equals a fresh engine's. `tests/integration/replay/`, `frizzle_replay`, run when edits, the graph or the turn loop change. | Built |
| Browser, end to end | In a real browser: download and verify the pinned file, generate a reply, pin neurons in two layers, and get the edited reply beside the original. | Planned |
| Site build | The Hugo site builds and publishes with the app inside it. | Planned |

One slot is unassigned.

## Contract suites

| Suite | Contract it guards | Status |
| --- | --- | --- |
| Tokenizer | Prompts tokenize to the reference token IDs for Gemma 3's vocabulary, and token IDs decode back to the same text. `tests/contracts/tokenizer/`, over the model's vocabulary alone in `tests/fixtures/tokenizer/` (`scripts/make_tokenizer_fixture.py`). | Built |
| Model file | A malformed or hostile GGUF is refused by name, never read past its bounds or misinterpreted. `tests/contracts/gguf/`, over the files `scripts/make_gguf_fixtures.py` writes. | Built |
| Kernels | A kernel alone on the GPU, with no model, against a CPU reference. The edit kernel, against `edit::apply` bit for bit: the listed neurons of the last row become `apply`'s result, in a step that scores a token (a prefill step ending the prompt, or a one-token decode step, so edits hold at every generated token) and in no other; a layer's launch applies its own layer's edits and no other's; more than 64 edits in a layer all apply; Scale × 1 keeps a subnormal's bits; every other value keeps its bits. `tests/contracts/kernels/`, `frizzle_kernels`, in the routine run; it needs a GPU and fails without one. The ported kernel families (gather, norm, matmul, rope, attention, top-k) get no tests of their own: the oracle covers them end to end, by decision. Captures join in slice 4. | Built |
| Edit specs | An invalid edit (unknown kind or layer, neuron out of range, non-finite value, a Zero with a value, a repeat, too many) is refused by name; Scale × 1 is an exact no-op; the table's layout. `tests/contracts/edits/`. | Built |
