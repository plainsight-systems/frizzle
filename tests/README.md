# Test inventory

This list is the whole test suite. Each entry names the contract it guards.
Follow the [test rules](../docs/process/workflow.md#tests): every test is shown
to fail before it is kept, refactors never change tests, and adding or removing
an entry is a recorded decision.

**Status: planned.** None of these tests exists yet. An entry moves to
"built" only when the test runs and its deliberate breakage has been seen to
fail it.

## Integration scenarios (four of five)

| Scenario | Contract it guards | Status |
| --- | --- | --- |
| Logits vs llama.cpp | The native engine, on Dawn, computes Gemma 3 1B's logits for fixed token sequences within a frozen tolerance of llama.cpp's for the same file. | Planned |
| Replay and edits | With no edits, or Scale × 1, a rerun reproduces the original prediction bit for bit, including after other edits were made and removed. Several edits across layers apply together. Edits starting at layer L change the last position only from L onward and leave earlier positions' cache entries untouched. Edits are held at every generated token. | Planned |
| Browser, end to end | In a real browser: download and verify the pinned file, generate a reply, pin neurons in two layers, and get the edited reply beside the original. | Planned |
| Site build | The Hugo site builds and publishes with the app inside it. | Planned |

One slot is unassigned.

## Contract suites

| Suite | Contract it guards | Status |
| --- | --- | --- |
| Tokenizer | Prompts tokenize to the reference token IDs for Gemma 3's vocabulary, and token IDs decode back to the same text. | Planned |
| Model file | A malformed or hostile GGUF is refused by name, never read past its bounds or misinterpreted. | Planned |
| Kernels | Each kernel family (gather, norm, matmul, rope, attention, top-k), and the new capture and edit outputs, matches a CPU reference on small inputs. | Planned |
| Edit specs | An invalid edit (unknown layer, neuron out of range, non-finite value) is refused by name; Scale × 1 is an exact no-op. | Planned |
