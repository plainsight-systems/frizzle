# Test inventory

This list is the whole test suite. Each entry locks the contract it names.
Follow the [test rules](../docs/process/workflow.md#tests): refactors never
change tests, and adding or removing an entry is a recorded decision.

An entry is "built" when the test exists and passes.

**Running them.** The contract suites are one executable, `frizzle_tests`,
which `python3 scripts/check.py test` (and `all`) builds and runs, with the
app's TypeScript suites and type check; that check covers the browser
scenario's driver too, so it needs `npm ci` in `app/` and in
`tests/integration/browser/`. The oracle
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
| Browser, end to end | In a real browser with WebGPU, the app as it ships: fetch and verify the pinned file, generate a reply, pin neurons in three layers, one of each kind, and get the edited reply, from the same seed, beside the original, different from it and ended by the model; then read each pinned neuron back from the page: in the edited run it holds exactly what its pin sent (a set's value, zero, or the original's value scaled, in f32), so every edit crosses the page, the worker and the module as sent, and every run's captures come back from the engine. Values at unpinned neurons, and the original's own values, have no independent check here (the scenario's header lists what still passes). `tests/integration/browser/`, its own npm project: puppeteer-core driving Chrome for Testing at its pin, headless, over the app's production build served as static files. The pinned URL is answered from the local copy (`.cache/models/`, `scripts/fetch_model.py`) and every other host is blocked, so the app's own fetch, OPFS and hash check run with no network, into a fresh profile each run. Runs before every push as the pre-push hook (`scripts/pre_push.py`), which first rebuilds the browser module; not in CI. By hand: `npm ci` and `npm run install-browser` once, then `npm run scenario`, in that directory. | Built |
| Site build | The Hugo site builds and publishes with the app inside it. | Planned |

One slot is unassigned.

## Contract suites

| Suite | Contract it guards | Status |
| --- | --- | --- |
| Tokenizer | Prompts tokenize to the reference token IDs for Gemma 3's vocabulary, and token IDs decode back to the same text. `tests/contracts/tokenizer/`, over the model's vocabulary alone in `tests/fixtures/tokenizer/` (`scripts/make_tokenizer_fixture.py`). | Built |
| Model file | A malformed or hostile GGUF is refused by name, never read past its bounds or misinterpreted. `tests/contracts/gguf/`, over the files `scripts/make_gguf_fixtures.py` writes. | Built |
| Kernels | A kernel alone on the GPU, with no model, against a CPU reference. The edit kernel, against `edit::apply` bit for bit: the listed neurons of the last row become `apply`'s result, in a step that scores a token (a prefill step ending the prompt, or a one-token decode step, so edits hold at every generated token) and in no other; a layer's launch applies its own layer's edits and no other's; more than 64 edits in a layer all apply; Scale × 1 keeps a subnormal's bits; every other value keeps its bits. `tests/contracts/kernels/`, `frizzle_kernels`, in the routine run; it needs a GPU and fails without one. The capture, with its reader, against the activation it copies, bit for bit: the activation's last row lands in its layer's slot of the store in the step that ends the prompt (a prefill block, or the one-token recompute of the last prompt position) and in no other, never outside its slot; the store reads back as its bits; each layer's slot follows the earlier layers'. The ported kernel families (gather, norm, matmul, rope, attention, top-k) get no tests of their own: the oracle covers them end to end, by decision. | Built |
| Edit specs | An invalid edit (unknown kind or layer, neuron out of range, non-finite value, a Zero with a value, a repeat, too many) is refused by name; Scale × 1 is an exact no-op; the table's layout. `tests/contracts/edits/`. | Built |
| Chat template | Gemma 3's template, from the model file, renders one user message as `<bos><start_of_turn>user\n{message, trimmed}<end_of_turn>\n<start_of_turn>model\n`; a conversation renders each turn, assistant as `model`; a system prompt joins the first user turn; turns that do not alternate are refused by name. `tests/contracts/template/`, TypeScript, against Charlotte's frozen Gemma references (`tests/fixtures/template/`) and the template in the committed tokenizer fixture. | Built |
| Delivery | SHA-256 matches published test vectors, whole and fed in pieces; the pinned file is kept only when it arrives whole, as many bytes as the pin says, with the pin's SHA-256, and a copy already kept is not downloaded again; a stated size that differs, a download that ends early or runs long, or a hash that differs is refused by name and nothing is kept; a download aborted partway keeps nothing and discards what it wrote. `tests/contracts/delivery/`, TypeScript; the store and the server are in-memory stand-ins at the module's own interfaces, the real fetch and OPFS the browser scenario's. | Built |
| Shape | What the deck shows at a cell is the neuron folded there, a heat map of its size, whichever its sign: a slot folds 96 to a row in reading order, and each cell back to its neuron, a short last row ending in empty cells, and a box of cells to its neurons, clipped to the grid; each neuron's pixel is at its cell; v and −v are one colour; 0 is the ramp's first colour, the slot's scale (its 99th-percentile finite magnitude, or the largest when that is 0) its last, and a larger magnitude never further back along the ramp than a smaller one; infinities are full heat and NaN its own colour, and none sets the scale; what is not a grid, a neuron or the slot's width, or a ramp of fewer than two colours, is refused; heatOf is paint's colour, and an overlay colours its cells at the slot's own scale and nothing else. The strongest neurons rank by magnitude, either sign, ties in order, NaN never. Bytes read as whole megabytes, values with their sign and two places, a scale's end with two significant digits. The gamma and the deck's own palette and scene are not locked. `tests/contracts/shape/`, TypeScript, plain arrays. | Built |
| Pins | The pins held for the next run: at most one edit a neuron, in the order first pinned; adding replaces in place and adding what is held changes nothing; removing takes only the named neurons of the named layer; chips group by layer and edit, -0 apart from 0, in the order each began; a set sends its value, a zero 0, a scale the f32 product. `tests/contracts/pins/`, TypeScript, plain arrays. | Built |
| Protocol | Every request settles once, with its result or a named WorkerError, after its progress or text in order; the latest generation wins: a later one settles the earlier as Superseded, cancels it in the worker, and drops what the worker still sends for it. `tests/contracts/protocol/`, TypeScript, over a real MessageChannel with the test as the worker's end. | Built |
