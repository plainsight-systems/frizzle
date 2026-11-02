// Axis O: changes with a new captured value, or what is sent.
//
// The copy kernel (capture/capture.h): one row of a source buffer at the
// last prompt position, copied into a slot of the store. The graph uses it
// for a layer's neurons, after the layer's edits and before its down
// projection.
//
// Contract
//   - In the step that ends the prompt (step.fed 0; the program runs this
//     launch only in steps that ask for logits), copies the
//     source's last row, `width` words, to store[first ..], word for word.
//   - In a fed step, a decode step, writes nothing.
//   - Writes nothing outside the slot.
//   - Preconditions: first + width is within the store; the source's rows
//     are `width` words.
//
// Algorithm, one workgroup of workgroup_size (256)
//   1. If step.fed is 1, return.
//   2. Invocation j copies words j, j + 256, ... below width, from row
//      tokens − 1.
//
// Design and cost
//   - Both buffers are bound as array<u32>: the copy moves bits, never
//     floats.
//   - A launch over the last token (kernels/interface.h): it declares
//     last_token, which it does not need, as the edit kernel does: it works
//     on row tokens − 1 always.
//   - One dispatch a layer a scoring step; in a decode step it reads one
//     word of the step and stops. No two invocations write one word, so no
//     barrier is needed.

struct CaptureLaunch {
    first: u32,   // the layer's slot: its first word in the store
    width: u32,   // the activation's row, and the slot's length
}

override workgroup_size: u32 = 256;
override last_token: bool = false;     // set by the program; see above

@group(0) @binding(1) var<uniform> launch: CaptureLaunch;
@group(0) @binding(2) var<storage, read> activation: array<u32>;
@group(0) @binding(3) var<storage, read_write> store: array<u32>;

@compute @workgroup_size(workgroup_size)
fn main(@builtin(local_invocation_index) index: u32) {
    if (step.fed == 1u) {
        return;
    }
    let row = (step.tokens - 1u) * launch.width;
    for (var i = index; i < launch.width; i += workgroup_size) {
        store[launch.first + i] = activation[row + i];
    }
}
