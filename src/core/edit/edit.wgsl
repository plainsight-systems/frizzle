// Axis Q: changes with a kind of edit, or what one does.
//
// The edit kernel (edit/edit_kernel.h): a layer's pinned neurons, applied to
// the feed-forward activation's last row, between gate·up and down.
//
// Contract
//   - Reads the layer's [first, count] and its edits from the table
//     (edit/edit.h): edit i at words first + 3i .. first + 3i + 2, as
//     [neuron, kind, value's bits], kind Set 0, Scale 1, Zero 2.
//   - Rewrites activation[(tokens − 1) × width + neuron] for each: Set and
//     Zero store the value (Zero's is 0), Scale multiplies by it once. A
//     Scale of exactly 1 keeps the neuron's bits: WGSL lets an
//     implementation flush a subnormal product to zero, so ×1 is not
//     multiplied. Elsewhere the result is edit::apply's, as WGSL's f32
//     multiply is correctly rounded on normal values.
//   - Writes nothing else. A layer with no edits writes nothing.
//   - Preconditions: the table holds a checked set (edit::check), so every
//     neuron is below `width` and no two edits name one neuron.
//
// Algorithm, one workgroup of workgroup_size (64)
//   1. Read the layer's [first, count].
//   2. Invocation j takes edits j, j + 64, ... below count, and rewrites its
//      neuron in the last row.
//
// Design and cost
//   - A launch over the last token (kernels/interface.h): the program runs
//     it only in steps that score, and sets last_token, which this kernel
//     declares and does not need: it works on row tokens − 1 always, as the
//     draw does.
//   - One dispatch a layer a scoring step; with no edits it reads two words.
//     No two invocations write one value, so no barrier is needed.

struct EditLaunch {
    layer: u32,   // the layer this launch edits
    width: u32,   // the activation's row: the layer's feed-forward width
}

override workgroup_size: u32 = 64;
override last_token: bool = false;     // set by the program; see above

@group(0) @binding(1) var<uniform> launch: EditLaunch;
@group(0) @binding(2) var<storage, read> table: array<u32>;
@group(0) @binding(3) var<storage, read_write> activation: array<f32>;

const kScale: u32 = 1u;   // edit/edit.h's EditKind::Scale

@compute @workgroup_size(workgroup_size)
fn main(@builtin(local_invocation_index) index: u32) {
    let first = table[2u * launch.layer];
    let count = table[2u * launch.layer + 1u];
    let row = (step.tokens - 1u) * launch.width;
    for (var i = index; i < count; i += workgroup_size) {
        let at = first + 3u * i;
        let neuron = table[at];
        let kind = table[at + 1u];
        let value = bitcast<f32>(table[at + 2u]);
        let old = activation[row + neuron];
        var edited = value;  // Set, and Zero, whose value is 0
        if (kind == kScale) {
            edited = select(old * value, old, value == 1.0);
        }
        activation[row + neuron] = edited;
    }
}
