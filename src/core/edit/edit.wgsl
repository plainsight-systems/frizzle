// Axis Q: changes with a kind of edit, or what one does.
//
// The edit kernel (edit/edit_kernel.h): a layer's pinned neurons, applied to
// the feed-forward activation's rows whose positions their ranges hold,
// between gate·up and down.
//
// Contract
//   - Reads the layer's [first, count] and its edits from the table
//     (edit/edit.h): edit i at words first + 5i .. first + 5i + 4, as
//     [neuron, kind, value's bits, first position, end position], kind Set
//     0, Scale 1, Zero 2, the end exclusive.
//   - For each edit and each row r of the step whose position,
//     step.position + r, is in [first position, end position), rewrites
//     activation[r × width + neuron]: Set and Zero store the value (Zero's
//     is 0), Scale multiplies by it once. A
//     Scale of exactly 1 keeps the neuron's bits: WGSL lets an
//     implementation flush a subnormal product to zero, so ×1 is not
//     multiplied. Elsewhere the result is edit::apply's, as WGSL's f32
//     multiply is correctly rounded on normal values.
//   - Writes nothing else. A layer with no edits writes nothing.
//   - Preconditions: the table holds a checked set (edit::check), so every
//     neuron is below `width` and no two edits of one neuron hold at one
//     position.
//
// Algorithm, one workgroup of workgroup_size (64) a row
//   1. Workgroup r takes row r, at position step.position + r; read the
//      layer's [first, count].
//   2. Invocation j takes edits j, j + 64, ... below count, and rewrites its
//      neuron in row r when the edit's range holds the row's position.
//
// Design and cost
//   - A launch over every token (kernels/interface.h): the program runs it
//     in every step, and sets last_token false, which this kernel declares
//     and does not need: it reads each row's position from the step.
//   - One dispatch a layer a step, a workgroup a row; with no edits each
//     reads two words. No two invocations write one value — one row a
//     workgroup, one neuron an edit at a position — so no barrier is
//     needed.

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
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_index) index: u32) {
    let first = table[2u * launch.layer];
    let count = table[2u * launch.layer + 1u];
    let position = step.position + group.x;
    let row = group.x * launch.width;
    for (var i = index; i < count; i += workgroup_size) {
        let at = first + 5u * i;
        if (position < table[at + 3u] || position >= table[at + 4u]) {
            continue;
        }
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
