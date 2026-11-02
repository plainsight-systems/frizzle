// Axis O: changes with a new captured value, or what is sent.
//
// The attention weights (capture/weights.h): in the step that ends the
// prompt, each query head's scores at the last position, in the layer's
// weights slot, become exp2(s − m) / l in place, with the head's m and l
// from the stats slot; the slot's keys past the ones read become 0.
//
// Contract
//   - In the step that ends the prompt (step.fed 0; the program runs this
//     launch only in steps that ask for logits), for head h = the
//     workgroup's index: weights[h × keys + i] = exp2(s − m) / l for i below
//     count, 0 from count to keys, where s is the word already there.
//   - In a fed step, a decode step, writes nothing.
//   - Writes nothing outside the weights slot; reads the stats slot.
//   - Preconditions: both slots within the store; keys and query_heads the
//     layer's (capture/capture.h, layout).
//
// Algorithm, one workgroup of workgroup_size (256) a head
//   1. If step.fed is 1, return.
//   2. p = position + tokens − 1; earliest = p + 1 − min(p + 1, keys);
//      count = p − earliest + 1.
//   3. Invocation j writes i = j, j + 256, ... below keys.
//
// Design and cost
//   - The store is bound as array<f32>: the scores and the weights are
//     floats, and every word read or written here is one.
//   - No two invocations write one word, so no barrier is needed.
//   - Constants (binding 1): struct WeightsLaunch { weights: u32, stats: u32,
//     keys: u32, query_heads: u32 }, the first two word offsets in the
//     store.

struct WeightsLaunch {
    weights: u32,       // the layer's weights slot: its first word
    stats: u32,         // the layer's stats slot: its first word
    keys: u32,          // keys a head
    query_heads: u32,
}

override workgroup_size: u32 = 256;
override last_token: bool = false;     // set by the program; a launch over the last token

@group(0) @binding(1) var<uniform> launch: WeightsLaunch;
@group(0) @binding(2) var<storage, read_write> store: array<f32>;

@compute @workgroup_size(workgroup_size)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) index: u32) {
    if (step.fed == 1u) {
        return;
    }
    let h = group.x;
    if (h >= launch.query_heads) {
        return;
    }
    // The keys the step's last position read: count = p − earliest + 1.
    let p = step.position + step.tokens - 1u;
    let count = min(p + 1u, launch.keys);
    let m = store[launch.stats + 2u * h];
    let l = store[launch.stats + 2u * h + 1u];
    let base = launch.weights + h * launch.keys;
    for (var i = index; i < launch.keys; i += workgroup_size) {
        var w = 0.0;
        if (i < count) {
            w = exp2(store[base + i] - m) / l;
        }
        store[base + i] = w;
    }
}
