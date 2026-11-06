// Axis E: changes with a new or optimized kernel.
//
// Contract, algorithm, cost and corpus guidance: norm.h. This file defines
// what that header states.
//
// RMSNorm with the residual add before it (kernels/norm/norm.h). One
// workgroup per row: the row is read once into registers, reduced in a fixed
// tree in workgroup memory, and written normalized.
//
// For a row x of width n, gain g and the file's epsilon:
//
//   rms(x)   = sqrt((x₁² + x₂² + … + xₙ²) / n + epsilon)
//   normed   = (x / rms(x)) × g
//
// computed as x × r with r = inverseSqrt(mean of squares + epsilon), then × g:
// normalize first, then the gain. Where the launch adds, the block's output y
// is added into x first, x written back as the new hidden row; with Gemma 3's
// post-norm, y is itself normalized, with its own gain, and multiplied by
// post_scale before the add: 1 for Gemma 3, a multiply kept for the bits
// (norm.h).
//
// The capture (norm.h, Capture): in the step that ends the prompt (logits 1,
// fed 0), the workgroup of the step's last row also writes x, once the add
// is done and before it is normalized, into the store's
// residual slot (`capture_residual`), and y as it was added into the added
// slot (`capture_added`). Nothing else changes: the same x and normed are
// written either way.
//
// inverseSqrt, dot, select and workgroupBarrier are WGSL built-ins; reduce is
// below.

struct Norm {
    epsilon: f32,
    // What the post-normed y is multiplied by before the add: 1, as Gemma 3's
    // graph passes it. The multiply changes no value; without it the bits
    // change (norm.h).
    post_scale: f32,
    // The capture's slots in the store, first words (norm.h, Capture).
    residual_slot: u32,
    added_slot: u32,
}

// The program sets workgroup_size from the launcher (norm.cpp, 256); the
// default only lets a tool that reads this file alone size what uses it.
override workgroup_size: u32 = 256;
override last_token: bool = false;
override add: bool = false;
override post_norm: bool = false;
override capture_residual: bool = false;
override capture_added: bool = false;
// The row's width, set by the program from the launcher; with no default, a
// pipeline that is not given it fails to build.
override width: u32;

@group(0) @binding(1) var<uniform> norm: Norm;
@group(0) @binding(2) var<storage, read> gain: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> post_gain: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> output: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> hidden: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> normed: array<vec4<f32>>;
// The capture store, by words: a slot need not start on a vec4.
@group(0) @binding(7) var<storage, read_write> store: array<u32>;

// Rows up to 4,096 wide: at most 4 vec4s an invocation of 256.
const kMaxVec4s = 4u;

var<workgroup> partial: array<f32, workgroup_size>;

// The workgroup's sum of `value`, in a fixed order: halving the workgroup at
// each level. Every invocation returns the same total. A caller that reduces
// again must first barrier, so every invocation has read partial[0] before
// it is written. A tree in workgroup memory, not subgroupAdd: subgroups are
// optional in WebGPU, and for cross-browser compatibility the harness
// requires no optional feature (norm.h).
fn reduce(index: u32, value: f32) -> f32 {
    partial[index] = value;
    workgroupBarrier();
    for (var stride = workgroup_size / 2u; stride > 0u; stride = stride / 2u) {
        if (index < stride) {
            partial[index] = partial[index] + partial[index + stride];
        }
        workgroupBarrier();
    }
    return partial[0];
}

// Writes vec4 i of a row into the store slot at `slot`, word for word.
fn write_captured(slot: u32, i: u32, v: vec4<f32>) {
    let at = slot + 4u * i;
    store[at] = bitcast<u32>(v.x);
    store[at + 1u] = bitcast<u32>(v.y);
    store[at + 2u] = bitcast<u32>(v.z);
    store[at + 3u] = bitcast<u32>(v.w);
}

@compute @workgroup_size(workgroup_size)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) index: u32) {
    // The row: this workgroup's, or the step's last for the final norm.
    let row = select(group.x, step.tokens - 1u, last_token);
    if (row >= step.tokens) {
        return;
    }
    let vec4s = width / 4u;
    let base = row * vec4s;
    let n = f32(width);
    // The vec4s each invocation takes, a constant once the pipeline is built:
    // 2 for Gemma 3 1B's 288. Every loop below runs that many times, not
    // kMaxVec4s.
    let per_invocation = (vec4s + workgroup_size - 1u) / workgroup_size;
    // The capture's row: the step's last, in the step that ends the prompt.
    let captures = step.logits == 1u && step.fed == 0u &&
        row == step.tokens - 1u;

    // Step 1: read the row into registers, vec4s index, index + 256, … — read
    // once, used by every step below.
    var x: array<vec4<f32>, kMaxVec4s>;
    for (var k = 0u; k < per_invocation; k++) {
        let i = index + k * workgroup_size;
        if (i < vec4s) {
            x[k] = hidden[base + i];
        }
    }

    // Step 2, where the launch adds: x = x + y, the block's output, and x
    // written back as the new hidden row.
    if (add) {
        var y: array<vec4<f32>, kMaxVec4s>;
        for (var k = 0u; k < per_invocation; k++) {
            let i = index + k * workgroup_size;
            if (i < vec4s) {
                y[k] = output[base + i];
            }
        }
        // Gemma 3's post-norm, before the add: y = ((y × r_y) × post_gain) ×
        // post_scale, with r_y = inverseSqrt(mean of y² + epsilon) — the same
        // steps as below, on y — and post_scale 1, kept for the bits.
        if (post_norm) {
            var squares = 0.0;
            for (var k = 0u; k < per_invocation; k++) {
                if (index + k * workgroup_size < vec4s) {
                    squares = squares + dot(y[k], y[k]);
                }
            }
            let r = inverseSqrt(reduce(index, squares) / n + norm.epsilon);
            workgroupBarrier();   // partial[0] read before Step 3 reduces again
            for (var k = 0u; k < per_invocation; k++) {
                let i = index + k * workgroup_size;
                if (i < vec4s) {
                    y[k] = ((y[k] * r) * post_gain[i]) * norm.post_scale;
                }
            }
        }
        // y as it is added, below.
        if (capture_added && captures) {
            for (var k = 0u; k < per_invocation; k++) {
                let i = index + k * workgroup_size;
                if (i < vec4s) {
                    write_captured(norm.added_slot, i, y[k]);
                }
            }
        }
        for (var k = 0u; k < per_invocation; k++) {
            let i = index + k * workgroup_size;
            if (i < vec4s) {
                x[k] = x[k] + y[k];
                hidden[base + i] = x[k];
            }
        }
    }
    // X as the layer's later steps read it: after the add, or as read where
    // the launch does not add.
    if (capture_residual && captures) {
        for (var k = 0u; k < per_invocation; k++) {
            let i = index + k * workgroup_size;
            if (i < vec4s) {
                write_captured(norm.residual_slot, i, x[k]);
            }
        }
    }
    // Step 3: the sum of squares. Each invocation sums its own vec4s' squares,
    // then reduce sums the 256 partial sums across the workgroup.
    var squares = 0.0;
    for (var k = 0u; k < per_invocation; k++) {
        if (index + k * workgroup_size < vec4s) {
            squares = squares + dot(x[k], x[k]);
        }
    }
    // Step 4: r = 1 / sqrt(mean of squares + epsilon). Every invocation has
    // the same total, so every one computes the same r.
    let r = inverseSqrt(reduce(index, squares) / n + norm.epsilon);
    // Step 5: normed = (x × r) × g, written once.
    for (var k = 0u; k < per_invocation; k++) {
        let i = index + k * workgroup_size;
        if (i < vec4s) {
            normed[base + i] = (x[k] * r) * gain[i];
        }
    }
}
