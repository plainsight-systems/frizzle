// Axis B: changes with the Q8_0 format: its layout on the device or its
// unpack.
//
// Contract, algorithm, cost and corpus guidance: q8_0.h. This file defines
// what that header states: Q8_0's unpack (formats/format.h). A block is 32
// weights: an fp16 scale d and 32 signed 8-bit codes qs; weight j is
// qs[j] * d. On the device a piece holds every block's 32 code bytes, then
// every block's scale (device_layout.h): eight words of codes a block, then
// two scales a word. `weights` is the kernel's binding; bitcast and
// unpack2x16float are WGSL built-ins.
fn unpack(base: u32, blocks_in_piece: u32, group: u32) -> array<vec4<f32>, 8> {
    let d = unpack2x16float(weights[base + blocks_in_piece * 8u + group / 2u])[group % 2u];
    var out: array<vec4<f32>, 8>;
    for (var v = 0u; v < 8u; v++) {
        // Codes 4v .. 4v + 3, low byte first, each sign-extended.
        let word = weights[base + group * 8u + v];
        let q = vec4<i32>(bitcast<i32>(word << 24u), bitcast<i32>(word << 16u), bitcast<i32>(word << 8u),
                          bitcast<i32>(word)) >> vec4<u32>(24u);
        out[v] = vec4<f32>(q) * d;
    }
    return out;
}
