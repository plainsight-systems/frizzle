// Toolchain self-check kernel.
//
// This exists to prove the build pipeline end to end: WGSL is embedded into
// the binary at build time, reaches the GPU through webgpu.h, and its result
// is read back and asserted. It is not a model kernel and is not on any hot
// path.

@group(0) @binding(0) var<storage, read>       lhs    : array<f32>;
@group(0) @binding(1) var<storage, read>       rhs    : array<f32>;
@group(0) @binding(2) var<storage, read_write> result : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    // Guard the tail: dispatch is rounded up to whole workgroups, so the
    // final workgroup runs invocations past the end of the data.
    if (i >= arrayLength(&result)) {
        return;
    }
    result[i] = lhs[i] + rhs[i];
}
