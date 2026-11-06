// Axis E: changes with a new or optimized kernel.
//
// Contract, algorithm, cost and corpus guidance: rope.h. This file defines
// what that header states. It is the first of the one form's two parts: the
// constants, the gains, the queries, the angle tables, the norm's tree and
// the rotation. rope_cache.wgsl, joined after it (rope.cpp), holds the key,
// value and cache bindings and the entry point.
//
// Invocation i of a head holds its vec4s i and i + d / 8; a workgroup is
// workgroup_size / (d / 8) heads of one token, one for Gemma 3 1B.
//
// inverseSqrt, dot, round, cos, sin and workgroupBarrier are WGSL built-ins;
// reduce_head, angles and rotate are below.

struct Rope {
    slots: u32,
    epsilon: f32,
    // Each pair's θ_k / 2π, turns a position, four to a vec4: d / 2 used.
    turns: array<vec4<f32>, 32>,
}

// The program sets workgroup_size from the launcher (rope.cpp, workgroup_for:
// 32 for Gemma 3 1B), and the launcher every other override on every launch,
// so no default takes effect. The defaults are Gemma 3 1B's — d = 256, 4
// query heads, 1 key-value head, a workgroup of 32 — and only let a tool
// that reads this file alone size what uses them.
override workgroup_size: u32 = 32;
override head_dimension: u32 = 256;
override query_heads: u32 = 4;
override key_value_heads: u32 = 1;
// d / 2, the pairs a head has, which size the workgroup's angle tables. Set
// on every launch, as WebKit requires of an override sizing a workgroup
// array.
override pairs_per_head: u32 = 128u;

@group(0) @binding(1) var<uniform> rope: Rope;
@group(0) @binding(2) var<storage, read> query_gain: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> key_gain: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> query: array<vec4<f32>>;
const kTau = 6.283185307179586;

var<workgroup> partial: array<f32, workgroup_size>;
var<workgroup> cosines: array<f32, pairs_per_head>;
var<workgroup> sines: array<f32, pairs_per_head>;

// The sum of `value` over the `lanes` invocations of this invocation's head,
// in a fixed order: halving the head at each level. Every invocation of the
// head returns the same total. One barrier before the tree and one after
// each level; the first also publishes whatever the workgroup wrote before
// the call.
fn reduce_head(local: u32, lane: u32, lanes: u32, value: f32) -> f32 {
    partial[local] = value;
    workgroupBarrier();
    for (var stride = lanes / 2u; stride > 0u; stride = stride / 2u) {
        if (lane < stride) {
            partial[local] = partial[local] + partial[local + stride];
        }
        workgroupBarrier();
    }
    return partial[local - lane];
}

fn turn(k: u32) -> f32 {
    return rope.turns[k / 4u][k % 4u];
}

// The workgroup's token: its row and position, and whether its heads
// rotate. A row is a whole number of workgroups (rope.h), so these are the
// same for every invocation of a workgroup — uniform, as the barriers after
// them require: from the workgroup's index alone.
struct Token {
    row: u32,
    p: u32,
    rotates: bool,
}

fn token_of(group: u32) -> Token {
    let lanes = head_dimension / 8u;
    let rotating_heads = query_heads + key_value_heads;
    let per_row = (rotating_heads + key_value_heads) * lanes;
    let first = group * workgroup_size;
    let row = first / per_row;
    return Token(row, step.position + row, (first % per_row) / lanes < rotating_heads);
}

// An invocation's head and lane, and the first vec4 of its head's row in
// its buffer.
struct Head {
    head: u32,
    lane: u32,
    is_query: bool,
    is_value: bool,
    kv_head: u32,
    base: u32,
}

fn head_of(group: u32, local: u32, row: u32) -> Head {
    let lanes = head_dimension / 8u;
    let quarter = head_dimension / 4u;              // vec4s in a head
    let rotating_heads = query_heads + key_value_heads;
    let per_row = (rotating_heads + key_value_heads) * lanes;
    let index = group * workgroup_size + local;
    var h: Head;
    h.head = (index % per_row) / lanes;
    h.lane = index % lanes;
    h.is_query = h.head < query_heads;
    h.is_value = h.head >= rotating_heads;
    h.kv_head = h.head - query_heads;
    if (h.is_value) {
        h.kv_head = h.head - rotating_heads;
    }
    if (h.is_query) {
        h.base = (row * query_heads + h.head) * quarter;
    } else {
        h.base = (row * key_value_heads + h.kv_head) * quarter;
    }
    return h;
}

// The token's cos and sin for each pair, once, shared out among the
// workgroup; published by the next barrier.
fn angles(p: u32, local: u32) {
    for (var k = local; k < head_dimension / 2u; k += workgroup_size) {
        let t = f32(p) * turn(k);
        let angle = kTau * (t - round(t));      // in [−π, π]
        cosines[k] = cos(angle);
        sines[k] = sin(angle);
    }
}

// Each pair of a query or key head turned by its angle: a and b hold four
// whole pairs (4i + c, 4i + c + d / 2).
fn rotate(a: vec4<f32>, b: vec4<f32>, lane: u32) -> array<vec4<f32>, 2> {
    let k = 4u * lane;
    let c = vec4<f32>(cosines[k], cosines[k + 1u], cosines[k + 2u], cosines[k + 3u]);
    let s = vec4<f32>(sines[k], sines[k + 1u], sines[k + 2u], sines[k + 3u]);
    return array<vec4<f32>, 2>(a * c - b * s, a * s + b * c);
}
