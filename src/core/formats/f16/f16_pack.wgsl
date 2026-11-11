// Axis B: changes with the F16 format: its layout on the device, its unpack
// or its pack.
//
// Contract, algorithm, cost and corpus guidance: f16.h. This file defines
// what that header states: F16's pack and its inverse (formats/format.h),
// reading and writing no binding. pack: four values to the two words that
// store them, the lower half of each word first, as unpack reads them. Each
// value is first saturated to F16's largest finite, ±65,504: WGSL leaves
// converting a value outside F16's range indeterminate, so the saturation
// makes the result the nearest finite value rather than an infinity. A
// deliberate narrowing (ES.46), one rounding a value, to nearest on the
// target as format.h assumes. clamp, pack2x16float and unpack2x16float are
// WGSL built-ins.
const kPackF16Max = 65504.0;

fn pack(values: vec4<f32>) -> vec2<u32> {
    let v = clamp(values, vec4<f32>(-kPackF16Max), vec4<f32>(kPackF16Max));
    return vec2<u32>(pack2x16float(v.xy), pack2x16float(v.zw));
}

// The four values two words store, exact: every f16 is an f32. The KV
// cache's reader, attention, decodes with it.
fn unpack4(words: vec2<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(words.x), unpack2x16float(words.y));
}
