// Axis H (shape): ranking a layer's neurons by strength, for the
// inspector's "Strongest in layer" list. Plain TypeScript.
//
// Contract
//   - strongest(values, count): the indices of the `count` neurons with the
//     largest magnitude, largest first; equal magnitudes in neuron order.
//     Infinities rank above every finite value; NaN is never ranked. Fewer
//     than `count` when fewer values qualify. Throws a RangeError for a
//     count that is not a whole number of at least 0.
//
// Design and cost
//   - Magnitude, not signed value: a strongly negative neuron is as strong
//     as a strongly positive one, and the list shows each with its sign.
//   - A sort of the qualifying indices: 6,912 a layer, when the shown layer
//     or captures change, never per frame.

export function strongest(values: Float32Array, count: number): number[] {
  if (!Number.isInteger(count) || count < 0) {
    throw new RangeError(`${count} is not a count of neurons`);
  }
  const indices: number[] = [];
  values.forEach((v, i) => {
    if (!Number.isNaN(v)) indices.push(i);
  });
  indices.sort(
    (a, b) => Math.abs(values[b] ?? 0) - Math.abs(values[a] ?? 0) || a - b,
  );
  return indices.slice(0, count);
}
