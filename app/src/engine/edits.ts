// Axis Q: the edit kinds, as the page holds and sends them. Changes with a
// kind of edit, together with src/core/edit/ (edit.h, edit.wgsl).
//
// An edit pins one neuron of one layer's feed-forward activation: set it to
// a value, scale it by a factor, or zero it. A set is any number of them,
// in any layers, sent whole with every run (logical-overview.md, invariant
// 4). Whether a set is valid for the model, its layers, neurons and
// values, is the engine's to say (edit::check), refused by name; nothing
// here checks it, so there is one judge.

export type EditKind = "set" | "scale" | "zero";

export interface Edit {
  readonly layer: number;
  readonly neuron: number;
  readonly kind: EditKind;
  readonly value: number; // the set value or the scale factor; 0 for zero
}
