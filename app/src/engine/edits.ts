// Axis Q: the edit kinds, as the page holds and sends them. Changes with a
// kind of edit, together with src/core/edit/ (edit.h, edit.wgsl).
//
// An edit pins one neuron of one layer's feed-forward activation: set it to
// a value, scale it by a factor, or zero it, over a range of the run's
// steps (RangedEdit). A set is any number of them, in any layers, sent
// whole with every run (logical-overview.md, invariant 4). Whether a set is valid for the model, its layers, neurons and
// values, is the engine's to say (edit::check), refused by name; nothing
// here checks it, so there is one judge.

export type EditKind = "set" | "scale" | "zero";

export interface Edit {
  readonly layer: number;
  readonly neuron: number;
  readonly kind: EditKind;
  readonly value: number; // the set value or the scale factor; 0 for zero
}

// An edit as a run sends it: a pin over the steps [from, until) of the
// run, until null for every later step (src/core/edit/edit.h). Step s is
// the position that draws reply token s: step 0 is the rendered text's
// last token. The page pins neurons (Edit) and makes these for each run
// (session/branch.ts): an accepted reply's pins up to the branch, the pins
// held now from it.
export interface RangedEdit extends Edit {
  readonly from: number;
  readonly until: number | null;
}
