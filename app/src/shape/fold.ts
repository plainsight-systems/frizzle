// Axis H (shape): folding a layer's neurons, one flat run of words, into
// the grid the deck draws. Plain TypeScript, no React, no DOM.
//
// Contract
//   - fold(width, columns = DECK_COLUMNS): the grid for a slot of `width`
//     neurons, `columns` to a row: rows = ceil(width / columns). The width
//     comes from the captures (a slot's width, messages.ts), never from this
//     file. Throws a RangeError for a width or column count that is not a
//     whole number of at least 1.
//   - cellOf(grid, neuron): where a neuron sits, in reading order: column
//     neuron % columns, row floor(neuron / columns); row 0 is the top row.
//     Throws a RangeError for a neuron that is not a whole number in
//     [0, width).
//   - neuronAt(grid, column, row): the neuron at that cell, the inverse of
//     cellOf; null for a cell past the last neuron (the last row may be
//     short) or outside the grid.
//   - neuronsIn(grid, a, b): the neurons in the box with corners a and b,
//     both included, whichever corner is which: every cell from the lesser
//     column to the greater and the lesser row to the greater, the box
//     clipped to the grid first, so a corner may lie outside it (a drag
//     that leaves the card). Ascending; cells past the last neuron are
//     skipped; empty when the box misses the grid.
//
// Design
//   - 96 columns: Gemma 3 1B's 6,912 neurons fold to 96 × 72, the deck's
//     card (docs/architecture/interface.md), a little wider than tall. The count
//     of columns is this file's choice; how many neurons there are is the
//     model's.
//   - The deck paints the grid with paint.ts, picks a neuron from a cell
//     with neuronAt, and a box of them, dragged, with neuronsIn.

export const DECK_COLUMNS = 96;

export interface Grid {
  readonly width: number; // neurons in the slot
  readonly columns: number;
  readonly rows: number;
}

export interface Cell {
  readonly column: number;
  readonly row: number;
}

export function fold(width: number, columns: number = DECK_COLUMNS): Grid {
  if (!isCount(width)) {
    throw new RangeError(`a slot of ${width} neurons is not a grid`);
  }
  if (!isCount(columns)) {
    throw new RangeError(`${columns} columns is not a grid`);
  }
  return { width, columns, rows: Math.ceil(width / columns) };
}

export function cellOf(grid: Grid, neuron: number): Cell {
  if (!Number.isInteger(neuron) || neuron < 0 || neuron >= grid.width) {
    throw new RangeError(
      `neuron ${neuron} is not one of the slot's ${grid.width}`,
    );
  }
  return {
    column: neuron % grid.columns,
    row: Math.floor(neuron / grid.columns),
  };
}

export function neuronAt(
  grid: Grid,
  column: number,
  row: number,
): number | null {
  if (!Number.isInteger(column) || column < 0 || column >= grid.columns) {
    return null;
  }
  if (!Number.isInteger(row) || row < 0) return null;
  const neuron = row * grid.columns + column;
  return neuron < grid.width ? neuron : null;
}

export function neuronsIn(grid: Grid, a: Cell, b: Cell): number[] {
  const left = Math.max(0, Math.floor(Math.min(a.column, b.column)));
  const right = Math.min(
    grid.columns - 1,
    Math.floor(Math.max(a.column, b.column)),
  );
  const top = Math.max(0, Math.floor(Math.min(a.row, b.row)));
  const bottom = Math.min(grid.rows - 1, Math.floor(Math.max(a.row, b.row)));
  const found: number[] = [];
  for (let row = top; row <= bottom; row++) {
    for (let column = left; column <= right; column++) {
      const neuron = row * grid.columns + column;
      if (neuron < grid.width) found.push(neuron);
    }
  }
  return found;
}

// A whole number of at least 1.
function isCount(n: number): boolean {
  return Number.isInteger(n) && n >= 1;
}
