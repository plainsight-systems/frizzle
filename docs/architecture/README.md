# Architecture

Frizzle as it is: diagrams first, then tables, then as few words as they
need. Every arrow says what it means.

| Document | Covers in full |
| --- | --- |
| [Logical overview](logical-overview.md) | The parts and what each owns; the model; the operations; loading; the cache; a layer's order of work with its capture and edit points; edits; captures, when written and read and how they reach the page; the loop; what must stay true; the pinned versions |
| [Interface](interface.md) | The page: its layout, a session's states and controls, the deck and how it draws, the look |
| [Change axes](change-axes.md) | The independent reasons a file changes, how they depend on each other, and the boundary tests |
| [File mapping](file-mapping.md) | How the axes become directories and files, and the import rules `check.py boundaries` enforces |
