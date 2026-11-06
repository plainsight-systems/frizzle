# src

The C++23 engine. It compiles to WebAssembly for the browser and natively, on
Dawn, for analysis and tests. Its layout is
[the file mapping](../docs/architecture/file-mapping.md). The starting point
is Charlotte's Gemma 3 path, listed in the
[inventory](../docs/research/2026-10-27-charlotte-inventory.md) and ported
mechanically by `scripts/port_charlotte.py`; `ported.json` records each file
as ported, so a file still matching it is known to be unchanged.
