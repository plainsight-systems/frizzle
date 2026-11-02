# Design concepts

`concepts/` holds the interface and theme concepts reviewed on 2026-10-27,
kept as they were made. Their values are made up.

- `matrices.html`: the pass map, every stage drawn at its real width. It
  predates the last-position decision and still draws every token's row.
- `mlp.html`: the first MLP workbench, one layer at a time; the deck replaces
  it. Its token thumbnails and this-token or all-tokens scope predate the
  last-position decision.
- `deck.html`: the layer deck, agreed later the same day as the MLP
  workbench. The MLP neurons at the last position as 26 cards, one per layer,
  that slide away as you scroll; pins stay on their own cards, and the ranked
  list, neuron card and pins follow the front layer. Negative values are a
  cool blue.
- `themes.html`: the three theme directions; the site uses A and the app B.
- `transition.html`: the agreed hand-over from the site to the app.
- `layouts.html` (2026-10-31, not yet chosen): three wireframe directions
  for where the workbench and past tries live, so a reply can be watched
  while working on neurons and earlier tries are kept. It loads the app's
  own Plex fonts (`app/src/theme/fonts/`).
- `explore.html` (2026-10-31, interactive, under review): four views over
  one shared state (the reply, the branch point, the pins, the tries), the
  prompt with Run above them all, tabs between them and links where
  attention already is. Poke: the deck at a word, the front card's
  strongest neurons as a line over it, the inspector as a drawer. Follow:
  the replies tinted by one neuron, and a row per followed neuron, word by
  word. Compare: every try against the reply, with its pins and branch
  word. Map: every layer at the point, its MLP neurons as a strip;
  attention and the residual marked not captured. The values come from a
  hash and the steered replies are canned word swaps; nothing runs a
  model.

They are static pages: serve the parent `tech-demos` folder over HTTP (for
example `python3 -m http.server`) and open
`/frizzle/docs/design/concepts/…`. The theme pages load IBM Plex fonts from a
sibling Dolittle checkout (`../../../../dolittle/web/static/fonts/`) until
Frizzle has its own copies, and the 3D concepts load three.js r128 from a CDN.
Neither is a pinned dependency of Frizzle. Without the sibling checkout the
pages fall back to system fonts.
