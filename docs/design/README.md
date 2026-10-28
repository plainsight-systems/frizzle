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

They are static pages: serve the parent `tech-demos` folder over HTTP (for
example `python3 -m http.server`) and open
`/frizzle/docs/design/concepts/…`. The theme pages load IBM Plex fonts from a
sibling Dolittle checkout (`../../../../dolittle/web/static/fonts/`) until
Frizzle has its own copies, and the 3D concepts load three.js r128 from a CDN.
Neither is a pinned dependency of Frizzle. Without the sibling checkout the
pages fall back to system fonts.
