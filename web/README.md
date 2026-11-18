# web

The project site, in Hugo, in the Plainsight family shell (the concepts in
`../docs/design/concepts/themes.html` and `transition.html`). It hosts the
built app from `../app/` at `app/`.

```sh
python3 scripts/site.py build   # from the repository root
python3 scripts/site.py serve
```

[scripts/README.md](../scripts/README.md#the-site) says what each does and
what they need.

- `hugo.toml`: the address, the site's description, and the mounts: the
  app's fonts (`../app/src/theme/fonts/`), the README banner
  (`../docs/assets/`) and the built app (`../app/dist/`), read where they
  live, never copied.
- `content/`: Home (`_index.md`), the tutorial and About.
- `assets/tutorial/`: the tutorial's screenshots, taken from the app on the
  pinned model; retake them when the Ride or Deck changes how they look.
- `layouts/`: the shell (`_default/baseof.html`), Home, the reading page,
  the 404 page, the share image made from the banner, and the `shot`
  shortcode for a screenshot with its caption.
- `assets/site.css`: the site's look.

`.github/workflows/site.yml` builds and checks the site on every pull request
and publishes it to GitHub Pages from `main`, at the address in `hugo.toml`.

Everything under `content/`, `layouts/` and `assets/` is public. The site
has no client-side script of its own, no analytics and no cookies.
