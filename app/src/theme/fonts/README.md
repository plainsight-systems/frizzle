# Fonts

IBM Plex, kept in the app so no font is fetched from anywhere else
(`../theme.css`). Licensed under the SIL Open Font License 1.1: `LICENSE.txt`,
IBM's, the same text in both packages.

Copied from IBM's npm packages, from `fonts/complete/woff2/`, on 2026-10-29.
Each package archive matched the registry's published integrity before the
files were taken from it. The packages are not dependencies of the app: both
depend on @ibm/telemetry-js, which would enter the lockfile with an install
script that only `.npmrc` keeps from running.

| File                         | Package               | SHA-256                                                            |
| ---------------------------- | --------------------- | ------------------------------------------------------------------ |
| `IBMPlexSerif-Regular.woff2` | @ibm/plex-serif 2.0.0 | `024ebce13cec984b46e350dd85fa7c01105c777e116bfe95f097ad7fa93f39f2` |
| `IBMPlexSerif-Medium.woff2`  | @ibm/plex-serif 2.0.0 | `f66376aeb33a5b32116558e0de53b76b48531f6388e352aeeb35ad82ea05cfaa` |
| `IBMPlexMono-Regular.woff2`  | @ibm/plex-mono 2.5.0  | `ba204497f16b6d334cee9d1e963a831b73e3a56e1d6300a8489d18df7214b350` |
| `IBMPlexMono-Medium.woff2`   | @ibm/plex-mono 2.5.0  | `33faf307fa6031fb4062276d7320a6d632de890cbb347576fd80cfa01077bc25` |

The package archives' integrity, as the registry publishes it:

- @ibm/plex-serif 2.0.0: `sha512-xVu9JsC18tBoQF2M6fofpsWrHj5a94saxlOZbYmXoC0E3UfjgS1JlibgrnaUJjnkLlEyCpMgI2PdhkDxijw0gA==`
- @ibm/plex-mono 2.5.0: `sha512-STBJIPxPomOYPmBMO7z5TKPJUotAF9u3gAUumTqVgwgrAO+K4FRNh0MlhsoJjKhJKsMbBJR10/bk4inkj/wc1w==`

The complete files carry every script the font has, not only Latin; at
about 240 KB for the four, splitting them was not worth a second copy step.
