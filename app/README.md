# app

The interface: React, TypeScript and Vite, in the "night glass" theme. It
runs the engine in a worker and, from slice 4, draws the workbenches; the
site in `../web/` hosts the built app.

Slice 3, step 1: the scaffold. The page says only that nothing runs yet.

```sh
cd app
npm ci             # the lockfile's packages; Node 24.12.0 (.nvmrc)
npm run dev        # the dev server
npm run build      # dist/
```

The checks run from the repository's root, through `python3 scripts/check.py`
(format, boundaries, and the type check in `test`). The import rules are the
[file mapping's](../docs/architecture/file-mapping.md).
