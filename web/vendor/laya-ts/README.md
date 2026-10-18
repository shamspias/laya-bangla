# laya-ts (vendored build)

Compiled JavaScript from [`laya-ts`](https://github.com/NandhaKishorM/laya/tree/main/laya-ts),
the TypeScript runtime of Laya by Convai Innovations, licensed under Apache 2.0 (`LICENSE`).
It is not published on npm, so the static playground ships this build unchanged.

- Source commit: `d74e8be46ef5241e4ba2b16cd25e896f24f71373` (laya 0.4.11)
- Built with `npm ci && npm run build` in `laya-ts/` (TypeScript 7.0.2, from its lockfile)
- `onnxruntime-web` 1.30.0 is loaded from jsDelivr through the page's import map.

To update: rebuild at a new commit, copy `dist/*.js` here, and record the commit above.
