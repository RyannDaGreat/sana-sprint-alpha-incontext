# WebGPU app — plan

The one-step matte → image model (manifest §12) runs in the browser with ONNX Runtime Web (WebGPU EP, WASM fallback). Vite + vanilla JS (smaller than Svelte; no reactivity needed). The build goes to `journal/webgpu/`, so the journal links to it, and `run_web.sh` serves `journal/` at `/` and `models/web/` at `/models/web/` (HTTPS with a self-signed cert: WebGPU and OPFS need a secure context, and plain HTTP on a LAN IP is not one).

## Layout (simplified Swiss)

- 12-column grid, one gutter size, flush-left type, no rounded corners, no shadows. Hairline rules separate sections.
- Jost (bundled, `@fontsource-variable/jost`): weight 200 for titles, 400 for text, tabular numerals for timings.
- Color: near-black / near-white, one gray for secondary text, one accent (signal red) for the primary button, progress bars and the selected tool. Dark and light modes follow `prefers-color-scheme`; a header toggle overrides it (stored in localStorage).
- Header row: title "Matte → Image", version picker, theme toggle.
- Row 1, left (6 cols): **Matte** (the 1024² white-on-black canvas, shown square). Right (6 cols): **Output** (RGB on gray | RGBA on checkerboard tabs, download buttons, per-stage timings).
- Row 2, left: tool strip under the matte. Right: **Prompt** box, Seed, Generate, Real-time toggle.
- Row 3: **Models** (version list, per-file progress bars, cache size, clear cache).

## Matte tools

Upload (alpha channel if the PNG has one, else luminance; fit inside 1024²) · Brush · Eraser (size slider) · Text (types white words; font + size; placed at click) · Rectangle / Ellipse · Move (drag objects; corner handle resizes; Delete removes) · Invert · Clear. Model: one raster layer (upload + strokes) plus a list of vector objects (text, shapes) drawn over it; invert is applied last.

## Pipeline

`prompt → tokenizer (JS, SANA template) → text_encoder → embeddings (IndexedDB cache by text encoder + prompt)`, `matte → vae_encoder → × scaling_factor`, `noise (seeded) + matte latent + embeddings → transformer (one step, Sprint math inside the graph) → ÷ scaling_factor → vae_decoder → RGB on gray`. RGBA = un-composite with the matte: `fg = (rgb − (1 − a)·0.5) / a`.
Real-time (default off): each edit schedules a debounced run. A mask edit re-runs vae_encoder + transformer + decoder only; a prompt edit re-runs the text encoder only if the embedding is not cached. Edits during a run queue one follow-up run (latest state wins).

## Models

`models/web/index.json` = `{"versions": ["<version>", ...]}`. Each `models/web/<version>/manifest.json` lists its files (path, bytes, dtype, external data) and the sampler constants; see `web/src/models.js` for the schema. Files download with streamed fetch: per-file bar, MB, MB/s, ETA. Each file is streamed into OPFS (`sana-web-models/`, named by SHA-256 of URL + file sha256 or size; Cache Storage fails on multi-GB files); the second load reads only OPFS. `?version=<id>` preselects a version (gallery items link to it).

## Test

Puppeteer (`agents/.scratchpad/web/`): fake version (tiny random ONNX graphs with the real names and shapes, made by a scratchpad script), light + dark screenshots, every matte tool, generate, second load from cache with no model-file network requests.
