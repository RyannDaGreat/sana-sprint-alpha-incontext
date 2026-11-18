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

**Undo / threshold / fonts / mask gallery** (m00643): every matte edit (stroke, shape add/move/resize/delete, text commit, upload, photo → matte, mask load, invert, clear, threshold) is one undo step: a snapshot of raster + objects + invert, the raster shared between steps that did not touch it; `HISTORY_LIMIT` = 40 steps (≤ 160 MB of rasters). Buttons plus Ctrl/Cmd+Z, Ctrl/Cmd+Shift+Z, Ctrl+Y (text fields keep native undo). Threshold binarizes the raster layer at a gray level with live preview; Softness > 0 keeps a linear ramp of that width (levels). Apply = one undo step; any other edit cancels the preview. Vector objects are already binary. Text fonts: `src/fonts.js` — built-ins plus ~28 Google Fonts (sans, serif, script, blackletter, mono, display), stylesheet loaded on demand in CORS mode (the local server sends COEP require-corp). Mask gallery: "Masks…" opens a full-screen `<dialog>`: category filter, square tiles, credit + source on hover, click = load as a fresh matte (one undo step). Data: `public/masks/<category>/index.json` = `{"name"?, "items": [{file, name, source_url, license, author}]}`; `scripts/mask_index.mjs` (run by `npm run build`) writes `public/masks/categories.json` from the folders that have an index. `magick/` = the 29 held-out showcase mattes (CC BY-NC 4.0, MAGICK dataset).

**Photo → matte** (m00632): upload any photo; BEN2 (MIT, `onnx-community/BEN2-ONNX` fp16, 219 MB, from the HF hub into the same OPFS cache with a progress bar) finds the foreground in the browser — WebGPU when the adapter has shader-f16, else WASM; the status line names which. The photo is stretched to 1024² for the model; the soft alpha is resized back and letterboxed like an upload, replacing the raster (objects and invert are reset), so every tool still edits it. "Show photo" puts the letterboxed photo under a translucent matte. BiRefNet_lite was rejected: its Split nodes (up to 33 outputs) exceed WebGPU's storage buffers per shader, and its fp32 run hit std::bad_alloc on WASM.

**Trimask editing** (m00684, details and the user study in `TRIMASK_GUI.md`): the canvas is a trimask (white object, gray 0.5 unknown, black background, fuzzy values allowed); a plain white-on-black matte is the special case. A value palette sets what brush, text, rectangle, ellipse, polygon and line paint (eraser = black; with Move it recolors the selected object). Polygon (click corners, click the first / double-click / Enter closes, Esc cancels) and Line (polyline, width = brush size) are movable objects stored as a box + relative points. Filters (`src/trimask_ops.js`, pure, examples run by `agents/.scratchpad/web/test_trimask_ops.mjs`): soften, harden, value remaps, dilate, erode, dilate/erode into gray, edge band. A filter acts on the whole visible matte, so it flattens objects and inversion into the raster (one undo step; Undo restores the shapes). `?gui=A|B|C|D` picks the control design (`src/trimask_ui.js`); D, the user-study merge, is the default. The title line shows the live white / gray / black percentages; the gallery has a name search.

## Pipeline

`prompt → tokenizer (JS, SANA template) → text_encoder → embeddings (IndexedDB cache by text encoder + prompt)`, `matte → vae_encoder → × scaling_factor`, `noise (seeded) + matte latent + embeddings → transformer (one step, Sprint math inside the graph) → ÷ scaling_factor → vae_decoder → RGB on gray`. RGBA = un-composite with the matte: `fg = (rgb − (1 − a)·0.5) / a`.
Real-time (default off): each edit schedules a debounced run. A mask edit re-runs vae_encoder + transformer + decoder only; a prompt edit re-runs the text encoder only if the embedding is not cached. Edits during a run queue one follow-up run (latest state wins).

## Models

`models/web/index.json` = `{"versions": ["<version>", ...]}`. Each `models/web/<version>/manifest.json` lists its files (path, bytes, dtype, external data) and the sampler constants; see `web/src/models.js` for the schema. Files download with streamed fetch: per-file bar, MB, MB/s, ETA. Each file is streamed into OPFS (`sana-web-models/`, named by SHA-256 of URL + file sha256 or size; Cache Storage fails on multi-GB files); the second load reads only OPFS. `?version=<id>` preselects a version (gallery items link to it).

## Test

Puppeteer (`agents/.scratchpad/web/`): fake version (tiny random ONNX graphs with the real names and shapes, made by a scratchpad script), light + dark screenshots, every matte tool, generate, second load from cache with no model-file network requests.
