# Trimask editor GUI — three designs (m00684)

The matte canvas becomes a **trimask**: white = object, black = background, gray 0.5 = unknown (the model decides); fuzzy values are allowed. Plain white-on-black mattes still work unchanged. All operations are pure functions on a Float32 (H, W) mask in [0, 1] (`src/trimask_ops.js`); the three designs differ only in how they are offered. `?gui=A|B|C` picks one.

## Shared by every design

- **Value palette** (White / Gray / Black): every drawing tool paints the chosen value — brush, rectangle, ellipse, text, polygon, line. Eraser always paints black. With **Move**, picking a value recolors the selected object (one undo step).
- **Polygon**: click corners; click the first corner, double-click or press Enter to close; Esc cancels. **Line / polyline**: click points, double-click or Enter to finish; width = the brush size slider. Both become movable, resizable objects (box + relative points), like rectangles.
- **Undo**: every filter is one undo step. A filter works on what you see, so it **flattens** objects, text and inversion into the raster; Undo brings the editable shapes back. Drawing on the canvas cancels any preview.
- **Photo → matte, upload, mask gallery** replace the raster as before; their soft edges are fuzzy trimask values, so Harden or Edge band turn them into a clean trimask.
- **Threshold** (existing) stays: it binarizes the raster layer only.

| Operation | Definition (m = mask, r = size px) |
|---|---|
| Soften | Gaussian blur, σ = r / 2.5 (hard edge → gray ramp ≈ r wide) |
| Harden | ≤ 1/3 → black, ≥ 2/3 → white, else gray (B: both thresholds are sliders) |
| Remap | piecewise-linear map of black, gray, white to any of the three (white→gray, gray→black, black→white, swap, …) |
| Dilate / erode | max / min over a disk of radius r |
| Dilate into gray | max(m, min(dilate(m), 0.5)): the grown rim is gray |
| Erode into gray | min(m, max(erode(m), 0.5)): the removed rim is gray |
| Edge band | erode into gray r/2, then dilate into gray r/2: a gray band of width r centered on every edge |

## A — Filter buttons (technical names, one click)

A Size slider and three rows of buttons: *Edges* (Soften, Harden, Edge → gray band), *Shape* (Dilate, Erode, Dilate into gray, Erode into gray), *Values* (White → gray, Gray → black, Gray → white, Black → gray, Black → white, White → black, Swap). A click applies at once; dragging Size right after re-runs that filter from the same start (it replaces its own undo step). Fast and discoverable; risk: image-processing words (dilate, erode) for non-experts.

## B — Remap matrix + morphology (preview, then Apply)

A 3 × 3 radio matrix (rows *from* white/gray/black, columns *to*), a Morphology block (Dilate | Erode | Edge band, an **Into gray** switch, Size), and Soften / Harden sliders (black-below and white-above thresholds). Any change previews live on the matte; one Apply / Cancel pair commits it. Expressive (any of the 27 remaps, tunable harden); risk: more steps, forgetting Apply, matrix reading.

## C — Verbs + lasso (plain language, local)

The palette reads *Sure object / Unsure / Sure background*; the buttons are verbs: Blur edges, Snap to sure / unsure, Make edge unsure, Grow, Shrink, Grow an unsure rim, Shrink (rim unsure), Object → unsure, Unsure → background, … , Everything unsure. An Amount slider re-tunes the last verb as in A. A **Lasso** tool limits every verb to the drawn region (e.g. make only a roof unsure); "Whole matte" drops it. Risk: verbs are less precise; the lasso is one more mode.

## Agentic user study

Fresh Haiku participants (`claude -p --bare`, one folder each under `agents/.scratchpad/user_study/`, read/run access only to their own folder), one design each, no plan or expected answers. They drive the real app headlessly through `ui.js` (Puppeteer CLI: list / click / set / canvas click|dblclick|drag / key / look → screenshot they Read / submit). Seven tasks (heart edge → 20 px gray band from the gallery; polygon house with a gray roof; gray → black; soften the edge; erode-into-gray, undo, dilate-into-gray; a black line cutting a disc in two; harden a blurry heart). Each submitted matte is scored by `tools/score.mjs` against objective checks (band width from gray area / perimeter, exact agreement, component counts, …); an oracle run passes all seven and a do-nothing run fails all seven. Results: see the bottom of this file.

## Results (2026-11-18)

Pilot r1 (9 participants) only debugged the harness. Main round r2: 4 Haiku participants per design A, B, C; confirmation round r3: 4 on D. Pass = the submitted matte meets the task's objective check (`agents/.scratchpad/user_study/rounds/r*/score.txt`). B3 quit before its first command (a denied `cat`, harness), so B has 3 completers.

| Design | n | T1 band | T2 house | T3 gray→black | T4 soften | T5 erode/undo/dilate | T6 line cut | T7 harden | median commands (T1…T7) |
|---|---|---|---|---|---|---|---|---|---|
| A buttons | 4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 11 21 6 6 11 10 15 |
| B matrix | 3 (+1 dropout) | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 17 21 9 7 16 10 9 |
| C verbs | 4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 11 20 5 5 12 10 10 |
| **D merge** | 4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 13 21 6 6 11 16 6 |

Every design reached the goal, so the comparison rests on effort and on the participants' notes:
- **B** cost the most commands (T1 17 vs 11, T5 16 vs 11), and 3 of 3 noted that a preview looks applied when it is not, that the shared Apply sits far from the matrix, and that "Into gray" is unexplained.
- **A / C**: T7 users of A did not see "Harden" as the three-level tool (all four built it from threshold + edge band, 15 commands); C's "Snap to sure / unsure" fared better (10). Everyone asked what Size / Amount means (radius or width? inside or outside the edge?), missed that line width is the brush slider, and wanted a gallery search and a value readout.

**D** = A's buttons with technical names plus meaning ("Dilate (grow)", "Harden → 3 levels", palette "Gray · unknown"), a tooltip on every filter that says exactly what Filter size does (e.g. "band of total width Filter size centered on every edge: half inside, half outside"), the slider renamed "Filter size" and the brush slider "Size (brush, eraser, line width)", C's lasso ("Lasso (filter region)"), gallery search, and a live "white · gray · black %" readout under the matte title. In r3, T7 dropped to 6 commands with no confusion about harden; the other tasks stayed at A/C cost. Remaining notes are about the headless harness (no cursor coordinates), not the app. D is the default; `?gui=A|B|C` keep the others.

Known harness flaw (fixed after r3, same for every design): `ui.js` left stale control numbers on the page, so a few clicks inside the gallery dialog could land on the wrong control.
