# TODO

## H3 MAGIC (manifest §22, m00912–m00917) — deadline: last video row by 9:00 PM ET 2026-11-22
| # | Item | Product LOC | Status |
|---|---|---|---|
| H1 | Transcripts m00912–m00917, manifest §22, this checklist | 0 | done 3:55 PM |
| H2 | RAM guard 75%, 1 ms poll (`h3magic/ram_guard.py`, tmux ram_guard) | +80 | running since 3:48 PM; not committed |
| H3 | Pexels selection: 10k real, decent motion, first 6 s; Sonnet audit sub-frenzy + manual check | — | pending research |
| H4 | BlueJay chunk "H3 MAGIC PROJECT": tables + processors (clips by path, 512² crop, latents, captions, text embeddings, eval) | — | pending |
| H5 | H3 localized (repo models/ + node /models/) | — | pending |
| H6 | 20 curated eval prompts (H3 style, no audio) | — | pending |
| H7 | Captions: H3 style, no audio, 10k Pexels + 10k MAGICK; Muse vs Gemini by measured speed (m00916); notify start + finish | — | pending |
| H8 | Row 1: stock H3 t2v, 20 prompts, BlueJay inference, GPUs 6–7 | — | pending |
| H9 | Pexels adapter (6 GPUs), taste + stop at plateau; row 2 | — | pending |
| H10 | Image-gen adapter on baked Pexels, 7 frames per clip at 0–6 s (m00917); row 3 (images) | — | pending |
| H11 | Gray magic adapter on baked Pexels + image-gen; row 4 (images) | — | pending |
| H12 | Video rows 5a (all on), 5b (gray magic only), 5c (Pexels + gray magic) | — | pending; due 9:00 PM |
| H13 | Journal page journal/h3magic.html, published periodically | — | pending |
| H14 | Journal: Pexels selection page (m00930) with ~50 cut 6 s 512² clips (m00932, m00933); notify once when live (m00931) | — | Sonnet agent running |
| H15 | Journal: one dataset page per adapter (Pexels clips, image-gen frames, gray MAGICK), each with examples + captions and the adapter lineage (trained on top of which baked adapters) (m00934) | — | after each dataset is cut + captioned |

### Schedule (ET, 2026-11-22)
- 3:00–4:15 read, plan, data selection, BlueJay chunk, H3 localize, eval prompts
- 4:15 stock H3 row on GPUs 6–7; captions start (N1); VAE encode on GPUs 0–5
- ~5:15 captions done (N2); text embeddings
- 5:20–6:20 Pexels adapter on GPUs 0–5 (N3: started + stock row live)
- 6:20–7:05 image-gen adapter (N4: Pexels row live)
- 7:05–8:00 gray magic adapter (N5: image-gen row live)
- 8:00–8:45 rows 4, 5a, 5b, 5c on all 8 GPUs (N6: gray magic row live)
- 8:45–9:00 publish journal (N7: final rows live). N8–N10 reserved for problems.

## REVIEW CHECKLIST (only the user checks items off)
| # | Item | Product LOC | Status |
|---|---|---|---|
| 1–44 | Day 1: curation, adapter v1/v2, control v1/v2, galleries, journal on Pages, WebGPU demo + controls + masks | see git log | pushed / live |
| 48 | Final matte model: control_v2 step 6000 (Sprint 1-step IoU 0.95, 2-step 0.97) | — | live as web default |
| 120 | Splitter data: BlueJay white/black composite encode (§17) | +30 (processor), +3 dtab | `51d04d1`, `c4287b1`, `b8cedff` (skip 3 RGB-only rows), not pushed; full encode running since 11:25 PM |
| 121 | Splitter LoRA training + sampling + RGBA solve (§17) | not measured | Fork SplitTrain running |
| 122 | Splitter training from step 7500, fresh optimizer, 5,000 steps, GPUs 0–5 (m00760: GPUs 6–7 for the fast decoder; none idle) | — | after the step-7500 export |
| 123 | Splitter in the browser: export, γ slider, 4 views, turbo | not measured | Fork SplitWeb running (started ~11:32 PM) |
| 124 | Two progress bars: per step and per model part (m00738) | not measured | Fork ProgressBars running |
| 125 | FIBO2 step 7500 to the web (m00741, m00750): export + verify on 2 GPUs in parallel before training, then publish + live test with the compare on CPU during training | — | after the encode |
| 126 | UI polish: icons for editor tools, aligned equal-width buttons, dilate/erode before-after icons with labels, font dropdown previews fonts; Sonnet sub-frenzy (3 auditors) then implement, push (m00747) ; model menu cache dots: yellow = partly, green = fully cached (m00748) ; blind Haiku slop test vs professional UIs, pass ≤ 65% (m00768) | not measured | Fork UIPolish running (worktree) |
| 127 | Sneak peek (§18, m00755): Tweedie previews per step, tiny decoder (research sub-frenzy), per-block stop-gradient shortcut heads, async in the app, ≤ 10% slower ; readable notification once the plan is researched and confident (m00759) | not measured | Fork SneakPeek running |
| 130 | Alpha + color decontamination by iterative (box-constrained) least squares, validated against gradient descent, any K background colors (m00826) | not measured | Fork AlphaILS running |
| 131 | GitHub Pages site (journal + WebGPU app) as a progressive web app: installable, works offline after one visit (m00836) | not measured | Fork PWA running |
| 135 | Splitter checkpoints to the web every 1000 steps (turbo + dev, γ, int8, per-block, highway graph; verify λ/γ 0 and 1; publish; live test) (m00849) | not measured | Fork SplitShip running |
| 136 | Generate button disabled after load (live bug); resizable 1px panes (svelte-lib behavior); Svelte port plan (m00850) | not measured | Fork PaneFix running |
| 137 | Output view "Timeline" tab next to RGB/RGBA/Alpha: scrub every sneak peek snapshot after a generation (m00875) | not measured | Fork PeekScrub running |
| 138 | Gallery tab: every past generation stored in browser disk storage (lossless outputs + full settings), grid, open 1:1, load settings, delete, clear (m00876) | not measured | Fork PeekScrub (second commit) |
| 139 | Sneak peek sawtooth with the trained highway: recalibrate blend/skip tables in pixel space (perceptual), gate monotone (PeekDemo finding) | not measured | Fork PeekCalib running |
| 140 | Sneak peek options: "Every layer" switch (default off), preview decoder small | big (default small = r2a3_xs, big = r5a3_long) (m00880) | not measured | PeekScrub (UI) + PeekCalib (manifests) |
| 141 | Pad: prompts not hidden, one-line boxes, view tabs; gallery Load settings without model; Gallery as a top-level mode (m00883–m00888) | not measured | Fork PeekScrub |
| 142 | Pan/zoom + minimap on image displays (m00889) | not measured | Fork PanZoom running |
| 143 | Matte adapter (μ): 4th head predicting the alpha matte; BlueJay matte encode; train after splitter 5000; app Alpha view (m00892, §21) | not measured | Fork Matte running |
| 144 | λ/γ/μ override inputs (any value); Generate turns into Cancel + Pause/Resume during a run (m00893, m00894) | not measured | Fork PeekScrub |
| 50–52 | Trimask: table (337,934 rows, 19 methods, 0 violations, 4 resolutions), encode, training code | see git log | pushed |
| 53 | control_tri_scratch: from scratch, multires | — | stopped 1:29 PM at step 15840 by user (m00695); last save 15500; 15000 live |
| 54 | Checkpoint script agents/.scratchpad/tri_checkpoint.sh (merge, λ int4 web, verify, publish) | b09151d 742c2ac dec4e00 | pushed; step 3000 + 6000 live, notified 8:13/8:33 AM; 9000 queued (tmux tri_ckpt9) |
| 55 | Trimask galleries | 5257bb4 (rows) | deferred: GPUs go to FIBO2 (m00695); GALLERY=1 when GPUs free |
| 56 | Trimask GUI (design D, user study) | 852a8f6..9895285 | live |
| 57 | Web demo trimask class (cond field) | 1f3d238 (+38) | live; not tested on real WebGPU |
| 58 | Mask presets x10 | data | agent 6 running; consolidation (dedup, merge overlapping categories) pending; uncleaned folders already went live (NOT ASKED) |
| 59 | Trimask journal page | — | after training |
| 61 | FIBO2: train same way (all res, all augmentations), ckpt every 500, notify at 5 milestones (m00694, m00695) | da804e4 | adapter done (1000 steps); control fibo2_tri on GPUs 0-6 since 6:33 PM (global batch 28), GPU 7 = eval GPU for reports (NOT ASKED); 5 notifications sent |
| 62 | SANA vs FIBO2 reports, journal/compare/ (m00698, m00699) | see git log | M0 live 11:02 PM + notified; M1 (FIBO2 ctrl 2000) via fork Fibo2Eval on GPU 7 |
| 63 | FIBO2 in the browser (m00688, m00693, m00694, m00710) | — | LIVE on Pages (gh-pages 22ee1f4): turbo 4-step int8 (49 dB live vs PyTorch), dev int8 (34-38 dB), dev int4; all live-tested; notified 4:58 AM + 8:03 AM. Open: turbo verify 2 items at 28.6 dB (cause unknown) |
| 64 | Full resolution everywhere (m00705) | — | DONE: Pages serves original PNGs only (gh-pages a68dcca); compare atlases dropped; NOT ASKED: step0 JPEGs remain (no generator in repo); repo size past GitHub 5 GB guidance |
| 67 | FIBO2 few-step (m00711, m00712) | — | DONE: A turbo transfer WORKS and ships (IoU 0.590 vs dev4 0.490); B own DMD2 student works (sharp; regression term 8cb0bf9 fixes drift; level with dev 30-step on 11 items) but turbo beats it by 0.138 for the control; DMD paused at ~step 1000 to free GPU 7 |
| 68 | FIBO2 full eval set + report (m00713, m00714) | see git log | turbo 1024 pass (1273) LIVE in compare/scrub (gh-pages dab1da9), notified 11:39 AM; 768/512/256 + dev 30-step teacher rendering |
| 70 | FIBO2 step 4000 in browser (dev + turbo int8) | — | LIVE + live-tested (47 / 51 dB vs PyTorch), notified 11:39 AM |
| 71 | Web UI generation progress bar: steps + evenly spaced layer chunks (m00715), push when done | — | fork Progress running |
| 72 | Trimask methods `original` + `threshold` via BlueJay; training mix 30% solid outlines / 70% trimasks (m00718, m00720, m00721 — "masters single outlines"), restart FIBO2 training | — | fork TrimaskPlus running; manifest §13 + concerns updated |
| 73 | Reports: every image reachable (all-methods default view, counts on hub); journal index newest first with dates (m00719) | — | fork FullRes running |
| 74 | GitHub over size quota -> big files to ONE Hugging Face repo (OneOverZero), shrink GitHub, push everything, publish, live test; notify when ready to play (m00727, m00728) | — | fork HFMove running |

## Open
- [ ] Checkpoints 9000, 12000, 15000, 18000, 20000: push + manual notify each
- [ ] After training: galleries on all GPUs (GALLERY=1), journal page
- [ ] Mask consolidation pass, publish, push, notify (m00686)
- Deadline 2:15 PM ET 2026-11-18 (m00644)

## Notification points (m00725, m00728, m00731, m00732, m00741, m00745, m00759, m00764, m00765) — every message starts "Sana:"
1. Encode finished + step-7500 export started
2. Splitter training started (after the first real steps)
3. Step-7500 exports finished (verified, published, live-tested, link)
4. Everything pushed, ready to play with
5. Sneak peek plan researched and confident (readable, amnesia)
6. Fast-decoder training started (sneak peek) + ETA; then a measured-ETA follow-up
7. Sneak peek highway attached to training (training restart) + ETA; then a follow-up
8. Splitter evaluation: how good or bad (readable, amnesia)
11. Fast decoder: a clearly better set of candidates worth looking at (beats fal Tiny AE color or fal-lite D speed with Tiny-AE look), with the link (m00788)
10. Morning (~8 AM): fast-decoder report, every candidate sorted best → worst, ≥ 48 examples, link (m00786)
12. Morning (~8 AM): sneak peek UI done in the app, with the per-generation preview-timeline graph in a closed accordion (m00822)
9. Problems at once
14. UI things live (Timeline, Gallery mode, pad fixes, Greek-letter overrides, Cancel/Pause, pan/zoom) (m00895)
13. NWD (m00884): "Done" when all running work is finished (publish, PeekScrub, PeekCalib, ship loop to step 5000, training)

## Highway + matte notifications (m00903): at most 5, spaced evenly, one big meaning each, short and readable
- [sent 7:40 PM] H1: matte W pilots done: which W, does joint training help previews, does final quality drop (λ = 0 and λ = 1)
- [sent 7:50 PM] H2: Highway2 (learned blend, m00901) pilot numbers vs the old highway, and which highway the long run uses
- [sent 07:46 PM] H3: the long matte + highway run is launched, with a measured ETA
- [sent 11:27 PM] H4 (a few hours in, m00896): is it working, and does final image quality degrade
- H5: the long run finished and shipped to the web app, with the link

## 128 — Cross-step activation reuse / layer skipping (m00820) — after everything else; no permission needed to start
- Phase 1 done 2026-11-21 (manifest §19, journal/cross_step_reuse.html): whole-step reuse wins; turbo step 2 reuse = 25% of blocks for LPIPS 0.014; dev FORA3 = 55% for 0.034 / 0.013. Phase 2 (trained) only worth it for turbo.
- Phase 2 done 2026-11-21 (§19): reuse every block at turbo steps 2–3 (50%) + a zero-init connector distilled to x̂0: LPIPS 0.098 / 0.069 -> 0.038 / 0.031 (v4). Open: Learning-to-Cache router (not done), app port + wall time.
- Measure redundancy: per-block cosine similarity / relative change of hidden states between consecutive steps (dev 20 steps, turbo 4), and the error of reusing step s−1's block output at step s (training-free caching, FORA/Δ-DiT style).
- Trained version: store every block's activations at step s−1; at step s feed them through zero-init (ControlNet-like) connectors; schedule-dependent layer skipping (middle steps skip the first half, late steps skip ~80%), distilled to the full model's x̂0 (Learning-to-Cache style).
- Report speedup vs quality (SSIM/LPIPS/ΔE + VLM) in a journal report; log in manifest §19.

## 129 — Superseded by item 143 (matte adapter μ, m00892)
- Encode the alpha matte as a white-on-black image (FLUX.2 VAE, 1024/768/512/256) for every training row (new encode_fibo2 job, BlueJay).
- 4 heads: gray, white, black, alpha; fuse with the least-squares fit (decontaminated color stays from the fit).

## 132 — Persistent drawing + prompt, Reset-all (m00848)
- The drawn mask, prompt and settings survive a reload (stored in the browser); a "Reset all" button clears them.

## 133 — Four-way pad mode (m00848) — after the sneak peek UI lands
- A tab switches between the normal mode and the four-way pad; the mask editor and hyperparameters are shared.
- Up to 4 prompt/seed slots with the same steps/settings; all 4 diffuse at once, taking turns (round-robin), each with its sneak peek preview, prompt and seed shown below it.
- Click a slot: it goes 50% transparent and pauses; the others get its share and run faster; click again to re-enable.
- Editing a slot's prompt or seed restarts only that slot; editing the mask restarts all.

## 134 — Real-time mode with cancel/restart (m00848)
- While drawing, a running generation is cancelled partway and restarted with the new condition at once (sneak peek makes partial results useful).

## 145 — Greek-letter tooltips + small agentic user study (m00896)
## 146 — Pan/zoom on the trimask editor, drawing unaffected (m00896)
## 147 — μ adapter training with the highway stop-gradient removed; coefficient pilots; quality analysis + notification a few hours in (m00896)
## m00900 / m00902 — layout, top tabs, gallery freeze, model card Loading, 5 px panes (fork Layout)
- [x] Gallery freeze: not reproduced headless (no WebGPU here; 20 stored generations open in 3–8 ms, with and without a model load); every swallowed error made loud, IndexedDB `blocked` now rejects, main-thread blocks > 200 ms logged
- [x] Top tabs Editor | Gallery | Models; Four-way pad is a switch in the output header
- [x] Output above the trimask (resizable split); prompt and settings at full height on the right
- [x] Model card says Loading… with the live status line at once
- [x] Every pane ≥ 5 px (m00902)
- [ ] User: try Gallery on the live site; if it freezes again, the console now says where
## 161 — Gallery items keep their sneak peek timeline (m00906)
- Lossless PNG frames saved with the item; a timeline tab in the full view (scrubber, Play, error curve); per-generation reels. Verified headlessly with a stored item and a reload (galtl_probe.js); not yet with a real generation (no WebGPU on the node).
