# TODO

## REVIEW CHECKLIST (only the user checks items off)
| # | Item | Product LOC | Status |
|---|---|---|---|
| 1 | Aviary checkout in data/, aviary.sh, setup.sh | +30 | pushed |
| 2 | Curation processors + run (Step 0) | +550 | done: 140,938 latents; pushed |
| 3 | Eval set, localize, generate/metrics, Step 1 baseline | +599 | pushed |
| 4 | Adapter training (full fine-tune), adapter_v1 | +698 | pushed. NOT ASKED: full fine-tune instead of LoRA |
| 5 | GateControl port + control training (rank 64) | +792 | pushed. NOT ASKED: rank 64 (paper: 16) |
| 7 | Eval for adapter/control/final checkpoints | +180 | pushed |
| 8 | Web UI (gradio) + run_server.sh / run_diffusion.sh | small | pushed; browser-tested |
| 10 | Step 5: control moved onto Sprint (merge_final --student) | +46 | pushed |
| 17 | adapter_v2 (random-color backgrounds) + control_v2 | — | pushed. NOT ASKED: changes the adapter spec (run as a variant); it is the version that works |
| 21 | Showcase set (67), detailed prompts, galleries v1 / v2 (500, 1500, 3000) | — | pushed, live on Pages |
| 26 | WebGPU app (web/, run_web.sh) + Photo → matte (BEN2) | +2784 web | live on Pages; generation untested in a real browser |
| 25 | ONNX export, SVDQuant-style int4 / int8, models branch | — | pushed; int8 not pushed (repo size; your call) |
| 29 | Sparrow views mirroring galleries; run_sparrow.fish | — | pushed |
| 31 | Journal on GitHub Pages (gh-pages + journal-assets) | — | live |
| 39 | Web controls: steps, guidance, λ, resolution; OPFS >2 GB read fix | — | agent running |

## Open
- [ ] Control v2 to step 4000 (~5:15 PM ET), eval, export to web if better
- [ ] Web controls agent: steps / guidance / λ / resolution + NotReadableError fix (m00638–m00640)
- [ ] User decision: push int8 web models (models branch 5.5 → 10.7 GB)?
- [ ] Final wrap-up notification (6 of 6) at 5:16 PM ET
- Deadline 5:16 PM ET 2026-11-17; keep training after it (m00607)
