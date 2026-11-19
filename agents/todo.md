# TODO

## REVIEW CHECKLIST (only the user checks items off)
| # | Item | Product LOC | Status |
|---|---|---|---|
| 1–44 | Day 1: curation, adapter v1/v2, control v1/v2, galleries, journal on Pages, WebGPU demo + controls + masks | see git log | pushed / live |
| 48 | Final matte model: control_v2 step 6000 (Sprint 1-step IoU 0.95, 2-step 0.97) | — | live as web default |
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
| 63 | FIBO2 in the browser: memory64 ORT-web build, >4 GB proof, int4 export (m00688, m00693, m00694) | — | fork Fibo2Web running (CPU only) |
| 64 | Full resolution everywhere (m00705): train previews full 1024 (Fibo2Train), compare-report atlases 256 px + 512 thumbs -> original PNGs, journal-assets WebP q95 -> lossless | — | previews fix sent; reports + journal not started |

## Open
- [ ] Checkpoints 9000, 12000, 15000, 18000, 20000: push + manual notify each
- [ ] After training: galleries on all GPUs (GALLERY=1), journal page
- [ ] Mask consolidation pass, publish, push, notify (m00686)
- Deadline 2:15 PM ET 2026-11-18 (m00644)
