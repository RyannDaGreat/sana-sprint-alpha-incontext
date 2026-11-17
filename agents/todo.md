# TODO

## REVIEW CHECKLIST (only the user checks items off)
| # | Item | Product LOC | Status |
|---|---|---|---|
| 1 | Aviary checkout in data/, aviary.sh, setup.sh | — | Fork running |
| 2 | Processors + run curation (Step 0) | — | Fork running |
| 3 | Models, eval set, fixed noise, baseline (Step 1); eval as Bluejay plan_eval/generate_eval; models.localize | — | Done, uncommitted, awaiting review |
| 4 | Adapter training code (Step 2) | — | Fork running |
| 5 | GateControl port + control training code (Step 3) | — | Fork running |

## Done (planning)
- [x] Manifest, dtab, fish scripts, research (see git log)

## Open
- [ ] Step 0 curate MAGICK via BlueJay
- [ ] ≤1 h throughput tuning, big batches
- [ ] Step 1 baseline
- [ ] Step 2 adapter → Checkpoint 1 notify (babysit)
- [ ] Step 3 control, GateControl-based (babysit)
- [ ] Step 4 base − adapter + control
- [ ] Journal entries per step
- [ ] Step 5 distill / Step 6 web UI (if time)
- Deadline: 17:16 EST 2026-11-17 (agenda: manifest §0)
