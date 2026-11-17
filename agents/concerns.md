# Concerns — append-only history

## 2026-11-17 03:27 — Project start
- Dump renamed from "Untitled Dump Tue Nov 17, 2026 at 3-09-36AM" to "SanaSprint Alpha InContext Plan" (user request, m00018). A symlink with the old name was left in `~/CleanCode/Dumps/` so the running session's cwd stays valid. The user may delete it.
- Planning only on the Mac. Training will run on an 8× A100 80GB node via a GitHub repo.
- Sonnet sub-frenzy launched (5 agents): MAGICK dataset, Sprint fine-tuning, AnimateDiff adapter, in-context conditioning, RGBA prior art. Outputs: `agents/.scratchpad/frenzy/`.

## Known risks
- Fine-tuning the Sprint student with a plain diffusion loss may destroy its few-step ability (unverified). The alternative is training the teacher and re-distilling.
- bf16 weight arithmetic (`base − adapter + control`) loses precision. Do the subtraction in fp32.
- MAGICK is CC BY-NC 4.0. Fine for research.
