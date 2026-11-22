# Pexels selection (H3 Magic, item H3, m00912)

`pexels_selection.parquet`: 10,000 `train` + 20 `val` (validation loss only, never trained on) + 1,503 `spare` real Pexels clips with decent motion. Columns: `pexels_id, source_path, split, duration, fps, width, height, flow_mean_magnitude, birdseye_caption`. Seed 20261122. Only the first 6 s of each clip are used.

## Source
- Metadata: BirdsEye `dataset_plugins/pexels/_aggregation/dataset/_fields.sqlite` (read-only), processor `pexels`: 81,766 videos.
- Videos: `/fsx_scanline/from_eyeline/ning_video_genai/datasets/ryan/Pexels/Pexels/Attempt_A/<pexels_id>` (extensionless MP4, S3-backed). All 65,578 pool files exist.
- `birdseye_caption` is the old BirdsEye caption of the WHOLE video, not of the first 6 s. Kept for reference; the training captions are the new H3-style ones.

## Filters, in order
1. `duration >= 6.5` s, short side >= 512 px, caption longer than 20 characters: 75,818.
2. Caption keywords (case-insensitive) removed: animat, cartoon, 3d render, rendered, `render `, cgi, computer-generated, computer generated, illustration, digital art, graphic, motion graphic, `text `, logo, screen, screenshot, animation, abstract, particles, glitch, simulation, virtual, game, slideshow, infographic, hologram: 65,578.
3. `flow_mean_magnitude >= 1.5` (RAFT on 25 frames of the whole video at 256 px) and extra keywords promotional, black friday, stop-motion, stop motion, mockup, mock-up, advertisement.
4. Random draw (seed above); each drawn file checked with ffprobe (readable video stream, duration >= 6.5 s, short side >= 512).

## Audit (5 Sonnet agents + a manual check, 2026-11-22)
- 203 candidates after filters 1–2, 29 per flow stratum (0.3–0.6, 0.6–1, 1–1.5, 1.5–2.5, 2.5–5, 5–10, ≥ 10). Each judged from a contact sheet of the first 6 s (frames at 0–5 s): real camera footage, and motion static / low / decent / high.
- Not real: 4 of 203 (2.0%): a solar-carport render, a Black Friday gift-box render, a stop-motion balloon "LOVE" (real objects), coils falling on white (doubtful). By eye: the two renders are CGI; the other two look like real studio footage. So the true synthetic rate after the keyword filter is about 1–2%.
- Static (first 6 s nearly frozen): 4/29 at flow 0.3–0.6, 2/29 at 0.6–1, 1/29 at 1–1.5, **0/116 at flow >= 1.5**. Hence the threshold 1.5.
- "Low" motion stays about 30–60% in every stratum: the whole-video flow does not predict the first 6 s well. The clip cutter should measure the motion of the cut 512² first 6 s itself and drop the stillest clips, replacing them from `spare`.
- Manual check (h3magic): 6 low-flow sheets (all real, low but visible motion) and the 4 flagged + 4 random passed sheets; agreed with the agents except the two borderline studio clips noted above.
- Raw: `agents/.scratchpad/pexelspick/` (`audit_joined.parquet`, `sheets/`, `agent_N/verdicts.json`; not saved in git).
