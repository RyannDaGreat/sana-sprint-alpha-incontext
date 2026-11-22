# Pexels selection (H3 Magic, item H3, m00912)

`pexels_selection.parquet`: 10,000 `train` + 20 `val` (validation loss only, never trained on) + 11,503 `spare` real Pexels clips with decent motion: rows 1–11,523 from the first draw, then 10,000 more `spare` rows appended (same filters, ids disjoint, seed 20261122 + 1, ffprobe-checked) so the clip cutter can keep the 10,000 most-moving clips of the first 6 s. Columns: `pexels_id, source_path, split, duration, fps, width, height, flow_mean_magnitude, birdseye_caption`. Seed 20261122. Only the first 6 s of each clip are used.

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

## Which BirdsEye metrics predict the audit (m00926)
Second audit batch: 105 random `train` rows, 3 more Sonnet agents; 306 audited clips in total (165 decent/high motion, 6 not real). Per metric, on the audited clips: AUC for "decent or high motion in the first 6 s", Spearman with the motion level (static 0 … high 3), AUC for "real". Coverage = BirdsEye pexels videos that have the field (of 81,764).

| Metric | AUC moving | Spearman motion | AUC real | Coverage |
|---|---|---|---|---|
| phash_max_change | 0.740 | 0.49 | 0.44 | 81,764 |
| phash_std_change | 0.738 | 0.48 | 0.42 | 81,764 |
| phash_mean_change | 0.731 | 0.47 | 0.50 | 81,764 |
| phash_temporal_std | 0.724 | 0.45 | 0.47 | 81,764 |
| clip_std | 0.708 | 0.41 | 0.38 | 81,764 |
| gve_std | 0.659 | 0.30 | — | 37,126 |
| flow_std_magnitude | 0.617 | 0.29 | 0.39 | 81,764 |
| flow_temporal_std | 0.616 | 0.29 | 0.50 | 81,764 |
| flow_max_magnitude | 0.609 | 0.27 | 0.42 | 81,764 |
| flow_mean_magnitude | 0.585 | 0.22 | 0.47 | 81,764 |
| flow_min_magnitude | 0.570 | 0.18 | 0.59 | 81,764 |
| height / width / duration / fps / num_frames / file_size_bytes | 0.46–0.53 | ≈ 0 | 0.46–0.66 | 81,764 |
| bits_per_pixel_second (derived: file size · 8 / (duration · width · height)) | 0.49 | 0.02 | 0.78 | 81,764 |
| dwpose_*, siglip_std, nb_pro3_* | too few audited rows (≤ 18) | | | 5,123 / 800 / 491 |

- The perceptual-hash change metrics and clip_std predict first-6-s motion clearly better than the RAFT flow we filtered on (0.74 vs 0.59).
- No metric reliably predicts "real": only 6 negatives. bits_per_pixel_second 0.78 hints that renders compress well (low bitrate), but n = 6 is too few to filter on.
- **`birdseye_score`** (new column, every row): the mean of the percentile ranks (over all 81,764 BirdsEye pexels videos) of phash_std_change, phash_max_change and clip_std; 0 = least motion, 1 = most. In-sample AUC for moving 0.758. Among audited clips, the top 50% by score are 73% moving, the bottom 50% 35%; the top 75% are 66% moving, the bottom 25% 18%. Use it with the cutter's own first-6-s motion measurement to keep the 10,000 most-moving clips.
