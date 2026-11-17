# MAGICK dataset findings (agent 1)

Sources: project page https://ryanndagreat.github.io/MAGICK/ ; HF https://huggingface.co/datasets/OneOverZero/MAGICK ; tree https://huggingface.co/datasets/OneOverZero/MAGICK/tree/main ; HF API https://huggingface.co/api/datasets/OneOverZero/MAGICK ; paper https://openaccess.thecvf.com/content/CVPR2024/papers/Burgert_MAGICK_A_Large-scale_Captioned_Dataset_from_Matting_Generated_Images_using_CVPR_2024_paper.pdf (downloaded and read via pdftotext). I also downloaded the full magick_index.tsv and one PNG and measured them directly.

## Size / resolution / format
- Index TSV has 148,193 data rows (148,194 lines incl. header) -> 148,193 images. HF card says "over 140,000"; paper says 150,000 objects (the paper number is approximate / pre-filter). Use 148,193 (measured).
  - nsfw=1: 3,000 rows; nsfw=0: 145,193.
- All 1024x1024, 8-bit RGBA, non-interlaced PNG (verified on one sample file: 1024x1024 8-bit/color RGBA, ~1.2 MB).
- On-disk: NOT parquet, NOT tar. Plain files `images/{page_id[:2]}/{page_id}.png` plus `magick_index.tsv`. Alpha is embedded in the PNG (4th channel), no separate alpha file. HF repo tag "format:imagefolder"; the Hub viewer/datasets-server is broken for it (info shows 1 split "validation", 1 example, image column only; first-rows returns 404). Do NOT use `load_dataset("OneOverZero/MAGICK")`: you would get 1 image, not the dataset (unverified beyond the info endpoint; the info JSON is what I saw).
- Repo size: 191 GB shown on tree page (card says ~200 GB for a clone). API usedStorage = 319,373,822,224 bytes (includes LFS/xet history; not the download size).
- Other files: README.md, .gitattributes, .gitignore, `explorer/` (explorer_full.html, explorer_handpicked.html, handpicked_ids.txt, make_explorer_webpages.py), `images/00..zz/` subfolders. magick_index.tsv = 16.1 MB (xet).

## Columns (magick_index.tsv, tab-separated, header row)
`page_id, sim_score, color, nsfw, subject, tag_idk, picked`
- page_id: 10-char alphanumeric id = PNG filename stem.
- sim_score: MS-SSIM-based agreement of the 3 alpha extractors (paper Eq. 2); measured min 0.42, mean 0.991, max 0.9997. Paper: median 0.984 (that is over the paper's set; the released index skews higher).
- color: chroma background color used: green 102,693; blue 40,946; violet 3,757; red 515; yellow 209; orange 73.
- nsfw: 0/1.
- subject: THE CAPTION (see below).
- tag_idk: all "0" (unused).
- picked: "auto" 136,325 (automatic matte selection, pixel-wise chroma key) vs "hand" 11,868 (human-picked best matte, typically glass/water/smoke/fire). (auto,nsfw=0)=133,534; (hand,nsfw=0)=11,659.
- Note: the TSV contains quote characters in subjects; read with csv QUOTE_NONE (I did) or pandas `quoting=3`, otherwise parsing breaks.

## Captions
Stored only in the `subject` column of magick_index.tsv. Short descriptive subject phrases (no background description). Sources (paper 3.1.1): GPT-4/ChatGPT-written subjects, procedurally templated human descriptions, and subjects extracted from a proprietary image-caption dataset with tags like "clipping path", "greenscreen", "on a white background" removed. Consequence: captions from the third source often have artifacts such as a dropped word ("standing against", "models against", "over a .", "on dark .", "and over  covering").
Real examples copied from the TSV:
1. `red boiled crayfish on a plate .`  (id Hcx82jcUpM)
2. `honey dripping from a spoon`  (id UHotBh7CLs, blue bg)
3. `chainsaw`  (id lvYBjIwQRu)
4. `a serene yogi, their hair flowing freely as they practice mindfulness and meditation.`  (id elgcjMYJCZ, LLM style)
5. `nonbinary doctor person hazel eyes sunglasses wearing gown`  (id oI24ottDXo, procedural human template)
6. `close-up of attractive man in trendy suit, tasting champagne, looking at glass, standing against`  (id 001mx1prfP)
7. `a beautiful bottle of luxurious perfume  with smoke`  (id qezdl53j4d)
Many human captions contain stock-photo junk ("copy space for text or design", "template", "mockup", "concept").

## Generator
(paper Sec 3.1.3) Pipeline: prompt = subject + "isolated on a solid <least-common-hue> background" (hue picked from an SDXL sample's hue histogram) -> DeepFloyd IF text-to-image -> Photoshop Subject Selection matte, composite on a brighter saturated same-hue background -> SDEdit img2img with Stable Diffusion XL (strength 0.95, background color as negative prompt to suppress spill) -> alpha extracted by 3 methods (per-pixel color-difference chroma key with inpainted per-pixel background; an internal deep chroma-key model; Photoshop Subject Selection) and best chosen. So final images = SDXL img2img (HF diffusers SDEdit) output over DeepFloyd; not plain SD/Midjourney.

## License
Project page: CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/). The HF card has NO license metadata (HF shows "empty or missing yaml metadata in repo card"). Non-commercial.

## Download
- Full (~191-200 GB): `hf download OneOverZero/MAGICK --repo-type dataset --local-dir MAGICK` (the old `huggingface-cli` is deprecated; hf_hub 1.27 prints "use hf instead"). Or `git clone https://huggingface.co/datasets/OneOverZero/MAGICK` (card's own suggestion; needs git-lfs/xet).
- Index only: `hf download OneOverZero/MAGICK magick_index.tsv --repo-type dataset --local-dir MAGICK`
- Subset: `hf download OneOverZero/MAGICK --repo-type dataset --include "images/0*/*" --local-dir MAGICK` (not run; standard hf flag).
- Single image URL: `https://huggingface.co/datasets/OneOverZero/MAGICK/resolve/main/images/{page_id[:2]}/{page_id}.png`

## Compositing RGBA onto 0.5 gray
- Alpha is STRAIGHT (non-premultiplied): measured on sample 001mx1prfP, where alpha==0 pixels hold nonzero RGB (mean ~75,59,57), which would be 0 if premultiplied. Paper Eq. (3)/(1) treats F as "pure foreground color" with background removed, so RGB = F. Composite as `out = a*rgb + (1-a)*0.5`. Do not multiply by alpha again.
- Color spill: paper claims spill is suppressed by the SDEdit negative prompt and that the pixel-wise chroma key does color decontamination, but admits the pipeline is imperfect (residual green in hair etc., tinted foregrounds). Not verified visually beyond one image. Spill risk is higher for color=green/blue rows with sim_score low; filtering sim_score >= ~0.95 reduces it. Rows with picked=hand skew to translucent objects (glass/smoke/water) where alpha is mostly partial.
- The authors themselves composite on gray for ControlNet training: F = (I - G)/alpha + G with G gray (paper section 4). So gray-background training on this dataset has precedent.
- Sample stats (1 image): alpha=0 55%, alpha=1 37%, partial 8%.

## Not verified
- Exact total download size to the byte (191 GB is the HF tree number). 
- Whether all 148,193 listed page_ids actually exist as PNGs (I only fetched one).
- Per-image spill beyond one sample; the appendix (Google Drive) was not read.
- Datasets-server /splits, /first-rows: first-rows 404; info shows only 1 validation row.
