# Discovery: is a "use the printed solved picture" option worth shipping?

## Question

Some online puzzle pictures print a small copy of the solved picture on the frame. Matching the tiles
against it (`src/picture/reference.ts`, `matchReference`; a prototype, not wired into the app) does not
depend on tile edges continuing. Is it clearly better than edge assembly (`assemblePicture`) once the
corners and the empty slot are corrected by hand, so that it is worth a UI (a second four-corner
placement for the print)?

## Method

`scripts/gen_correction_photos.py` writes three sets of photos, 48 each (24 3x3, 24 4x4), recreating
failures of owner-reported online pictures (the owner's images are not available and are not used):
`angled-blank`, `white-frame-thumbnail` (a printed thumbnail on a wide white frame) and `line-art`
(low detail, thumbnail on the frame). The `dev` split (seed 20261123, `data/corrections-dev`) was used to
look at failures; the `eval` split (seed 20261124, pictures never used for tuning) is the held-out set
below. Thumbnails are 58-141 px per side on the white frame (median 95) and 76-135 px on line art
(median 106). `scripts/eval-corrections.ts` measures, per photo, the whole solved layout being exactly
right ("exact", which includes the empty slot's home). The corrected rows use the app's own path,
`readPictureBoard(image, { quad, size })` then `assemblePicture`. The reference rows cut the cells from
that same corner path with the labelled empty slot, and cut the print from its labelled corners
(exact, or each corner moved up to 3% of the print's side).

Run 2026-11-25, Node v22.23.3, 4 cores, ~6 min for the eval:

```
npm run pictures:fetch
bash scripts/python-env.sh scripts/gen_correction_photos.py --split dev --count 48 --seed 20261123 --out data/corrections-dev
npm run gen:corrections
npm run eval:corrections
```

## Held-out results (seed 20261124, 48 photos per set)

The reference columns go through the app's reachability check (`checkArrangement`); the edge column is
scored directly. That favours the reference, so where the edge column wins the result holds a fortiori.

| set                   | grid | n   | edge assembly, corners + blank given | reference | reference, print's corners ±3% |
| --------------------- | ---- | --- | ------------------------------------ | --------- | ------------------------------ |
| white-frame-thumbnail | 3x3  | 24  | 75.0%                                | 66.7%     | 50.0%                          |
| white-frame-thumbnail | 4x4  | 24  | 75.0%                                | 54.2%     | 45.8%                          |
| white-frame-thumbnail | all  | 48  | 75.0%                                | 60.4%     | 47.9%                          |
| line-art              | 3x3  | 24  | 8.3%                                 | 50.0%     | 50.0%                          |
| line-art              | 4x4  | 24  | 4.2%                                 | 8.3%      | 12.5%                          |
| line-art              | all  | 48  | 6.3%                                 | 29.2%     | 31.3%                          |

The edge assembly column is the final assembler (two scorers with ramp removal, README "Picture puzzles" and "Reference picture (not shipped)");
the reference columns do not depend on it.

One photo is 2.1 points; the 95% interval of a rate near 50% on 48 photos is about ±14 points, so only
differences of 20+ points mean anything.

Reference matching by its mean confidence, held out (a cut is the lowest mean per-position confidence
kept; the dev set suggested 0.5 for white-frame, which held out confirms):

| set                   | min mean confidence | photos kept | reference exact | edge assembly (same photos) |
| --------------------- | ------------------- | ----------- | --------------- | --------------------------- |
| white-frame-thumbnail | 0                   | 48/48       | 60.4%           | 75.0%                       |
| white-frame-thumbnail | 0.5                 | 31/48       | 83.9%           | 83.9%                       |
| line-art              | 0                   | 48/48       | 29.2%           | 6.3%                        |
| line-art              | 0.4                 | 20/48       | 60.0%           | 15.0%                       |
| line-art              | 0.5                 | 14/48       | 64.3%           | 14.3%                       |

(The full table for cuts 0 to 0.6 is in the eval's output.) On white-frame, dev kept 35/48 at 0.5 with the
reference 82.9% right (edge assembly 82.9% on the same photos), so 0.5 is stable there. On line art the
held-out exact rate rises with every cut (29, 30, 44, 54, 60, 64 and 83% at 0, 0.1, 0.2, 0.3, 0.4, 0.5
and 0.6, the last on 6 photos), so confidence does rank the results; on dev it does too (0.4 keeps 23 of
48 at 47.8%, 0.5 keeps 16 at 62.5%). It does not make the option useful: a cut that reaches 60-64% keeps
14-20 of 48 photos.

## Other finding: corner jitter and snapping

A finger places corners a few percent off. Cutting the cells from corners moved by up to 3% of the side,
as placed with no grid fit or snap ("raw ±3% (no snap)" in the eval), assembles almost nothing
(held-out angled-blank / white-frame / line-art: 2.1% / 8.3% / 0.0%). The app's path (fit the grid inside
the quad, snap to the seams twice, within 4% of the side) gets 29.2% / 68.8% / 4.2% from the same moved
corners ("corners ±3%"), against 29.2% / 75.0% / 4.2% from the labelled corners: it loses nothing on
angled-blank and line-art and 6.3 points on white-frame.

## Recommendation

**Do not ship a "use the solved picture as reference" option.**

- White-frame: with the print's corners placed exactly the reference is 14.6 points worse than edge
  assembly (60.4% vs 75.0%); with them moved by 3%, as a person would place them, it is 27.1 points worse
  (47.9% vs 75.0%). It does not repay a second corner-placement screen. Even the mean-confidence cut 0.5
  gives the reference no edge (83.9% vs 83.9% for edge assembly on the same photos).
- Line art: the clear win (29% vs 6%), but still wrong two times in three, and only on 3x3 (50% vs
  8%; 4x4 8.3%). Confidence does rank the results (on the photos kept at a cut of 0.4 or 0.5 the
  reference is 60-64% right), but those are at most 42% of the set, so most photos would still need another way.

Revisit if the print could be found automatically (no second placement) or with a larger sample.
