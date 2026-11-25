# Failure-mode photos for the manual corrections

One generated photo from each set (the line-art one from the dev set, see below) of `scripts/gen_correction_photos.py` (`npm run
gen:corrections`, seed 20261124; pictures are public-domain, see `picture-sources.toml`, or drawn by the
generator). Each is a case where the automatic reading fails and the app's manual correction gets the
whole solved layout right, so `test/picture-corrections.test.ts` fails if either regresses. The labels
are the generator's (`tilesQuad`, `blank`, `solved`; `thumbQuad` is the printed solved picture, not used
by the app). No owner photo is here or may be added.

- `angled-blank.jpg` (3x3, `0004` of the set): board tilted 30-50 degrees. The automatic reading picks
  the wrong cell as the empty slot; **Change empty slot** (`reassemblePicture`) fixes it.
- `white-frame.jpg` (4x4, `0015`): board on a wide white frame with a printed thumbnail of the solved
  picture. The automatic reading takes the wrong area for the tiles; **Adjust corners** fixes it, and
  the empty slot is then found by the app.
- `line-art.jpg` (3x3, `0042` of the **dev** set, `gen_correction_photos.py --split dev --count 48 --seed
20261123`): a line drawing on a flat background, little for edges to continue. The automatic reading
  orders the tiles wrong; reading from the labelled corners orders them right. That is luck of a few
  pixels, not a fix: corners rescue only a handful of line-art photos (4 of 48 dev). It is a 3x3
  because the 4x4 line-art photos are rarely rescued at all. It is from the dev split, so the held-out
  set stays unseen; `0042` is the one with a JPEG under 200 KB where, with the current code, automatic
  fails and the corrected path is exact.

Chosen as photos where automatic fails and corners (plus the
labelled blank, for the angled one) work. Each JPEG is the generator's output, copied unchanged and under
200 KB.
