# slide-puzzle-solver

Live: https://slide-puzzle-solver.vercel.app/

Reads a photographed 3x3 or 4x4 sliding-tile puzzle (numbers printed on tiles, or pieces of one
picture with no numbers; one empty slot) and solves it optimally. The reader and solver are pure TypeScript with no DOM, so the same code runs in
Node and in the browser; `src/app` is a mobile-first web app around them (Vite, no UI framework).

## Run

```
bash checks.sh               # npm ci, typecheck, lint, format check, tests, build, Python checks
                             # (ruff, mypy, pytest; venv in ~/.cache), browser e2e
bash scripts/ui-evidence.sh  # the browser e2e alone, listing its screenshots
npm run dev                  # app on http://localhost:5173 (camera needs HTTPS or localhost)
npm run build                # static site in dist/; `npm run preview` serves it
npm run e2e                  # builds, serves, and drives the app in Chromium (see below)
npm run build-pattern-db     # regenerate src/puzzle/pattern-db.bin.gz (~10 s, ~100 MB RAM)
npm run gen                  # 240 held-out synthetic photos -> data/eval (needs python3; builds a venv in ~/.cache)
npm run eval                 # real pipeline over data/eval -> prints metrics, writes data/eval-results.json
npm run eval:seeds           # generates (if missing) and evaluates seeds 20240601, 7, 99 -> per-seed table (~25 min, ~15 if generated)
npm run real:fetch           # downloads the real photos listed in real-photos.toml -> data/real (sha256-checked)
npm run real:eval            # writes their labels and evaluates them -> data/real-results.json
npm run train                # retrain the tile classifier -> src/vision/weights.json (minutes, CPU)
```

## How it works

- `src/puzzle`: board type, parity solvability, `checkReading` (permutation + confidence + parity;
  rejects with a specific reason), and optimal solvers. 3x3: IDA* with Manhattan + linear conflict
  (instant). 4x4: IDA* with an additive 6-6-3 pattern database (`pattern-db.ts`, `pattern-solver.ts`;
  the 2.4 MB table `pattern-db.bin.gz` is built by `npm run build-pattern-db`, and the second 6-tile
  group is its mirror image across the diagonal). If the optimal search passes a time budget
  (20 s) it falls back to weighted IDA\* and the result is labelled "not guaranteed shortest".
  Unsolvable boards return immediately.
- `src/app`: the web app. Two Web Workers keep the page responsive: `vision.worker.ts` runs
  `readBoard`; `solve.worker.ts` runs `solve`. Cancelling a solve terminates and restarts only the
  solver worker. `review.ts` re-runs `checkReading` on manual corrections and says which cells to blame.
  The page fetches the pattern table once at load and hands it to the solver worker, so reading and
  solving make no network requests (the e2e test checks this offline). The solver worker is bundled
  inline, so Cancel restarts it from a blob without a fetch and works offline.
- `src/vision`, `readBoard(rgbaImage, weights)`:
  1. `find-board.ts` thresholds brightness and colourfulness at many levels and proposes quads: board
     silhouettes, and clusters of tile-like blobs (for boards whose frame blends into the table). It
     looks at four views of the photo: grey and saturation, each also lighting-flattened (a box-blur
     background subtracted), because a gradient or a hard shadow edge breaks a board up in the plain
     views. `refine.ts` snaps silhouette sides to the strongest straight edge; the raw and the snapped
     quad are both read.
  2. Each candidate is rectified (`grid.ts`) and the tile grid is fitted per axis: a comb of N+1 evenly
     spaced lines slides over the horizontal and the vertical edge profile separately, so a board that is
     a few percent off on one side shifts the grid on that axis only. Each candidate is read as 3x3 and
     as 4x4.
  3. `classifier.ts` is a 150k-parameter CNN (pure-TS inference) over 16 classes: blank and the
     whole tile number 1..15. Weights are 1.5 MB JSON, bundled into the vision worker (not made smaller: ~0.5 MB gzipped is fine for a same-origin asset).
  4. Joint decoding (`decode.ts`): the cells of a board hold each of 0..N²-1 exactly once, with blank
     as a class like any tile, so instead of taking each cell's best class alone the reader solves an
     assignment problem (Hungarian algorithm) over the cells' class probabilities and picks the
     whole-board reading with the highest total log-likelihood. A cell's confidence is how much likelier
     that board is than the best board that puts a different value in that cell (a logistic of the
     log-likelihood gap), so it is the weakest link of the permutation, not one softmax output; a tie is 0.5.
  5. The best reading wins: consistent readings rank before inconsistent ones, then by mean confidence,
     and the search stops early at a consistent reading with mean confidence 0.9. The winning candidate is
     then re-read with seven crops per cell (centred, two scales, four shifts; class log-probabilities
     averaged) for both grid sizes, because a plain read may have picked the wrong size. That is 3x3
     versus 4x4: whichever of the two re-reads is a consistent permutation, and if both or neither are,
     the higher mean confidence.
  6. `checkReading` accepts the result only if every cell's confidence is at least `MIN_CELL_CONFIDENCE`
     = 0.95, the cells are a permutation, and the permutation is solvable. The threshold was chosen on two
     development sets (seeds 4242 and 31337), never on the report seeds or on real photos. Solvability is
     checked after decoding and is never a constraint on it: forcing it would turn a genuinely swapped
     board into a confident wrong answer. Anything else is rejected with a reason; low-confidence cells
     are never accepted.
- No vision library: OpenCV.js is ~8-10 MB, too heavy for a phone page; the pipeline is ~1500 lines of TypeScript.
- Type-check: `tsconfig.core.json` checks `src/vision` and `src/puzzle` without the DOM lib (part of
  `npm run typecheck`), so browser APIs cannot creep into the core that Node and the workers share.
- Training data (`scripts/synth.py`) is rendered tiles in 41 open-licence fonts (`fonts/train`; licence and copyright notices in `fonts/OFL.txt` and `fonts/COPYRIGHTS.md`) with
  per-board weight, slant, stretch, old-style digit offsets, blur, noise, lighting and JPEG, plus the
  real-photo styles below.

### Synthetic realism (`synth.py`)

The generator is shared by `gen_photos.py` (eval photos), `train_classifier.py` (training crops) and
`npm run gen`, seeded and deterministic. Besides flat printed digits it draws, per board unless noted:

- digit styles (`DIGIT_STYLES`, weighted): flat 40%, embossed and debossed (light and dark offset edges),
  engraved outline, engraved with a thin inner shadow, metallic foil (gradient fill in gold, silver or
  bronze, a highlight band, a dark rim) and low contrast (digit 18-40 luminance levels from its tile);
- tile surface: glossy gradient (35% of boards), specular blobs or streaks anywhere on a tile so they can
  cross a digit (on glossy and metal boards), soft drop shadows between tiles (40%);
- board base texture: flat, wood grain, brushed metal or plastic mottle (frame and empty slot);
- per photo: uneven lighting with a hard shadow edge (20%), barrel lens distortion (30%; the labelled
  corners are moved with it), and a partial thumb at the board's outer edge (12%) that reaches at most
  0.3 of a tile past the frame, so no tile is ever covered whole.

The eval split still uses only `fonts/eval`; none of this touches that separation. Generation takes about
0.9 s per 960x720 photo and a training board about 0.7 s.
The committed weights were trained on these styles (see "Training provenance").

Training also jitters each crop: gamma, contrast and brightness, an occasional saturating highlight
(`synth.augment_grey_window`) and a white-balance cast before the grey conversion.

### Picture-tile puzzles (`gen_picture_photos.py`)

Picture photos reuse the same camera model (perspective, blur, glare, noise, JPEG) on a board whose
tiles are cut from an openly licensed painting or photo listed in `picture-sources.toml` (6 dev
pictures for tuning, 12 held out for the eval; `npm run pictures:fetch` downloads and checks them). The
empty slot's home is bottom-right on 70% of boards, another corner on 18%, anywhere on the rest; boards
are scrambled by a random walk of the slot, so every one is solvable.

## Picture puzzles

Some puzzles have no numbers: the tiles carry pieces of one picture. The app handles them too.

1. **Which kind?** The numbered reader runs first. Its reading is kept as a numbered board if it was
   accepted, or if it was rejected but at least a third of the cells were read with confidence 0.95
   (`MIN_CONFIDENT_SHARE` in `src/app/routing.ts`; measured: a picture puzzle reads 1 of 16 cells
   confidently, the numbered low-confidence fixture 4 of 9). Otherwise the picture reader runs. If it
   finds no board either, the numbered reader's message is shown. Both review screens have a button
   ("Treat as picture puzzle" / "Treat as numbered puzzle") that re-reads the same photo the other way.
2. **Reading the board** (`src/picture/read.ts`): the board is found as for numbered puzzles, the tile
   grid is fitted, and every cell is cut to a 64x64 colour crop. The empty slot is the cell that looks
   most like the frame.
3. **Assembly** (`src/picture/assemble.ts`): the crops are arranged into one picture by how well their
   edges continue into each other. Sliding puzzles only reach half of all arrangements, so the result is
   checked for parity (`checkArrangement`); if it is unreachable and two tiles look the same
   (near-identical crops), those two are exchanged, which cannot be seen; the screen says so and outlines both tiles.
4. **Confirming**: the review screen shows the proposed solved picture and, below it, the board as
   photographed, every tile drawn from the photo with its number: the number is the tile's place in the
   solved picture (1, 2, 3, ... in reading order, skipping the empty slot's home). Tiles the assembler
   is unsure of (confidence below 0.15, `LOW_TILE_CONFIDENCE` in `src/app/picture-review.ts`) have a
   dashed outline. Tap one tile of the solved picture, then another, to swap them (buttons, so the
   keyboard works too); tapping the empty square and a tile moves where the empty slot ends up. The
   arrangement is re-checked after every swap and the reason shown if it cannot be solved.
5. **Solving** uses the same solver worker; moves are named by tile number ("Move 5 right") and the
   board shows the photo's tiles with their numbers.
6. **Saved puzzles**: after a solve, the puzzle's tiles (16x16 thumbnails, about 15 KB for 4x4, at most
   five puzzles) are saved in `localStorage` on this device only. A later picture photo is matched
   against them first (`matchProfile`, which copes with a new scramble and different light); a match
   skips assembly and says "Recognised your saved puzzle", with "Not this puzzle" to assemble from
   scratch. "Forget saved puzzles" deletes them. If the browser blocks or fills the storage, older
   puzzles are dropped or nothing is saved, and the screen says so.

Reading and assembly run in the vision worker, not on the page. The solver handles an empty slot whose
home is anywhere; on a 4x4 board with the home in a corner it uses the pattern table and is optimal,
with the home on an edge or in the middle it uses plain IDA\* within the 20 s budget and may label the
answer "not guaranteed shortest". Limits: tiles must really be
pieces of one picture with some detail (flat-coloured tiles cannot be told apart, and the parity check
may then blame the wrong tile, so check the proposal against your puzzle); a tile that looks like
a neighbour's continuation can be misplaced, which is why the proposal is shown for confirmation
rather than solved silently; the owner's photo in `test/fixtures/owner-picture` is test data and is
not part of the built site.

## Deploy

The site is a static build, deployed by Vercel from `main` to https://slide-puzzle-solver.vercel.app/.
Vercel settings: Framework preset Vite, install command `npm ci`, build command `npm run build`, output
directory `dist`, Node 22 (`engines` in `package.json`). Nothing else is needed: no Python and no venv,
because the pattern table (`src/puzzle/pattern-db.bin.gz`) and the classifier weights
(`src/vision/weights.json`) are committed and bundled by Vite. Assets load by relative URL (`base: "./"`),
so `dist/` also works from any path on a static host. No `vercel.json` is needed (one page, no routes).
`npx tsx scripts/check-dist.ts` (run by `checks.sh` after the build) fails if the pattern table or the
weights are missing from `dist/` or if anything from `test/` was bundled.

## Measured accuracy

Held-out eval sets are synthetic photos from `npm run gen` (seed 20240601) and `npm run eval:seeds`
(seeds 20240601, 7 and 99), 240 photos each, half 3x3 and half 4x4, about 8% unsolvable. Eval photos use
only the four fonts in `fonts/eval` (Comic Neue Bold, Playfair Display old-style digits, Slabo 27px,
Josefin Sans), which no training code reads. Nothing was tuned on seeds 7 or 99.

Run on 2026-11-22 (`rm -rf data/eval-seed* && npm run eval:seeds`, Node v22.23.3,
`nproc` 4, seeds evaluated one after another; table pasted as printed):

| seed     | images | exact-read | false-accept | unsolvable rejected | detection | frame-tight | grid-size | per-cell | mean ms/image |
| -------- | ------ | ---------- | ------------ | ------------------- | --------- | ----------- | --------- | -------- | ------------- |
| 20240601 | 240    | 90.9%      | 0.0%         | 21/21               | 93.8%     | 67.9%       | 99.2%     | 97.4%    | 1096          |
| 7        | 240    | 93.2%      | 0.0%         | 18/18               | 97.5%     | 76.7%       | 99.6%     | 98.8%    | 1161          |
| 99       | 240    | 90.0%      | 0.0%         | 20/20               | 93.3%     | 64.2%       | 97.9%     | 96.5%    | 1253          |

The generator is seeded, yet its photos are not byte-identical across machines (most likely
floating-point library routines that differ by CPU), so the figures differ slightly between machines. A reviewer's run of the same command on another machine got
20240601: 90.4% / 0.0% / 21/21 / 98.8% / 97.3% / 2102 ms; 7: 93.7% / 0.0% / 18/18 / 99.6% / 98.2% /
1547 ms; 99: 90.5% / 0.0% / 20/20 / 97.9% / 96.5% / 1420 ms (exact-read / false-accept / unsolvable
rejected / grid-size / per-cell / mean ms). Expect exact-read within about a point of the table, and no
false accepts, on any machine. Disabling numpy's AVX/AVX2/FMA paths (`NPY_DISABLE_CPU_FEATURES`) did not
change the photos here, so it is not used.

Exact-read means a solvable board was read exactly and accepted. False-accept means accepted but wrong.
The remaining misses are almost all rejections (low confidence on one or two cells, which the user can
confirm with one tap each, or a wrong grid size), not wrong boards: no seed accepted a wrong reading. The
weights, the 0.95 threshold and every pipeline parameter were fixed before these seeds were run.

Detection rate: a detection is a found quad whose every corner is within 12% of the board side of
the labelled outer-frame corner. The generator's frame margin is at most 8% of the edge, so a quad
hugging the tile grid is at most 11.3% away; the tolerance admits frame quads and tile-cluster quads
but not a quad that is too big, too small or on the wrong object. "Frame-tight" is the stricter count of
quads whose every corner is within 5% of the board side, i.e. quads on the outer frame itself.

### Confirming uncertain cells

When the only reason a board is rejected is low confidence, i.e. the best guesses already form a solvable
permutation, the review screen highlights every unsure cell (dashed outline, guess shown on the board)
and lists each one as "Check this cell: row r, column c. Best guess: N" with a **Yes, it is N** button and
a **Change** button (opens the number pad). Tapping Yes keeps the guess and sets that cell's confidence to
1; the consistency check re-runs, and Solve is enabled once no unsure cell is left. A confirmed or
corrected cell counts as confidence 1 (`review.ts`: `confirmCell`, `setCell`). Nothing is ever accepted
without the tap, and if the guesses are not a valid permutation (a duplicate, two blanks, unsolvable) no
confirm button is offered: the real reason is shown and cells are fixed with the number pad as before.
The e2e fixture `test/fixtures/photo-lowconf.jpg` is a generated 3x3 photo (`gen_photos.py --split eval
--count 120 --seed 5151`, image 44) that the committed weights read with every cell right but the weakest at
confidence 0.53, so it is rejected only for low confidence. Keep that margin well below 0.95: the browser
decodes JPEGs with its own decoder, which differs from the `jpeg-js` used in Node by up to ~17 levels per
channel, and an earlier fixture at 0.885 read 0.956 (accepted) in Chromium. `test/app-parity.test.ts` runs
what the vision worker and the review screen run on the fixture and checks the status, the flagged cells and
that the weakest cell is at least 0.05 under the threshold. After a retrain or a pipeline change, if it
fails, print `readBoard` results for generated photos (`gen_photos.py --count 120 --out /tmp/x`), pick a 3x3
photo whose cells are all correct, rejected only for low confidence, with a weakest cell under about 0.85, and
replace the fixture and its label (under 200 KB; run prettier on the json).

### Real photos

`real-photos.toml` lists real photographs (openly licensed, from Wikimedia Commons) with hand-written
labels; `npm run real:fetch` downloads them and `npm run real:eval` evaluates them with `scripts/eval.ts`
(large photos are downscaled with the app's 1280 px rule, `MAX_PHOTO_SIDE`). The photos are a held-out set:
nothing was tuned on them. Labels have no corner coordinates, so detection rate is "n/a".

**This set is far too small to measure anything.** Commons holds few photographs of numbered sliding
puzzles; of about 40 candidates only 4 show a whole board whose every tile I could read with certainty
(loose tiles, boxes, drawings, screenshots and one giant wooden board where 6 and 9 were ambiguous were
left out). Flickr/Openverse searches added nothing usable. Photos 1 and 2 carry Commons' plain
"Attribution" tag, not a named CC licence; drop them if that is not acceptable.

Run on 2026-11-22 (Node 22, `npm run real:eval`):

| photos | solvable | exact-read | false-accept | unsolvable rejected | grid-size | per-cell | mean ms/image |
| ------ | -------- | ---------- | ------------ | ------------------- | --------- | -------- | ------------- |
| 4      | 3        | 2/3        | 0/4          | 1/1                 | 4/4       | 100%     | 1286          |

Every board was found and read as 4x4 with all 64 cells right. The colour-column board and the solved
red-white board were accepted; the keychain (shiny embossed metal digits) was rejected for low confidence on
one cell (0.82 for the 3, which the user would confirm with one tap); the red-white board with 14 and 15
swapped was rejected as unsolvable, for the right reason. Four photos show the pipeline is not limited to the
generator, but they are not a measurement.

To measure real-photo accuracy properly, collect 30+ labelled photos of the target puzzle under
different lighting, evaluate them with `npx tsx scripts/eval.ts <dir> <dir>/results.json`, and re-check the
confidence threshold on one half before reporting the other.

### Picture puzzles

`npm run pictures:fetch && npm run gen:picture && npm run eval:picture`: 144 synthetic photos (seed
20261122, half 3x3 and half 4x4) of the held-out openly licensed paintings and photos in
`picture-sources.toml` (licence and source per picture), cut into tiles, scrambled by sliding and
photographed with perspective, blur, glare and JPEG noise. The main table measures **assembly alone**:
the crops are cut from the labelled tile area and the labelled empty slot is given, so it says nothing
about finding the board in the photo. "exact" is the whole solved layout right, "neighb" the share of
right neighbour pairs, "home" the empty slot's home right. Run on 2026-11-23, Node v22.23.3:

| set | n   | exact | neighb | home  | mean ms |
| --- | --- | ----- | ------ | ----- | ------- |
| 3x3 | 72  | 43.1% | 70.3%  | 51.4% | 96      |
| 4x4 | 72  | 45.8% | 74.0%  | 56.9% | 580     |
| all | 144 | 44.4% | 72.9%  | 54.2% | 338     |

The same run also measures the app's **real path**, `readPictureBoard` then `assemblePicture`, with
nothing labelled given (board, grid size and empty slot all found from the photo; time is both steps).
"exact" is the whole solved layout right, "size" the grid size found right, "slot" the empty slot found
at the right cell. In 10 of the 144 photos no board was found at all; they count as wrong everywhere.

| set | n   | exact | size  | slot  | mean ms |
| --- | --- | ----- | ----- | ----- | ------- |
| 3x3 | 72  | 20.8% | 87.5% | 50.0% | 649     |
| 4x4 | 72  | 16.7% | 97.2% | 38.9% | 1141    |
| all | 144 | 18.8% | 92.4% | 44.4% | 895     |

So 64 of 144 photos (44%) come out right from the labelled area with no help, and 27 of 144 (19%) when
the app has to find the board itself. Of the other 80 in the assembly-only run (`data/picture-eval-results.json`),
18 are one or two swaps from right (the review screen's edit: exchange two solved positions); 13 are three or four;
49 need five or more, and in 66 of the 80 the empty slot's home is wrong too, so fixing one on the review
screen often means rearranging much of the picture. The assembly constants were chosen on the owner's photo and its decode variants (the tests in
`test/picture-*.test.ts`) and checked against this set so as not to lose accuracy on it; it was not a
fully blind held-out set for those choices. The owner's 4x4 photo (`test/fixtures/owner-picture`) is
read and assembled exactly, from Node's decode and from Chromium's at 960x1280 (the size the app uses), and solved in 24
moves; two of its tiles (solved places 7 and 11) are a near tie and are outlined for checking.

## Solver speed (Node 22, one core)

Measured by `test/pattern-solver.test.ts`: Korf's first ten instances (optimal 46-59 moves) take
2 ms to 1.0 s each; ten seeded uniform random states take 3 ms to 0.9 s (median under 0.1 s). Optimal
results match published optima. The hardest known states (Korf #88, h = 62) do not finish optimally in
minutes; after the budget the weighted fallback returns 88 moves in about 50 ms. The
Manhattan + linear-conflict search (3x3, and 4x4 without the table) takes ~30 s on some 60-move states.

## Browser e2e

`npm run e2e` (Playwright, pinned) builds the app, serves it with `vite preview`, and at 390x844
uploads committed fixtures (3x3, 4x4, one unsolvable photo, one low-confidence photo, and, in
`e2e/picture.e2e.ts`, a picture puzzle: picture review, swapping tiles, solving, recognising a saved puzzle,
switching reading mode), checks the board shown,
the first move and move count of the solution, the rejection reason and highlighted cells, manual correction,
confirming or changing each unsure cell, a fake camera capture (a fixture served as the camera's video), a
denied camera, tapping "Use camera" twice (one stream, every track ended on Close), cancelling a long solve
offline, and a new upload or a cell correction during a solve (the old solve is dropped and never shown). Waits for a photo to be read use a 30 s timeout (a read takes
1-3 s, a rejected one 2-3x that), not Playwright's 5 s default.
Every test fails on console errors; the read/solve tests go offline after load and fail on any request.
Screenshots land in `e2e-artifacts/` (gitignored). It needs Chromium and its system libraries
(`npx playwright install --with-deps chromium`, root). `checks.sh` runs it and fails when Chromium cannot
be launched; `bash scripts/ui-evidence.sh` runs the same suite on its own.
`npx playwright test --list` shows every test.

## Limitations

- Real photos are barely tested: 4 photos (see "Real photos"), 2 of 3 solvable boards accepted, 0 wrong
  boards accepted. The generator imitates embossed, engraved and foil digits, glare, wood grain and thumbs
  and the weights are trained on it, but real digits can still differ from it; uncertain cells go to the
  user to confirm.
- Accuracy figures come from Node, which decodes JPEGs with `jpeg-js`; the browser uses its own decoder,
  which differs by a few levels per pixel, so a cell near the 0.95 threshold can land on the other side in
  the app. That only moves a cell between "accepted" and "confirm with one tap".
- Board must be roughly upright (up to ~20-45 degrees); sideways photos are not tried.
- The board must contrast with its surroundings, or its tiles with its frame; some clutter touching
  the board or very low contrast still fail (the read is then rejected, not guessed).
- Training is seeded but torch results may differ across CPUs/thread counts; committed weights are
  the reference.

## Training provenance

`src/vision/weights.json` comes from `npm run train` with its defaults: seed 7, 9000 synthetic boards
from the 41 fonts in `fonts/train`, 14 epochs, no `--extra` crop sets. `npm run train` writes
`src/vision/weights.meta.json` next to the weights (`--out`'s name with a `.meta.json` suffix): the exact
command line, seed, boards, epochs, `--extra` sets, validation accuracy, time and library versions. That file
is the record of the committed weights. Torch results may differ across CPUs and thread counts, so a
retrain is not guaranteed to reproduce the file byte for byte; the committed file is the reference. No
eval-split photo or font was used: the training code reads `fonts/train` only.

`--extra` takes `.npz` crop sets (arrays `x`, `y`, `fonts`, `source`) for hard examples.
`synth.check_training_crops` refuses a set that lacks `fonts` or `source`, names a font from
`fonts/eval` or any font not in `fonts/train`, has `eval` in its `source`, or lives under `data/eval*`.
An `.npz` holds only pixels, so this is a declaration check: crops cut from eval photos must not be
declared as anything else.
