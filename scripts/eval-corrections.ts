// Measures how much the manual corrections would fix on the three failure-mode photo sets of
// scripts/gen_correction_photos.py (angled-blank, white-frame-thumbnail, line-art). Per photo:
//  - auto: readPictureBoard + assemblePicture, nothing given;
//  - blank only: the automatic reading, but with the empty slot given (assemblePicture with the
//    labelled blank: what the "Change empty slot" chooser alone gives);
//  - corners: the app's corner path, readPictureBoard(image, { quad, size }) (which fits the grid in
//    the quad and snaps it to the seams) then assemblePicture, the empty slot still found by the app;
//    with the labelled corners and with them moved by up to 3% of the side (as a finger would place them);
//  - raw quad (no snap): the same moved corners cut as placed (readPictureCells), for comparison;
//  - corners + blank: the corner path with the empty slot given as well (exact and moved);
//  - reference (photos with a printed solved picture): the cells of the corner path (labelled corners),
//    the labelled empty slot and the print's corners given (exact and moved by up to 3% of the print's
//    side), tiles matched to the print (reference.ts).
// "exact" is the whole solved layout right, which includes the empty slot's home and (since the
// empty-slot cell is not in the label's layout) the right cell taken as the slot.
// Usage: tsx scripts/eval-corrections.ts [dir] [out.json] [every-nth: evaluate a subset while iterating]
import { mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { pathToFileURL } from "node:url";
import { checkArrangement } from "../src/picture/arrangement.ts";
import { assemblePicture } from "../src/picture/assemble.ts";
import { cutTileCrops, readPictureBoard, readPictureCells } from "../src/picture/read.ts";
import { matchReference } from "../src/picture/reference.ts";
import { type Arrangement, type TileCrop } from "../src/picture/types.ts";
import { type GridSize } from "../src/puzzle/board.ts";
import { type Quad } from "../src/vision/geometry.ts";
import { type RgbaImage } from "../src/vision/image.ts";
import { parsePictureLabel, type PictureLabel } from "./labels.ts";
import { createRandom, jitterQuad, measureSide } from "./random.ts";
import { DETECTION_TOLERANCE } from "./eval.ts";
import { readPhoto } from "./read-photo.ts";

/** Pass-bar thresholds (see formatBars); the bar text is printed from the same numbers. */
const BAR1_MIN_RATE = 70;
const BAR_MIN_GAIN = 25;
const BAR3_MAX_GAP = 10;
const JITTER = 0.03;
const SETS = ["angled-blank", "white-frame-thumbnail", "line-art"] as const;
/** Mean-confidence cut-offs tried for the reference matcher (see docs/reference-picture.md). */
const CONFIDENCE_CUTS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6];

interface ReferenceOutcome {
  exact: boolean;
  /** Mean of the matcher's per-position confidences. */
  confidence: number;
}

/** Assembly exact from one tile-area quad, with the empty slot found by the app and with it given. */
interface QuadOutcome {
  foundBlank: boolean;
  givenBlank: boolean;
}

export interface Outcome {
  set: string;
  name: string;
  size: GridSize;
  autoExact: boolean;
  detectionRight: boolean;
  blankRight: boolean;
  /** Automatic corners, labelled empty slot. */
  blankOnly: boolean;
  corners: QuadOutcome;
  cornersJittered: QuadOutcome;
  /** The jittered corners cut as placed, no grid fit or snap (not the app's path). */
  rawJittered: QuadOutcome;
  /** Absent for sets with no printed solved picture. */
  reference?: ReferenceOutcome;
  referenceJittered?: ReferenceOutcome;
}

function hashText(text: string): number {
  let hash = 2166136261;
  for (let i = 0; i < text.length; i++) hash = Math.imul(hash ^ text.charCodeAt(i), 16777619);
  return hash;
}

/** Largest distance between a found and the labelled corner, as a share of the labelled quad's side. */
function measureWorstCorner(found: Quad, truth: Quad): number {
  const side = measureSide(truth);
  return Math.max(
    ...truth.map((p, i) => Math.hypot(p.x - (found[i]?.x ?? 0), p.y - (found[i]?.y ?? 0)) / side),
  );
}

const isSameLayout = (found: readonly number[], truth: readonly number[]): boolean =>
  found.every((cell, position) => cell === truth[position]);

/**
 * What the app does with an arrangement before the owner sees it: a layout sliding cannot reach is
 * fixed by exchanging two look-alike tiles, or counts as wrong.
 */
function isExactAfterReview(
  size: GridSize,
  cells: TileCrop[],
  blank: number,
  arrangement: Arrangement,
  label: PictureLabel,
): boolean {
  const checked = checkArrangement(
    { size, quad: label.tilesQuad, cells, blank, blankConfidence: 1 },
    arrangement,
  );
  return (
    (checked.status === "ok" || checked.status === "fixed") &&
    isSameLayout(checked.arrangement.solved, label.solved)
  );
}

function evaluateAuto(
  image: RgbaImage,
  label: PictureLabel,
): Pick<Outcome, "autoExact" | "detectionRight" | "blankRight" | "blankOnly"> {
  const reading = readPictureBoard(image);
  if ("status" in reading) {
    return { autoExact: false, detectionRight: false, blankRight: false, blankOnly: false };
  }
  const found = assemblePicture(reading.cells, reading.size, reading.blank);
  const sizeRight = reading.size === label.size;
  return {
    autoExact: sizeRight && isSameLayout(found.solved, label.solved),
    // A wrong grid size makes the labelled slot meaningless (it may not even be a cell).
    blankOnly:
      sizeRight &&
      isSameLayout(assemblePicture(reading.cells, reading.size, label.blank).solved, label.solved),
    detectionRight:
      sizeRight && measureWorstCorner(reading.quad, label.tilesQuad) <= DETECTION_TOLERANCE,
    blankRight: sizeRight && reading.blank === label.blank,
  };
}

/** The app's path from user corners: readPictureBoard with { quad, size }. */
function evaluateQuad(image: RgbaImage, label: PictureLabel, quad: Quad): QuadOutcome {
  const reading = readPictureBoard(image, { quad, size: label.size });
  if ("status" in reading) return { foundBlank: false, givenBlank: false };
  return scoreCells(label, reading.cells, reading.blank);
}

/** Cuts `quad` exactly as placed. */
function evaluateRawQuad(image: RgbaImage, label: PictureLabel, quad: Quad): QuadOutcome {
  const { cells, blank } = readPictureCells(image, quad, label.size);
  return scoreCells(label, cells, blank);
}

function scoreCells(label: PictureLabel, cells: TileCrop[], blank: number): QuadOutcome {
  const isExactAt = (slot: number): boolean =>
    isSameLayout(assemblePicture(cells, label.size, slot).solved, label.solved);
  const givenBlank = isExactAt(label.blank);
  return { givenBlank, foundBlank: blank === label.blank ? givenBlank : isExactAt(blank) };
}

function evaluateReference(
  image: RgbaImage,
  label: PictureLabel,
  thumbQuad: Quad,
): ReferenceOutcome {
  const reading = readPictureBoard(image, { quad: label.tilesQuad, size: label.size });
  if ("status" in reading) return { exact: false, confidence: 0 };
  const cells = reading.cells;
  const reference = cutTileCrops(image, thumbQuad, label.size);
  const arrangement = matchReference(cells, label.blank, reference);
  return {
    exact: isExactAfterReview(label.size, cells, label.blank, arrangement, label),
    confidence: arrangement.confidence.reduce((a, b) => a + b, 0) / arrangement.confidence.length,
  };
}

function evaluatePhoto(dir: string, set: string, name: string): Outcome {
  const label = parsePictureLabel(
    JSON.parse(readFileSync(join(dir, `${name}.json`), "utf8")),
    name,
  );
  const image = readPhoto(join(dir, `${name}.jpg`));
  const random = createRandom(hashText(`${set}/${name}`));
  const jittered = jitterQuad(label.tilesQuad, JITTER, random);
  const outcome: Outcome = {
    set,
    name,
    size: label.size,
    ...evaluateAuto(image, label),
    corners: evaluateQuad(image, label, label.tilesQuad),
    cornersJittered: evaluateQuad(image, label, jittered),
    rawJittered: evaluateRawQuad(image, label, jittered),
  };
  if (label.thumbQuad) {
    outcome.reference = evaluateReference(image, label, label.thumbQuad);
    outcome.referenceJittered = evaluateReference(
      image,
      label,
      jitterQuad(label.thumbQuad, JITTER, random),
    );
  }
  return outcome;
}

const formatPercent = (part: number, whole: number): string =>
  whole === 0 ? "-" : `${((100 * part) / whole).toFixed(1)}%`;

type Row = [string, GridSize | "all", Outcome[]];

/** Rows of a set: 3x3, 4x4 and both. */
function listRows(outcomes: Outcome[], sets: readonly string[]): Row[] {
  return sets.flatMap((set) => {
    const inSet = outcomes.filter((o) => o.set === set);
    if (inSet.length === 0) return [];
    return [
      [set, 3, inSet.filter((o) => o.size === 3)],
      [set, 4, inSet.filter((o) => o.size === 4)],
      [set, "all", inSet],
    ] satisfies Row[];
  });
}

function formatMarkdown(header: string[], rows: string[][]): string {
  return [header, header.map(() => "---"), ...rows]
    .map((row) => `| ${row.join(" | ")} |`)
    .join("\n");
}

const formatShare = (rowOutcomes: Outcome[], pick: (o: Outcome) => boolean | undefined): string =>
  formatPercent(rowOutcomes.filter((o) => pick(o) === true).length, rowOutcomes.length);

function formatTable(outcomes: Outcome[]): string {
  const rows = listRows(outcomes, SETS).map(([set, grid, group]) => {
    const formatGroupShare = (pick: (o: Outcome) => boolean): string => formatShare(group, pick);
    return [
      set,
      `${String(grid)}${grid === "all" ? "" : `x${String(grid)}`}`,
      String(group.length),
      formatGroupShare((o) => o.autoExact),
      formatGroupShare((o) => o.detectionRight),
      formatGroupShare((o) => o.blankRight),
      formatGroupShare((o) => o.blankOnly),
      formatGroupShare((o) => o.corners.foundBlank),
      formatGroupShare((o) => o.cornersJittered.foundBlank),
      formatGroupShare((o) => o.rawJittered.foundBlank),
      formatGroupShare((o) => o.corners.givenBlank),
      formatGroupShare((o) => o.cornersJittered.givenBlank),
    ];
  });
  return formatMarkdown(
    [
      "set",
      "grid",
      "n",
      "auto exact",
      "detection",
      "blank found",
      "blank only",
      "corners",
      "corners ±3%",
      "raw ±3% (no snap)",
      "corners+blank",
      "corners+blank ±3%",
    ],
    rows,
  );
}

/** Photos with a printed solved picture: reference matching against edge assembly, same given corners and blank. */
function formatReferenceTable(outcomes: Outcome[]): string {
  const rows = listRows(outcomes, SETS)
    .filter(([, , group]) => group.some((o) => o.reference))
    .map(([set, grid, group]) => [
      set,
      `${String(grid)}${grid === "all" ? "" : `x${String(grid)}`}`,
      String(group.length),
      formatShare(group, (o) => o.corners.givenBlank),
      formatShare(group, (o) => o.reference?.exact),
      formatShare(group, (o) => o.referenceJittered?.exact),
    ]);
  return formatMarkdown(
    ["set", "grid", "n", "edge assembly", "reference", "reference, print's corners ±3%"],
    rows,
  );
}

/** Per set with a reference: how often it is right at or above each mean-confidence cut, and edge assembly on the same photos. */
function formatConfidenceTable(outcomes: Outcome[]): string {
  const rows: string[][] = [];
  for (const set of SETS) {
    const inSet = outcomes.filter((o) => o.set === set && o.reference);
    for (const cut of inSet.length === 0 ? [] : CONFIDENCE_CUTS) {
      const kept = inSet.filter((o) => (o.reference?.confidence ?? 0) >= cut);
      rows.push([
        set,
        String(cut),
        `${String(kept.length)}/${String(inSet.length)}`,
        formatShare(kept, (o) => o.reference?.exact),
        formatShare(kept, (o) => o.corners.givenBlank),
      ]);
    }
  }
  return formatMarkdown(
    [
      "set",
      "min mean confidence",
      "photos kept",
      "reference exact",
      "edge assembly exact (same photos)",
    ],
    rows,
  );
}

/**
 * The pass bars, fixed before any held-out run (both grid sizes pooled):
 *  1. white-frame-thumbnail: exact with the labelled corners (empty slot found by the app) >= 70%
 *     and >= auto + 25 points;
 *  2. angled-blank: exact with the labelled empty slot (automatic corners, or labelled corners)
 *     >= auto + 25 points;
 *  3. every set: exact with 3%-jittered corners within 10 points of exact with labelled corners
 *     (empty slot found by the app; the same is printed with the slot given).
 */
export function formatBars(outcomes: Outcome[]): string {
  const computeRate = (group: Outcome[], pick: (o: Outcome) => boolean): number =>
    group.length === 0 ? 0 : (100 * group.filter(pick).length) / group.length;
  const formatVerdict = (pass: boolean): string => (pass ? "PASS" : "FAIL");
  const listOutcomesOfSet = (set: string): Outcome[] => outcomes.filter((o) => o.set === set);
  const lines: string[] = [];
  const white = listOutcomesOfSet("white-frame-thumbnail");
  const whiteAuto = computeRate(white, (o) => o.autoExact);
  const whiteCorners = computeRate(white, (o) => o.corners.foundBlank);
  lines.push(
    `1. white-frame-thumbnail corners ${whiteCorners.toFixed(1)}% (>= ${String(BAR1_MIN_RATE)} and >= auto ${whiteAuto.toFixed(1)} + ${String(BAR_MIN_GAIN)}): ${formatVerdict(whiteCorners >= BAR1_MIN_RATE && whiteCorners >= whiteAuto + BAR_MIN_GAIN)}`,
  );
  const angled = listOutcomesOfSet("angled-blank");
  const angledAuto = computeRate(angled, (o) => o.autoExact);
  const blankOnly = computeRate(angled, (o) => o.blankOnly);
  const cornersBlank = computeRate(angled, (o) => o.corners.givenBlank);
  lines.push(
    `2. angled-blank blank only ${blankOnly.toFixed(1)}%, corners+blank ${cornersBlank.toFixed(1)}% (best >= auto ${angledAuto.toFixed(1)} + ${String(BAR_MIN_GAIN)}): ${formatVerdict(Math.max(blankOnly, cornersBlank) >= angledAuto + BAR_MIN_GAIN)}`,
  );
  for (const set of SETS) {
    const group = listOutcomesOfSet(set);
    const gap =
      computeRate(group, (o) => o.corners.foundBlank) -
      computeRate(group, (o) => o.cornersJittered.foundBlank);
    const gapGiven =
      computeRate(group, (o) => o.corners.givenBlank) -
      computeRate(group, (o) => o.cornersJittered.givenBlank);
    lines.push(
      `3. ${set} jitter gap ${gap.toFixed(1)} points (slot given: ${gapGiven.toFixed(1)}) (within ${String(BAR3_MAX_GAP)}): ${formatVerdict(Math.abs(gap) <= BAR3_MAX_GAP)}`,
    );
  }
  return lines.join("\n");
}

/** The per-photo stride from the optional third argument: every `nth` photo of each set is evaluated. */
function parseStride(text: string): number {
  const stride = Number(text);
  if (!Number.isInteger(stride) || stride < 1) {
    throw new Error(
      `the third argument (every nth photo) must be a whole number >= 1, got "${text}"`,
    );
  }
  return stride;
}

/** What to record of how the run was invoked: the script and its arguments, and the per-photo stride. */
export function describeRun(
  argv: string[],
  stride: number,
  cwd: string = process.cwd(),
): { command: string; stride: number } {
  // tsx passes the script as an absolute path; the record keeps it relative so no machine's path is committed.
  const [, script = "", ...rest] = argv;
  return { command: [relative(cwd, script), ...rest].join(" "), stride };
}

function main(): void {
  const [dir = "data/corrections", out = "data/corrections-results.json", nth = "1"] =
    process.argv.slice(2);
  const stride = parseStride(nth);
  const outcomes: Outcome[] = [];
  const sets: { set: string; photos: number; elapsedMs: number; msPerPhoto: number }[] = [];
  for (const set of SETS) {
    const names = readdirSync(join(dir, set))
      .filter((file) => file.endsWith(".json"))
      .map((file) => file.slice(0, -".json".length))
      .sort()
      .filter((_, index) => index % stride === 0);
    if (names.length === 0) throw new Error(`no labelled photos in ${join(dir, set)}`);
    const started = performance.now();
    outcomes.push(...names.map((name) => evaluatePhoto(join(dir, set), set, name)));
    const elapsedMs = Math.round(performance.now() - started);
    sets.push({
      set,
      photos: names.length,
      elapsedMs,
      msPerPhoto: Math.round(elapsedMs / names.length),
    });
    console.error(`${set}: ${String(names.length)} photos in ${String(elapsedMs)} ms`);
  }
  console.log(formatTable(outcomes));
  console.log(`\n${formatReferenceTable(outcomes)}`);
  console.log(`\n${formatConfidenceTable(outcomes)}`);
  console.log(`\n${formatBars(outcomes)}`);
  mkdirSync(dirname(out), { recursive: true });
  const results = {
    date: new Date().toISOString(),
    node: process.version,
    ...describeRun(process.argv, stride),
    dir,
    jitter: { share: JITTER, rng: "createRandom(hashText(`${set}/${name}`)) per photo" },
    sets,
    outcomes,
  };
  writeFileSync(out, JSON.stringify(results, null, 1));
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) main();
