// Runs picture assembly over a labelled photo set (see scripts/gen_picture_photos.py) two ways:
//  - assembly alone: crops are cut from the LABELLED tile quad and the labelled empty slot is given;
//    reports the exact rate, neighbour accuracy, home-of-the-empty-slot accuracy, time, and how many
//    tile swaps (the review screen's edit) the result is from the truth;
//  - the app's real path: readPictureBoard finds board, grid size and empty slot in the photo, then
//    assemblePicture; reports whether each was right, and the time of both steps together.
// Usage: tsx scripts/eval-picture.ts [dir] [out.json] [every-nth: evaluate a subset while iterating]
import { mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { pathToFileURL } from "node:url";
import { type GridSize } from "../src/puzzle/board.ts";
import { assemblePicture } from "../src/picture/assemble.ts";
import { type Quad } from "../src/vision/geometry.ts";
import { type RgbaImage } from "../src/vision/image.ts";
import { cutTileCrops, readPictureBoard } from "../src/picture/read.ts";
import { readPhoto } from "./read-photo.ts";

interface PictureLabel {
  size: GridSize;
  blank: number;
  solved: number[];
  tilesQuad: Quad;
}

/** What the app's own path made of a photo, with no labels given. */
interface RealPathOutcome {
  /** False when no board was found at all; the other fields are then false too. */
  boardFound: boolean;
  sizeCorrect: boolean;
  blankCorrect: boolean;
  exact: boolean;
  millis: number;
}

interface Outcome {
  name: string;
  size: GridSize;
  exact: boolean;
  neighboursCorrect: number;
  neighbours: number;
  homeCorrect: boolean;
  millis: number;
  /** Fewest exchanges of two solved positions that turn the assembly-only result into the truth. */
  swapsToTruth: number;
  realPath: RealPathOutcome;
}

function parseLabel(raw: unknown, where: string): PictureLabel {
  if (typeof raw !== "object" || raw === null) throw new Error(`${where}: label must be an object`);
  const { size, blank, solved, tilesQuad } = raw as Record<string, unknown>;
  if (size !== 3 && size !== 4) throw new Error(`${where}: size must be 3 or 4`);
  if (typeof blank !== "number") throw new Error(`${where}: blank must be a number`);
  if (!Array.isArray(solved) || solved.length !== size * size)
    throw new Error(`${where}: solved must have ${String(size * size)} entries`);
  if (!Array.isArray(tilesQuad) || tilesQuad.length !== 4)
    throw new Error(`${where}: tilesQuad must have 4 corners`);
  // Hand-written or generated outside the type system: shape checked above, values trusted below.
  return { size, blank, solved: solved as number[], tilesQuad: tilesQuad as unknown as Quad };
}

/** Adjacent tile pairs of a layout as "a>b" (b right of a) and "a^b" (b below a); the home has none. */
function listNeighbours(solved: readonly number[], size: number): Set<string> {
  const pairs = new Set<string>();
  solved.forEach((tile, position) => {
    const right = solved[position + 1];
    const below = solved[position + size];
    if (tile < 0) return;
    if ((position + 1) % size !== 0 && right !== undefined && right >= 0)
      pairs.add(`${String(tile)}>${String(right)}`);
    if (below !== undefined && below >= 0) pairs.add(`${String(tile)}^${String(below)}`);
  });
  return pairs;
}

/** Fewest exchanges of two positions that turn `found` into `truth`: positions minus permutation cycles. */
function countSwaps(found: readonly number[], truth: readonly number[]): number {
  const positionOfTile = new Map(truth.map((tile, position) => [tile, position]));
  const target = found.map((tile) => positionOfTile.get(tile) ?? -1);
  if (target.includes(-1)) throw new Error("layouts do not hold the same tiles");
  const seen = new Set<number>();
  let cycles = 0;
  for (let start = 0; start < target.length; start++) {
    if (seen.has(start)) continue;
    cycles++;
    for (let at = start; !seen.has(at); at = target[at] ?? start) seen.add(at);
  }
  return target.length - cycles;
}

function evaluateRealPath(image: RgbaImage, label: PictureLabel): RealPathOutcome {
  const started = performance.now();
  const reading = readPictureBoard(image);
  if ("status" in reading) {
    return {
      boardFound: false,
      sizeCorrect: false,
      blankCorrect: false,
      exact: false,
      millis: performance.now() - started,
    };
  }
  const found = assemblePicture(reading.cells, reading.size, reading.blank);
  const millis = performance.now() - started;
  const sizeCorrect = reading.size === label.size;
  return {
    boardFound: true,
    sizeCorrect,
    blankCorrect: reading.blank === label.blank,
    // A different grid size cannot match the label, and comparing layouts of different lengths is moot.
    exact: sizeCorrect && found.solved.every((tile, position) => tile === label.solved[position]),
    millis,
  };
}

function evaluatePhoto(dir: string, name: string): Outcome {
  const label = parseLabel(JSON.parse(readFileSync(join(dir, `${name}.json`), "utf8")), name);
  const image = readPhoto(join(dir, `${name}.jpg`));
  const crops = cutTileCrops(image, label.tilesQuad, label.size);
  const started = performance.now();
  const found = assemblePicture(crops, label.size, label.blank);
  const millis = performance.now() - started;
  const truthPairs = listNeighbours(label.solved, label.size);
  const foundPairs = listNeighbours(found.solved, label.size);
  return {
    name,
    size: label.size,
    exact: found.solved.every((tile, position) => tile === label.solved[position]),
    neighboursCorrect: [...truthPairs].filter((pair) => foundPairs.has(pair)).length,
    neighbours: truthPairs.size,
    homeCorrect: found.solved.indexOf(-1) === label.solved.indexOf(-1),
    millis,
    swapsToTruth: countSwaps(found.solved, label.solved),
    realPath: evaluateRealPath(image, label),
  };
}

const percent = (part: number, whole: number): string =>
  `${((100 * part) / whole).toFixed(1).padStart(5)}%`;
const mean = (values: number[]): number => values.reduce((sum, v) => sum + v, 0) / values.length;

function formatAssemblyRow(label: string, outcomes: Outcome[]): string {
  const count = outcomes.length;
  const exact = outcomes.filter((o) => o.exact).length;
  const neighboursCorrect = outcomes.reduce((sum, o) => sum + o.neighboursCorrect, 0);
  const neighbours = outcomes.reduce((sum, o) => sum + o.neighbours, 0);
  const home = outcomes.filter((o) => o.homeCorrect).length;
  const meanMillis = mean(outcomes.map((o) => o.millis));
  return `${label.padEnd(8)} ${String(count).padStart(4)}  ${percent(exact, count)} ${percent(neighboursCorrect, neighbours)} ${percent(home, count)}  ${meanMillis.toFixed(0).padStart(6)}`;
}

function formatRealPathRow(label: string, outcomes: Outcome[]): string {
  const count = outcomes.length;
  const share = (pick: (r: RealPathOutcome) => boolean): string =>
    percent(outcomes.filter((o) => pick(o.realPath)).length, count);
  const meanMillis = mean(outcomes.map((o) => o.realPath.millis));
  return `${label.padEnd(8)} ${String(count).padStart(4)}  ${share((r) => r.exact)} ${share((r) => r.sizeCorrect)} ${share((r) => r.blankCorrect)}  ${meanMillis.toFixed(0).padStart(6)}`;
}

/** How many photos need 0, 1-2, 3-4 and 5+ swaps, and in how many of the rest the home is wrong too. */
function formatSwapSummary(outcomes: Outcome[]): string {
  const within = (low: number, high: number): number =>
    outcomes.filter((o) => o.swapsToTruth >= low && o.swapsToTruth <= high).length;
  const wrong = outcomes.filter((o) => !o.exact);
  const wrongHome = wrong.filter((o) => !o.homeCorrect).length;
  return `assembly alone, swaps to the truth: 0: ${String(within(0, 0))}, 1-2: ${String(within(1, 2))}, 3-4: ${String(within(3, 4))}, 5+: ${String(within(5, Infinity))}; of ${String(wrong.length)} not exact, ${String(wrongHome)} have the empty slot's home wrong`;
}

function main(): void {
  const [dir = "data/picture-eval", out = "data/picture-eval-results.json", nth = "1"] =
    process.argv.slice(2);
  const names = readdirSync(dir)
    .filter((file) => file.endsWith(".json"))
    .map((file) => file.slice(0, -".json".length))
    .sort()
    .filter((_, index) => index % Number(nth) === 0);
  if (names.length === 0) throw new Error(`${dir}: no labelled photos (npm run gen:picture)`);
  const outcomes = names.map((name) => evaluatePhoto(dir, name));
  const sets = [
    ...([3, 4] as const).map((size) => ({
      label: `${String(size)}x${String(size)}`,
      outcomes: outcomes.filter((o) => o.size === size),
    })),
    { label: "all", outcomes },
  ];
  console.log("assembly alone (labelled tile area and empty slot)");
  console.log("set      n     exact  neighb  home   mean ms");
  for (const { label, outcomes: rows } of sets) console.log(formatAssemblyRow(label, rows));
  console.log("\nreal path (readPictureBoard, then assemblePicture)");
  console.log("set      n     exact  size    slot   mean ms");
  for (const { label, outcomes: rows } of sets) console.log(formatRealPathRow(label, rows));
  console.log(`\n${formatSwapSummary(outcomes)}`);
  for (const o of outcomes.filter((o) => !o.exact))
    console.log(`  assembly not exact: ${o.name} (${String(o.swapsToTruth)} swaps)`);
  for (const o of outcomes.filter((o) => !o.realPath.exact))
    console.log(`  real path not exact: ${o.name}`);
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, JSON.stringify({ outcomes }, null, 1));
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) main();
