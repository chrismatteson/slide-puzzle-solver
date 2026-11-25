import { type Board, GRID_SIZES, type GridSize } from "../puzzle/board.ts";
import { type CellReading, type Rejection, checkReading } from "../puzzle/consistency.ts";
import { type ClassifierWeights } from "./classifier.ts";
import { decodePermutation } from "./decode.ts";
import { type BoardCandidate, findBoardCandidates } from "./find-board.ts";
import { type Point, type Quad, orderQuadCorners } from "./geometry.ts";
import { refineQuad } from "./refine.ts";
import {
  OUTLINE_INSET_RANGE,
  TILES_INSET_RANGE,
  findGrid,
  type GridLayout,
  AUGMENTED_CROPS,
  CENTRED_CROP,
  type CropVariant,
  readCellProbabilities,
  rectifyQuad,
} from "./grid.ts";
import {
  type GreyImage,
  type RgbaImage,
  flattenGrey,
  shrinkGrey,
  convertToGrey,
  convertToSaturation,
} from "./image.ts";

const DETECTION_SIDE = 320;
/** Box-blur radius (detection pixels) of the lighting-flattened views: about a tenth of a tile. */
const FLATTEN_RADIUS = 4;
/** Candidates read per kind, best first. */
const MAX_CANDIDATES_PER_KIND = 8;
/** Stop looking at further candidates once one reads this well and consistently. */
const GOOD_ENOUGH_CONFIDENCE = 0.9;

export interface Attempt {
  size: GridSize;
  quad: Quad;
  cells: CellReading[];
  /** Mean cell confidence. */
  confidence: number;
  rejection?: Rejection;
}

export type ReadResult =
  | { status: "accepted"; board: Board; quad: Quad; confidence: number; cells: CellReading[] }
  | { status: "rejected"; attempt: Attempt; message: string }
  | { status: "no-board"; message: string };

function computeMeanConfidence(cells: readonly CellReading[]): number {
  return cells.reduce((sum, c) => sum + c.confidence, 0) / cells.length;
}

/** An attempt plus what is needed to read the same grid again with more crops. */
interface Reading {
  attempt: Attempt;
  rect: GreyImage;
  grids: Record<GridSize, GridLayout>;
}

function buildAttempt(
  rect: GreyImage,
  quad: Quad,
  size: GridSize,
  grid: GridLayout,
  weights: ClassifierWeights,
  variants: readonly CropVariant[],
): Attempt {
  const decoded = decodePermutation(readCellProbabilities(rect, size, grid, weights, variants));
  const cells = decoded.values.map((value, i) => ({
    value,
    confidence: decoded.confidences[i] ?? 0,
  }));
  const attempt: Attempt = { size, quad, cells, confidence: computeMeanConfidence(cells) };
  const checked = checkReading(size, cells);
  if (!checked.accepted) attempt.rejection = checked.rejection;
  return attempt;
}

/** Reads the board in `quad` (full-image coordinates) as each of `sizes`, one crop per cell. */
function readQuad(
  grey: GreyImage,
  quad: Quad,
  kind: BoardCandidate["kind"],
  weights: ClassifierWeights,
  sizes: readonly GridSize[],
): Reading[] {
  const rect = rectifyQuad(grey, quad);
  const range = kind === "tiles" ? TILES_INSET_RANGE : OUTLINE_INSET_RANGE;
  const grids = { 3: findGrid(rect, 3, range), 4: findGrid(rect, 4, range) };
  return sizes.map((size) => ({
    attempt: buildAttempt(rect, quad, size, grids[size], weights, CENTRED_CROP),
    rect,
    grids,
  }));
}

/** Ranks attempts: consistent readings first, then by mean confidence. */
function isBetter(a: Attempt, b: Attempt | undefined): boolean {
  if (!b) return true;
  const aOk = a.rejection === undefined;
  const bOk = b.rejection === undefined;
  if (aOk !== bOk) return aOk;
  return a.confidence > b.confidence;
}

function takeBest(candidates: BoardCandidate[], kind: BoardCandidate["kind"]): BoardCandidate[] {
  return candidates.filter((c) => c.kind === kind).slice(0, MAX_CANDIDATES_PER_KIND);
}

/** Board candidates of a photo, with what is needed to refine them and map them to full-size pixels. */
export interface DetectedCandidates {
  grey: GreyImage;
  /** The detection-size views the candidates were found in (plain and saturation, no flattened ones). */
  views: GreyImage[];
  candidates: BoardCandidate[];
  /** Converts a quad in detection pixels to full-image pixels. */
  scaleToFullSize: (quad: Quad) => Quad;
}

/** Shared by the numbered and the picture reader: finds candidate board quads, best first per kind. */
export function detectCandidates(image: RgbaImage): DetectedCandidates {
  const grey = convertToGrey(image);
  const small = shrinkGrey(grey, DETECTION_SIDE);
  const saturation = shrinkGrey(convertToSaturation(image), DETECTION_SIDE);
  const views = [small.image, saturation.image];
  // The flattened views find boards that a lighting gradient or shadow edge breaks up in the plain views.
  const found = findBoardCandidates([
    ...views,
    ...views.map((view) => flattenGrey(view, FLATTEN_RADIUS)),
  ]);
  const candidates = [...takeBest(found, "silhouette"), ...takeBest(found, "tiles")];
  // map() keeps the length (four points) but widens the tuple to Point[], hence the cast.
  const scaleToFullSize = (quad: Quad): Quad =>
    quad.map((p): Point => ({ x: p.x / small.scale, y: p.y / small.scale })) as unknown as Quad;
  return { grey, views, candidates, scaleToFullSize };
}

/** The user's own board outline for a re-read: `quad` replaces the board search, `size` the size guess. */
export interface ReadOptions {
  /** Corners of the tile area in image pixels, in any order; `prepareUserQuad` orders them. */
  quad?: Quad;
  size?: GridSize;
}

/**
 * The corners the user placed, in TL, TR, BR, BL order, checked before any pixel is read: a corner outside
 * the photo or a crossed quad would give a reading of nothing, and a NaN corner never ends. Throws with
 * a message that says what is wrong.
 */
export function prepareUserQuad(quad: Quad, image: RgbaImage): Quad {
  const ordered = orderQuadCorners(quad);
  if (ordered.some((p) => p.x < 0 || p.y < 0 || p.x > image.width || p.y > image.height))
    throw new Error("the corners must lie inside the photo");
  return ordered;
}

export function readBoard(
  image: RgbaImage,
  weights: ClassifierWeights,
  options: ReadOptions = {},
): ReadResult {
  const sizes = options.size === undefined ? GRID_SIZES : [options.size];
  let best: Reading | undefined;
  if (options.quad) {
    // The user placed the corners on the tiles, so there is nothing to search and nothing to refine.
    const quad = prepareUserQuad(options.quad, image);
    const grey = convertToGrey(image);
    for (const reading of readQuad(grey, quad, "tiles", weights, sizes)) {
      if (isBetter(reading.attempt, best?.attempt)) best = reading;
    }
  } else {
    best = readBestCandidate(detectCandidates(image), weights, sizes);
    if (!best)
      return { status: "no-board", message: "No board-like shape was found in the photo." };
  }
  if (!best) throw new Error("no attempt produced");
  // Only the winner is read with augmented crops: it costs several times a plain read.
  // Every size asked for is re-read, since the plain reading may have picked the wrong one.
  const { rect, grids, attempt: first } = best;
  let final: Attempt | undefined;
  for (const size of sizes) {
    const attempt = buildAttempt(rect, first.quad, size, grids[size], weights, AUGMENTED_CROPS);
    if (isBetter(attempt, final)) final = attempt;
  }
  return buildReadResult(final ?? first);
}

/** Reads the detected candidates in order and returns the best reading; undefined if there are none. */
function readBestCandidate(
  { grey, views, candidates, scaleToFullSize }: DetectedCandidates,
  weights: ClassifierWeights,
  sizes: readonly GridSize[],
): Reading | undefined {
  let best: Reading | undefined;
  for (const candidate of candidates) {
    // The raw hull and its edge-snapped refinement are both read; the reading decides which is right.
    const quads = [candidate.quad];
    if (candidate.kind === "silhouette") quads.push(refineQuad(views, candidate.quad));
    for (const quad of quads) {
      for (const reading of readQuad(grey, scaleToFullSize(quad), candidate.kind, weights, sizes)) {
        if (isBetter(reading.attempt, best?.attempt)) best = reading;
      }
    }
    if (
      best &&
      best.attempt.rejection === undefined &&
      best.attempt.confidence >= GOOD_ENOUGH_CONFIDENCE
    )
      break;
  }
  return best;
}

function buildReadResult(best: Attempt): ReadResult {
  if (best.rejection) return { status: "rejected", attempt: best, message: best.rejection.message };
  const checked = checkReading(best.size, best.cells);
  if (!checked.accepted) throw new Error("accepted attempt failed re-check");
  return {
    status: "accepted",
    board: checked.board,
    quad: best.quad,
    confidence: best.confidence,
    cells: best.cells,
  };
}
