import { type GridSize } from "../puzzle/board.ts";
import { type ClassifierWeights, classifyCropProbabilities } from "./classifier.ts";
import { CROP_WINDOW, sampleCrop } from "./crop.ts";
import {
  type Homography,
  type Point,
  type Quad,
  applyHomography,
  computeHomography,
} from "./geometry.ts";
import { type GreyImage, sampleBilinear } from "./image.ts";

export const RECTIFIED_SIZE = 320;

/** Warps the quad (corners in `grey` pixel coordinates) to a RECTIFIED_SIZE square, top-left corner first. */
export function rectifyQuad(grey: GreyImage, quad: Quad): GreyImage {
  const edge = RECTIFIED_SIZE;
  const square: Quad = [
    { x: 0, y: 0 },
    { x: edge, y: 0 },
    { x: edge, y: edge },
    { x: 0, y: edge },
  ];
  const h: Homography = computeHomography(square, quad);
  const data = new Float32Array(edge * edge);
  for (let y = 0; y < edge; y++) {
    for (let x = 0; x < edge; x++) {
      const source: Point = applyHomography(h, { x: x + 0.5, y: y + 0.5 });
      data[y * edge + x] = sampleBilinear(grey, source.x - 0.5, source.y - 0.5);
    }
  }
  return { width: edge, height: edge, data };
}

/** Plausible frame width, as a fraction of the board edge, for a quad hugging the board's outline... */
export const OUTLINE_INSET_RANGE = { min: 0.02, max: 0.14 };
/** ...and for a quad hugging the tiles, which leaves at most half a gap of frame. */
export const TILES_INSET_RANGE = { min: 0, max: 0.05 };
const INSET_STEP = 0.005;
/** Outer strip ignored when looking for grid lines: the quad's own silhouette edge lives there. */
const EDGE_EXCLUSION = 0.015;

/** Absolute central-difference gradient summed along each column (vertical lines) and row (horizontal lines). */
function computeEdgeProfiles(rect: GreyImage): { columns: Float32Array; rows: Float32Array } {
  const { width, height, data } = rect;
  const columns = new Float32Array(width);
  const rows = new Float32Array(height);
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      columns[x] =
        (columns[x] ?? 0) +
        Math.abs((data[y * width + x + 1] ?? 0) - (data[y * width + x - 1] ?? 0));
      rows[y] =
        (rows[y] ?? 0) +
        Math.abs((data[(y + 1) * width + x] ?? 0) - (data[(y - 1) * width + x] ?? 0));
    }
  }
  const strip = Math.round(EDGE_EXCLUSION * width);
  for (let i = 0; i < strip; i++) {
    columns[i] = 0;
    columns[width - 1 - i] = 0;
    rows[i] = 0;
    rows[height - 1 - i] = 0;
  }
  return { columns, rows };
}

function findPeakNear(profile: Float32Array, position: number): number {
  const centre = Math.round(position);
  let peak = 0;
  for (let i = centre - 2; i <= centre + 2; i++)
    peak = Math.max(peak, profile[Math.min(profile.length - 1, Math.max(0, i))] ?? 0);
  return peak;
}

/** Where the tile grid sits in a rectified board, in rectified pixels: first line and cell pitch per axis. */
export interface GridLayout {
  x0: number;
  y0: number;
  pitchX: number;
  pitchY: number;
}

/** Best (start, end) inset of N+1 evenly spaced lines over one axis's edge profile, as fractions of the edge. */
function fitAxis(
  profile: Float32Array,
  size: GridSize,
  range: { min: number; max: number },
): { start: number; pitch: number } {
  const edge = profile.length;
  let best = { start: edge * range.min, pitch: (edge * (1 - 2 * range.min)) / size };
  let bestScore = -1;
  for (let first = range.min; first <= range.max + 1e-9; first += INSET_STEP) {
    for (let last = range.min; last <= range.max + 1e-9; last += INSET_STEP) {
      const start = edge * first;
      const pitch = (edge * (1 - first - last)) / size;
      let score = 0;
      for (let k = 0; k <= size; k++) score += findPeakNear(profile, start + k * pitch);
      if (score > bestScore) {
        bestScore = score;
        best = { start, pitch };
      }
    }
  }
  return best;
}

/**
 * Finds the tile grid in a rectified board by sliding a comb of N+1 grid lines over the gradient
 * profiles and keeping the placement that sits on the most edge energy: tile borders line up, digit
 * strokes do not. The two axes are fitted independently, each with its own start and end inset (the
 * frame width, as a fraction of the edge, within `range`), because the quad is rarely exact: a quad
 * a few percent off on one side shifts the grid on that axis only.
 */
export function findGrid(
  rect: GreyImage,
  size: GridSize,
  range: { min: number; max: number },
): GridLayout {
  const { columns, rows } = computeEdgeProfiles(rect);
  const x = fitAxis(columns, size, range);
  const y = fitAxis(rows, size, range);
  return { x0: x.start, y0: y.start, pitchX: x.pitch, pitchY: y.pitch };
}

export interface CropVariant {
  /** Shift of the crop centre, in cell pitches. */
  dx: number;
  dy: number;
  /** Multiplier on the crop window. */
  scale: number;
}

export const CENTRED_CROP: readonly CropVariant[] = [{ dx: 0, dy: 0, scale: 1 }];

/**
 * Test-time augmentation: the grid is only found to within a pixel or two, and the crop window is a
 * compromise between tile sizes, so a cell is also classified a little shifted and rescaled and the
 * class log-probabilities are averaged. Seven crops cost seven times as much, so this is for the
 * final reading only, not for ranking candidates.
 */
export const AUGMENTED_CROPS: readonly CropVariant[] = [
  ...CENTRED_CROP,
  { dx: 0, dy: 0, scale: 0.9 },
  { dx: 0, dy: 0, scale: 1.1 },
  { dx: 0.04, dy: 0, scale: 1 },
  { dx: -0.04, dy: 0, scale: 1 },
  { dx: 0, dy: 0.04, scale: 1 },
  { dx: 0, dy: -0.04, scale: 1 },
];

/** Class probabilities of every cell (row-major) of a rectified board with the given grid. */
export function readCellProbabilities(
  rect: GreyImage,
  size: GridSize,
  grid: GridLayout,
  weights: ClassifierWeights,
  variants: readonly CropVariant[] = CENTRED_CROP,
): number[][] {
  const window = ((grid.pitchX + grid.pitchY) / 2) * CROP_WINDOW;
  const probabilities: number[][] = [];
  for (let row = 0; row < size; row++) {
    for (let col = 0; col < size; col++) {
      const sum = new Array<number>(size * size).fill(0);
      for (const variant of variants) {
        const cx = grid.x0 + (col + 0.5 + variant.dx) * grid.pitchX;
        const cy = grid.y0 + (row + 0.5 + variant.dy) * grid.pitchY;
        const crop = sampleCrop(rect, cx, cy, window * variant.scale);
        classifyCropProbabilities(weights, crop, size * size).forEach((p, v) => {
          sum[v] = (sum[v] ?? 0) + Math.log(Math.max(p, 1e-9)) / variants.length;
        });
      }
      const odds = sum.map((x) => Math.exp(x));
      const total = odds.reduce((a, b) => a + b, 0);
      probabilities.push(odds.map((o) => o / total));
    }
  }
  return probabilities;
}
