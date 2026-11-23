// Reads a photo of a picture-tile puzzle: finds the board, fits the tile grid, cuts every cell to a
// colour TileCrop and says which cell is the empty slot. There is no digit classifier to judge a
// reading by, so candidates are ranked by grid-line evidence alone (see `findGridWithStrength`).

import { GRID_SIZES, type GridSize } from "../puzzle/board.ts";
import { type Point, type Quad, applyHomography, computeHomography } from "../vision/geometry.ts";
import {
  OUTLINE_INSET_RANGE,
  RECTIFIED_SIZE,
  TILES_INSET_RANGE,
  type GridLayout,
  findGridWithStrength,
  rectifyQuad,
  rectifyQuadRgb,
} from "../vision/grid.ts";
import { type RgbaImage } from "../vision/image.ts";
import { detectCandidates } from "../vision/read-board.ts";
import { refineQuad } from "../vision/refine.ts";
import { snapQuadToSeams } from "./seams.ts";
import { type PictureReading, TILE_SIDE, type TileCrop } from "./types.ts";

/**
 * Fraction of a cell trimmed from each side of its crop, so tile gaps and the frame stay out. Kept small:
 * assembly compares the crops' edge pixels, and every trimmed pixel widens the gap it has to bridge
 * (6% made assembly of the owner's photo fall apart, 1% did not; see test/picture-pipeline.test.ts).
 */
const CROP_TRIM = 0.01;
/** Where the frame colour is sampled: between these distances (in cell pitches) outside the tile area. */
const FRAME_RING = { near: 0.05, far: 0.12 };
/** Samples per side and distance; samples falling outside the photo are skipped. */
const FRAME_SAMPLES_PER_SIDE = 24;

export type PictureReadResult = PictureReading | { status: "no-board"; message: string };

/**
 * Cuts the cells of a board whose tile area is `tilesQuad` (corners TL, TR, BR, BL in `image` pixels,
 * hugging the tiles) to TILE_SIDE crops, row-major. Each crop is warped straight from the photo,
 * so it is resampled once and perspective is removed.
 */
export function cutTileCrops(image: RgbaImage, tilesQuad: Quad, size: GridSize): TileCrop[] {
  const unit: Quad = [
    { x: 0, y: 0 },
    { x: 1, y: 0 },
    { x: 1, y: 1 },
    { x: 0, y: 1 },
  ];
  const h = computeHomography(unit, tilesQuad);
  const at = (u: number, v: number): Point => applyHomography(h, { x: u / size, y: v / size });
  const crops: TileCrop[] = [];
  for (let row = 0; row < size; row++) {
    for (let col = 0; col < size; col++) {
      const near = CROP_TRIM;
      const far = 1 - CROP_TRIM;
      const cell: Quad = [
        at(col + near, row + near),
        at(col + far, row + near),
        at(col + far, row + far),
        at(col + near, row + far),
      ];
      crops.push({ side: TILE_SIDE, rgb: rectifyQuadRgb(image, cell, TILE_SIDE, TILE_SIDE).data });
    }
  }
  return crops;
}

interface Fit {
  size: GridSize;
  tilesQuad: Quad;
  strength: number;
}

/** The tile area, in photo pixels, of a grid fitted in the rectified view of `quad`. */
function layoutToTilesQuad(quad: Quad, layout: GridLayout, size: GridSize): Quad {
  const edge = RECTIFIED_SIZE;
  const square: Quad = [
    { x: 0, y: 0 },
    { x: edge, y: 0 },
    { x: edge, y: edge },
    { x: 0, y: edge },
  ];
  const h = computeHomography(square, quad);
  const x1 = layout.x0 + size * layout.pitchX;
  const y1 = layout.y0 + size * layout.pitchY;
  return [
    applyHomography(h, { x: layout.x0, y: layout.y0 }),
    applyHomography(h, { x: x1, y: layout.y0 }),
    applyHomography(h, { x: x1, y: y1 }),
    applyHomography(h, { x: layout.x0, y: y1 }),
  ];
}

/** Median of each RGB channel over a ring just outside the tile area: the board's base colour. */
function measureFrameColour(image: RgbaImage, tilesQuad: Quad, size: GridSize): number[] {
  const unit: Quad = [
    { x: 0, y: 0 },
    { x: 1, y: 0 },
    { x: 1, y: 1 },
    { x: 0, y: 1 },
  ];
  const h = computeHomography(unit, tilesQuad);
  const channels: number[][] = [[], [], []];
  const sample = (u: number, v: number): void => {
    const p = applyHomography(h, { x: u, y: v });
    const x = Math.round(p.x);
    const y = Math.round(p.y);
    if (x < 0 || y < 0 || x >= image.width || y >= image.height) return;
    for (let c = 0; c < 3; c++) channels[c]?.push(image.data[(y * image.width + x) * 4 + c] ?? 0);
  };
  for (let k = 0; k < FRAME_SAMPLES_PER_SIDE; k++) {
    const t = (k + 0.5) / FRAME_SAMPLES_PER_SIDE;
    for (const d of [FRAME_RING.near, (FRAME_RING.near + FRAME_RING.far) / 2, FRAME_RING.far]) {
      const out = d / size;
      sample(t, -out);
      sample(t, 1 + out);
      sample(-out, t);
      sample(1 + out, t);
    }
  }
  return channels.map((values) => {
    values.sort((a, b) => a - b);
    return values[Math.floor(values.length / 2)] ?? 0;
  });
}

/** Per-crop numbers the slot test compares. */
interface CellStats {
  mean: number[];
  /** Mean of max(R,G,B) - min(R,G,B). */
  saturation: number;
  /** Mean absolute grey difference between neighbouring pixels. */
  texture: number;
}

function measureCell(crop: TileCrop): CellStats {
  const { side, rgb } = crop;
  const sums = [0, 0, 0];
  let saturation = 0;
  let texture = 0;
  const grey = new Float32Array(side * side);
  for (let i = 0; i < side * side; i++) {
    const r = rgb[i * 3] ?? 0;
    const g = rgb[i * 3 + 1] ?? 0;
    const b = rgb[i * 3 + 2] ?? 0;
    sums[0] = (sums[0] ?? 0) + r;
    sums[1] = (sums[1] ?? 0) + g;
    sums[2] = (sums[2] ?? 0) + b;
    saturation += Math.max(r, g, b) - Math.min(r, g, b);
    grey[i] = 0.299 * r + 0.587 * g + 0.114 * b;
  }
  for (let y = 0; y < side - 1; y++) {
    for (let x = 0; x < side - 1; x++) {
      const here = grey[y * side + x] ?? 0;
      texture +=
        Math.abs((grey[y * side + x + 1] ?? 0) - here) +
        Math.abs((grey[(y + 1) * side + x] ?? 0) - here);
    }
  }
  const n = side * side;
  return {
    mean: sums.map((s) => s / n),
    saturation: saturation / n,
    texture: texture / (2 * (side - 1) * (side - 1)),
  };
}

/**
 * How unlike an empty slot a cell is, 0 = exactly like one. The slot shows the board's base, so it
 * is the cell closest to the frame colour and also flat and unsaturated. Printed tiles, even a plain
 * blue one, are far from the frame colour or saturated; the sum needs all three to agree.
 */
function scoreAgainstSlot(stats: CellStats, frame: readonly number[]): number {
  const distance = Math.hypot(...stats.mean.map((m, c) => m - (frame[c] ?? 0)));
  return distance / 60 + stats.saturation / 10 + stats.texture / 10;
}

/** The blank cell index and its margin over the runner-up (0..1). */
function findBlank(
  crops: readonly TileCrop[],
  frame: readonly number[],
): { blank: number; blankConfidence: number } {
  const scores = crops.map((crop) => scoreAgainstSlot(measureCell(crop), frame));
  let blank = 0;
  scores.forEach((s, i) => {
    if (s < (scores[blank] ?? Infinity)) blank = i;
  });
  const best = scores[blank] ?? 0;
  const runnerUp = Math.min(...scores.filter((_, i) => i !== blank));
  return { blank, blankConfidence: runnerUp > 0 ? Math.max(0, (runnerUp - best) / runnerUp) : 0 };
}

/** Fits every grid size (or just `size`) to `quad`; the strongest grid-line evidence wins. */
function fitQuad(
  grey: ReturnType<typeof detectCandidates>["grey"],
  quad: Quad,
  kind: "silhouette" | "tiles",
  sizes: readonly GridSize[],
): Fit[] {
  const rect = rectifyQuad(grey, quad);
  const range = kind === "tiles" ? TILES_INSET_RANGE : OUTLINE_INSET_RANGE;
  return sizes.map((size) => {
    const { layout, strength } = findGridWithStrength(rect, size, range);
    return { size, strength, tilesQuad: layoutToTilesQuad(quad, layout, size) };
  });
}

export function readPictureBoard(image: RgbaImage, size?: GridSize): PictureReadResult {
  const { grey, views, candidates, scaleToFullSize } = detectCandidates(image);
  if (candidates.length === 0) {
    return { status: "no-board", message: "No board-like shape was found in the photo." };
  }
  const sizes = size === undefined ? GRID_SIZES : [size];
  let best: Fit | undefined;
  for (const candidate of candidates) {
    const quads = [candidate.quad];
    if (candidate.kind === "silhouette") quads.push(refineQuad(views, candidate.quad));
    for (const quad of quads) {
      for (const fit of fitQuad(grey, scaleToFullSize(quad), candidate.kind, sizes)) {
        if (fit.strength > (best?.strength ?? -Infinity)) best = fit;
      }
    }
  }
  if (!best) throw new Error("no fit produced");
  // The comb fit works in a view rectified from a quad that is a few percent off; the seams fix that.
  const tilesQuad = snapQuadToSeams(grey, best.tilesQuad, best.size);
  const cells = cutTileCrops(image, tilesQuad, best.size);
  const frame = measureFrameColour(image, tilesQuad, best.size);
  return { size: best.size, quad: tilesQuad, cells, ...findBlank(cells, frame) };
}
