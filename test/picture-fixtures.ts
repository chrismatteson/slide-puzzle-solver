// Procedural picture-tile puzzles for tests: a smooth picture made of coloured blobs, cut into
// crops the way the reader would, with optional lighting change and crop shift.

import {
  type Arrangement,
  type PictureReading,
  TILE_SIDE,
  type TileCrop,
} from "../src/picture/types.ts";
import { type GridSize } from "../src/puzzle/board.ts";

export function createRandom(seed: number): () => number {
  let state = seed;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
}

type Sampler = (x: number, y: number) => [number, number, number];

/** A picture of `size` x `size` tiles: a base gradient plus blobs and stripes, all seeded. */
export function createPicture(size: GridSize, seed: number): Sampler {
  const random = createRandom(seed);
  const side = size * TILE_SIDE;
  const base = [random() * 255, random() * 255, random() * 255];
  const blobs = Array.from({ length: 6 * size * size }, () => ({
    x: random() * side,
    y: random() * side,
    radius: 10 + random() * 30,
    colour: [random() * 255, random() * 255, random() * 255],
  }));
  return (x, y) => {
    const out = [0, 0, 0];
    for (let c = 0; c < 3; c++) out[c] = (base[c] ?? 0) * (0.5 + (0.5 * (x + y)) / (2 * side));
    for (const blob of blobs) {
      const weight = Math.exp(-((x - blob.x) ** 2 + (y - blob.y) ** 2) / (2 * blob.radius ** 2));
      for (let c = 0; c < 3; c++)
        out[c] = (out[c] ?? 0) * (1 - weight) + (blob.colour[c] ?? 0) * weight;
    }
    return [out[0] ?? 0, out[1] ?? 0, out[2] ?? 0];
  };
}

export interface CutOptions {
  /** Pixels the crop window is shifted right and down. */
  shift?: number;
  /** Multiplies every channel (a lighting change). */
  gain?: number;
  /** Added to every channel. */
  bias?: number;
}

/** Crop of the picture's tile `tile` (row-major in the picture). */
export function cutTile(
  picture: Sampler,
  size: GridSize,
  tile: number,
  options: CutOptions = {},
): TileCrop {
  const { shift = 0, gain = 1, bias = 0 } = options;
  const rgb = new Uint8ClampedArray(TILE_SIDE * TILE_SIDE * 3);
  const x0 = (tile % size) * TILE_SIDE + shift;
  const y0 = Math.floor(tile / size) * TILE_SIDE + shift;
  for (let y = 0; y < TILE_SIDE; y++) {
    for (let x = 0; x < TILE_SIDE; x++) {
      const colour = picture(x0 + x, y0 + y);
      for (let c = 0; c < 3; c++) rgb[(y * TILE_SIDE + x) * 3 + c] = (colour[c] ?? 0) * gain + bias;
    }
  }
  return { side: TILE_SIDE, rgb };
}

/** A flat-colour crop. */
export function createFlatCrop(r: number, g: number, b: number): TileCrop {
  const rgb = new Uint8ClampedArray(TILE_SIDE * TILE_SIDE * 3);
  for (let i = 0; i < rgb.length; i += 3) rgb.set([r, g, b], i);
  return { side: TILE_SIDE, rgb };
}

/**
 * A photo of the picture scrambled by `order`: `order[cell]` is the picture tile shown at that photo
 * cell, with -1 for the empty slot (drawn as a dark flat crop). Also returns the solved arrangement.
 */
export function createReading(
  picture: Sampler,
  size: GridSize,
  order: readonly number[],
  homeTile: number,
  options: CutOptions = {},
): { reading: PictureReading; arrangement: Arrangement } {
  const cells = order.map((tile) =>
    tile < 0 ? createFlatCrop(20, 20, 20) : cutTile(picture, size, tile, options),
  );
  const blank = order.indexOf(-1);
  const solved = Array.from({ length: size * size }, (_, p) =>
    p === homeTile ? -1 : order.indexOf(p),
  );
  const quad = [
    { x: 0, y: 0 },
    { x: 1, y: 0 },
    { x: 1, y: 1 },
    { x: 0, y: 1 },
  ] as const;
  return {
    reading: { size, quad, cells, blank, blankConfidence: 1 },
    arrangement: { size, solved, confidence: solved.map(() => 1) },
  };
}

/**
 * A reachable scramble of an identity picture whose empty slot is picture tile `homeTile`: slides the
 * empty slot around `moves` times. Returns `order` for createReading.
 */
export function scrambleOrder(
  size: GridSize,
  homeTile: number,
  moves: number,
  seed: number,
): number[] {
  const random = createRandom(seed);
  const n = size;
  const order = Array.from({ length: n * n }, (_, i) => (i === homeTile ? -1 : i));
  let blank = homeTile;
  for (let i = 0; i < moves; i++) {
    const options = [blank - n, blank + n, blank - 1, blank + 1].filter(
      (p) =>
        p >= 0 &&
        p < n * n &&
        !(blank % n === 0 && p === blank - 1) &&
        !(blank % n === n - 1 && p === blank + 1),
    );
    const from = options[Math.floor(random() * options.length)] ?? 0;
    order[blank] = order[from] ?? 0;
    order[from] = -1;
    blank = from;
  }
  return order;
}
