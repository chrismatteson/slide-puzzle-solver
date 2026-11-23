// Shared types of picture-tile mode: puzzles whose tiles carry pieces of one picture and no numbers.
// Reading (`read.ts`) produces a PictureReading; assembly (`assemble.ts`) and saved puzzles
// (`profile.ts`) turn it into an Arrangement; `toBoard` in `arrangement.ts` makes a solver Board of it.

import { type GridSize } from "../puzzle/board.ts";
import { type Quad } from "../vision/geometry.ts";

/** Side, in pixels, of every rectified tile crop. */
export const TILE_SIDE = 64;

/** One rectified cell of the photo: RGB, row-major, TILE_SIDE x TILE_SIDE, 3 bytes per pixel. */
export interface TileCrop {
  side: number;
  rgb: Uint8ClampedArray;
}

export interface PictureReading {
  size: GridSize;
  /** The tile area's quad (corners TL, TR, BR, BL) in the input image's pixels, after snapping to the seams. */
  quad: Quad;
  /** Row-major photo cells, length size². The crop at `blank` is the empty slot. */
  cells: TileCrop[];
  /** Photo cell index of the empty slot. */
  blank: number;
  /** 0..1: how much more slot-like the chosen blank cell is than the runner-up. */
  blankConfidence: number;
}

/**
 * A solved layout of a photo. `solved[p]` is the photo cell index of the tile that belongs at solved
 * position p (row-major), or -1 at the solved position of the empty slot (its "home").
 */
export interface Arrangement {
  size: GridSize;
  solved: number[];
  /** Per solved position, 0..1 confidence that the tile there is right; 1 at the empty-slot home. */
  confidence: number[];
}
