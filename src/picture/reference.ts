// Prototype (not wired into the app): arranges tiles by matching them against a picture of the solved
// puzzle, e.g. the small print on the frame or the box art, instead of by how their edges continue
// (assemble.ts). See docs/reference-picture.md for how well it works and when to trust it.

import { solveAssignment } from "../vision/decode.ts";
import { type Arrangement, TILE_SIDE, type TileCrop } from "./types.ts";

/** Features per crop side: a crop is averaged down to FEATURE_SIDE x FEATURE_SIDE colour cells. */
const FEATURE_SIDE = 8;
/**
 * Rounds of re-fitting the tiles' colours to the cells they were assigned to, then assigning again: the
 * first pass runs on plain standardised colours, which a print that is tinted or lit differently
 * throws off. Measured on the dev photos (docs/reference-picture.md): one round helps, three add nothing.
 */
const COLOUR_FIT_ROUNDS = 1;

/**
 * Colour of a crop averaged over blocks, which is what a 100 px blurry print and a sharp tile still
 * share: 192 numbers, row-major, three channels per block.
 */
function averageBlockColours(crop: TileCrop): Float64Array {
  if (crop.side !== TILE_SIDE) throw new Error(`crops must be ${String(TILE_SIDE)} px`);
  const block = TILE_SIDE / FEATURE_SIDE;
  const out = new Float64Array(FEATURE_SIDE * FEATURE_SIDE * 3);
  for (let y = 0; y < TILE_SIDE; y++) {
    for (let x = 0; x < TILE_SIDE; x++) {
      const target = (Math.floor(y / block) * FEATURE_SIDE + Math.floor(x / block)) * 3;
      for (let c = 0; c < 3; c++)
        out[target + c] = (out[target + c] ?? 0) + (crop.rgb[(y * TILE_SIDE + x) * 3 + c] ?? 0);
    }
  }
  return out.map((sum) => sum / (block * block));
}

/**
 * Features of every crop with each channel standardised over the whole set, so a print that is duller,
 * lighter or tinted compared to the tiles in the photo (or a photo lit unevenly) still lines up.
 */
function standardise(crops: readonly TileCrop[]): Float64Array[] {
  const features = crops.map(averageBlockColours);
  for (let c = 0; c < 3; c++) {
    let sum = 0;
    let count = 0;
    for (const f of features)
      for (let i = c; i < f.length; i += 3) {
        sum += f[i] ?? 0;
        count++;
      }
    const mean = sum / count;
    let variance = 0;
    for (const f of features)
      for (let i = c; i < f.length; i += 3) variance += ((f[i] ?? 0) - mean) ** 2;
    // A flat picture (all one colour) has no spread to scale by; keep the scale at 1 there.
    const spread = Math.sqrt(variance / count) || 1;
    for (const f of features)
      for (let i = c; i < f.length; i += 3) f[i] = ((f[i] ?? 0) - mean) / spread;
  }
  return features;
}

/** Mean squared difference of every tile's features to every reference cell's: tiles x positions. */
function priceMatches(own: readonly Float64Array[], wanted: readonly Float64Array[]): number[][] {
  return own.map((f) =>
    wanted.map((target) => {
      let sum = 0;
      for (let i = 0; i < f.length; i++) sum += ((f[i] ?? 0) - (target[i] ?? 0)) ** 2;
      return sum / f.length;
    }),
  );
}

/**
 * Hungarian assignment of tiles to positions. The empty slot is no tile; as one more row costing the
 * same everywhere it takes the position left over. Returns the position of each tile (row of `cost`).
 */
function assignTiles(cost: readonly number[][], count: number): number[] {
  const homeCost = cost.map((row) => Math.min(...row));
  const flat = homeCost.reduce((a, b) => a + b, 0) / homeCost.length;
  return solveAssignment([...cost, new Array<number>(count).fill(flat)]);
}

/** Moves each channel of the tiles' features by the gain and offset that best fit them to their assigned cells. */
function adjustColours(
  own: Float64Array[],
  wanted: readonly Float64Array[],
  assignment: readonly number[],
): void {
  for (let c = 0; c < 3; c++) {
    let n = 0,
      sx = 0,
      sy = 0,
      sxx = 0,
      sxy = 0;
    own.forEach((f, row) => {
      const target = wanted[assignment[row] ?? 0] as Float64Array;
      for (let i = c; i < f.length; i += 3) {
        const x = f[i] ?? 0;
        const y = target[i] ?? 0;
        n++;
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
      }
    });
    const gain = (n * sxy - sx * sy) / (n * sxx - sx * sx || 1);
    const offset = (sy - gain * sx) / n;
    for (const f of own) for (let i = c; i < f.length; i += 3) f[i] = gain * (f[i] ?? 0) + offset;
  }
}

/**
 * The solved arrangement of a photo from a picture of the solved puzzle. `cells` are the photo's
 * crops (row-major, `blank` the empty slot's cell) and `reference` the solved picture cut into the
 * same grid (row-major, same crop size; the cell at the empty slot's home may be blank or picture).
 * Every tile goes to the reference cell it looks most like, all together (Hungarian assignment), and
 * the cell left over is the empty slot's home. Confidence per position is how much worse the tile's
 * next-best position would have been, relative to it: 0 when two positions look alike (a flat patch of
 * sky), towards 1 when only one fits. It does not know which layouts sliding can reach; pass the result
 * through `checkArrangement`.
 */
export function matchReference(
  cells: readonly TileCrop[],
  blank: number,
  reference: readonly TileCrop[],
): Arrangement {
  const count = cells.length;
  const size = Math.round(Math.sqrt(count));
  if (size !== 3 && size !== 4) throw new Error(`${String(count)} cells is not a 3x3 or 4x4 grid`);
  if (reference.length !== count) throw new Error("reference must have as many cells as the photo");
  if (!Number.isInteger(blank) || blank < 0 || blank >= count)
    throw new Error(`empty slot ${String(blank)} is not a cell of the photo`);

  const cellOfTile = cells.map((_, cell) => cell).filter((cell) => cell !== blank);
  // The empty slot's flat crop would skew the tiles' statistics, so it is left out.
  const own = standardise(cellOfTile.map((cell) => cells[cell] as TileCrop));
  const wanted = standardise(reference);
  let cost = priceMatches(own, wanted);
  let assignment = assignTiles(cost, count);
  for (let round = 0; round < COLOUR_FIT_ROUNDS; round++) {
    adjustColours(own, wanted, assignment);
    cost = priceMatches(own, wanted);
    assignment = assignTiles(cost, count);
  }
  const solved = new Array<number>(count).fill(-1);
  const confidence = new Array<number>(count).fill(1);
  cellOfTile.forEach((cell, row) => {
    const position = assignment[row] ?? 0;
    const assigned = cost[row]?.[position] ?? 0;
    const other = Math.min(...(cost[row] ?? []).filter((_, p) => p !== position));
    solved[position] = cell;
    confidence[position] = other > 0 ? Math.min(Math.max((other - assigned) / other, 0), 1) : 0;
  });
  return { size, solved, confidence };
}
