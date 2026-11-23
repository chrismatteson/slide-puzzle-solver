// Remembers a confirmed picture puzzle so a later photo of the same tiles in a new scramble is read
// without asking again. A profile keeps a small thumbnail of every tile; matching assigns the photo's
// cells to the remembered tiles by thumbnail similarity (an assignment problem, solved jointly so that
// two similar tiles cannot both claim the same place).

import { type GridSize } from "../puzzle/board.ts";
import { solveAssignment } from "../vision/decode.ts";
import { type Arrangement, type PictureReading, TILE_SIDE, type TileCrop } from "./types.ts";

/** Side of the stored thumbnail: TILE_SIDE / 4, so a 2 px shift of a crop moves it half a thumbnail pixel. */
const THUMB_SIDE = 16;
const THUMB_BYTES = THUMB_SIDE * THUMB_SIDE * 3;
const BLOCK = TILE_SIDE / THUMB_SIDE;
/** After lighting normalisation every channel of the photo's tiles averages this. */
const TARGET_MEAN = 128;

/**
 * Highest mean thumbnail difference (0..255, averaged over the assigned tiles) still taken as the
 * same puzzle. Measured with test/profile.test.ts (2026-11-23): the same procedural tiles under a 0.75
 * gain, +6 bias and 2 px shift score 3.3 at most, the owner's photo re-composited at 0.8 gain and a
 * 2 px shift scores 6.0, and unrelated pictures score 28 or more (39.8 for the owner's tiles
 * against a procedural picture). 12 sits between with margin on both sides.
 */
export const MAX_MEAN_DIFFERENCE = 12;

export interface PuzzleProfile {
  version: 1;
  size: GridSize;
  /** Solved position of the empty slot. */
  blankHome: number;
  /**
   * One lighting-normalised thumbnail per tile (16x16 RGB, base64), in goal numbering: entry k is the
   * tile at solved position k before the home and k + 1 after it. 15 tiles of 4x4 are about 15 KB of JSON.
   */
  tiles: string[];
}

export type ProfileMatch =
  | {
      status: "match";
      arrangement: Arrangement;
      /** Lowest per-tile confidence, 0..1. */
      confidence: number;
    }
  | { status: "mismatch"; reason: string };

/** Thumbnail of a crop as 768 channel means. */
function shrinkCrop(crop: TileCrop): Float32Array {
  const thumb = new Float32Array(THUMB_BYTES);
  for (let y = 0; y < TILE_SIDE; y++) {
    for (let x = 0; x < TILE_SIDE; x++) {
      const target = (Math.floor(y / BLOCK) * THUMB_SIDE + Math.floor(x / BLOCK)) * 3;
      for (let c = 0; c < 3; c++) {
        thumb[target + c] =
          (thumb[target + c] ?? 0) + (crop.rgb[(y * TILE_SIDE + x) * 3 + c] ?? 0) / (BLOCK * BLOCK);
      }
    }
  }
  return thumb;
}

/**
 * Thumbnails of `crops` scaled per channel so that, over the whole photo, each channel averages
 * TARGET_MEAN. That cancels exposure and white balance of the photo as a whole, which is what
 * differs between two photos of one puzzle; local glare is left to the tolerance.
 */
function normaliseThumbnails(crops: readonly TileCrop[]): Uint8Array[] {
  const thumbs = crops.map(shrinkCrop);
  const sums = [0, 0, 0];
  for (const thumb of thumbs) {
    for (let i = 0; i < THUMB_BYTES; i++) sums[i % 3] = (sums[i % 3] ?? 0) + (thumb[i] ?? 0);
  }
  const gains = sums.map((sum) =>
    sum === 0 ? 1 : (TARGET_MEAN * thumbs.length * THUMB_SIDE * THUMB_SIDE) / sum,
  );
  return thumbs.map((thumb) =>
    Uint8Array.from(thumb, (v, i) => Math.round(v * (gains[i % 3] ?? 1))),
  );
}

function encodeThumbnail(thumb: Uint8Array): string {
  // btoa/atob exist in browsers and Node alike; Buffer would not run in the app.
  return btoa(String.fromCharCode(...thumb));
}

function decodeThumbnail(text: string): Uint8Array {
  const bytes = Uint8Array.from(atob(text), (ch) => ch.charCodeAt(0));
  if (bytes.length !== THUMB_BYTES)
    throw new Error(
      `profile thumbnail has ${String(bytes.length)} bytes, expected ${String(THUMB_BYTES)}`,
    );
  return bytes;
}

function measureDifference(a: Uint8Array, b: Uint8Array): number {
  let sum = 0;
  for (let i = 0; i < THUMB_BYTES; i++) sum += Math.abs((a[i] ?? 0) - (b[i] ?? 0));
  return sum / THUMB_BYTES;
}

/** Photo cells holding a tile, in order: all but the empty slot. */
function listTileCells(reading: PictureReading): number[] {
  return reading.cells.flatMap((_, cell) => (cell === reading.blank ? [] : [cell]));
}

/** Remembers the tiles of a confirmed `arrangement` of `reading`. */
export function createProfile(reading: PictureReading, arrangement: Arrangement): PuzzleProfile {
  const blankHome = arrangement.solved.indexOf(-1);
  if (blankHome < 0) throw new Error("arrangement has no empty-slot position (-1)");
  // Normalise over the photo's tiles only, as matchProfile does; the empty slot would skew the means.
  const cells = listTileCells(reading);
  const thumbs = normaliseThumbnails(cells.map((cell) => reading.cells[cell] as TileCrop));
  const thumbOfCell = new Map(cells.map((cell, i) => [cell, thumbs[i] as Uint8Array]));
  const tiles = arrangement.solved.flatMap((cell) => {
    if (cell === -1) return [];
    const thumb = thumbOfCell.get(cell);
    if (!thumb) throw new Error(`photo cell ${String(cell)} is not a tile of the photo`);
    return [encodeThumbnail(thumb)];
  });
  return { version: 1, size: arrangement.size, blankHome, tiles };
}

/**
 * Assigns each tile cell of `reading` to a solved position of the remembered puzzle. Rejects the
 * photo when the tiles are too different from the profile's. Per-tile confidence is how much closer
 * the chosen tile is than the next best choice for that cell or for that tile (0 when two tiles look
 * the same, so the user is asked to confirm those).
 */
export function matchProfile(reading: PictureReading, profile: PuzzleProfile): ProfileMatch {
  if (reading.size !== profile.size) {
    return {
      status: "mismatch",
      reason: `the photo is ${String(reading.size)}x${String(reading.size)} but the saved puzzle is ${String(profile.size)}x${String(profile.size)}`,
    };
  }
  const cells = listTileCells(reading);
  const remembered = profile.tiles.map(decodeThumbnail);
  if (cells.length !== remembered.length) {
    throw new Error(
      `profile has ${String(remembered.length)} tiles, the photo has ${String(cells.length)}`,
    );
  }
  const thumbs = normaliseThumbnails(cells.map((cell) => reading.cells[cell] as TileCrop));
  const cost = thumbs.map((thumb) => remembered.map((tile) => measureDifference(thumb, tile)));
  const tileOfRow = solveAssignment(cost);
  const total = tileOfRow.reduce((sum, tile, row) => sum + (cost[row]?.[tile] ?? 0), 0);
  const meanDifference = total / cells.length;
  if (meanDifference > MAX_MEAN_DIFFERENCE) {
    return {
      status: "mismatch",
      reason: `the tiles differ from the saved puzzle's by ${meanDifference.toFixed(1)} on average (limit ${String(MAX_MEAN_DIFFERENCE)}): not this puzzle`,
    };
  }
  const solved = new Array<number>(profile.size * profile.size).fill(-1);
  const confidence = new Array<number>(solved.length).fill(1);
  const tileConfidences: number[] = [];
  tileOfRow.forEach((tile, row) => {
    const position = tile < profile.blankHome ? tile : tile + 1;
    const own = cost[row]?.[tile] ?? 0;
    const otherForCell = Math.min(...(cost[row] ?? []).filter((_, t) => t !== tile));
    const otherForTile = Math.min(
      ...cost.filter((_, r) => r !== row).map((costs) => costs[tile] ?? Infinity),
    );
    const rival = Math.min(otherForCell, otherForTile);
    solved[position] = cells[row] ?? -1;
    confidence[position] = rival + own === 0 ? 0 : Math.max(0, (rival - own) / (rival + own));
    tileConfidences.push(confidence[position]);
  });
  return {
    status: "match",
    arrangement: { size: profile.size, solved, confidence },
    confidence: Math.min(...tileConfidences),
  };
}

/**
 * The saved puzzle `reading` matches best (the highest lowest-tile confidence), with the arrangement
 * it gives. The first of two equally confident ones wins.
 */
export function findBestProfileMatch(
  reading: PictureReading,
  profiles: readonly PuzzleProfile[],
): { profile: PuzzleProfile; arrangement: Arrangement } | undefined {
  let best: { profile: PuzzleProfile; arrangement: Arrangement; confidence: number } | undefined;
  for (const profile of profiles) {
    let match;
    try {
      match = matchProfile(reading, profile);
    } catch {
      continue; // a damaged saved puzzle must not stop the photo being read
    }
    if (match.status === "match" && match.confidence > (best?.confidence ?? -1))
      best = { profile, arrangement: match.arrangement, confidence: match.confidence };
  }
  return best && { profile: best.profile, arrangement: best.arrangement };
}
