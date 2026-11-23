// Pure logic of the picture review screen: what the user can do to a proposed arrangement.

import { createGoalBoard } from "../puzzle/board.ts";
import { type Arrangement, type PictureReading, type TileCrop } from "../picture/types.ts";

/**
 * A solved position whose tile the assembler is less sure of than this is flagged. Confidence is
 * 0..1: how much worse the layout gets if the tile is swapped with its best look-alike, over what a
 * clear-cut tile loses (`assemblePicture`). On the owner's photo, assembled fully correctly, whether
 * read from the file or from Chromium's decode (2026-11-23), the near-tied pair (solved positions 7 and
 * 11) scores 0 - swapping them is free, only the parity of the puzzle tells them apart - and every other
 * tile 0.28-0.66. 0.15 flags that pair and nothing else there. Re-measure if the assembler's scoring
 * changes (test/picture-review.test.ts pins the value, not its meaning).
 */
export const LOW_TILE_CONFIDENCE = 0.15;

/** The number shown on the tile at solved position `position`: its place in reading order skipping the home. */
export function tileNumberAt(arrangement: Arrangement, position: number): number {
  return createGoalBoard(arrangement.size, arrangement.solved.indexOf(-1)).cells[position] ?? 0;
}

/** Solved positions to flag, in reading order. The empty-slot home is never flagged. */
export function listDoubtfulPositions(arrangement: Arrangement): number[] {
  return arrangement.confidence.flatMap((confidence, position) =>
    arrangement.solved[position] !== -1 && !(confidence >= LOW_TILE_CONFIDENCE) ? [position] : [],
  );
}

/**
 * Exchanges what is at solved positions `a` and `b` (either may be the empty-slot home, which is how
 * the user moves it). Both are then trusted fully, as a hand-set cell is in the numbered review.
 */
export function swapPositions(arrangement: Arrangement, a: number, b: number): Arrangement {
  const solved = [...arrangement.solved];
  const confidence = [...arrangement.confidence];
  [solved[a], solved[b]] = [solved[b] as number, solved[a] as number];
  confidence[a] = 1;
  confidence[b] = 1;
  return { ...arrangement, solved, confidence };
}

/** The crop of each tile number 1..N²-1 in `arrangement`, to draw tiles on a board. */
export function mapTileCrops(
  reading: PictureReading,
  arrangement: Arrangement,
): Map<number, TileCrop> {
  const crops = new Map<number, TileCrop>();
  arrangement.solved.forEach((cell, position) => {
    if (cell !== -1)
      crops.set(tileNumberAt(arrangement, position), reading.cells[cell] as TileCrop);
  });
  return crops;
}
