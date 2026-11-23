// Turns a confirmed Arrangement of a photo into a solver Board, and checks that an arrangement can
// be a real puzzle state before the solver sees it.

import { type Board, createGoalBoard, isSolvable } from "../puzzle/board.ts";
import { type Arrangement, type PictureReading, type TileCrop } from "./types.ts";

/**
 * Two tiles whose crops differ by less than this (mean absolute RGB difference, 0..255) look the
 * same to the owner, so exchanging them is invisible. The most alike pair of genuinely different
 * tiles in the owner's photo (two lettered tiles of the bottom row) differs by 20.9, so 8 is well
 * below anything that is visibly different, and above the ~2-4 that JPEG noise and glare add between
 * two crops of one flat patch of picture.
 */
export const NEAR_IDENTICAL_DIFFERENCE = 8;

export type ArrangementCheck =
  | { status: "ok"; arrangement: Arrangement }
  /** `swapped` are the two photo cells whose positions in the arrangement were exchanged. */
  | { status: "fixed"; arrangement: Arrangement; swapped: [number, number] }
  | { status: "invalid" | "unsolvable"; reason: string };

/** What is wrong with `arrangement` as a layout of `reading`'s cells, or undefined if nothing. */
function findProblem(reading: PictureReading, arrangement: Arrangement): string | undefined {
  const count = reading.size * reading.size;
  if (arrangement.size !== reading.size) {
    return `arrangement is ${String(arrangement.size)}x${String(arrangement.size)} but the photo is ${String(reading.size)}x${String(reading.size)}`;
  }
  if (arrangement.solved.length !== count) {
    return `arrangement has ${String(arrangement.solved.length)} positions, expected ${String(count)}`;
  }
  const homes = arrangement.solved.filter((cell) => cell === -1).length;
  if (homes !== 1) return `arrangement has ${String(homes)} empty-slot positions (-1), expected 1`;
  const seen = new Set<number>();
  for (const cell of arrangement.solved) {
    if (cell === -1) continue;
    if (!Number.isInteger(cell) || cell < 0 || cell >= count) {
      return `photo cell ${String(cell)} is not a cell of the photo`;
    }
    if (cell === reading.blank) return `photo cell ${String(cell)} is the empty slot, not a tile`;
    if (seen.has(cell)) return `photo cell ${String(cell)} is placed twice`;
    seen.add(cell);
  }
  return undefined;
}

/**
 * The board the photo shows, with the solved layout as its goal: the tile at solved position p is
 * numbered as p is in the goal (reading order skipping the home), and the empty slot is 0.
 */
export function toBoard(reading: PictureReading, arrangement: Arrangement): Board {
  const problem = findProblem(reading, arrangement);
  if (problem) throw new Error(`cannot make a board: ${problem}`);
  const blankHome = arrangement.solved.indexOf(-1);
  const goal = createGoalBoard(reading.size, blankHome).cells;
  const cells = new Array<number>(reading.cells.length).fill(0);
  arrangement.solved.forEach((cell, position) => {
    if (cell !== -1) cells[cell] = goal[position] ?? 0;
  });
  return { size: reading.size, blankHome, cells };
}

function measureDifference(a: TileCrop, b: TileCrop): number {
  let sum = 0;
  for (let i = 0; i < a.rgb.length; i++) sum += Math.abs((a.rgb[i] ?? 0) - (b.rgb[i] ?? 0));
  return sum / a.rgb.length;
}

/** The two photo cells (not the empty slot) that look most alike, and how much they differ. */
function findMostSimilarPair(
  reading: PictureReading,
): { cells: [number, number]; difference: number } | undefined {
  let best: { cells: [number, number]; difference: number } | undefined;
  for (let a = 0; a < reading.cells.length; a++) {
    for (let b = a + 1; b < reading.cells.length; b++) {
      if (a === reading.blank || b === reading.blank) continue;
      const difference = measureDifference(
        reading.cells[a] as TileCrop,
        reading.cells[b] as TileCrop,
      );
      if (!best || difference < best.difference) best = { cells: [a, b], difference };
    }
  }
  return best;
}

/**
 * Checks `arrangement` is a permutation of the photo's tiles and that sliding can reach it. If the
 * parity is wrong, exchanging two tiles that look the same fixes it without the owner being able to
 * tell, so that is done, reported, and the pair is marked unsure so the review outlines it;
 * otherwise the arrangement is rejected as unsolvable.
 */
export function checkArrangement(
  reading: PictureReading,
  arrangement: Arrangement,
): ArrangementCheck {
  const problem = findProblem(reading, arrangement);
  if (problem) return { status: "invalid", reason: problem };
  if (isSolvable(toBoard(reading, arrangement))) return { status: "ok", arrangement };
  const pair = findMostSimilarPair(reading);
  if (!pair || pair.difference >= NEAR_IDENTICAL_DIFFERENCE) {
    return {
      status: "unsolvable",
      reason:
        "the tiles cannot be slid into this arrangement (two tiles are swapped or misread), and no two tiles look alike enough to blame",
    };
  }
  const [a, b] = pair.cells;
  const solved = arrangement.solved.map((cell) => (cell === a ? b : cell === b ? a : cell));
  const confidence = arrangement.confidence.map((value, position) =>
    solved[position] === a || solved[position] === b ? 0 : value,
  );
  return {
    status: "fixed",
    arrangement: { ...arrangement, solved, confidence },
    swapped: pair.cells,
  };
}
