import { type Board, type GridSize, type Move } from "../puzzle/board.ts";
import { type CellReading, MIN_CELL_CONFIDENCE, checkReading } from "../puzzle/consistency.ts";

export type BoardCheck =
  | { accepted: true; board: Board }
  /**
   * `offending` are row-major cell indexes to highlight; `message` is checkReading's reason.
   * `confirmable`: the only problem is low confidence, i.e. the best guesses already form a solvable
   * board, so the user can confirm each flagged guess instead of retyping it.
   */
  | { accepted: false; message: string; offending: number[]; confirmable: boolean };

function listLeastConfident(cells: readonly CellReading[], count: number): number[] {
  return (
    cells
      // NaN would make the comparison undefined; it is the least trustworthy value, so it sorts first.
      .map((cell, index) => ({
        index,
        confidence: Number.isNaN(cell.confidence) ? -Infinity : cell.confidence,
      }))
      .sort((a, b) => a.confidence - b.confidence || a.index - b.index)
      .slice(0, count)
      .map((entry) => entry.index)
      .sort((a, b) => a - b)
  );
}

function listUnsureCells(cells: readonly CellReading[]): number[] {
  // Same negation as checkReading, so a NaN confidence counts as unsure.
  return cells.flatMap((cell, index) => (!(cell.confidence >= MIN_CELL_CONFIDENCE) ? [index] : []));
}

function listCellsWithValue(cells: readonly CellReading[], value: number): number[] {
  return cells.flatMap((cell, index) => (cell.value === value ? [index] : []));
}

/**
 * Runs the same consistency check the reader uses and says which cells to blame. Reasons that name
 * no cell (no blank, unsolvable) point at the least confident ones: that is where a misread most
 * likely is.
 */
export function checkBoard(size: GridSize, cells: readonly CellReading[]): BoardCheck {
  const result = checkReading(size, cells);
  if (result.accepted) return { accepted: true, board: result.board };
  let { rejection } = result;
  let confirmable = false;
  if (rejection.reason === "low-confidence") {
    // Would the guesses be accepted if every cell were trusted? Then confirming is enough. If not,
    // the real problem (a duplicate, say) is more useful to show than the first unsure cell.
    const trusting = checkReading(size, cells, 0);
    if (trusting.accepted) confirmable = true;
    else rejection = trusting.rejection;
  }
  let offending: number[];
  switch (rejection.reason) {
    case "duplicate":
      offending = listCellsWithValue(cells, rejection.value);
      break;
    case "two-blanks":
      offending = listCellsWithValue(cells, 0);
      break;
    case "out-of-range":
      offending = [rejection.row * size + rejection.col];
      break;
    case "low-confidence":
      offending = listUnsureCells(cells);
      break;
    case "no-blank":
      offending = listLeastConfident(cells, 1);
      break;
    case "unsolvable":
      offending = listLeastConfident(cells, 2);
      break;
  }
  return { accepted: false, message: rejection.message, offending, confirmable };
}

/** A cell the user set by hand is trusted fully: confidence 1. */
export function setCell(
  cells: readonly CellReading[],
  index: number,
  value: number,
): CellReading[] {
  return cells.map((cell, i) => (i === index ? { value, confidence: 1 } : cell));
}

/** The user tapped "yes, that is right" on a guessed cell: keep its value, trust it fully. */
export function confirmCell(cells: readonly CellReading[], index: number): CellReading[] {
  return cells.map((cell, i) => (i === index ? { value: cell.value, confidence: 1 } : cell));
}

export function describeMove(move: Move): string {
  return `Move ${String(move.tile)} ${move.direction}`;
}
