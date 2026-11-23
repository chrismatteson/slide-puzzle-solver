import { type Board, type GridSize, isSolvable } from "./board.ts";

/** One classified cell: 0 for the empty slot, else the tile number, with the classifier's confidence in [0, 1]. */
export interface CellReading {
  value: number;
  confidence: number;
}

export type Rejection =
  // A duplicate always leaves some tile missing (every cell holds exactly one value), so one reason names both.
  | { reason: "duplicate"; value: number; missing: number; message: string }
  | { reason: "no-blank"; message: string }
  | { reason: "two-blanks"; message: string }
  | { reason: "out-of-range"; row: number; col: number; message: string }
  | { reason: "low-confidence"; row: number; col: number; message: string }
  | { reason: "unsolvable"; message: string };

export type ConsistencyResult =
  { accepted: true; board: Board } | { accepted: false; rejection: Rejection };

/**
 * A cell below this is never silently accepted. A cell's confidence is how much likelier the best
 * whole-board reading is than the best one that changes this cell (see vision/decode.ts), so it is
 * the weakest link of the permutation, not one softmax output. Chosen on two held-out synthetic
 * sets (seeds 4242 and 31337, 460 photos, neither a report seed): at 0.95 no wrongly read board was
 * accepted, at 0.9 one was; above 0.96 correct boards start being rejected (README, "Measured accuracy").
 */
export const MIN_CELL_CONFIDENCE = 0.95;

/**
 * Accepts a reading only if it is a permutation of 1..N²-1 plus exactly one blank, every cell is
 * confident, and the permutation is reachable from the goal. Anything else is rejected with the
 * first specific reason found; an unsolvable permutation means a cell was most likely misread.
 */
export function checkReading(
  size: GridSize,
  cells: readonly CellReading[],
  minConfidence: number = MIN_CELL_CONFIDENCE,
): ConsistencyResult {
  const total = size * size;
  if (cells.length !== total) {
    throw new Error(
      `expected ${String(total)} cells for a ${String(size)}x${String(size)} grid, got ${String(cells.length)}`,
    );
  }
  const reject = (rejection: Rejection): ConsistencyResult => ({
    accepted: false,
    rejection,
  });

  for (const [index, cell] of cells.entries()) {
    const row = Math.floor(index / size);
    const col = index % size;
    if (!Number.isInteger(cell.value) || cell.value < 0 || cell.value >= total) {
      return reject({
        reason: "out-of-range",
        row,
        col,
        message: `cell ${String(row)},${String(col)} read ${String(cell.value)}, not a tile on a ${String(size)}x${String(size)} board`,
      });
    }
    // Written as a negation so that NaN or a missing confidence is rejected, not accepted.
    if (!(cell.confidence >= minConfidence)) {
      return reject({
        reason: "low-confidence",
        row,
        col,
        message: `low confidence at cell ${String(row)},${String(col)} (${Number.isFinite(cell.confidence) ? cell.confidence.toFixed(2) : "no confidence"} for ${String(cell.value)})`,
      });
    }
  }

  const counts = new Array<number>(total).fill(0);
  for (const cell of cells) counts[cell.value] = (counts[cell.value] ?? 0) + 1;
  if ((counts[0] ?? 0) === 0) return reject({ reason: "no-blank", message: "no empty slot found" });
  const blanks = counts[0] ?? 0;
  if (blanks > 1) {
    const many = blanks === 2 ? "two" : String(blanks);
    return reject({ reason: "two-blanks", message: `${many} empty slots found` });
  }
  for (let value = 1; value < total; value++) {
    if ((counts[value] ?? 0) > 1) {
      const missing = counts.findIndex((count, v) => v > 0 && count === 0);
      return reject({
        reason: "duplicate",
        value,
        missing,
        message: `duplicate tile ${String(value)}, tile ${String(missing)} missing`,
      });
    }
  }

  const board: Board = { size, cells: cells.map((c) => c.value) };
  if (!isSolvable(board)) {
    return reject({
      reason: "unsolvable",
      message: "this arrangement cannot be solved, so a tile was probably misread",
    });
  }
  return { accepted: true, board };
}
