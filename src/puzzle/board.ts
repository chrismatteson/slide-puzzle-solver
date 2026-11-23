// Board representation shared by the solver, the consistency check and the vision pipeline.
// A board is row-major; 0 is the empty slot. Goal: 1..N²-1, then the blank at bottom-right.

export type GridSize = 3 | 4;
export type Direction = "up" | "down" | "left" | "right";

/** One move: the numbered tile slides in `direction` into the empty slot. */
export interface Move {
  tile: number;
  direction: Direction;
}

export interface Board {
  size: GridSize;
  /** Row-major, length size². 0 is the empty slot. */
  cells: readonly number[];
}

export const GRID_SIZES: readonly GridSize[] = [3, 4];

export function createGoalBoard(size: GridSize): Board {
  const cells = Array.from({ length: size * size }, (_, i) => (i + 1) % (size * size));
  return { size, cells };
}

export function isGoalBoard(board: Board): boolean {
  const goal = createGoalBoard(board.size).cells;
  return board.cells.every((v, i) => v === goal[i]);
}

function countInversions(cells: readonly number[]): number {
  let inversions = 0;
  for (let i = 0; i < cells.length; i++) {
    const a = cells[i] ?? 0;
    if (a === 0) continue;
    for (let j = i + 1; j < cells.length; j++) {
      const b = cells[j] ?? 0;
      if (b !== 0 && a > b) inversions++;
    }
  }
  return inversions;
}

/**
 * Parity test for a board that is already a valid permutation. Odd width: the inversion count
 * must be even. Even width: each blank row move flips inversion parity, so inversions plus the
 * blank's row (from the top) must have the same parity as the goal's (inversions 0, row N-1).
 */
export function isSolvable(board: Board): boolean {
  const inversions = countInversions(board.cells);
  if (board.size % 2 === 1) return inversions % 2 === 0;
  const blankRow = Math.floor(board.cells.indexOf(0) / board.size);
  return (inversions + blankRow) % 2 === (board.size - 1) % 2;
}

const DELTAS: Record<Direction, { dr: number; dc: number }> = {
  // The tile slides this way, so the blank is on that side of the tile before the move.
  up: { dr: -1, dc: 0 },
  down: { dr: 1, dc: 0 },
  left: { dr: 0, dc: -1 },
  right: { dr: 0, dc: 1 },
};

/** Applies one move, throwing if the tile is not next to the blank in that direction. */
export function applyMove(board: Board, move: Move): Board {
  const n = board.size;
  const from = board.cells.indexOf(move.tile);
  const blank = board.cells.indexOf(0);
  if (from < 0) throw new Error(`tile ${String(move.tile)} is not on the board`);
  const { dr, dc } = DELTAS[move.direction];
  const row = Math.floor(from / n) + dr;
  const col = (from % n) + dc;
  if (row < 0 || row >= n || col < 0 || col >= n || row * n + col !== blank) {
    throw new Error(`tile ${String(move.tile)} cannot slide ${move.direction}: blank is not there`);
  }
  const cells = [...board.cells];
  cells[blank] = move.tile;
  cells[from] = 0;
  return { size: board.size, cells };
}

export function applyMoves(board: Board, moves: readonly Move[]): Board {
  return moves.reduce(applyMove, board);
}
