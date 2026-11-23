// Board representation shared by the solver, the consistency check and the vision pipeline.
// A board is row-major; 0 is the empty slot. Goal: tiles 1..N²-1 in reading order with the blank at its
// home cell, by default the bottom-right one (numbered boards) but any cell for picture boards.

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
  /** Cell index where the blank sits in the goal; default size²-1 (bottom-right). */
  blankHome?: number;
}

export const GRID_SIZES: readonly GridSize[] = [3, 4];

export function getBlankHome(board: Board): number {
  return board.blankHome ?? board.size * board.size - 1;
}

/** Cell where `tile` belongs when the blank's home is `home`: tiles fill the other cells in reading order. */
export function getGoalPosition(tile: number, home: number): number {
  return tile <= home ? tile - 1 : tile;
}

export function createGoalBoard(size: GridSize, blankHome = size * size - 1): Board {
  const count = size * size;
  if (!Number.isInteger(blankHome) || blankHome < 0 || blankHome >= count) {
    throw new Error(
      `blankHome ${String(blankHome)} is not a cell of a ${String(size)}x${String(size)} board`,
    );
  }
  const cells = Array.from({ length: count }, (_, i) =>
    i === blankHome ? 0 : i < blankHome ? i + 1 : i,
  );
  return blankHome === count - 1 ? { size, cells } : { size, cells, blankHome };
}

export function isGoalBoard(board: Board): boolean {
  const goal = createGoalBoard(board.size, getBlankHome(board)).cells;
  return board.cells.every((v, i) => v === goal[i]);
}

/**
 * Parity test for a board that is already a valid permutation. Every move swaps the blank with a
 * tile (flipping the permutation's parity) and moves the blank one cell (changing its taxicab
 * distance from home by one), so permutation parity plus that distance keeps its parity; the goal
 * has both zero. Parity of a permutation is (cells - cycles) mod 2.
 */
export function isSolvable(board: Board): boolean {
  const n = board.size;
  const home = getBlankHome(board);
  // Where each cell's content belongs.
  const target = board.cells.map((tile) => (tile === 0 ? home : getGoalPosition(tile, home)));
  const visited = new Array<boolean>(target.length).fill(false);
  let cycles = 0;
  for (let start = 0; start < target.length; start++) {
    if (visited[start]) continue;
    cycles++;
    for (let at = start; !visited[at]; at = target[at] ?? start) visited[at] = true;
  }
  const blank = board.cells.indexOf(0);
  const distance =
    Math.abs(Math.floor(blank / n) - Math.floor(home / n)) + Math.abs((blank % n) - (home % n));
  return (target.length - cycles + distance) % 2 === 0;
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
  return { ...board, cells };
}

export function applyMoves(board: Board, moves: readonly Move[]): Board {
  return moves.reduce(applyMove, board);
}
