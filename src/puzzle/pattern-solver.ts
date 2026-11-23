import { type Board, type Direction, type Move } from "./board.ts";
import {
  DIAGONAL_TILES,
  type PatternTables,
  UPPER_TILES,
  mirrorPosition,
  mirrorTile,
} from "./pattern-db.ts";

// IDA* for the 4x4 puzzle with the additive 6-6-3 pattern database (see pattern-db.ts). The three
// table indexes are updated with one addition per move and the heuristic is carried down the
// recursion as an argument, so there is nothing to undo but the board itself.

export interface PatternSolution {
  moves: Move[];
  /** False when the time budget ran out and a weighted (faster, longer) search produced the moves. */
  optimal: boolean;
}

/**
 * Weighted IDA* multiplies h by this. On the hardest state tried (h = 62) weight 1.5 gave 88 moves
 * in 50 ms, 1.25 gave 82 in 0.5 s, 1.15 had not finished in 80 s, and 2 gave a poor 116 moves.
 */
const FALLBACK_WEIGHT = 1.5;
/** Nodes between clock reads; reading the clock per node would dominate the search. */
const CLOCK_INTERVAL = 1 << 16;

const GROUP_COUNT = 3;
const CORNER = 0;
const MIRRORED_CORNER = 1;
const DIAGONAL = 2;

const NEIGHBOURS: readonly (readonly number[])[] = Array.from({ length: 16 }, (_, p) => {
  const row = p >> 2;
  const col = p & 3;
  const found: number[] = [];
  if (row > 0) found.push(p - 4);
  if (row < 3) found.push(p + 4);
  if (col > 0) found.push(p - 1);
  if (col < 3) found.push(p + 1);
  return found;
});

/** group[tile], and contribution[tile * 16 + position]: that tile's share of its group's table index. */
function createIndexParts(): { group: Uint8Array; contribution: Int32Array } {
  const group = new Uint8Array(16);
  const contribution = new Int32Array(16 * 16);
  const assign = (tile: number, groupId: number, slot: number, mirrored: boolean): void => {
    group[tile] = groupId;
    for (let position = 0; position < 16; position++) {
      contribution[tile * 16 + position] =
        (mirrored ? mirrorPosition(position) : position) << (4 * slot);
    }
  };
  UPPER_TILES.forEach((tile, slot) => {
    assign(tile, CORNER, slot, false);
    assign(mirrorTile(tile), MIRRORED_CORNER, slot, true);
  });
  DIAGONAL_TILES.forEach((tile, slot) => {
    assign(tile, DIAGONAL, slot, false);
  });
  return { group, contribution };
}

const INDEX_PARTS = createIndexParts();

class PatternSearch {
  private readonly cells: Int8Array;
  private readonly tables: readonly Uint8Array[];
  private readonly index = new Int32Array(GROUP_COUNT);
  private readonly pathTile = new Int8Array(128);
  private readonly pathFrom = new Int8Array(128);
  private readonly pathTo = new Int8Array(128);
  private nodes = 0;
  private bound = 0;
  private nextBound = Infinity;
  private blank: number;
  pathLength = 0;
  found = false;
  timedOut = false;

  constructor(
    board: Board,
    tables: PatternTables,
    private readonly weight: number,
    private readonly deadline: number,
  ) {
    this.cells = Int8Array.from(board.cells);
    this.blank = this.cells.indexOf(0);
    this.tables = [tables.corner, tables.corner, tables.diagonal];
    this.cells.forEach((tile, position) => {
      if (tile !== 0) {
        const group = INDEX_PARTS.group[tile] ?? 0;
        this.index[group] =
          (this.index[group] ?? 0) + (INDEX_PARTS.contribution[tile * 16 + position] ?? 0);
      }
    });
  }

  private lookup(group: number): number {
    return this.tables[group]?.[this.index[group] ?? 0] ?? 0;
  }

  /** Sum of the three tables for the current board. */
  estimateCurrentDistance(): number {
    return this.lookup(CORNER) + this.lookup(MIRRORED_CORNER) + this.lookup(DIAGONAL);
  }

  /** One deepening iteration; returns the smallest f above `bound` seen, which is the next bound. */
  deepen(bound: number): number {
    this.bound = bound;
    this.nextBound = Infinity;
    this.descend(0, this.estimateCurrentDistance(), -1);
    return this.nextBound;
  }

  /** `cameFrom` is where the blank was before the last move: sliding that tile back is never useful. */
  private descend(depth: number, heuristic: number, cameFrom: number): void {
    if (heuristic === 0) {
      this.found = true;
      this.pathLength = depth;
      return;
    }
    if ((++this.nodes & (CLOCK_INTERVAL - 1)) === 0 && performance.now() > this.deadline) {
      this.timedOut = true;
      return;
    }
    const blank = this.blank;
    for (const from of NEIGHBOURS[blank] ?? []) {
      if (from === cameFrom) continue;
      const tile = this.cells[from] ?? 0;
      const group = INDEX_PARTS.group[tile] ?? 0;
      const oldIndex = this.index[group] ?? 0;
      const newIndex =
        oldIndex +
        (INDEX_PARTS.contribution[tile * 16 + blank] ?? 0) -
        (INDEX_PARTS.contribution[tile * 16 + from] ?? 0);
      const table = this.tables[group];
      const newHeuristic = heuristic - (table?.[oldIndex] ?? 0) + (table?.[newIndex] ?? 0);
      const f = depth + 1 + this.weight * newHeuristic;
      if (f > this.bound) {
        if (f < this.nextBound) this.nextBound = f;
        continue;
      }
      this.pathTile[depth] = tile;
      this.pathFrom[depth] = from;
      this.pathTo[depth] = blank;
      this.cells[blank] = tile;
      this.cells[from] = 0;
      this.index[group] = newIndex;
      this.blank = from;
      this.descend(depth + 1, newHeuristic, blank);
      if (this.found || this.timedOut) return;
      this.blank = blank;
      this.index[group] = oldIndex;
      this.cells[from] = tile;
      this.cells[blank] = 0;
    }
  }

  readMoves(): Move[] {
    const moves: Move[] = [];
    for (let i = 0; i < this.pathLength; i++) {
      const delta = (this.pathTo[i] ?? 0) - (this.pathFrom[i] ?? 0);
      const direction: Direction =
        delta === -4 ? "up" : delta === 4 ? "down" : delta === -1 ? "left" : "right";
      moves.push({ tile: this.pathTile[i] ?? 0, direction });
    }
    return moves;
  }
}

/** Runs IDA* until a solution, or until `deadline` (a performance.now() time); then `timedOut` is set. */
function runSearch(
  board: Board,
  tables: PatternTables,
  weight: number,
  deadline: number,
): PatternSearch {
  const search = new PatternSearch(board, tables, weight, deadline);
  let bound = weight * search.estimateCurrentDistance();
  while (!search.found && !search.timedOut) bound = search.deepen(bound);
  return search;
}

/**
 * Optimal solution of a solvable 4x4 board, or, if `budgetMs` passes first, a weighted-IDA*
 * solution labelled `optimal: false`. The fallback has no deadline of its own: with h weighted it
 * takes milliseconds even on the hardest states (see test/pattern-solver.test.ts).
 */
export function solveWithPatterns(
  board: Board,
  tables: PatternTables,
  budgetMs: number,
): PatternSolution {
  const optimal = runSearch(board, tables, 1, performance.now() + budgetMs);
  if (optimal.found) return { moves: optimal.readMoves(), optimal: true };
  const fallback = runSearch(board, tables, FALLBACK_WEIGHT, Infinity);
  return { moves: fallback.readMoves(), optimal: false };
}
