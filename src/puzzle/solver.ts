import { type Board, type Direction, type Move, isSolvable } from "./board.ts";
import { type PatternTables } from "./pattern-db.ts";
import { solveWithPatterns } from "./pattern-solver.ts";

export type SolveResult =
  | {
      status: "solved";
      moves: Move[];
      /** Present (true) only when the time budget ran out and the moves are not guaranteed to be the fewest. */
      notGuaranteedShortest?: true;
    }
  | { status: "unsolvable" };

export interface SolveOptions {
  /** Pattern database for 4x4 boards; without it 4x4 uses the slower Manhattan + linear-conflict search. */
  tables?: PatternTables;
  /** Milliseconds of optimal search on a 4x4 board before falling back (default 20 s). */
  budgetMs?: number;
}

export const DEFAULT_BUDGET_MS = 20_000;

// IDA* with Manhattan distance + linear conflict. The heuristic is evaluated incrementally:
// Manhattan distance by a per-(tile, position) table, linear conflict by per-line lookup tables
// indexed by the line's contents. 3x3 uses this search alone. 4x4 normally uses the 6-6-3 pattern
// database, which is built ahead of time and shipped with the app (see pattern-solver.ts), and falls
// back to this search only when no table is supplied.

interface Tables {
  n: number;
  cellCount: number;
  /** distance[tile * cellCount + position]: Manhattan distance of a tile at a position from its goal. */
  distance: Uint8Array;
  /** Linear-conflict penalty (already ×2) for a row / column, indexed by its contents in base cellCount. */
  rowConflict: Uint8Array[];
  colConflict: Uint8Array[];
  /** place[position]: base-cellCount weight of that position within its row / column code. */
  rowWeight: Int32Array;
  colWeight: Int32Array;
}

const tablesBySize = new Map<number, Tables>();

function countLineConflicts(
  values: readonly number[],
  goalLine: number,
  isRow: boolean,
  n: number,
): number {
  // Tiles in their goal line, with their goal index along it; conflicting pairs are inverted.
  const goalIndexes: number[] = [];
  for (const tile of values) {
    if (tile === 0) continue;
    const goalPos = tile - 1;
    const line = isRow ? Math.floor(goalPos / n) : goalPos % n;
    if (line === goalLine) goalIndexes.push(isRow ? goalPos % n : Math.floor(goalPos / n));
    else goalIndexes.push(-1);
  }
  // Greedy removal of the most-conflicted tile: each removal costs 2 extra moves.
  const present = goalIndexes.map((g) => g >= 0);
  let penalty = 0;
  for (;;) {
    let worst = -1;
    let worstCount = 0;
    for (let i = 0; i < goalIndexes.length; i++) {
      if (!present[i]) continue;
      let count = 0;
      for (let j = 0; j < goalIndexes.length; j++) {
        if (i === j || !present[j]) continue;
        const gi = goalIndexes[i] ?? 0;
        const gj = goalIndexes[j] ?? 0;
        if ((i < j && gi > gj) || (i > j && gi < gj)) count++;
      }
      if (count > worstCount) {
        worstCount = count;
        worst = i;
      }
    }
    if (worst < 0) return penalty;
    present[worst] = false;
    penalty += 2;
  }
}

function buildTables(n: number): Tables {
  const cellCount = n * n;
  const distance = new Uint8Array(cellCount * cellCount);
  for (let tile = 1; tile < cellCount; tile++) {
    for (let pos = 0; pos < cellCount; pos++) {
      const goal = tile - 1;
      distance[tile * cellCount + pos] =
        Math.abs(Math.floor(goal / n) - Math.floor(pos / n)) + Math.abs((goal % n) - (pos % n));
    }
  }
  const codes = cellCount ** n;
  const rowConflict: Uint8Array[] = [];
  const colConflict: Uint8Array[] = [];
  for (const isRow of [true, false]) {
    for (let line = 0; line < n; line++) {
      const table = new Uint8Array(codes);
      for (let code = 0; code < codes; code++) {
        const values: number[] = [];
        let rest = code;
        for (let i = 0; i < n; i++) {
          values.push(rest % cellCount);
          rest = Math.floor(rest / cellCount);
        }
        // Codes holding a tile twice never occur; whatever they score is never read.
        table[code] = countLineConflicts(values, line, isRow, n);
      }
      (isRow ? rowConflict : colConflict).push(table);
    }
  }
  const rowWeight = new Int32Array(cellCount);
  const colWeight = new Int32Array(cellCount);
  for (let pos = 0; pos < cellCount; pos++) {
    rowWeight[pos] = cellCount ** (pos % n);
    colWeight[pos] = cellCount ** Math.floor(pos / n);
  }
  return {
    n,
    cellCount,
    distance,
    rowConflict,
    colConflict,
    rowWeight,
    colWeight,
  };
}

function getTables(n: number): Tables {
  let tables = tablesBySize.get(n);
  if (!tables) {
    tables = buildTables(n);
    tablesBySize.set(n, tables);
  }
  return tables;
}

const DIRECTIONS: readonly Direction[] = ["up", "down", "left", "right"];

/** Search state mutated in place and undone on backtrack. */
class Search {
  readonly cells: Int8Array;
  readonly rowCode: Int32Array;
  readonly colCode: Int32Array;
  readonly path: Int8Array; // tile slid at each depth
  readonly pathDirection: Int8Array; // index into DIRECTIONS
  blank: number;
  manhattan = 0;
  conflict = 0;
  nextBound = Infinity;
  found = false;

  constructor(
    private readonly t: Tables,
    board: Board,
  ) {
    const { n, cellCount } = t;
    this.cells = Int8Array.from(board.cells);
    this.rowCode = new Int32Array(n);
    this.colCode = new Int32Array(n);
    this.path = new Int8Array(256);
    this.pathDirection = new Int8Array(256);
    this.blank = this.cells.indexOf(0);
    for (let pos = 0; pos < cellCount; pos++) {
      const tile = this.cells[pos] ?? 0;
      const row = Math.floor(pos / n);
      const col = pos % n;
      this.rowCode[row] = (this.rowCode[row] ?? 0) + tile * (t.rowWeight[pos] ?? 0);
      this.colCode[col] = (this.colCode[col] ?? 0) + tile * (t.colWeight[pos] ?? 0);
      if (tile !== 0) this.manhattan += t.distance[tile * cellCount + pos] ?? 0;
    }
    for (let i = 0; i < n; i++) {
      this.conflict +=
        (t.rowConflict[i]?.[this.rowCode[i] ?? 0] ?? 0) +
        (t.colConflict[i]?.[this.colCode[i] ?? 0] ?? 0);
    }
  }

  estimateDistance(): number {
    return this.manhattan + this.conflict;
  }

  /** Slides the tile at `from` into the blank; returns nothing, state is updated incrementally. */
  private swap(from: number): void {
    const { n, cellCount, distance, rowConflict, colConflict, rowWeight, colWeight } = this.t;
    const to = this.blank;
    const tile = this.cells[from] ?? 0;
    const fromRow = Math.floor(from / n);
    const fromCol = from % n;
    const toRow = Math.floor(to / n);
    const toCol = to % n;
    // Remove the old conflict of every line that changes, then add the new one.
    const touchedRows = fromRow === toRow ? [fromRow] : [fromRow, toRow];
    const touchedCols = fromCol === toCol ? [fromCol] : [fromCol, toCol];
    for (const r of touchedRows) this.conflict -= rowConflict[r]?.[this.rowCode[r] ?? 0] ?? 0;
    for (const c of touchedCols) this.conflict -= colConflict[c]?.[this.colCode[c] ?? 0] ?? 0;
    this.manhattan +=
      (distance[tile * cellCount + to] ?? 0) - (distance[tile * cellCount + from] ?? 0);
    this.rowCode[fromRow] = (this.rowCode[fromRow] ?? 0) - tile * (rowWeight[from] ?? 0);
    this.colCode[fromCol] = (this.colCode[fromCol] ?? 0) - tile * (colWeight[from] ?? 0);
    this.rowCode[toRow] = (this.rowCode[toRow] ?? 0) + tile * (rowWeight[to] ?? 0);
    this.colCode[toCol] = (this.colCode[toCol] ?? 0) + tile * (colWeight[to] ?? 0);
    for (const r of touchedRows) this.conflict += rowConflict[r]?.[this.rowCode[r] ?? 0] ?? 0;
    for (const c of touchedCols) this.conflict += colConflict[c]?.[this.colCode[c] ?? 0] ?? 0;
    this.cells[to] = tile;
    this.cells[from] = 0;
    this.blank = from;
  }

  /** Depth-first search to `bound`; `last` is the direction index of the previous move (to skip undoing it). */
  descend(depth: number, bound: number, last: number): void {
    const f = depth + this.estimateDistance();
    if (f > bound) {
      if (f < this.nextBound) this.nextBound = f;
      return;
    }
    if (this.manhattan === 0) {
      this.found = true;
      this.pathLength = depth;
      return;
    }
    const n = this.t.n;
    const blank = this.blank;
    const row = Math.floor(blank / n);
    const col = blank % n;
    for (let d = 0; d < 4; d++) {
      // d is the direction the tile slides; the tile sits on the opposite side of the blank.
      if (last >= 0 && d === (last ^ 1)) continue;
      let from: number;
      if (d === 0)
        from = row + 1 < n ? blank + n : -1; // up: tile is below the blank
      else if (d === 1)
        from = row > 0 ? blank - n : -1; // down: tile is above
      else if (d === 2)
        from = col + 1 < n ? blank + 1 : -1; // left: tile is right of blank
      else from = col > 0 ? blank - 1 : -1; // right: tile is left of blank
      if (from < 0) continue;
      this.path[depth] = this.cells[from] ?? 0;
      this.pathDirection[depth] = d;
      this.swap(from);
      this.descend(depth + 1, bound, d);
      if (this.found) return;
      this.swap(blank); // the tile now sits where the blank was; slide it back
    }
  }

  pathLength = 0;
}

/** Without this a duplicated tile makes the search run forever, since no goal is reachable. */
function checkPermutation(board: Board): void {
  const total = board.size * board.size;
  const seen = new Set(board.cells);
  const valid =
    board.cells.length === total &&
    seen.size === total &&
    board.cells.every((v) => Number.isInteger(v) && v >= 0 && v < total);
  if (!valid) {
    throw new Error(
      `board is not a permutation of 0..${String(total - 1)}: [${board.cells.join(", ")}]`,
    );
  }
}

export function solve(board: Board, options: SolveOptions = {}): SolveResult {
  checkPermutation(board);
  if (!isSolvable(board)) return { status: "unsolvable" };
  if (board.size === 4 && options.tables) {
    const solution = solveWithPatterns(
      board,
      options.tables,
      options.budgetMs ?? DEFAULT_BUDGET_MS,
    );
    return solution.optimal
      ? { status: "solved", moves: solution.moves }
      : { status: "solved", moves: solution.moves, notGuaranteedShortest: true };
  }
  const search = new Search(getTables(board.size), board);
  let bound = search.estimateDistance();
  while (!search.found) {
    search.nextBound = Infinity;
    search.descend(0, bound, -1);
    bound = search.nextBound;
  }
  const moves: Move[] = [];
  for (let i = 0; i < search.pathLength; i++) {
    moves.push({
      tile: search.path[i] ?? 0,
      direction: DIRECTIONS[search.pathDirection[i] ?? 0] ?? "up",
    });
  }
  return { status: "solved", moves };
}
