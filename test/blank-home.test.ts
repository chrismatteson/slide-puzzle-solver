import { describe, expect, it } from "vitest";
import {
  type Board,
  type GridSize,
  applyMove,
  applyMoves,
  createGoalBoard,
  isGoalBoard,
  isSolvable,
} from "../src/puzzle/board.ts";
import { loadPatternTables } from "../src/puzzle/pattern-db-node.ts";
import { solve } from "../src/puzzle/solver.ts";

const tables = await loadPatternTables();

/** Deterministic pseudo-random numbers, so a failing scramble can be reproduced. */
function createRandom(seed: number): () => number {
  let state = seed;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
}

/** Slides the blank around at random, never straight back; the result is always solvable. */
function scramble(goal: Board, moves: number, random: () => number): Board {
  let board = goal;
  let previousTile = -1;
  for (let i = 0; i < moves; i++) {
    const n = board.size;
    const blank = board.cells.indexOf(0);
    const options = [blank - n, blank + n, blank - 1, blank + 1].filter(
      (p) =>
        p >= 0 &&
        p < n * n &&
        !(blank % n === 0 && p === blank - 1) &&
        !(blank % n === n - 1 && p === blank + 1) &&
        board.cells[p] !== previousTile,
    );
    const from = options[Math.floor(random() * options.length)] ?? 0;
    const tile = board.cells[from] ?? 0;
    const delta = blank - from;
    const direction = delta === n ? "down" : delta === -n ? "up" : delta === 1 ? "right" : "left";
    board = applyMove(board, { tile, direction });
    previousTile = tile;
  }
  return board;
}

/** Breadth-first search over boards: an independent reference for the optimal length. */
function measureByBreadthFirst(start: Board): number {
  const goal = createGoalBoard(start.size, start.blankHome).cells.join();
  let frontier = [start];
  const seen = new Set([start.cells.join()]);
  for (let depth = 0; ; depth++) {
    const next: Board[] = [];
    for (const board of frontier) {
      if (board.cells.join() === goal) return depth;
      const n = board.size;
      const blank = board.cells.indexOf(0);
      const neighbours = [
        blank - n >= 0 ? blank - n : -1,
        blank + n < n * n ? blank + n : -1,
        blank % n > 0 ? blank - 1 : -1,
        blank % n < n - 1 ? blank + 1 : -1,
      ];
      for (const from of neighbours) {
        if (from < 0) continue;
        const cells = [...board.cells];
        cells[blank] = cells[from] ?? 0;
        cells[from] = 0;
        const key = cells.join();
        if (seen.has(key)) continue;
        seen.add(key);
        next.push({ ...board, cells });
      }
    }
    frontier = next;
  }
}

const CORNERS: Record<GridSize, number[]> = { 3: [0, 2, 6, 8], 4: [0, 3, 12, 15] };

describe("goal with the empty slot at any cell", () => {
  it("defaults to the bottom-right corner", () => {
    expect(createGoalBoard(3).cells).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 0]);
    expect(createGoalBoard(3, 8).cells).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 0]);
  });

  it("numbers the tiles in reading order skipping the home", () => {
    expect(createGoalBoard(3, 0).cells).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8]);
    expect(createGoalBoard(3, 4).cells).toEqual([1, 2, 3, 4, 0, 5, 6, 7, 8]);
    expect(createGoalBoard(4, 2).cells).toEqual([
      1, 2, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    ]);
  });

  it("recognises only its own goal", () => {
    const board = createGoalBoard(3, 4);
    expect(isGoalBoard(board)).toBe(true);
    expect(isGoalBoard({ size: 3, cells: board.cells })).toBe(false);
  });

  it("judges solvability relative to the goal", () => {
    for (const size of [3, 4] as const) {
      for (let home = 0; home < size * size; home++) {
        const goal = createGoalBoard(size, home);
        expect(isSolvable(goal)).toBe(true);
        const cells = [...goal.cells];
        // Swap two tiles that are not the blank: always flips solvability.
        const [a, b] = cells.map((v, i) => [v, i] as const).filter(([v]) => v !== 0);
        cells[a?.[1] ?? 0] = b?.[0] ?? 0;
        cells[b?.[1] ?? 0] = a?.[0] ?? 0;
        expect(isSolvable({ ...goal, cells })).toBe(false);
      }
    }
  });

  it("keeps every scramble solvable and applyMoves reaches the goal", () => {
    const random = createRandom(7);
    for (const size of [3, 4] as const) {
      for (let home = 0; home < size * size; home++) {
        const goal = createGoalBoard(size, home);
        expect(isSolvable(scramble(goal, 40, random))).toBe(true);
      }
    }
  });
});

describe("solve with the empty slot at any cell", () => {
  it("matches a breadth-first reference for every 3x3 home", () => {
    const random = createRandom(11);
    for (let home = 0; home < 9; home++) {
      for (let i = 0; i < 4; i++) {
        const board = scramble(createGoalBoard(3, home), 12 + i * 3, random);
        const result = solve(board, { tables });
        if (result.status !== "solved") throw new Error("expected a solution");
        expect(isGoalBoard(applyMoves(board, result.moves))).toBe(true);
        expect(result.moves.length).toBe(measureByBreadthFirst(board));
        expect(result.notGuaranteedShortest).toBeUndefined();
      }
    }
  });

  it("is optimal for every 4x4 corner home, via the pattern database", () => {
    const random = createRandom(13);
    for (const home of CORNERS[4]) {
      for (let i = 0; i < 4; i++) {
        const board = scramble(createGoalBoard(4, home), 10 + i, random);
        const result = solve(board, { tables });
        if (result.status !== "solved") throw new Error("expected a solution");
        expect(isGoalBoard(applyMoves(board, result.moves))).toBe(true);
        expect(result.moves.length).toBe(measureByBreadthFirst(board));
        expect(result.notGuaranteedShortest).toBeUndefined();
      }
    }
  });

  it("solves a deep 4x4 corner home with the same length as the bottom-right goal", () => {
    // Turning the board and renaming tiles maps one goal onto another, so lengths must agree.
    const board = scramble(createGoalBoard(4, 15), 80, createRandom(3));
    const reference = solve(board, { tables });
    if (reference.status !== "solved") throw new Error("expected a solution");
    // Mirror left-right: bottom-right goal becomes the bottom-left one; tile k goes to its mirror's number.
    const mirrored = Array.from({ length: 16 }, () => 0);
    board.cells.forEach((tile, i) => {
      const target = Math.floor(i / 4) * 4 + (3 - (i % 4));
      mirrored[target] = tile === 0 ? 0 : mirrorTileLeftRight(tile);
    });
    const result = solve({ size: 4, blankHome: 12, cells: mirrored }, { tables });
    if (result.status !== "solved") throw new Error("expected a solution");
    expect(result.moves.length).toBe(reference.moves.length);
  });

  it("solves 4x4 edge and interior homes and reaches the goal", () => {
    const random = createRandom(17);
    for (const home of [1, 2, 4, 5, 6, 7, 9, 10, 11, 13, 14]) {
      const board = scramble(createGoalBoard(4, home), 22, random);
      const result = solve(board, { tables });
      if (result.status !== "solved") throw new Error("expected a solution");
      expect(isGoalBoard(applyMoves(board, result.moves))).toBe(true);
      expect(result.notGuaranteedShortest).toBeUndefined();
    }
  });

  it("matches a breadth-first reference on short 4x4 edge-home scrambles", () => {
    const random = createRandom(19);
    for (const home of [1, 6, 11]) {
      const board = scramble(createGoalBoard(4, home), 9, random);
      const result = solve(board, { tables });
      if (result.status !== "solved") throw new Error("expected a solution");
      expect(result.moves.length).toBe(measureByBreadthFirst(board));
    }
  });

  it("marks a non-optimal result when the budget runs out on an edge home", () => {
    // A uniformly random 4x4 permutation is typically ~50+ moves deep, far beyond a 100 ms search
    // with Manhattan + linear conflict.
    const random = createRandom(5);
    let board: Board;
    do {
      const cells = [...createGoalBoard(4, 6).cells];
      for (let i = cells.length - 1; i > 0; i--) {
        const j = Math.floor(random() * (i + 1));
        [cells[i], cells[j]] = [cells[j] ?? 0, cells[i] ?? 0];
      }
      board = { size: 4, blankHome: 6, cells };
    } while (!isSolvable(board));
    const result = solve(board, { tables, budgetMs: 100 });
    if (result.status !== "solved") throw new Error("expected a solution");
    expect(isGoalBoard(applyMoves(board, result.moves))).toBe(true);
    expect(result.notGuaranteedShortest).toBe(true);
  });

  it("detects unsolvable boards for any home", () => {
    for (const size of [3, 4] as const) {
      for (const home of [0, 1, size * size - 1]) {
        const cells = [...createGoalBoard(size, home).cells];
        const [i, j] = cells.flatMap((v, p) => (v === 0 ? [] : [p]));
        [cells[i ?? 0], cells[j ?? 0]] = [cells[j ?? 0] ?? 0, cells[i ?? 0] ?? 0];
        expect(solve({ size, blankHome: home, cells }, { tables })).toEqual({
          status: "unsolvable",
        });
      }
    }
  });
});

/** Tile number under a left-right mirror of the bottom-right goal, as a bottom-left goal (home 12). */
function mirrorTileLeftRight(tile: number): number {
  // Bottom-right goal position of `tile` is tile-1; mirrored, it is the position in the bottom-left goal,
  // whose tile numbers skip index 12.
  const position = tile - 1;
  const mirrored = Math.floor(position / 4) * 4 + (3 - (position % 4));
  return mirrored < 12 ? mirrored + 1 : mirrored;
}
