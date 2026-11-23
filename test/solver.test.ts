import { describe, expect, it } from "vitest";
import {
  type Board,
  applyMoves,
  createGoalBoard,
  isGoalBoard,
  isSolvable,
} from "../src/puzzle/board.ts";
import { loadPatternTables } from "../src/puzzle/pattern-db-node.ts";
import { solve } from "../src/puzzle/solver.ts";

// The 4x4 tests use the pattern-database solver, the one the app runs; without tables solve() uses the
// much slower Manhattan + linear-conflict fallback.
const tables = await loadPatternTables();

function makeBoard3(cells: number[]): Board {
  return { size: 3, cells };
}
function makeBoard4(cells: number[]): Board {
  return { size: 4, cells };
}

function solveOptimally(board: Board): number {
  const result = solve(board, { tables });
  if (result.status !== "solved") throw new Error("expected a solution");
  expect(isGoalBoard(applyMoves(board, result.moves))).toBe(true);
  return result.moves.length;
}

describe("solvability", () => {
  it("accepts the goal and rejects a swapped pair, for both sizes", () => {
    expect(isSolvable(createGoalBoard(3))).toBe(true);
    expect(isSolvable(createGoalBoard(4))).toBe(true);
    expect(isSolvable(makeBoard3([2, 1, 3, 4, 5, 6, 7, 8, 0]))).toBe(false);
    expect(isSolvable(makeBoard4([2, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]))).toBe(
      false,
    );
  });

  it("applies the even-width rule with the blank off the bottom row", () => {
    // One slide up from the goal: the blank moves to row 2, a legal position.
    expect(isSolvable(makeBoard4([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 13, 14, 15, 12]))).toBe(
      true,
    );
    // The famous Sam Loyd 14-15 swap is unsolvable.
    expect(isSolvable(makeBoard4([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 14, 0]))).toBe(
      false,
    );
  });
});

describe("3x3 solver", () => {
  it("returns no moves for the goal", () => {
    expect(solveOptimally(createGoalBoard(3))).toBe(0);
  });

  it("finds known 1-, 2- and 4-move solutions", () => {
    expect(solveOptimally(makeBoard3([1, 2, 3, 4, 5, 6, 7, 0, 8]))).toBe(1);
    expect(solveOptimally(makeBoard3([1, 2, 3, 4, 0, 6, 7, 5, 8]))).toBe(2);
    // The blank walked up, left, up, left from the goal corner.
    expect(solveOptimally(makeBoard3([0, 1, 3, 4, 2, 5, 7, 8, 6]))).toBe(4);
  });

  it("solves the 31-move worst case optimally", () => {
    expect(solveOptimally(makeBoard3([8, 6, 7, 2, 5, 4, 3, 0, 1]))).toBe(31);
  });

  it("reports the first move as tile and slide direction", () => {
    const result = solve(makeBoard3([1, 2, 3, 4, 5, 6, 7, 0, 8]));
    expect(result).toEqual({
      status: "solved",
      moves: [{ tile: 8, direction: "left" }],
    });
  });
});

describe("unsolvable input", () => {
  it("is reported without searching", () => {
    const start = performance.now();
    expect(solve(makeBoard3([1, 2, 3, 4, 5, 6, 8, 7, 0]))).toEqual({
      status: "unsolvable",
    });
    expect(solve(makeBoard4([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 14, 0]))).toEqual({
      status: "unsolvable",
    });
    expect(performance.now() - start).toBeLessThan(50);
  });
});

function breadthFirstDepth(start: Board): number {
  // Independent reference: plain BFS over board strings, only used for shallow states.
  const goal = createGoalBoard(start.size).cells.join(",");
  let frontier = [start];
  const seen = new Set([start.cells.join(",")]);
  for (let depth = 0; ; depth++) {
    if (frontier.some((b) => b.cells.join(",") === goal)) return depth;
    const next: Board[] = [];
    for (const b of frontier) {
      const blank = b.cells.indexOf(0);
      const n = b.size;
      const neighbours = [
        blank - n,
        blank + n,
        blank % n > 0 ? blank - 1 : -1,
        blank % n < n - 1 ? blank + 1 : -1,
      ];
      for (const p of neighbours) {
        if (p < 0 || p >= n * n) continue;
        const cells = [...b.cells];
        cells[blank] = cells[p] ?? 0;
        cells[p] = 0;
        const key = cells.join(",");
        if (!seen.has(key)) {
          seen.add(key);
          next.push({ size: n, cells });
        }
      }
    }
    frontier = next;
  }
}

describe("4x4 solver", () => {
  it("matches an independent breadth-first search on shallow states", () => {
    const shallow = [
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 15],
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 12, 13, 14, 11, 15],
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 10, 12, 13, 14, 11, 15],
      [1, 2, 3, 4, 5, 0, 7, 8, 9, 6, 10, 12, 13, 14, 11, 15],
      [5, 1, 2, 3, 9, 6, 7, 4, 13, 10, 11, 8, 0, 14, 15, 12],
    ];
    for (const cells of shallow) {
      const b = makeBoard4(cells);
      expect(solveOptimally(b)).toBe(breadthFirstDepth(b));
    }
  });

  // Measured (Node 22, 4-core container, idle-ish): 2.0 s for both cases together with the pattern
  // database. Timeout is 10x that: a loaded machine took 4.7 s, and vitest's default 5 s killed it.
  it(
    "solves states of 48 and 52 optimal moves well inside the 20 s search budget",
    { timeout: 20_000 },
    () => {
      const cases: [number[], number][] = [
        [[2, 14, 0, 11, 10, 6, 7, 1, 9, 8, 15, 3, 5, 13, 4, 12], 48],
        [[3, 4, 2, 7, 9, 8, 10, 11, 12, 13, 0, 1, 15, 14, 6, 5], 52],
      ];
      for (const [cells, expected] of cases) {
        const start = performance.now();
        const result = solve(makeBoard4(cells), { tables });
        if (result.status !== "solved") throw new Error("expected a solution");
        expect(result.notGuaranteedShortest).toBeUndefined();
        expect(result.moves.length).toBe(expected);
        expect(performance.now() - start).toBeLessThan(10_000);
      }
    },
  );
});

describe("solve on invalid boards", () => {
  it("throws instead of searching when a tile is duplicated", () => {
    expect(() => solve(makeBoard3([1, 1, 3, 4, 5, 6, 7, 8, 0]))).toThrow(/not a permutation/);
  });

  it("throws on a wrong cell count and on out-of-range values", () => {
    expect(() => solve(makeBoard3([1, 2, 3, 4, 5, 6, 7, 0]))).toThrow(/not a permutation/);
    expect(() => solve(makeBoard3([1, 2, 3, 4, 5, 6, 7, 8, 9]))).toThrow(/not a permutation/);
    expect(() => solve(makeBoard3([1, 2, 3, 4, 5, 6, 7, 8, Number.NaN]))).toThrow(
      /not a permutation/,
    );
  });
});
