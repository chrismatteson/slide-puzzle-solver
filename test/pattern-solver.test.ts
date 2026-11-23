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

const tables = await loadPatternTables();

function makeBoard4(cells: number[]): Board {
  return { size: 4, cells };
}

/** Solves with the pattern database and checks the moves really reach the goal. */
function solveChecked(board: Board, budgetMs?: number) {
  const result = solve(board, { tables, ...(budgetMs === undefined ? {} : { budgetMs }) });
  if (result.status !== "solved") throw new Error("expected a solution");
  expect(isGoalBoard(applyMoves(board, result.moves))).toBe(true);
  return result;
}

/**
 * Korf's instances use the goal "blank, 1..15". Turning the board half a turn and renaming tile k
 * to 16 - k maps that goal onto this project's goal (1..15, blank last) and keeps every distance.
 */
function convertFromKorf(cells: number[]): number[] {
  return cells.map((v) => (v === 0 ? 0 : 16 - v)).reverse();
}

// Korf (1985), 100 random 15-puzzle instances, with their published optimal lengths.
const KORF: [number, number[], number][] = [
  [1, [14, 13, 15, 7, 11, 12, 9, 5, 6, 0, 2, 1, 4, 8, 10, 3], 57],
  [2, [13, 5, 4, 10, 9, 12, 8, 14, 2, 3, 7, 1, 0, 15, 11, 6], 55],
  [3, [14, 7, 8, 2, 13, 11, 10, 4, 9, 12, 5, 0, 3, 6, 1, 15], 59],
  [4, [5, 12, 10, 7, 15, 11, 14, 0, 8, 2, 1, 13, 3, 4, 9, 6], 56],
  [5, [4, 7, 14, 13, 10, 3, 9, 12, 11, 5, 6, 15, 1, 2, 8, 0], 56],
  [6, [14, 7, 1, 9, 12, 3, 6, 15, 8, 11, 2, 5, 10, 0, 4, 13], 52],
  [7, [2, 11, 15, 5, 13, 4, 6, 7, 12, 8, 10, 1, 9, 3, 14, 0], 52],
  [8, [12, 11, 15, 3, 8, 0, 4, 2, 6, 13, 9, 5, 14, 1, 10, 7], 50],
  [9, [3, 14, 9, 11, 5, 4, 8, 2, 13, 12, 6, 7, 10, 1, 15, 0], 46],
  [10, [13, 11, 8, 9, 0, 15, 7, 10, 4, 3, 6, 14, 5, 12, 2, 1], 59],
];

// Korf #88, among the hardest 4x4 states (h = 62): optimal search takes far longer than any budget here.
const VERY_HARD = convertFromKorf([15, 14, 8, 12, 10, 11, 9, 13, 2, 6, 5, 1, 3, 7, 4, 0]);

/** Small seeded generator so the timing states are the same on every run. */
function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A uniformly random solvable 4x4 state: shuffle all 16 cells, swap two tiles if that is unsolvable. */
function createRandomBoard(random: () => number): Board {
  const cells = Array.from({ length: 16 }, (_, i) => i);
  for (let i = cells.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [cells[i], cells[j]] = [cells[j] ?? 0, cells[i] ?? 0];
  }
  let board = makeBoard4(cells);
  if (!isSolvable(board)) {
    const [a, b] = cells.flatMap((v, i) => (v !== 0 ? [i] : [])).slice(0, 2) as [number, number];
    [cells[a], cells[b]] = [cells[b] ?? 0, cells[a] ?? 0];
    board = makeBoard4(cells);
  }
  return board;
}

describe("pattern database", () => {
  it("is zero only at the goal", () => {
    expect(tables.estimateDistance(createGoalBoard(4))).toBe(0);
    expect(
      tables.estimateDistance(makeBoard4([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 15])),
    ).toBe(1);
  });
});

describe("4x4 solver with pattern database", () => {
  it("is optimal on Korf's published instances", { timeout: 120_000 }, () => {
    // Measured (Node 22, one core), ms per instance 1-10: 312 21 1023 40 100 18 170 85 2 523.
    for (const [index, cells, optimal] of KORF) {
      const board = makeBoard4(convertFromKorf(cells));
      const result = solveChecked(board);
      expect(result.moves.length, `Korf #${String(index)}`).toBe(optimal);
      expect(tables.estimateDistance(board)).toBeLessThanOrEqual(optimal);
      expect(result.notGuaranteedShortest).toBeUndefined();
    }
  });

  it(
    "agrees with the Manhattan + linear-conflict solver on seeded shuffles",
    { timeout: 60_000 },
    () => {
      const random = createRandom(11);
      for (let i = 0; i < 12; i++) {
        // Short random walks from the goal stay easy enough for the old solver to check.
        let board = createGoalBoard(4);
        for (let step = 0; step < 30; step++) {
          const blank = board.cells.indexOf(0);
          const options = [blank - 4, blank + 4, blank - 1, blank + 1].filter(
            (p) =>
              p >= 0 &&
              p < 16 &&
              !(blank % 4 === 3 && p === blank + 1) &&
              !(blank % 4 === 0 && p === blank - 1),
          );
          const from = options[Math.floor(random() * options.length)] ?? 0;
          const cells = [...board.cells];
          cells[blank] = cells[from] ?? 0;
          cells[from] = 0;
          board = makeBoard4(cells);
        }
        const fast = solveChecked(board);
        const reference = solve(board);
        if (reference.status !== "solved") throw new Error("expected a solution");
        expect(fast.moves.length).toBe(reference.moves.length);
      }
    },
  );

  it("solves 10 seeded uniform random states", { timeout: 300_000 }, () => {
    // Measured (Node 22, one core), ms for the 10 states in order: 16 46 877 93 7 3 26 119 159 99.
    const random = createRandom(2024);
    const millis: number[] = [];
    for (let i = 0; i < 10; i++) {
      const board = createRandomBoard(random);
      const start = performance.now();
      const result = solveChecked(board);
      millis.push(performance.now() - start);
      expect(result.notGuaranteedShortest).toBeUndefined();
    }
    console.log("random 4x4 solve ms", millis.map((m) => Math.round(m)).join(" "));
    expect(Math.max(...millis)).toBeLessThan(30_000);
    expect(millis.sort((a, b) => a - b)[5] ?? 0).toBeLessThan(5_000);
  });

  it("falls back to a labelled non-optimal solution when the budget runs out", () => {
    // Without a budget this state takes minutes; the weighted fallback answers in tens of ms.
    const board = makeBoard4(VERY_HARD);
    const result = solveChecked(board, 50);
    expect(result.notGuaranteedShortest).toBe(true);
    expect(result.moves.length).toBeGreaterThanOrEqual(tables.estimateDistance(board));
  });

  it("reports unsolvable boards without needing the tables", () => {
    expect(
      solve(makeBoard4([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 14, 0]), { tables }),
    ).toEqual({
      status: "unsolvable",
    });
  });
});
