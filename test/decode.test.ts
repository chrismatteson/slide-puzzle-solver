import { describe, expect, it } from "vitest";
import { decodePermutation, solveAssignment } from "../src/vision/decode.ts";
import { checkReading } from "../src/puzzle/consistency.ts";

/** Probabilities that put `weight` on `value` and share the rest evenly. */
function makeRow(count: number, value: number, weight: number): number[] {
  return Array.from({ length: count }, (_, v) =>
    v === value ? weight : (1 - weight) / (count - 1),
  );
}

function makeProbabilities(values: number[], weight = 0.95): number[][] {
  return values.map((v) => makeRow(values.length, v, weight));
}

const SOLVABLE_3 = [1, 2, 3, 4, 5, 6, 7, 0, 8];

describe("solveAssignment", () => {
  it("finds the cheapest assignment", () => {
    // Greedy row-by-row would give row 0 its cheapest column 1 and total 1+3+... ; the optimum is 0,1,2 diag swap.
    const cost = [
      [4, 1, 3],
      [2, 0, 5],
      [3, 2, 2],
    ];
    const chosen = solveAssignment(cost);
    expect(new Set(chosen).size).toBe(3);
    expect(chosen.reduce((s, c, r) => s + (cost[r]?.[c] ?? 0), 0)).toBe(5);
  });

  it("matches brute force on random 5x5 matrices", () => {
    let seed = 12345;
    const random = (): number => {
      seed = (seed * 1103515245 + 12345) % 2147483648;
      return seed / 2147483648;
    };
    const permutations = (items: number[]): number[][] =>
      items.length <= 1
        ? [items]
        : items.flatMap((x, i) =>
            permutations(items.filter((_, j) => j !== i)).map((rest) => [x, ...rest]),
          );
    for (let trial = 0; trial < 20; trial++) {
      const cost = Array.from({ length: 5 }, () => Array.from({ length: 5 }, random));
      const total = (p: number[]): number => p.reduce((s, c, r) => s + (cost[r]?.[c] ?? 0), 0);
      const optimum = Math.min(...permutations([0, 1, 2, 3, 4]).map(total));
      expect(total(solveAssignment(cost))).toBeCloseTo(optimum, 9);
    }
  });
});

describe("decodePermutation", () => {
  it("returns the permutation when every cell is sure", () => {
    const decoded = decodePermutation(makeProbabilities(SOLVABLE_3));
    expect(decoded.values).toEqual(SOLVABLE_3);
    expect(Math.min(...decoded.confidences)).toBeGreaterThan(0.99);
  });

  it("settles a hesitant cell by the values the other cells use", () => {
    const probabilities = makeProbabilities(SOLVABLE_3);
    // Cell 8 (really 8) leans towards 7, which cell 6 already holds firmly; 8 is the only value left.
    probabilities[8] = makeRow(9, 7, 0.5).map((p, v) => (v === 8 ? 0.4 : p));
    const decoded = decodePermutation(probabilities);
    expect(decoded.values).toEqual(SOLVABLE_3);
    // Taken alone the cell would have been read as 7: a duplicate.
    const alone = probabilities.map((row) => row.indexOf(Math.max(...row)));
    expect(alone[8]).toBe(7);
  });

  it("is deterministic and reports a tie as a coin flip, which is not accepted", () => {
    const uniform = Array.from({ length: 9 }, () => new Array<number>(9).fill(1 / 9));
    const first = decodePermutation(uniform);
    expect(decodePermutation(uniform).values).toEqual(first.values);
    expect([...first.values].sort((a, b) => a - b)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8]);
    for (const c of first.confidences) expect(c).toBeCloseTo(0.5, 9);
    const cells = first.values.map((value, i) => ({
      value,
      confidence: first.confidences[i] ?? 0,
    }));
    expect(checkReading(3, cells).accepted).toBe(false);
  });

  it("does not accept a board whose two best assignments are nearly equally likely", () => {
    // Cells 0 and 1 each read 1 or 2 with the same odds: swapping them costs nothing.
    const probabilities = makeProbabilities(SOLVABLE_3);
    probabilities[0] = makeRow(9, 1, 0.45).map((p, v) => (v === 2 ? 0.45 : p));
    probabilities[1] = makeRow(9, 2, 0.45).map((p, v) => (v === 1 ? 0.44 : p));
    const decoded = decodePermutation(probabilities);
    const cells = decoded.values.map((value, i) => ({
      value,
      confidence: decoded.confidences[i] ?? 0,
    }));
    expect(decoded.confidences[0]).toBeLessThan(0.55);
    expect(checkReading(3, cells).accepted).toBe(false);
  });

  it("keeps a confidently swapped board unsolvable instead of repairing it", () => {
    const swapped = [1, 2, 3, 4, 5, 6, 8, 0, 7]; // two tiles exchanged: parity is odd
    const decoded = decodePermutation(makeProbabilities(swapped));
    expect(decoded.values).toEqual(swapped);
    const cells = decoded.values.map((value, i) => ({
      value,
      confidence: decoded.confidences[i] ?? 0,
    }));
    const result = checkReading(3, cells);
    expect(result.accepted).toBe(false);
    if (!result.accepted) expect(result.rejection.reason).toBe("unsolvable");
  });

  it("decodes the empty slot like any other class", () => {
    const decoded = decodePermutation(makeProbabilities([0, 1, 2, 3, 4, 5, 6, 7, 8]));
    expect(decoded.values[0]).toBe(0);
  });

  it("rejects a matrix that is not square", () => {
    expect(() => decodePermutation([[1, 0], [1], [1, 0]])).toThrow(/probabilities per cell/);
  });
});
