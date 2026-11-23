import { describe, expect, it } from "vitest";
import { type CellReading, checkReading } from "../src/puzzle/consistency.ts";

function makeReadings(values: number[], confidence = 0.99): CellReading[] {
  return values.map((value) => ({ value, confidence }));
}

const GOOD_3 = [8, 6, 7, 2, 5, 4, 3, 0, 1];

function findRejection(size: 3 | 4, cells: CellReading[]) {
  const result = checkReading(size, cells);
  if (result.accepted) throw new Error("expected a rejection");
  return result.rejection;
}

describe("checkReading", () => {
  it("accepts a solvable permutation", () => {
    const result = checkReading(3, makeReadings(GOOD_3));
    expect(result.accepted).toBe(true);
  });

  it("rejects a duplicate tile, naming it and the tile that is missing", () => {
    const rejection = findRejection(3, makeReadings([8, 6, 7, 2, 5, 4, 3, 0, 8]));
    expect(rejection).toMatchObject({
      reason: "duplicate",
      value: 8,
      missing: 1,
    });
    expect(rejection.message).toBe("duplicate tile 8, tile 1 missing");
  });

  it("rejects no blank", () => {
    const rejection = findRejection(3, makeReadings([8, 6, 7, 2, 5, 4, 3, 1, 1]));
    expect(rejection.reason).toBe("no-blank");
  });

  it("rejects two blanks", () => {
    expect(findRejection(3, makeReadings([8, 6, 7, 2, 0, 4, 3, 0, 1])).reason).toBe("two-blanks");
  });

  it("says how many blanks there are when there are more than two", () => {
    const rejection = findRejection(3, makeReadings([8, 0, 7, 2, 0, 4, 3, 0, 1]));
    expect(rejection.reason).toBe("two-blanks");
    expect(rejection.message).toBe("3 empty slots found");
  });

  it("rejects a value outside the grid's tile range", () => {
    const rejection = findRejection(3, makeReadings([8, 6, 7, 2, 5, 4, 3, 0, 12]));
    expect(rejection).toMatchObject({ reason: "out-of-range", row: 2, col: 2 });
  });

  it("rejects a low-confidence cell and says where", () => {
    const cells = makeReadings(GOOD_3);
    cells[4] = { value: 5, confidence: 0.3 };
    const rejection = findRejection(3, cells);
    expect(rejection).toMatchObject({
      reason: "low-confidence",
      row: 1,
      col: 1,
    });
  });

  it("rejects an unsolvable permutation as probably misread", () => {
    const rejection = findRejection(3, makeReadings([1, 2, 3, 4, 5, 6, 8, 7, 0]));
    expect(rejection.reason).toBe("unsolvable");
    expect(rejection.message).toContain("misread");
  });

  it("rejects a duplicate on a 4x4 board", () => {
    const cells = makeReadings([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14, 0]);
    expect(findRejection(4, cells)).toMatchObject({
      reason: "duplicate",
      value: 14,
      missing: 15,
    });
  });

  it("rejects a wrong cell count loudly", () => {
    expect(() => checkReading(3, makeReadings([1, 2, 3]))).toThrow(/expected 9 cells/);
  });

  it("rejects NaN and undefined confidence as low confidence", () => {
    for (const confidence of [Number.NaN, undefined as unknown as number]) {
      const cells = makeReadings(GOOD_3).map((c, i) => (i === 4 ? { ...c, confidence } : c));
      expect(findRejection(3, cells)).toMatchObject({ reason: "low-confidence", row: 1, col: 1 });
    }
  });
});
