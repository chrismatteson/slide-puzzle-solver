import { describe, expect, it } from "vitest";
import { type CellReading } from "../src/puzzle/consistency.ts";
import { checkBoard, confirmCell, describeMove, setCell } from "../src/app/review.ts";

function makeSureReadings(values: number[]): CellReading[] {
  return values.map((value) => ({ value, confidence: 1 }));
}

describe("checkBoard", () => {
  it("accepts a solvable permutation", () => {
    const result = checkBoard(3, makeSureReadings([1, 2, 3, 4, 5, 6, 7, 0, 8]));
    expect(result.accepted).toBe(true);
  });

  it("marks every cell holding a duplicated tile", () => {
    const result = checkBoard(3, makeSureReadings([1, 2, 2, 4, 5, 6, 7, 0, 8]));
    expect(result).toMatchObject({ accepted: false, offending: [1, 2] });
  });

  it("marks both blanks", () => {
    const result = checkBoard(3, makeSureReadings([1, 2, 3, 0, 5, 6, 7, 0, 8]));
    expect(result).toMatchObject({ accepted: false, offending: [3, 7] });
  });

  it("marks the unsure cell, and clears it once the user sets it", () => {
    const cells = makeSureReadings([1, 2, 3, 4, 5, 6, 7, 0, 8]);
    cells[4] = { value: 5, confidence: 0.3 };
    const rejected = checkBoard(3, cells);
    expect(rejected).toMatchObject({ accepted: false, offending: [4] });
    expect(checkBoard(3, setCell(cells, 4, 5)).accepted).toBe(true);
  });

  it("blames the two least confident cells of an unsolvable reading", () => {
    const cells = makeSureReadings([2, 1, 3, 4, 5, 6, 7, 0, 8]);
    cells[0] = { value: 2, confidence: 0.7 };
    cells[1] = { value: 1, confidence: 0.65 };
    expect(checkBoard(3, cells)).toMatchObject({ accepted: false, offending: [0, 1] });
  });

  it("blames the least confident cell when no blank was read", () => {
    const cells = makeSureReadings([1, 2, 3, 4, 5, 6, 7, 8, 8]);
    cells[7] = { value: 8, confidence: 0.61 };
    expect(checkBoard(3, cells)).toMatchObject({ accepted: false, offending: [7] });
  });

  it("flags a NaN confidence as unsure and does not offer to confirm it", () => {
    const cells = makeSureReadings([1, 2, 3, 4, 5, 6, 7, 0, 8]);
    cells[5] = { value: 6, confidence: Number.NaN };
    expect(checkBoard(3, cells)).toMatchObject({
      accepted: false,
      offending: [5],
      confirmable: false,
    });
  });
});

describe("confirming low-confidence cells", () => {
  function buildUnsureBoard(): CellReading[] {
    const cells = makeSureReadings([1, 2, 3, 4, 5, 6, 7, 0, 8]);
    cells[1] = { value: 2, confidence: 0.4 };
    cells[6] = { value: 7, confidence: 0.55 };
    return cells;
  }

  it("flags every unsure cell and says confirming can fix it", () => {
    expect(checkBoard(3, buildUnsureBoard())).toMatchObject({
      accepted: false,
      offending: [1, 6],
      confirmable: true,
    });
  });

  it("enables Solve only after every unsure cell is confirmed", () => {
    const once = confirmCell(buildUnsureBoard(), 1);
    expect(checkBoard(3, once)).toMatchObject({
      accepted: false,
      offending: [6],
      confirmable: true,
    });
    expect(checkBoard(3, confirmCell(once, 6)).accepted).toBe(true);
  });

  it("keeps the guess when confirming", () => {
    expect(confirmCell(buildUnsureBoard(), 1)[1]).toEqual({ value: 2, confidence: 1 });
  });

  it("does not offer confirmation when the guesses are not a permutation", () => {
    const cells = buildUnsureBoard();
    cells[2] = { value: 2, confidence: 0.4 };
    expect(checkBoard(3, cells)).toMatchObject({ accepted: false, confirmable: false });
  });

  it("does not offer confirmation when the guesses are unsolvable", () => {
    const cells = makeSureReadings([2, 1, 3, 4, 5, 6, 7, 0, 8]);
    cells[0] = { value: 2, confidence: 0.4 };
    expect(checkBoard(3, cells)).toMatchObject({ accepted: false, confirmable: false });
  });

  it("is not confirmable for other rejections", () => {
    const result = checkBoard(3, makeSureReadings([1, 2, 2, 4, 5, 6, 7, 0, 8]));
    expect(result).toMatchObject({ accepted: false, confirmable: false });
  });
});

describe("setCell", () => {
  it("changes one cell and leaves the input alone", () => {
    const cells = makeSureReadings([1, 2, 3, 4, 5, 6, 7, 0, 8]);
    const edited = setCell(cells, 0, 9);
    expect(edited[0]).toEqual({ value: 9, confidence: 1 });
    expect(cells[0]).toEqual({ value: 1, confidence: 1 });
  });
});

describe("describeMove", () => {
  it("names the tile and the way it slides", () => {
    expect(describeMove({ tile: 7, direction: "left" })).toBe("Move 7 left");
  });
});
