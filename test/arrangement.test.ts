import { describe, expect, it } from "vitest";
import { checkArrangement, toBoard } from "../src/picture/arrangement.ts";
import { type Arrangement, type PictureReading } from "../src/picture/types.ts";
import { isGoalBoard, isSolvable } from "../src/puzzle/board.ts";
import { createFlatCrop, createPicture, createReading, scrambleOrder } from "./picture-fixtures.ts";

const picture = createPicture(4, 1);

/** Grey crops 14 levels apart, so no two differ by less than the near-identical threshold (8). */
function createFlatGrey(index: number, offset = 0) {
  const level = 20 + index * 14 + offset;
  return createFlatCrop(level, level, level);
}

describe("toBoard", () => {
  it("numbers tiles by their solved position, skipping the home", () => {
    // Home at solved position 2: positions 0,1 are tiles 1,2; positions 3.. are tiles 3...
    const order = scrambleOrder(4, 2, 0, 1);
    const { reading, arrangement } = createReading(picture, 4, order, 2);
    const board = toBoard(reading, arrangement);
    expect(board.blankHome).toBe(2);
    expect(board.cells).toEqual([1, 2, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    expect(isGoalBoard(board)).toBe(true);
  });

  it("puts the blank at the photo's empty cell and numbers a scramble by solved position", () => {
    const order = scrambleOrder(4, 0, 30, 2);
    const { reading, arrangement } = createReading(picture, 4, order, 0);
    const board = toBoard(reading, arrangement);
    expect(board.cells.indexOf(0)).toBe(reading.blank);
    expect(board.cells).toEqual(order.map((tile) => (tile < 0 ? 0 : tile)));
    expect(isSolvable(board)).toBe(true);
  });

  it("refuses an arrangement that is not a layout of the photo", () => {
    const { reading, arrangement } = createReading(picture, 4, scrambleOrder(4, 0, 0, 1), 0);
    expect(() => toBoard(reading, { ...arrangement, solved: arrangement.solved.slice(1) })).toThrow(
      /positions/,
    );
  });
});

describe("checkArrangement", () => {
  const { reading, arrangement } = createReading(picture, 4, scrambleOrder(4, 5, 40, 3), 5);

  function expectReason(
    result: ReturnType<typeof checkArrangement>,
    status: "invalid" | "unsolvable",
    reason: RegExp,
  ): void {
    if (result.status !== status) throw new Error(`expected ${status}, got ${result.status}`);
    expect(result.reason).toMatch(reason);
  }

  it("accepts a reachable arrangement unchanged", () => {
    expect(checkArrangement(reading, arrangement)).toEqual({ status: "ok", arrangement });
  });

  it("rejects an arrangement that is not a permutation of the photo cells", () => {
    const twice = [...arrangement.solved];
    twice[0] = twice[1] ?? 0;
    expectReason(
      checkArrangement(reading, { ...arrangement, solved: twice }),
      "invalid",
      /placed twice/,
    );

    const noHome = arrangement.solved.map((c) => (c === -1 ? (arrangement.solved[0] ?? 0) : c));
    expectReason(
      checkArrangement(reading, { ...arrangement, solved: noHome }),
      "invalid",
      /0 empty-slot/,
    );

    const twoHomes = [...arrangement.solved];
    twoHomes[0] = -1;
    expectReason(
      checkArrangement(reading, { ...arrangement, solved: twoHomes }),
      "invalid",
      /2 empty-slot/,
    );

    const withBlankCell = arrangement.solved.map((c) =>
      c === arrangement.solved[0] ? reading.blank : c,
    );
    expectReason(
      checkArrangement(reading, { ...arrangement, solved: withBlankCell }),
      "invalid",
      /empty slot, not a tile/,
    );

    const outside = arrangement.solved.map((c) => (c === arrangement.solved[0] ? 99 : c));
    expectReason(
      checkArrangement(reading, { ...arrangement, solved: outside }),
      "invalid",
      /not a cell/,
    );

    const wrongSize: Arrangement = { ...arrangement, size: 3 };
    expectReason(checkArrangement(reading, wrongSize), "invalid", /3x3/);
  });

  it("rejects an unsolvable arrangement when no two tiles look alike", () => {
    // Every tile of the procedural picture is clearly different from the others.
    const swapped = [...arrangement.solved];
    const [i, j] = swapped.flatMap((c, p) => (c === -1 ? [] : [p]));
    [swapped[i ?? 0], swapped[j ?? 0]] = [swapped[j ?? 0] ?? 0, swapped[i ?? 0] ?? 0];
    expectReason(
      checkArrangement(reading, { ...arrangement, solved: swapped }),
      "unsolvable",
      /swapped or misread/,
    );
  });

  it("swaps two near-identical tiles to make an unsolvable arrangement solvable", () => {
    // Cells 0..14 flat greys 14 apart (clearly different), except cells 6 and 9 which differ by 1.
    const cells = Array.from({ length: 16 }, (_, i) => createFlatGrey(i));
    cells[9] = createFlatGrey(6, 1);
    const flat: PictureReading = { ...reading, cells, blank: 15 };
    // Identity layout is solvable; exchanging solved positions of cells 6 and 9 is not.
    const identity = Array.from({ length: 16 }, (_, p) => (p === 15 ? -1 : p));
    const broken = identity.map((c) => (c === 6 ? 9 : c === 9 ? 6 : c));
    const result = checkArrangement(flat, {
      size: 4,
      solved: broken,
      confidence: broken.map(() => 1),
    });
    if (result.status !== "fixed") throw new Error(`expected fixed, got ${result.status}`);
    expect([...result.swapped].sort()).toEqual([6, 9]);
    expect(result.arrangement.solved).toEqual(identity);
    expect(isSolvable(toBoard(flat, result.arrangement))).toBe(true);
    // The exchanged pair cannot be told apart, so it is marked unsure and the review outlines it.
    const positions = [6, 9].map((cell) => result.arrangement.solved.indexOf(cell));
    for (const p of positions) expect(result.arrangement.confidence[p]).toBe(0);
    for (let p = 0; p < 16; p++)
      if (!positions.includes(p)) expect(result.arrangement.confidence[p]).toBe(1);
  });

  it("does not swap tiles that merely look somewhat alike", () => {
    const cells = Array.from({ length: 16 }, (_, i) => createFlatGrey(i));
    const flat: PictureReading = { ...reading, cells, blank: 15 };
    const identity = Array.from({ length: 16 }, (_, p) => (p === 15 ? -1 : p));
    const broken = identity.map((c) => (c === 6 ? 9 : c === 9 ? 6 : c));
    const result = checkArrangement(flat, {
      size: 4,
      solved: broken,
      confidence: broken.map(() => 1),
    });
    expect(result.status).toBe("unsolvable");
  });

  it("ignores the empty slot's crop when looking for look-alikes", () => {
    // The blank cell's crop equals tile 3's, but exchanging with the blank is not a tile swap.
    const cells = Array.from({ length: 16 }, (_, i) => createFlatGrey(i));
    cells[15] = createFlatGrey(3);
    const flat: PictureReading = { ...reading, cells, blank: 15 };
    const identity = Array.from({ length: 16 }, (_, p) => (p === 15 ? -1 : p));
    const broken = identity.map((c) => (c === 6 ? 9 : c === 9 ? 6 : c));
    const result = checkArrangement(flat, {
      size: 4,
      solved: broken,
      confidence: broken.map(() => 1),
    });
    expect(result.status).toBe("unsolvable");
  });
});
