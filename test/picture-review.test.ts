import { describe, expect, it } from "vitest";
import {
  LOW_TILE_CONFIDENCE,
  listDoubtfulPositions,
  swapPositions,
  tileNumberAt,
} from "../src/app/picture-review.ts";
import { assemblePicture } from "../src/picture/assemble.ts";
import { checkArrangement } from "../src/picture/arrangement.ts";
import { type Arrangement } from "../src/picture/types.ts";
import { createPicture, createReading, scrambleOrder } from "./picture-fixtures.ts";

// 3x3, empty-slot home at position 4: positions 0-3 hold tiles 1-4, positions 5-8 hold tiles 5-8.
const arrangement: Arrangement = {
  size: 3,
  solved: [3, 1, 4, 0, -1, 5, 6, 7, 8],
  confidence: [1, 0.1, 1, 1, 1, 0.14, 0.15, 1, 1],
};

describe("picture review", () => {
  it("numbers tiles by solved position, skipping the empty-slot home", () => {
    expect(arrangement.solved.map((_, p) => tileNumberAt(arrangement, p))).toEqual([
      1, 2, 3, 4, 0, 5, 6, 7, 8,
    ]);
  });

  it("flags positions below the threshold, never the home", () => {
    expect(LOW_TILE_CONFIDENCE).toBe(0.15);
    expect(listDoubtfulPositions(arrangement)).toEqual([1, 5]);
    const homeUnsure = { ...arrangement, confidence: arrangement.confidence.with(4, 0) };
    expect(listDoubtfulPositions(homeUnsure)).toEqual([1, 5]);
  });

  it("swaps two positions, trusts both, and leaves the input alone", () => {
    const swapped = swapPositions(arrangement, 1, 6);
    expect(swapped.solved).toEqual([3, 6, 4, 0, -1, 5, 1, 7, 8]);
    expect(swapped.confidence[1]).toBe(1);
    expect(swapped.confidence[6]).toBe(1);
    expect(arrangement.solved[1]).toBe(1);
  });

  it("moves the empty-slot home by swapping it with a tile", () => {
    const moved = swapPositions(arrangement, 4, 8);
    expect(moved.solved.indexOf(-1)).toBe(8);
    expect(moved.solved[4]).toBe(8);
  });
});

describe("picture review of a near-identical exchange", () => {
  it("outlines exactly the two tiles the parity fix exchanged", () => {
    const picture = createPicture(4, 2);
    // Tile 6 (column 2, row 1) is a copy of tile 5, so the two cannot be told apart.
    const twins = (x: number, y: number) =>
      Math.floor(x / 64) === 2 && Math.floor(y / 64) === 1 ? picture(x - 64, y) : picture(x, y);
    const { reading } = createReading(twins, 4, scrambleOrder(4, 15, 200, 2), 15);
    const proposed = assemblePicture(reading.cells, 4, reading.blank);
    // The user swaps the two look-alikes, which breaks parity; the check exchanges a twin pair back.
    const result = checkArrangement(reading, swapPositions(proposed, 5, 6));
    if (result.status !== "fixed") throw new Error(`expected fixed, got ${result.status}`);
    const exchanged = result.swapped.map((cell) => result.arrangement.solved.indexOf(cell));
    expect(listDoubtfulPositions(result.arrangement)).toEqual(exchanged.sort((a, b) => a - b));
  });
});
