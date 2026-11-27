import { describe, expect, it } from "vitest";
import { matchReference } from "../src/picture/reference.ts";
import { type GridSize } from "../src/puzzle/board.ts";
import {
  createFlatCrop,
  createPicture,
  createReading,
  cutTile,
  scrambleOrder,
} from "./picture-fixtures.ts";

/** The solved picture cut into the grid, the way a reader cuts a printed copy. */
function cutReference(size: GridSize, seed: number, options = {}): ReturnType<typeof cutTile>[] {
  const picture = createPicture(size, seed);
  return Array.from({ length: size * size }, (_, tile) => cutTile(picture, size, tile, options));
}

describe("matchReference", () => {
  for (const size of [3, 4] as const) {
    it(`arranges a shuffled ${String(size)}x${String(size)} board exactly from a perfect reference`, () => {
      const homeTile = size * size - 1;
      const order = scrambleOrder(size, homeTile, 60, 11);
      const { reading, arrangement } = createReading(createPicture(size, 5), size, order, homeTile);
      const found = matchReference(reading.cells, reading.blank, cutReference(size, 5));
      expect(found.solved).toEqual(arrangement.solved);
      expect(found.size).toBe(size);
      expect(found.confidence[homeTile]).toBe(1);
      // A busy picture: every tile has one clear home.
      for (const confidence of found.confidence) expect(confidence).toBeGreaterThan(0.2);
    });
  }

  it("finds the empty slot's home wherever it is", () => {
    const homeTile = 4;
    const order = scrambleOrder(3, homeTile, 40, 3);
    const { reading, arrangement } = createReading(createPicture(3, 8), 3, order, homeTile);
    const found = matchReference(reading.cells, reading.blank, cutReference(3, 8));
    expect(found.solved.indexOf(-1)).toBe(4);
    expect(found.solved).toEqual(arrangement.solved);
  });

  it("copes with a print that is duller and lighter than the tiles, and a blank home cell", () => {
    const homeTile = 8;
    const order = scrambleOrder(3, homeTile, 50, 21);
    const { reading, arrangement } = createReading(createPicture(3, 2), 3, order, homeTile);
    const reference = cutReference(3, 2, { gain: 0.7, bias: 60 });
    reference[homeTile] = createFlatCrop(240, 240, 240);
    expect(matchReference(reading.cells, reading.blank, reference).solved).toEqual(
      arrangement.solved,
    );
  });

  it("gives low confidence to two tiles that look alike", () => {
    const flat = Array.from({ length: 9 }, () => createFlatCrop(200, 200, 200));
    const found = matchReference(flat, 8, flat);
    expect(Math.max(...found.confidence.slice(0, 8))).toBeLessThan(0.1);
  });

  it("rejects a reference of another size and an empty slot outside the photo", () => {
    const { reading } = createReading(createPicture(3, 1), 3, scrambleOrder(3, 8, 10, 1), 8);
    expect(() => matchReference(reading.cells, reading.blank, cutReference(4, 1))).toThrow(
      /as many cells/,
    );
    expect(() => matchReference(reading.cells, 9, cutReference(3, 1))).toThrow(/not a cell/);
  });
});
