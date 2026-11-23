import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { assemblePicture } from "../src/picture/assemble.ts";
import { TILE_SIDE, type TileCrop } from "../src/picture/types.ts";
import { type Quad } from "../src/vision/geometry.ts";
import { readPhoto } from "../scripts/read-photo.ts";
import { cutTileCrops } from "../src/picture/read.ts";

const DIR = "test/fixtures/owner-picture/";

describe("assemblePicture on the owner's photo", () => {
  const label = JSON.parse(readFileSync(`${DIR}picture-4x4-01.json`, "utf8")) as {
    blank: number;
    solved: number[];
    tilesQuad: Quad;
  };
  const crops = cutTileCrops(readPhoto(`${DIR}picture-4x4-01.jpg`), label.tilesQuad, 4);

  it("finds the labelled solved layout, with the empty slot's home at top-left", () => {
    const started = performance.now();
    const arrangement = assemblePicture(crops, 4, label.blank);
    expect(arrangement.solved).toEqual(label.solved);
    expect(arrangement.confidence).toHaveLength(16);
    expect(arrangement.confidence[0]).toBe(1);
    // The brief's 2 s assembly budget. Measured (Node 22.23.3, 4 cores): 0.55-0.63 s alone, about
    // 0.62-0.66 s inside the full `npm test`.
    expect(performance.now() - started).toBeLessThan(2000);
  });
});

/** A smooth, busy picture: every tile differs from every other along all four edges. */
function makeCrops(size: 3 | 4, solved: number[]): TileCrop[] {
  const side = size * TILE_SIDE;
  return solved.map((tile) => {
    const rgb = new Uint8ClampedArray(TILE_SIDE * TILE_SIDE * 3);
    const homeRow = Math.floor(tile / size);
    const homeCol = tile % size;
    for (let y = 0; y < TILE_SIDE; y++)
      for (let x = 0; x < TILE_SIDE; x++) {
        const u = (homeCol * TILE_SIDE + x) / side;
        const v = (homeRow * TILE_SIDE + y) / side;
        rgb[(y * TILE_SIDE + x) * 3] = 128 + 100 * Math.sin(7 * u + 3 * v);
        rgb[(y * TILE_SIDE + x) * 3 + 1] = 128 + 100 * Math.cos(5 * v - 2 * u + 1);
        rgb[(y * TILE_SIDE + x) * 3 + 2] = 128 + 100 * Math.sin(9 * u * v + 4 * v);
      }
    return { side: TILE_SIDE, rgb };
  });
}

describe("assemblePicture on a clean synthetic picture", () => {
  it("restores a scrambled 3x3 whose empty slot belongs at the bottom-right", () => {
    // Photo cell i shows the tile of home position photoToHome[i]; the slot is cell 4 (home 8).
    const photoToHome = [3, 0, 6, 1, 8, 2, 7, 5, 4];
    const crops = makeCrops(3, photoToHome);
    const arrangement = assemblePicture(crops, 3, 4);
    const expected = photoToHome.map((_, home) => photoToHome.indexOf(home));
    expected[8] = -1;
    expect(arrangement.solved).toEqual(expected);
    expect(arrangement.confidence[8]).toBe(1);
  });

  it("finds an empty-slot home that is not a corner", () => {
    // Home 4 (the middle). Slot at cell 8; it must be reachable: cell/home permutation of even parity.
    const photoToHome = [0, 1, 2, 3, 5, 6, 7, 8, 4];
    const crops = makeCrops(3, photoToHome);
    const arrangement = assemblePicture(crops, 3, 8);
    expect(arrangement.solved[4]).toBe(-1);
    expect(arrangement.solved.filter((cell) => cell >= 0)).toHaveLength(8);
  });

  it("rejects a cell list of the wrong length", () => {
    expect(() => assemblePicture(makeCrops(3, [0, 1, 2, 3, 4, 5, 6, 7]), 3, 0)).toThrow(/9 cells/);
  });

  it("rejects an empty slot that is not a cell of the photo", () => {
    const crops = makeCrops(3, [0, 1, 2, 3, 4, 5, 6, 7, 8]);
    for (const blank of [-1, 9, 1.5])
      expect(() => assemblePicture(crops, 3, blank)).toThrow(/empty slot/);
  });
});
