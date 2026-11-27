import { readFileSync } from "node:fs";
import jpeg from "jpeg-js";
import { describe, expect, it } from "vitest";
import { isSolvable } from "../src/puzzle/board.ts";
import { toBoard } from "../src/picture/arrangement.ts";
import { createProfile, findBestProfileMatch, matchProfile } from "../src/picture/profile.ts";
import { cutTileCrops } from "../src/picture/read.ts";
import { type PictureReading, TILE_SIDE } from "../src/picture/types.ts";
import { type Quad } from "../src/vision/geometry.ts";
import { createPicture, createReading, scrambleOrder } from "./picture-fixtures.ts";

function expectMatch(result: ReturnType<typeof matchProfile>) {
  if (result.status !== "match") throw new Error(`expected a match: ${result.reason}`);
  return result;
}

describe("puzzle profile from procedural pictures", () => {
  for (const [size, homeTile] of [
    [4, 0],
    [4, 6],
    [3, 8],
  ] as const) {
    it(`re-finds a ${String(size)}x${String(size)} puzzle (home ${String(homeTile)}) in a new scramble with other light and a shifted crop`, () => {
      const picture = createPicture(size, 11);
      const first = createReading(picture, size, scrambleOrder(size, homeTile, 60, 1), homeTile);
      const profile = createProfile(first.reading, first.arrangement);
      expect(profile.size).toBe(size);
      expect(profile.blankHome).toBe(homeTile);

      const second = createReading(picture, size, scrambleOrder(size, homeTile, 75, 2), homeTile, {
        shift: 2,
        gain: 0.75,
        bias: 6,
      });
      const result = expectMatch(matchProfile(second.reading, profile));
      expect(result.arrangement.solved).toEqual(second.arrangement.solved);
      expect(result.arrangement.size).toBe(size);
      expect(result.confidence).toBeGreaterThan(0.1);
      expect(result.arrangement.confidence[homeTile]).toBe(1);
    });
  }

  it("survives the profile being saved as JSON", () => {
    const picture = createPicture(4, 12);
    const first = createReading(picture, 4, scrambleOrder(4, 15, 50, 3), 15);
    const profile = createProfile(first.reading, first.arrangement);
    const saved = JSON.stringify(profile);
    // localStorage holds a few MB; a 4x4 profile must be a small fraction of that.
    expect(saved.length).toBeLessThan(20_000);
    const second = createReading(picture, 4, scrambleOrder(4, 15, 50, 4), 15);
    const result = expectMatch(matchProfile(second.reading, JSON.parse(saved) as typeof profile));
    expect(result.arrangement.solved).toEqual(second.arrangement.solved);
  });

  it("rejects the crops of an unrelated picture", () => {
    const first = createReading(createPicture(4, 13), 4, scrambleOrder(4, 0, 50, 5), 0);
    const profile = createProfile(first.reading, first.arrangement);
    for (const seed of [14, 15, 16]) {
      const other = createReading(createPicture(4, seed), 4, scrambleOrder(4, 0, 50, seed), 0);
      const result = matchProfile(other.reading, profile);
      expect(result.status).toBe("mismatch");
    }
  });

  it("rejects a photo of another size", () => {
    const first = createReading(createPicture(4, 13), 4, scrambleOrder(4, 0, 50, 5), 0);
    const profile = createProfile(first.reading, first.arrangement);
    const small = createReading(createPicture(3, 13), 3, scrambleOrder(3, 0, 20, 5), 0);
    const result = matchProfile(small.reading, profile);
    if (result.status !== "mismatch") throw new Error("expected a mismatch");
    expect(result.reason).toMatch(/3x3/);
  });

  it("gives a low confidence to tiles that look the same", () => {
    // A picture with two identical tiles: the match cannot tell them apart, and says so.
    const picture = createPicture(4, 17);
    const duplicated: typeof picture = (x, y) => picture(x % (2 * TILE_SIDE), y);
    const first = createReading(duplicated, 4, scrambleOrder(4, 0, 50, 6), 0);
    const profile = createProfile(first.reading, first.arrangement);
    const second = createReading(duplicated, 4, scrambleOrder(4, 0, 50, 7), 0);
    const result = expectMatch(matchProfile(second.reading, profile));
    expect(Math.min(...result.arrangement.confidence)).toBe(0);
    expect(result.confidence).toBe(0);
  });
});

describe("the empty slot of a recognised puzzle", () => {
  it("comes from the saved puzzle, not from the reader's guess", () => {
    const picture = createPicture(4, 21);
    // Saved after the user chose the empty slot (the reading's blank is the real one).
    const first = createReading(picture, 4, scrambleOrder(4, 6, 60, 1), 6);
    const profile = createProfile(first.reading, first.arrangement);

    const second = createReading(picture, 4, scrambleOrder(4, 6, 70, 2), 6, { gain: 0.8 });
    const wrongCell = second.reading.blank === 0 ? 1 : 0;
    const guessed = { ...second.reading, blank: wrongCell, blankConfidence: 0.2 };
    const result = expectMatch(matchProfile(guessed, profile));
    expect(result.blank).toBe(second.reading.blank);
    expect(result.arrangement.solved).toEqual(second.arrangement.solved);
    const board = toBoard({ ...guessed, blank: result.blank }, result.arrangement);
    expect(board.cells[second.reading.blank]).toBe(0);
    expect(isSolvable(board)).toBe(true);
    expect(findBestProfileMatch(guessed, [profile])?.blank).toBe(second.reading.blank);
  });

  it("is the reader's cell when that is right", () => {
    const picture = createPicture(3, 22);
    const first = createReading(picture, 3, scrambleOrder(3, 4, 30, 1), 4);
    const profile = createProfile(first.reading, first.arrangement);
    const second = createReading(picture, 3, scrambleOrder(3, 4, 30, 2), 4);
    expect(expectMatch(matchProfile(second.reading, profile)).blank).toBe(second.reading.blank);
  });
});

describe("findBestProfileMatch", () => {
  const picture = createPicture(4, 18);
  const first = createReading(picture, 4, scrambleOrder(4, 15, 50, 8), 15);
  const exact = createProfile(first.reading, first.arrangement);
  // Tile 1 remembered as a copy of tile 0: still the same puzzle, but two tiles can no longer be told apart.
  const blurred = {
    ...exact,
    tiles: exact.tiles.map((tile, k) => (k === 1 ? (exact.tiles[0] ?? tile) : tile)),
  };
  const second = createReading(picture, 4, scrambleOrder(4, 15, 50, 9), 15);

  it("picks the more confident of two matching saved puzzles, in either order", () => {
    expect(expectMatch(matchProfile(second.reading, blurred)).confidence).toBeLessThan(
      expectMatch(matchProfile(second.reading, exact)).confidence,
    );
    expect(findBestProfileMatch(second.reading, [blurred, exact])?.profile).toBe(exact);
    expect(findBestProfileMatch(second.reading, [exact, blurred])?.profile).toBe(exact);
  });

  it("skips a damaged saved puzzle and says nothing matches when none does", () => {
    const damaged = { ...exact, tiles: exact.tiles.slice(1) };
    expect(findBestProfileMatch(second.reading, [damaged, exact])?.profile).toBe(exact);
    expect(findBestProfileMatch(second.reading, [damaged])).toBeUndefined();
    expect(findBestProfileMatch(second.reading, [])).toBeUndefined();
  });
});

describe("puzzle profile from the owner's photo", () => {
  const decoded = jpeg.decode(
    readFileSync(new URL("fixtures/owner-picture/picture-4x4-01.jpg", import.meta.url)),
    { useTArray: true, formatAsRGBA: true },
  );
  const label = JSON.parse(
    readFileSync(new URL("fixtures/owner-picture/picture-4x4-01.json", import.meta.url), "utf8"),
  ) as { blank: number; solved: number[]; tilesQuad: Quad };
  const crops = cutTileCrops(
    { width: decoded.width, height: decoded.height, data: new Uint8ClampedArray(decoded.data) },
    label.tilesQuad,
    4,
  );
  const reading: PictureReading = {
    size: 4,
    quad: label.tilesQuad,
    cells: crops,
    blank: label.blank,
    blankConfidence: 1,
  };
  const arrangement = {
    size: 4 as const,
    solved: label.solved,
    confidence: label.solved.map(() => 1),
  };
  const profile = createProfile(reading, arrangement);

  /** The same tiles after sliding: the owner's crops re-composited in a new order, dimmer and shifted. */
  function rescramble(order: readonly number[]): PictureReading {
    // order[cell] = original photo cell shown at this cell; the empty slot stays an empty-slot crop.
    const cells = order.map((original) => {
      const source = crops[original];
      if (!source) throw new Error("bad order");
      const rgb = new Uint8ClampedArray(source.rgb.length);
      // Shift 2 px by reading the crop 2 px down and right (edge pixels repeat), then dim.
      for (let y = 0; y < TILE_SIDE; y++) {
        for (let x = 0; x < TILE_SIDE; x++) {
          const sx = Math.min(x + 2, TILE_SIDE - 1);
          const sy = Math.min(y + 2, TILE_SIDE - 1);
          for (let c = 0; c < 3; c++) {
            rgb[(y * TILE_SIDE + x) * 3 + c] =
              (source.rgb[(sy * TILE_SIDE + sx) * 3 + c] ?? 0) * 0.8;
          }
        }
      }
      return { side: TILE_SIDE, rgb };
    });
    return { ...reading, cells, blank: order.indexOf(label.blank) };
  }

  it("re-finds the tiles in a different scramble of them", () => {
    // Slide the empty slot: swap it with a neighbour a few times, tracking which original cell sits where.
    const order = Array.from({ length: 16 }, (_, i) => i);
    let blank = label.blank;
    for (const to of [1, 5, 4, 8, 9, 10, 6, 7, 3, 2, 1, 0, 4]) {
      [order[blank], order[to]] = [order[to] ?? 0, order[blank] ?? 0];
      blank = to;
    }
    const moved = rescramble(order);
    const result = expectMatch(matchProfile(moved, profile));
    // solved[p] was an original cell; in the new photo it sits at the cell showing that original.
    const expected = label.solved.map((cell) => (cell === -1 ? -1 : order.indexOf(cell)));
    expect(result.arrangement.solved).toEqual(expected);
  });

  it("rejects the crops of an unrelated picture", () => {
    const other = createReading(createPicture(4, 3), 4, scrambleOrder(4, 0, 50, 3), 0);
    expect(matchProfile(other.reading, profile).status).toBe("mismatch");
  });
});
