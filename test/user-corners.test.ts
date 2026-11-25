// Corners placed by the user replace the readers' board search (the "Adjust corners" screen).
import { readFileSync } from "node:fs";
import jpeg from "jpeg-js";
import { describe, expect, it } from "vitest";
import { assemblePicture } from "../src/picture/assemble.ts";
import { toBoard } from "../src/picture/arrangement.ts";
import { readPictureBoard, readPictureCells, setBlank } from "../src/picture/read.ts";
import { defaultWeights } from "../src/vision/default-weights.ts";
import { type Point, type Quad } from "../src/vision/geometry.ts";
import { type RgbaImage } from "../src/vision/image.ts";
import { prepareUserQuad, readBoard } from "../src/vision/read-board.ts";
import { createOpeningState } from "../src/app/corner-editor.ts";
import { isReadableQuad } from "../src/app/corner-view.ts";
import { createPicture, createReading, scrambleOrder } from "./picture-fixtures.ts";
import { createRandom, jitterQuad } from "../scripts/random.ts";
import { ownerLabel, ownerPhoto } from "./picture-variants.ts";

const photoLabel = JSON.parse(
  readFileSync(new URL("fixtures/photo-0000.json", import.meta.url), "utf8"),
) as { size: 3 | 4; state: number[]; corners: [number, number][] };
const decoded = jpeg.decode(readFileSync(new URL("fixtures/photo-0000.jpg", import.meta.url)), {
  useTArray: true,
  formatAsRGBA: true,
});
const photo: RgbaImage = {
  width: decoded.width,
  height: decoded.height,
  data: new Uint8ClampedArray(decoded.data),
};
// map gives Point[]; Quad is a 4-tuple, and the label file has exactly four corners.
const labelQuad = photoLabel.corners.map(([x, y]): Point => ({ x, y })) as unknown as Quad;

function side(quad: Quad): number {
  return Math.hypot(quad[1].x - quad[0].x, quad[1].y - quad[0].y);
}

describe("readBoard with the user's corners", () => {
  it("reads the labelled board from the labelled corners", { timeout: 25_000 }, () => {
    const result = readBoard(photo, defaultWeights, { quad: labelQuad });
    if (result.status !== "accepted") throw new Error(`not accepted: ${result.message}`);
    expect([...result.board.cells]).toEqual(photoLabel.state);
    expect(result.quad).toEqual(labelQuad);
  });

  for (const seed of [1, 2, 3]) {
    it(
      `still reads it with corners off by up to 2% (seed ${String(seed)})`,
      { timeout: 25_000 },
      () => {
        const result = readBoard(photo, defaultWeights, {
          quad: jitterQuad(labelQuad, 0.02, createRandom(seed)),
          size: photoLabel.size,
        });
        if (result.status !== "accepted") throw new Error(`not accepted: ${result.message}`);
        expect([...result.board.cells]).toEqual(photoLabel.state);
      },
    );
  }

  it("reads the quad it is given, not the board it would find", { timeout: 25_000 }, () => {
    // The top-left half of the board's area: a different region cannot read as the labelled board.
    const [a, b, , d] = labelQuad;
    const middle = (p: Point, q: Point): Point => ({ x: (p.x + q.x) / 2, y: (p.y + q.y) / 2 });
    const wrong: Quad = [
      a,
      middle(a, b),
      middle(middle(a, b), middle(d, labelQuad[2])),
      middle(a, d),
    ];
    const result = readBoard(photo, defaultWeights, { quad: wrong, size: photoLabel.size });
    if (result.status === "no-board") throw new Error("a quad is always read");
    const cells =
      result.status === "accepted"
        ? [...result.board.cells]
        : result.attempt.cells.map((cell) => cell.value);
    expect(cells).not.toEqual(photoLabel.state);
  });
});

describe("readPictureBoard with the user's corners", () => {
  it("finds the labelled blank and assembles as well as automatic", { timeout: 30_000 }, () => {
    const automatic = readPictureBoard(ownerPhoto, { size: 4 });
    const given = readPictureBoard(ownerPhoto, { quad: ownerLabel.tilesQuad, size: 4 });
    if ("status" in automatic || "status" in given) throw new Error("no reading");
    expect(given.blank).toBe(ownerLabel.blank);
    const same = (solved: number[]): number =>
      solved.filter((cell, position) => cell === ownerLabel.solved[position]).length;
    const wanted = same(assemblePicture(automatic.cells, 4, automatic.blank).solved);
    expect(same(assemblePicture(given.cells, 4, given.blank).solved)).toBeGreaterThanOrEqual(
      wanted,
    );
  });

  it(
    "keeps each corner within 4% of the user's, even when the corners are 2% off",
    { timeout: 30_000 },
    () => {
      const quad = jitterQuad(ownerLabel.tilesQuad, 0.02, createRandom(7));
      const reading = readPictureBoard(ownerPhoto, { quad, size: 4 });
      if ("status" in reading) throw new Error(reading.message);
      reading.quad.forEach((corner, i) => {
        const user = quad[i] as Point;
        expect(Math.hypot(corner.x - user.x, corner.y - user.y)).toBeLessThanOrEqual(
          0.04 * side(quad),
        );
      });
      expect(reading.blank).toBe(ownerLabel.blank);
    },
  );

  it("uses the quad: one placed on the wrong area reads differently", { timeout: 30_000 }, () => {
    const [a, b, c, d] = ownerLabel.tilesQuad;
    // Up, not left: moved left the quad would leave the photo, which the readers now refuse.
    const shift = (p: Point): Point => ({ x: p.x, y: p.y - 0.4 * side(ownerLabel.tilesQuad) });
    // map gives Point[]; Quad is a 4-tuple, and this maps four corners to four.
    const wrong = [a, b, c, d].map(shift) as unknown as Quad;
    const right = readPictureBoard(ownerPhoto, { quad: ownerLabel.tilesQuad, size: 4 });
    const moved = readPictureBoard(ownerPhoto, { quad: wrong, size: 4 });
    if ("status" in right || "status" in moved) throw new Error("no reading");
    expect(moved.quad).not.toEqual(right.quad);
    expect(moved.cells[0]?.rgb).not.toEqual(right.cells[0]?.rgb);
  });
});

describe("setBlank", () => {
  it("puts the empty slot at the chosen cell and is fully confident of it", () => {
    const picture = createPicture(4, 3);
    const { reading } = createReading(picture, 4, scrambleOrder(4, 0, 20, 5), 0);
    const chosen = reading.blank === 9 ? 10 : 9;
    const changed = setBlank(reading, chosen);
    expect(changed.blank).toBe(chosen);
    expect(changed.blankConfidence).toBe(1);
    const arrangement = assemblePicture(changed.cells, changed.size, changed.blank);
    expect(toBoard(changed, arrangement).cells[chosen]).toBe(0);
  });
});

/** A small photo-sized image; the corner checks must fire before any pixel is read. */
const smallImage: RgbaImage = {
  width: 500,
  height: 500,
  data: new Uint8ClampedArray(500 * 500 * 4).fill(128),
};
const at = (x: number, y: number): Point => ({ x, y });
const badQuads: [string, Quad][] = [
  ["outside the photo", [at(100, 100), at(600, 100), at(600, 400), at(100, 400)]],
  ["a corner left of the photo", [at(-5, 100), at(400, 100), at(400, 400), at(100, 400)]],
  ["a bow-tie", [at(100, 100), at(400, 400), at(400, 100), at(100, 400)]],
  ["a NaN corner", [at(100, 100), at(400, 100), at(400, 400), at(Number.NaN, 400)]],
  [
    "an infinite corner",
    [at(100, 100), at(400, 100), at(Number.POSITIVE_INFINITY, 400), at(100, 400)],
  ],
  [
    "three points",
    [at(100, 100), at(400, 100), at(400, 400)] as unknown as Quad, // three points: the type forbids it, the runtime check must too
  ],
];

describe("corners the readers refuse", () => {
  for (const [name, quad] of badQuads) {
    it(`readBoard rejects ${name}`, { timeout: 10_000 }, () => {
      expect(() => readBoard(smallImage, defaultWeights, { quad })).toThrow(/finite|cross|inside/);
    });
    it(`readPictureBoard rejects ${name}`, { timeout: 10_000 }, () => {
      expect(() => readPictureBoard(smallImage, { quad })).toThrow(/finite|cross|inside/);
    });
    it(`readPictureCells rejects ${name}`, { timeout: 10_000 }, () => {
      expect(() => readPictureCells(smallImage, quad, 3)).toThrow(/finite|cross|inside/);
    });
  }
});

describe("readPictureBoard with the labelled corners in another order", () => {
  const label = JSON.parse(
    readFileSync(new URL("fixtures/corrections/white-frame.json", import.meta.url), "utf8"),
  ) as { size: 3 | 4; blank: number; solved: number[]; tilesQuad: Point[] };
  const frame = jpeg.decode(
    readFileSync(new URL("fixtures/corrections/white-frame.jpg", import.meta.url)),
    { useTArray: true, formatAsRGBA: true },
  );
  const frameImage: RgbaImage = {
    width: frame.width,
    height: frame.height,
    data: new Uint8ClampedArray(frame.data),
  };
  const [a, b, c, d] = label.tilesQuad as [Point, Point, Point, Point];
  const orders: [string, Quad][] = [
    ["labelled", [a, b, c, d]],
    ["b,a,d,c (mirrored)", [b, a, d, c]],
    ["d,a,b,c (rotated)", [d, a, b, c]],
    ["c,d,a,b (rotated)", [c, d, a, b]],
  ];
  it(
    "opens the editor from a numbered outline past the photo's bottom with corners the readers accept",
    { timeout: 30_000 },
    () => {
      // The numbered reader puts one corner of this photo's outline below its last row.
      const reading = readBoard(frameImage, defaultWeights);
      const detected = reading.status === "rejected" ? reading.attempt.quad : undefined;
      if (!detected)
        throw new Error(`expected a rejected reading with an outline, got ${reading.status}`);
      expect(detected.some((p) => p.y > frameImage.height)).toBe(true);
      const state = createOpeningState(detected, 3, frameImage.width, frameImage.height);
      expect(isReadableQuad(state.quad, frameImage.width, frameImage.height)).toBe(true);
      expect(() => prepareUserQuad(state.quad, frameImage)).not.toThrow();
    },
  );

  for (const [name, quad] of orders) {
    it(`reads the labelled board from ${name}`, { timeout: 30_000 }, () => {
      const result = readPictureBoard(frameImage, { quad, size: label.size });
      if ("status" in result) throw new Error(result.message);
      expect(result.blank).toBe(label.blank);
      expect(assemblePicture(result.cells, label.size, label.blank).solved).toEqual(label.solved);
      // The reading reports the corners TL, TR, BR, BL whatever order they came in.
      const labelled = readPictureBoard(frameImage, { quad: [a, b, c, d], size: label.size });
      if ("status" in labelled) throw new Error(labelled.message);
      expect(result.quad).toEqual(labelled.quad);
    });
  }
});
