// Regression fixtures for the manual corrections (test/fixtures/corrections/README.md): on each photo
// the automatic reading is wrong, and the app's correction path gets the solved layout exactly right.
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { reassemblePicture } from "../src/app/routing.ts";
import { assemblePicture } from "../src/picture/assemble.ts";
import { readPictureBoard } from "../src/picture/read.ts";
import { type GridSize } from "../src/puzzle/board.ts";
import { type Quad } from "../src/vision/geometry.ts";
import { readPhoto } from "../scripts/read-photo.ts";

interface Label {
  size: GridSize;
  blank: number;
  solved: number[];
  tilesQuad: Quad;
}

const load = (name: string): { label: Label; image: ReturnType<typeof readPhoto> } => ({
  label: JSON.parse(readFileSync(`test/fixtures/corrections/${name}.json`, "utf8")) as Label,
  image: readPhoto(`test/fixtures/corrections/${name}.jpg`),
});

/** The automatic path as the app runs it; `undefined` when no board or the wrong grid size is found. */
function autoSolved(name: string): number[] | undefined {
  const { label, image } = load(name);
  const reading = readPictureBoard(image);
  if ("status" in reading || reading.size !== label.size) return undefined;
  return assemblePicture(reading.cells, reading.size, reading.blank).solved;
}

describe.each(["angled-blank", "white-frame", "line-art"])("correction fixture %s", (name) => {
  it("is not read right automatically", () => {
    expect(autoSolved(name)).not.toEqual(load(name).label.solved);
  }, 30_000);

  it("is exact with the corners and the empty slot given", () => {
    const { label, image } = load(name);
    const reading = readPictureBoard(image, { quad: label.tilesQuad, size: label.size });
    if ("status" in reading) throw new Error(reading.message);
    const corrected = reassemblePicture(reading, label.blank);
    expect(corrected.arrangement.solved).toEqual(label.solved);
  }, 30_000);
});

describe("what the corrections each fix", () => {
  it("angled-blank: the corners alone leave the wrong empty slot", () => {
    const { label, image } = load("angled-blank");
    const reading = readPictureBoard(image, { quad: label.tilesQuad, size: label.size });
    if ("status" in reading) throw new Error(reading.message);
    expect(reading.blank).not.toBe(label.blank);
  }, 30_000);

  it.each(["white-frame", "line-art"])(
    "%s: the corners alone are enough",
    (name) => {
      const { label, image } = load(name);
      const reading = readPictureBoard(image, { quad: label.tilesQuad, size: label.size });
      if ("status" in reading) throw new Error(reading.message);
      expect(reading.blank).toBe(label.blank);
      expect(assemblePicture(reading.cells, reading.size, reading.blank).solved).toEqual(
        label.solved,
      );
    },
    30_000,
  );
});
