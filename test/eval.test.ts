import { copyFileSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import jpeg from "jpeg-js";
import { PNG } from "pngjs";
import { describe, expect, it } from "vitest";
import { evaluateDirectory } from "../scripts/eval.ts";
import { parseBoardLabel, parsePictureLabel } from "../scripts/labels.ts";
import { fitWithin, shrinkRgba } from "../src/vision/image.ts";

const resolveFixturePath = (name: string): string =>
  new URL(`fixtures/${name}`, import.meta.url).pathname;

describe("evaluateDirectory with real-photo style labels", () => {
  // Measured (Node 22, 4-core container): 2.9 s for the two photos. Timeout 10x that, see vision.test.ts.
  it("reads a jpg and a png labelled with only size and state", { timeout: 30_000 }, () => {
    const dir = mkdtempSync(join(tmpdir(), "eval-"));
    const full = JSON.parse(readFileSync(resolveFixturePath("photo-0008.json"), "utf8")) as {
      size: number;
      state: number[];
    };
    const label = JSON.stringify({ size: full.size, state: full.state });
    copyFileSync(resolveFixturePath("photo-0008.jpg"), join(dir, "a.jpg"));
    writeFileSync(join(dir, "a.json"), label);

    const decoded = jpeg.decode(readFileSync(resolveFixturePath("photo-0008.jpg")), {
      useTArray: true,
      formatAsRGBA: true,
    });
    const png = new PNG({ width: decoded.width, height: decoded.height });
    png.data = Buffer.from(decoded.data);
    writeFileSync(join(dir, "b.png"), PNG.sync.write(png));
    writeFileSync(join(dir, "b.json"), label);

    const outcomes = evaluateDirectory(dir);
    expect(outcomes.map((o) => o.name)).toEqual(["a", "b"]);
    for (const o of outcomes) {
      expect(o.hasCorners).toBe(false);
      expect(o.cornerError).toBeUndefined();
      expect(o.font).toBe("unknown");
      expect(o.solvable).toBe(true);
      expect(o.exact).toBe(true);
    }
  });
});

describe("shrinkRgba", () => {
  it("scales the longer side down to the limit and never up", () => {
    expect(fitWithin(4000, 3000, 1280)).toEqual({ width: 1280, height: 960 });
    expect(fitWithin(800, 600, 1280)).toEqual({ width: 800, height: 600 });
    const image = { width: 40, height: 20, data: new Uint8ClampedArray(40 * 20 * 4).fill(200) };
    const small = shrinkRgba(image, 10);
    expect([small.width, small.height]).toEqual([10, 5]);
    expect(small.data.every((v) => v === 200)).toBe(true);
  });
});

describe("evaluateDirectory rejects bad labels", () => {
  // Labels are checked before any photo is read, so the photo bytes never matter here.
  const runWithLabel = (label: string | undefined): void => {
    const dir = mkdtempSync(join(tmpdir(), "eval-bad-"));
    writeFileSync(join(dir, "a.png"), "not read");
    if (label !== undefined) writeFileSync(join(dir, "a.json"), label);
    evaluateDirectory(dir);
  };

  it.each([
    ['{"size":5,"state":[5,1,3,7,8,2,6,0,4]}', /a\.json: size must be 3 or 4/],
    ['{"size":3,"state":[1,2,3]}', /a\.json: state is not a permutation/],
    ['{"size":3,"state":[0,1,2,3,4,5,6,8,8]}', /a\.json: state is not a permutation/],
    ['{"size":"4","state":[0,1,2,3]}', /a\.json: size must be 3 or 4/],
    ['{"size":3}', /a\.json: state must be an array/],
    ['{"size":3,"state":[0,1,2,3,4,5,6,7,"8"]}', /a\.json: state must hold integers/],
    ['{"size":3,"state":[0,1,2,3,4,5,6,7,8],"solvable":"yes"}', /a\.json: solvable must be/],
    ['{"size":3,"state":[0,1,2,3,4,5,6,7,8],"corners":[[0,0]]}', /a\.json: corners must be/],
    ["[1,2,3]", /a\.json: label must be a JSON object/],
    ["{not json", /a\.json: /],
  ])("%s", (label, reason) => {
    expect(() => {
      runWithLabel(label);
    }).toThrow(reason);
  });

  it("names the missing label file", () => {
    expect(() => {
      runWithLabel(undefined);
    }).toThrow(/a\.json: missing label for a\.png/);
  });
});

describe("evaluateDirectory refuses an empty run", () => {
  it("names a directory with no photos", () => {
    const dir = mkdtempSync(join(tmpdir(), "eval-empty-"));
    expect(() => evaluateDirectory(dir)).toThrow(/eval-empty-.*: no \.jpg or \.png photos/);
  });

  it.each([0, -1, 1.5, Number.NaN])("rejects every-nth %s", (every) => {
    const dir = mkdtempSync(join(tmpdir(), "eval-step-"));
    expect(() => evaluateDirectory(dir, every)).toThrow(/every-nth must be a whole number >= 1/);
  });
});

describe("parseBoardLabel", () => {
  it("returns every field of a full label", () => {
    const raw = {
      size: 3,
      state: [1, 2, 3, 4, 5, 6, 7, 8, 0],
      solvable: true,
      corners: [
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8.5],
      ],
      font: "DejaVuSans",
    };
    expect(parseBoardLabel(raw, "x.json")).toEqual(raw);
  });

  it("accepts a label with only size and state", () => {
    const raw = { size: 4, state: [...Array(16).keys()] };
    expect(parseBoardLabel(raw, "x.json")).toEqual(raw);
  });
});

describe("parsePictureLabel", () => {
  const quad = [
    { x: 0, y: 0 },
    { x: 1, y: 0 },
    { x: 1, y: 1 },
    { x: 0, y: 1 },
  ];
  const raw = { size: 3, blank: 4, solved: [0, 1, 2, 3, -1, 5, 6, 7, 8], tilesQuad: quad };

  it("returns a picture label, with the thumbnail quad only when there is one", () => {
    expect(parsePictureLabel(raw, "x")).toEqual(raw);
    expect(parsePictureLabel({ ...raw, thumbQuad: quad }, "x").thumbQuad).toEqual(quad);
  });

  it("names the file and the problem", () => {
    expect(() => parsePictureLabel({ ...raw, size: 5 }, "x")).toThrow("x: size must be 3 or 4");
    expect(() => parsePictureLabel({ ...raw, solved: [1] }, "x")).toThrow(
      "x: solved must have 9 entries",
    );
    expect(() => parsePictureLabel({ ...raw, tilesQuad: [] }, "x")).toThrow(
      "x: tilesQuad must have 4",
    );
  });

  it("rejects corners that are not four {x, y} points of finite numbers", () => {
    expect(() => parsePictureLabel({ ...raw, tilesQuad: [1, 2, 3, 4] }, "x")).toThrow(
      "x: tilesQuad corners must be {x, y} objects",
    );
    expect(() =>
      parsePictureLabel({ ...raw, tilesQuad: [quad[0], quad[1], quad[2], { x: 0 }] }, "x"),
    ).toThrow("x: tilesQuad corners must have finite x and y");
    expect(() =>
      parsePictureLabel({ ...raw, thumbQuad: [quad[0], quad[1], quad[2], { x: 0, y: null }] }, "x"),
    ).toThrow("x: thumbQuad corners must have finite x and y");
    expect(() =>
      parsePictureLabel({ ...raw, thumbQuad: [...quad.slice(0, 3), { x: Infinity, y: 0 }] }, "x"),
    ).toThrow("x: thumbQuad corners must have finite x and y");
  });

  it("rejects a solved layout that is not made of integers, or a label that is not an object", () => {
    expect(() =>
      parsePictureLabel({ ...raw, solved: [0, 1, 2, 3, -1, 5, 6, 7, "8"] }, "x"),
    ).toThrow("x: solved must hold integers only");
    expect(() =>
      parsePictureLabel({ ...raw, solved: [0, 1, 2, 3, -1, 5, 6, 7, 8.5] }, "x"),
    ).toThrow("x: solved must hold integers only");
    expect(() => parsePictureLabel([], "x")).toThrow("x: label must be an object");
    expect(() => parsePictureLabel(null, "x")).toThrow("x: label must be an object");
  });
});
