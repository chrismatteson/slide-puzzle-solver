// The owner's photo must assemble as labelled however it is decoded and however well its board was
// fitted. The browser decodes the JPEG differently from jpeg-js (IDCT, chroma upsampling), shrinks it to
// MAX_PHOTO_SIDE, and the fitted board is a few pixels off the hand-labelled one: assembly of the original
// scoring flipped the empty slot's home to the top-right in Chromium, see test/fixtures/owner-picture.
// The real Chromium decode is covered by e2e/picture.e2e.ts; here its differences are imitated.
import { readFileSync } from "node:fs";
import jpeg from "jpeg-js";
import { describe, expect, it } from "vitest";
import { assemblePicture } from "../src/picture/assemble.ts";
import { type Quad } from "../src/vision/geometry.ts";
import { MAX_PHOTO_SIDE, type RgbaImage, shrinkRgba } from "../src/vision/image.ts";
import { readPhoto } from "../scripts/read-photo.ts";
import { cutTileCrops } from "../src/picture/read.ts";

const DIR = "test/fixtures/owner-picture/";
const label = JSON.parse(readFileSync(`${DIR}picture-4x4-01.json`, "utf8")) as {
  blank: number;
  solved: number[];
  tilesQuad: Quad;
};
const original = readPhoto(`${DIR}picture-4x4-01.jpg`);

/** Deterministic noise in -1..1, so a failing jitter sample can be reproduced. */
function makeRandom(seed: number): () => number {
  let state = seed;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return (state / 2 ** 32) * 2 - 1;
  };
}

function scaleQuad(quad: Quad, factor: number): Quad {
  const [a, b, c, d] = quad.map((p) => ({ x: p.x * factor, y: p.y * factor }));
  return [a, b, c, d] as Quad;
}

function jitterQuad(quad: Quad, pixels: number, random: () => number): Quad {
  const [a, b, c, d] = quad.map((p) => ({
    x: p.x + pixels * random(),
    y: p.y + pixels * random(),
  }));
  return [a, b, c, d] as Quad;
}

function encodeAndDecode(image: RgbaImage, quality: number): RgbaImage {
  const encoded = jpeg.encode(
    { width: image.width, height: image.height, data: image.data },
    quality,
  );
  const decoded = jpeg.decode(encoded.data, { useTArray: true, formatAsRGBA: true });
  return {
    width: decoded.width,
    height: decoded.height,
    data: new Uint8ClampedArray(decoded.data),
  };
}

/**
 * What a different decoder does that matters here: chroma smoothed over 2x2 pixels (a browser upsamples
 * subsampled chroma with a filter, jpeg-js replicates it) and +-1 level rounding differences.
 */
function imitateBrowserDecode(image: RgbaImage): RgbaImage {
  const { width, height } = image;
  const data = new Uint8ClampedArray(image.data.length);
  const random = makeRandom(99);
  const at = (x: number, y: number, c: number): number =>
    image.data[(Math.min(y, height - 1) * width + Math.min(x, width - 1)) * 4 + c] ?? 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const luma = (xx: number, yy: number): number =>
        0.299 * at(xx, yy, 0) + 0.587 * at(xx, yy, 1) + 0.114 * at(xx, yy, 2);
      const ownLuma = luma(x, y);
      for (let c = 0; c < 3; c++) {
        let chroma = 0;
        for (const [dx, dy] of [
          [0, 0],
          [1, 0],
          [0, 1],
          [1, 1],
        ] as const)
          chroma += (at(x + dx, y + dy, c) - luma(x + dx, y + dy)) / 4;
        data[(y * width + x) * 4 + c] = ownLuma + chroma + Math.round(random());
      }
      data[(y * width + x) * 4 + 3] = 255;
    }
  }
  return { width, height, data };
}

function assembleWith(image: RgbaImage, quad: Quad): number[] {
  return assemblePicture(cutTileCrops(image, quad, 4), 4, label.blank).solved;
}

describe("the owner's photo assembles as labelled however it was decoded", () => {
  const shrunk = shrinkRgba(original, MAX_PHOTO_SIDE);
  const shrunkFactor = shrunk.width / original.width;
  const variants: [string, () => [RgbaImage, number]][] = [
    ["jpeg-js decode", () => [original, 1]],
    ["the app's downscale to MAX_PHOTO_SIDE", () => [shrunk, shrunkFactor]],
    ["re-encoded at JPEG quality 85", () => [encodeAndDecode(original, 85), 1]],
    ["a browser-like chroma upsampling", () => [imitateBrowserDecode(original), 1]],
    [
      "downscaled, then re-encoded at quality 85",
      () => [encodeAndDecode(shrunk, 85), shrunkFactor],
    ],
  ];
  for (const [name, make] of variants) {
    it(
      name,
      () => {
        const [image, factor] = make();
        expect(assembleWith(image, scaleQuad(label.tilesQuad, factor))).toEqual(label.solved);
      },
      30_000,
    );
  }
});

describe("the owner's photo with a badly fitted board", () => {
  it("assembles as labelled with every corner up to 4 px out, in each decode", () => {
    const random = makeRandom(4);
    const shrunk = shrinkRgba(original, MAX_PHOTO_SIDE);
    const factor = shrunk.width / original.width;
    for (let sample = 0; sample < 4; sample++) {
      const quad = jitterQuad(label.tilesQuad, 4, random);
      expect(assembleWith(original, quad), `original, sample ${String(sample)}`).toEqual(
        label.solved,
      );
      expect(
        assembleWith(shrunk, scaleQuad(quad, factor)),
        `downscaled, sample ${String(sample)}`,
      ).toEqual(label.solved);
    }
  }, 60_000);

  it("keeps the empty slot's home top-left and most tiles right with corners up to 8 px out", () => {
    // Not every sample can be exact at this error: 77% were when this was measured (48 samples, two
    // decodes). The bound only guards against a return to flipped homes and shifted rows.
    const random = makeRandom(8);
    let exact = 0;
    const samples = 12;
    for (let sample = 0; sample < samples; sample++) {
      const solved = assembleWith(original, jitterQuad(label.tilesQuad, 8, random));
      if (solved.every((cell, position) => cell === label.solved[position])) exact++;
    }
    expect(exact).toBeGreaterThanOrEqual(8);
  }, 60_000);
});
