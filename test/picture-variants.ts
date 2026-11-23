// Decode / scale variants of the owner's photo, shared by the corner-accuracy test and any ad-hoc
// measurement. Each variant carries the factor from full-resolution photo pixels to its own pixels.
import { readFileSync } from "node:fs";
import jpeg from "jpeg-js";
import { readPhoto } from "../scripts/read-photo.ts";
import { type Quad } from "../src/vision/geometry.ts";
import { type RgbaImage, shrinkRgba } from "../src/vision/image.ts";

const DIR = "test/fixtures/owner-picture/";

export interface OwnerLabel {
  blank: number;
  solved: number[];
  tilesQuad: Quad;
}

export const ownerLabel = JSON.parse(
  readFileSync(`${DIR}picture-4x4-01.json`, "utf8"),
) as OwnerLabel;
export const ownerPhoto = readPhoto(`${DIR}picture-4x4-01.jpg`);

export interface Variant {
  name: string;
  image: RgbaImage;
  /** Scale from full-resolution photo pixels to this variant's pixels. */
  factor: number;
}

function reencode(image: RgbaImage, quality: number): RgbaImage {
  const encoded = jpeg.encode(image, quality);
  const decoded = jpeg.decode(encoded.data, { useTArray: true, formatAsRGBA: true });
  return {
    width: decoded.width,
    height: decoded.height,
    data: new Uint8ClampedArray(decoded.data),
  };
}

/** 3x3 box blur, repeated, per channel; enough to soften seams the way a slightly soft photo does. */
function blur(image: RgbaImage, passes: number): RgbaImage {
  const { width, height } = image;
  let src = image.data;
  for (let p = 0; p < passes; p++) {
    const out = new Uint8ClampedArray(src.length);
    for (let y = 0; y < height; y++)
      for (let x = 0; x < width; x++)
        for (let c = 0; c < 4; c++) {
          let sum = 0;
          for (let dy = -1; dy <= 1; dy++)
            for (let dx = -1; dx <= 1; dx++) {
              const xx = Math.min(width - 1, Math.max(0, x + dx));
              const yy = Math.min(height - 1, Math.max(0, y + dy));
              sum += src[(yy * width + xx) * 4 + c] ?? 0;
            }
          out[(y * width + x) * 4 + c] = sum / 9;
        }
    src = out;
  }
  return { width, height, data: src };
}

/** Box-shrinks to exactly `width` x `height`, via shrinkRgba's maxSide rule (longer side). */
function shrinkTo(image: RgbaImage, maxSide: number): Variant["image"] {
  return shrinkRgba(image, maxSide);
}

export function makeVariants(): Variant[] {
  const full = ownerPhoto;
  const longer = Math.max(full.width, full.height);
  const at = (f: number): Variant => {
    const image = shrinkTo(full, Math.round(longer * f));
    return { name: `box-shrink ${String(f)}`, image, factor: image.width / full.width };
  };
  const small = shrinkTo(full, 1280);
  return [
    { name: "jpeg-js full", image: full, factor: 1 },
    at(0.75),
    at(0.5),
    { name: "960x1280", image: small, factor: small.width / full.width },
    { name: "re-encode q85", image: reencode(full, 85), factor: 1 },
    { name: "blur", image: blur(full, 1), factor: 1 },
  ];
}
