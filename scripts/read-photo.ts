// Loads a labelled photo from disk for the picture eval and tests. Crops are cut by the app's own
// cutTileCrops (src/picture/read.ts), so what they measure is what the app produces.
import { readFileSync } from "node:fs";
import jpeg from "jpeg-js";
import { type RgbaImage } from "../src/vision/image.ts";

/** Decodes a JPEG file at full size: the labelled quads are in its pixels, so it is not shrunk. */
export function readPhoto(path: string): RgbaImage {
  const decoded = jpeg.decode(readFileSync(path), { useTArray: true, formatAsRGBA: true });
  return {
    width: decoded.width,
    height: decoded.height,
    data: new Uint8ClampedArray(decoded.data),
  };
}
