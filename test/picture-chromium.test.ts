// The owner's photo as the app sees it in Chromium: createImageBitmap + canvas, downscaled to fit 1280 px
// (960x1280). jpeg-js's decode assembled correctly while this one did not: the glare on the right edge
// lifts the blue of two right-column tiles by different amounts than the green beside it, and the true
// seam between them cost more than a wrong one. The pixels are saved lossless (see the fixture README).
import { readFileSync } from "node:fs";
import { PNG } from "pngjs";
import { describe, expect, it } from "vitest";
import { assemblePicture } from "../src/picture/assemble.ts";
import { readPictureBoard } from "../src/picture/read.ts";
import { ownerLabel } from "./picture-variants.ts";

function readChromiumDecode(): { width: number; height: number; data: Uint8ClampedArray } {
  const png = PNG.sync.read(
    readFileSync(
      new URL("fixtures/owner-picture/picture-4x4-01.chromium-960.png", import.meta.url),
    ),
  );
  return { width: png.width, height: png.height, data: new Uint8ClampedArray(png.data) };
}

describe("the owner's photo as decoded by Chromium (960x1280)", () => {
  it("assembles as labelled", () => {
    const reading = readPictureBoard(readChromiumDecode(), 4);
    if ("status" in reading) throw new Error(reading.message);
    expect(reading.blank).toBe(ownerLabel.blank);
    const start = performance.now();
    const arrangement = assemblePicture(reading.cells, 4, reading.blank);
    expect(arrangement.solved).toEqual(ownerLabel.solved);
    // The brief's 2 s assembly budget. Measured (Node 22.23.3, 4 cores): assemblePicture alone on this
    // decode takes 0.54-0.66 s, about 0.64 s inside the full `npm test`, 0.72-0.88 s with the whole
    // suite on one core. If this fails, make assembly faster rather than raising the bound.
    expect(performance.now() - start).toBeLessThan(2000);
    // The whole test (read plus assembly) takes 1.2-1.6 s; the timeout is about 20x that.
  }, 30_000);
});
