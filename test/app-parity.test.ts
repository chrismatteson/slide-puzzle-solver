import { readFileSync } from "node:fs";
import jpeg from "jpeg-js";
import { describe, expect, it } from "vitest";
import { checkBoard } from "../src/app/review.ts";
import { MIN_CELL_CONFIDENCE } from "../src/puzzle/consistency.ts";
import { defaultWeights } from "../src/vision/default-weights.ts";
import { type RgbaImage } from "../src/vision/image.ts";
import { type ReadResult, readBoard } from "../src/vision/read-board.ts";

// The vision worker calls exactly `readBoard(image, defaultWeights)` and the page feeds the attempt's
// cells to `checkBoard`; this runs the same two calls on the low-confidence fixture. The browser decodes
// JPEGs with its own decoder, which differs from jpeg-js by up to ~17 levels (mean ~0.4, measured on
// this fixture in Chromium). The fixture's weakest cell must therefore sit well below the threshold:
// an earlier fixture at 0.885 read 0.956 (accepted) in the browser.
function loadFixture(): { image: RgbaImage; state: number[] } {
  const label = JSON.parse(
    readFileSync(new URL("fixtures/photo-lowconf.json", import.meta.url), "utf8"),
  ) as { state: number[] };
  const decoded = jpeg.decode(
    readFileSync(new URL("fixtures/photo-lowconf.jpg", import.meta.url)),
    {
      useTArray: true,
      formatAsRGBA: true,
    },
  );
  return {
    image: {
      width: decoded.width,
      height: decoded.height,
      data: new Uint8ClampedArray(decoded.data),
    },
    state: label.state,
  };
}

function expectLowConfidenceOnly(result: ReadResult, state: number[]): void {
  if (result.status !== "rejected") throw new Error(`expected a rejection, got ${result.status}`);
  const { cells } = result.attempt;
  expect(result.attempt.rejection?.reason).toBe("low-confidence");
  expect(cells.map((c) => c.value)).toEqual(state);
  const weakest = Math.min(...cells.map((c) => c.confidence));
  expect(weakest).toBeLessThan(MIN_CELL_CONFIDENCE - 0.05);

  const check = checkBoard(result.attempt.size, cells);
  const unsure = cells.flatMap((c, i) => (c.confidence < MIN_CELL_CONFIDENCE ? [i] : []));
  expect(check).toMatchObject({ accepted: false, confirmable: true, offending: unsure });
  expect(unsure.length).toBeGreaterThan(0);
}

describe("app and Node decide the same on the low-confidence photo", () => {
  it("rejects it, flags the unsure cells, and offers to confirm them", { timeout: 25_000 }, () => {
    const { image, state } = loadFixture();
    expectLowConfidenceOnly(readBoard(image, defaultWeights), state);
  });
});
