import { readFileSync } from "node:fs";
import jpeg from "jpeg-js";
import { describe, expect, it } from "vitest";
import {
  isNumberedReading,
  keepUserCorners,
  MIN_CONFIDENT_SHARE,
  reassemblePicture,
} from "../src/app/routing.ts";
import { toBoard } from "../src/picture/arrangement.ts";
import { createPicture, createReading, scrambleOrder } from "./picture-fixtures.ts";
import { MIN_CELL_CONFIDENCE } from "../src/puzzle/consistency.ts";
import { defaultWeights } from "../src/vision/default-weights.ts";
import { type Quad } from "../src/vision/geometry.ts";
import { type ReadResult, readBoard } from "../src/vision/read-board.ts";

const quad: Quad = [
  { x: 0, y: 0 },
  { x: 1, y: 0 },
  { x: 1, y: 1 },
  { x: 0, y: 1 },
];

/** A rejected 3x3 reading whose cells have the given confidences. */
function reject(confidences: number[]): ReadResult {
  const cells = confidences.map((confidence, i) => ({ value: i, confidence }));
  return {
    status: "rejected",
    message: "low confidence",
    attempt: { size: 3, quad, cells, confidence: 0.5, rejection: undefined as never },
  };
}

describe("isNumberedReading", () => {
  it("takes an accepted reading as numbered", () => {
    const result: ReadResult = {
      status: "accepted",
      board: { size: 3, cells: [1, 2, 3, 4, 5, 6, 7, 8, 0] },
      quad,
      confidence: 1,
      cells: [],
    };
    expect(isNumberedReading(result)).toBe(true);
  });

  it("does not take a photo with no board as numbered", () => {
    expect(isNumberedReading({ status: "no-board", message: "none" })).toBe(false);
  });

  it("keeps a rejected reading numbered when enough cells were read confidently", () => {
    const sure = MIN_CELL_CONFIDENCE;
    expect(isNumberedReading(reject([sure, sure, sure, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))).toBe(true);
  });

  it("sends a rejected reading to picture mode when almost no cell was read confidently", () => {
    expect(isNumberedReading(reject([1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))).toBe(false);
    expect(isNumberedReading(reject(new Array<number>(9).fill(NaN)))).toBe(false);
  });

  it("treats exactly the minimum share as numbered", () => {
    const sure = Math.ceil(9 * MIN_CONFIDENT_SHARE);
    const confidences = Array.from({ length: 9 }, (_, i) => (i < sure ? 1 : 0.1));
    expect(isNumberedReading(reject(confidences))).toBe(true);
  });
});

// The rule's two constants were chosen from these photos; if the classifier changes, this says so.
describe("isNumberedReading on the fixtures", () => {
  function read(path: string): ReadResult {
    const decoded = jpeg.decode(readFileSync(`${import.meta.dirname}/fixtures/${path}`), {
      useTArray: true,
    });
    const image = {
      width: decoded.width,
      height: decoded.height,
      data: new Uint8ClampedArray(decoded.data),
    };
    return readBoard(image, defaultWeights);
  }

  it.each([
    ["photo-0008.jpg", true],
    ["photo-0004.jpg", true], // rejected as unsolvable, but every digit is read with confidence
    ["photo-lowconf.jpg", true], // rejected for low confidence in a few cells
    ["owner-picture/picture-4x4-01.jpg", false],
  ])(
    "%s -> numbered: %s",
    (path, numbered) => {
      expect(isNumberedReading(read(path))).toBe(numbered);
    },
    30_000,
  );
});

describe("reassemblePicture", () => {
  it("arranges the reading with the chosen blank and does not call it recognised", () => {
    const { reading } = createReading(createPicture(3, 2), 3, scrambleOrder(3, 0, 10, 4), 0);
    const result = reassemblePicture(reading, 4);
    // Checked by tsc: the result is a picture reading without narrowing it first.
    const asPicture: { kind: "picture" } = result;
    expect(asPicture.kind).toBe("picture");
    expect(result.reading.blank).toBe(4);
    expect(result.reading.blankConfidence).toBe(1);
    expect(result.recognised).toBe(false);
    expect(toBoard(result.reading, result.arrangement).cells[4]).toBe(0);
  });
});

describe("keepUserCorners", () => {
  it("keeps the options of a read from corners the user placed", () => {
    const options = { quad, size: 3 as const };
    expect(keepUserCorners(options)).toBe(options);
  });

  it("keeps nothing of a read the app did alone", () => {
    expect(keepUserCorners({})).toBeUndefined();
    expect(keepUserCorners({ size: 4 })).toBeUndefined();
  });
});
