// The detected tile area of the owner's photo must sit on the hand-clicked one however the photo was
// decoded or scaled. Before the corners were snapped to the tile seams, the bottom-right corner sat
// 20-35 px off in the smaller scales and in a JPEG re-encode, and assembly then failed.
import { describe, expect, it } from "vitest";
import { assemblePicture } from "../src/picture/assemble.ts";
import { readPictureBoard } from "../src/picture/read.ts";
import { makeVariants, ownerLabel } from "./picture-variants.ts";

/** Allowed corner error as a fraction of the board side (~20 px at full resolution). */
const TOLERANCE = 0.02;

describe("readPictureBoard's tile area on the owner's photo", () => {
  for (const variant of makeVariants()) {
    it(`matches the hand label and assembles as labelled: ${variant.name}`, () => {
      const result = readPictureBoard(variant.image, 4);
      if ("status" in result) throw new Error(result.message);
      const [a, b] = ownerLabel.tilesQuad;
      const side = Math.hypot(b.x - a.x, b.y - a.y) * variant.factor;
      result.quad.forEach((corner, i) => {
        const hand = ownerLabel.tilesQuad[i];
        const error =
          Math.hypot(
            corner.x - (hand?.x ?? 0) * variant.factor,
            corner.y - (hand?.y ?? 0) * variant.factor,
          ) / side;
        expect(error, `corner ${String(i)}`).toBeLessThan(TOLERANCE);
      });
      expect(result.blank).toBe(ownerLabel.blank);
      expect(assemblePicture(result.cells, 4, result.blank).solved).toEqual(ownerLabel.solved);
    }, 30_000);
  }
});
