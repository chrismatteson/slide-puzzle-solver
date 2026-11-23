import { readFileSync } from "node:fs";
import jpeg from "jpeg-js";
import { describe, expect, it } from "vitest";
import { cutTileCrops, readPictureBoard } from "../src/picture/read.ts";
import { type PictureReading, TILE_SIDE, type TileCrop } from "../src/picture/types.ts";
import { type GridSize } from "../src/puzzle/board.ts";
import {
  type Point,
  type Quad,
  applyHomography,
  computeHomography,
} from "../src/vision/geometry.ts";
import { type RgbaImage } from "../src/vision/image.ts";

function readOwnerPhoto(): { image: RgbaImage; label: OwnerLabel } {
  const decoded = jpeg.decode(
    readFileSync(new URL("fixtures/owner-picture/picture-4x4-01.jpg", import.meta.url)),
    { useTArray: true, formatAsRGBA: true },
  );
  const label = JSON.parse(
    readFileSync(new URL("fixtures/owner-picture/picture-4x4-01.json", import.meta.url), "utf8"),
  ) as OwnerLabel;
  const image = {
    width: decoded.width,
    height: decoded.height,
    data: new Uint8ClampedArray(decoded.data),
  };
  return { image, label };
}

interface OwnerLabel {
  size: GridSize;
  blank: number;
  tilesQuad: Quad;
}

function meanAbsDifference(a: TileCrop, b: TileCrop): number {
  let sum = 0;
  for (let i = 0; i < a.rgb.length; i++) sum += Math.abs((a.rgb[i] ?? 0) - (b.rgb[i] ?? 0));
  return sum / a.rgb.length;
}

function expectReading(result: ReturnType<typeof readPictureBoard>): PictureReading {
  if ("status" in result) throw new Error(`no board: ${result.message}`);
  return result;
}

describe("readPictureBoard on the owner's photo", () => {
  const { image, label } = readOwnerPhoto();
  const reading = expectReading(readPictureBoard(image));

  it("finds a 4x4 board and the empty slot", () => {
    expect(reading.size).toBe(4);
    expect(reading.cells).toHaveLength(16);
    expect(reading.blank).toBe(label.blank);
    expect(reading.blankConfidence).toBeGreaterThan(0.2);
    expect(reading.blankConfidence).toBeLessThanOrEqual(1);
  });

  it("cuts every cell like the hand-labelled tile area does", () => {
    const expected = cutTileCrops(image, label.tilesQuad, 4);
    reading.cells.forEach((crop, i) => {
      expect(crop.side).toBe(TILE_SIDE);
      const errors = expected.map((e) => meanAbsDifference(crop, e));
      expect(errors.indexOf(Math.min(...errors))).toBe(i);
    });
  });
});

// ---- synthetic picture boards ----

/** A picture with smooth colour gradients and a few shapes; every region of it looks different. */
function pictureColour(u: number, v: number): [number, number, number] {
  let r = 60 + 150 * u;
  let g = 200 - 140 * v;
  let b = 90 + 100 * Math.sin(3 * u + 2 * v) ** 2;
  if (Math.hypot(u - 0.3, v - 0.35) < 0.18) [r, g, b] = [230, 60, 50];
  if (Math.abs(u - 0.7) < 0.1 && v > 0.2 && v < 0.85) [r, g, b] = [250, 240, 70];
  if (Math.hypot(u - 0.55, v - 0.75) < 0.12) [r, g, b] = [30, 30, 40];
  return [r, g, b];
}

const BOARD_EDGE = 600;
const FRAME = 40;
const GAP = 6;

interface SyntheticBoard {
  image: RgbaImage;
  blank: number;
}

/** A board photographed at an angle: wooden-ish table, frame, picture tiles in a shuffled order. */
function renderBoard(size: GridSize, slot: "dark" | "light", blank: number): SyntheticBoard {
  const frameColour: [number, number, number] = slot === "dark" ? [28, 28, 32] : [225, 222, 215];
  const tableColour = (x: number, y: number): [number, number, number] => [
    150 + 25 * Math.sin(x * 0.05 + 3 * Math.sin(y * 0.01)),
    95 + 15 * Math.sin(x * 0.05),
    60,
  ];
  const width = 900;
  const height = 800;
  const corners: Quad = [
    { x: 150, y: 120 },
    { x: 760, y: 150 },
    { x: 790, y: 700 },
    { x: 120, y: 660 },
  ];
  const board: Quad = [
    { x: 0, y: 0 },
    { x: BOARD_EDGE, y: 0 },
    { x: BOARD_EDGE, y: BOARD_EDGE },
    { x: 0, y: BOARD_EDGE },
  ];
  const toBoard = computeHomography(corners, board);
  const pitch = (BOARD_EDGE - 2 * FRAME) / size;
  // The picture is shuffled by a fixed stride permutation, so tile i shows picture cell (i * 5 + 3) mod n.
  const n = size * size;
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const p: Point = applyHomography(toBoard, { x: x + 0.5, y: y + 0.5 });
      let rgb: [number, number, number] = tableColour(x, y);
      if (p.x >= 0 && p.x < BOARD_EDGE && p.y >= 0 && p.y < BOARD_EDGE) {
        rgb = frameColour;
        const cx = (p.x - FRAME) / pitch;
        const cy = (p.y - FRAME) / pitch;
        const col = Math.floor(cx);
        const row = Math.floor(cy);
        const inGap =
          (cx - col) * pitch < GAP / 2 ||
          (col + 1 - cx) * pitch < GAP / 2 ||
          (cy - row) * pitch < GAP / 2 ||
          (row + 1 - cy) * pitch < GAP / 2;
        if (
          col >= 0 &&
          col < size &&
          row >= 0 &&
          row < size &&
          !inGap &&
          row * size + col !== blank
        ) {
          const source = ((row * size + col) * 5 + 3) % n;
          const sourceCol = source % size;
          const sourceRow = Math.floor(source / size);
          rgb = pictureColour((sourceCol + (cx - col)) / size, (sourceRow + (cy - row)) / size);
        }
      }
      const i = (y * width + x) * 4;
      data[i] = rgb[0];
      data[i + 1] = rgb[1];
      data[i + 2] = rgb[2];
      data[i + 3] = 255;
    }
  }
  return { image: { width, height, data }, blank };
}

describe("readPictureBoard on synthetic boards", () => {
  const cases: { size: GridSize; slot: "dark" | "light"; blank: number }[] = [
    { size: 3, slot: "dark", blank: 4 },
    { size: 3, slot: "light", blank: 7 },
    { size: 4, slot: "dark", blank: 5 },
    { size: 4, slot: "light", blank: 10 },
  ];
  for (const { size, slot, blank } of cases) {
    it(`reads a ${String(size)}x${String(size)} board with a ${slot} empty slot`, () => {
      const synthetic = renderBoard(size, slot, blank);
      const reading = expectReading(readPictureBoard(synthetic.image));
      expect(reading.size).toBe(size);
      expect(reading.blank).toBe(synthetic.blank);
      expect(reading.cells).toHaveLength(size * size);
      expect(reading.blankConfidence).toBeGreaterThan(0.2);
    });
  }

  it("takes the grid size as given", () => {
    const synthetic = renderBoard(3, "dark", 0);
    expect(expectReading(readPictureBoard(synthetic.image, 3)).size).toBe(3);
  });

  it("finds no board in a flat image", () => {
    const data = new Uint8ClampedArray(200 * 150 * 4).fill(128);
    expect(readPictureBoard({ width: 200, height: 150, data })).toMatchObject({
      status: "no-board",
    });
  });
});
