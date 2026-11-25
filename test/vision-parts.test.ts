import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { sampleCrop } from "../src/vision/crop.ts";
import {
  type Quad,
  applyHomography,
  computeHomography,
  computeConvexHull,
  isConvexQuad,
  orderCorners,
  orderQuadCorners,
  computePolygonArea,
  reduceToQuad,
} from "../src/vision/geometry.ts";
import {
  type GreyImage,
  convertToGrey,
  convertToSaturation,
  shrinkGrey,
} from "../src/vision/image.ts";
import { rectifyQuad, RECTIFIED_SIZE } from "../src/vision/grid.ts";

interface CropReference {
  size: number;
  cases: { cx: number; cy: number; side: number; crop: number[] }[];
}

/** Same formula as scripts/make_crop_reference.py. */
function buildReferenceImage(size: number): GreyImage {
  const data = new Float32Array(size * size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const value = 128 + 90 * Math.sin(x * 0.31) * Math.cos(y * 0.17) + ((x * 3 + y * 5) % 23);
      data[y * size + x] = Math.floor(Math.min(255, Math.max(0, value)));
    }
  }
  return { width: size, height: size, data };
}

describe("sampleCrop", () => {
  it("matches the Python crops the classifier was trained on", () => {
    const reference = JSON.parse(
      readFileSync(new URL("fixtures/crop-reference.json", import.meta.url), "utf8"),
    ) as CropReference;
    const image = buildReferenceImage(reference.size);
    for (const { cx, cy, side, crop } of reference.cases) {
      const actual = sampleCrop(image, cx, cy, side);
      expect(actual).toHaveLength(crop.length);
      actual.forEach((v, i) => {
        expect(Math.abs(v - (crop[i] ?? 0))).toBeLessThan(2e-3);
      });
    }
  });
});

describe("geometry", () => {
  const unit: Quad = [
    { x: 0, y: 0 },
    { x: 1, y: 0 },
    { x: 1, y: 1 },
    { x: 0, y: 1 },
  ];
  const skewed: Quad = [
    { x: 10, y: 20 },
    { x: 90, y: 10 },
    { x: 100, y: 95 },
    { x: 5, y: 80 },
  ];

  it("maps the four source corners onto the four destination corners and an interior point consistently", () => {
    const h = computeHomography(unit, skewed);
    skewed.forEach((target, i) => {
      const mapped = applyHomography(h, unit[i] as { x: number; y: number });
      expect(mapped.x).toBeCloseTo(target.x, 6);
      expect(mapped.y).toBeCloseTo(target.y, 6);
    });
    const centre = applyHomography(h, { x: 0.5, y: 0.5 });
    // The image of the centre is the intersection of the diagonals, not the vertex average.
    expect(centre.x).toBeGreaterThan(40);
    expect(centre.x).toBeLessThan(60);
  });

  it("reduces a convex polygon with extra vertices to its four corners", () => {
    const hull = computeConvexHull([
      { x: 0, y: 0 },
      { x: 5, y: 0.1 },
      { x: 10, y: 0 },
      { x: 10.1, y: 5 },
      { x: 10, y: 10 },
      { x: 5, y: 9.9 },
      { x: 0, y: 10 },
      { x: -0.1, y: 5 },
    ]);
    const quad = reduceToQuad(hull);
    expect(quad).toBeDefined();
    expect(computePolygonArea(quad ?? [])).toBeGreaterThan(99);
    expect(computePolygonArea(quad ?? [])).toBeLessThan(103);
  });

  it("orders corners top-left first for a board rotated by 20 degrees", () => {
    const angle = (20 * Math.PI) / 180;
    const rotated = [
      [-1, -1],
      [1, -1],
      [1, 1],
      [-1, 1],
    ].map(([x, y]) => ({
      x: 50 + (x ?? 0) * Math.cos(angle) - (y ?? 0) * Math.sin(angle),
      y: 50 + (x ?? 0) * Math.sin(angle) + (y ?? 0) * Math.cos(angle),
    }));
    const ordered = orderCorners([
      rotated[2],
      rotated[0],
      rotated[3],
      rotated[1],
    ] as typeof rotated);
    expect(ordered[0]).toEqual(rotated[0]);
    expect(ordered[1]).toEqual(rotated[1]);
    expect(ordered[2]).toEqual(rotated[2]);
    expect(ordered[3]).toEqual(rotated[3]);
  });
});

describe("image helpers", () => {
  it("rectifies a quad onto the whole square", () => {
    const image: GreyImage = { width: 100, height: 100, data: new Float32Array(10000) };
    for (let y = 0; y < 100; y++)
      for (let x = 0; x < 100; x++) image.data[y * 100 + x] = x < 50 ? 0 : 200;
    const quad: Quad = [
      { x: 20, y: 20 },
      { x: 80, y: 20 },
      { x: 80, y: 80 },
      { x: 20, y: 80 },
    ];
    const rect = rectifyQuad(image, quad);
    expect(rect.width).toBe(RECTIFIED_SIZE);
    expect(rect.data[10 * RECTIFIED_SIZE + 10]).toBe(0);
    expect(rect.data[10 * RECTIFIED_SIZE + RECTIFIED_SIZE - 10]).toBe(200);
  });

  it("shrinks by box averaging and reports the scale", () => {
    const image: GreyImage = {
      width: 8,
      height: 4,
      data: Float32Array.from({ length: 32 }, (_, i) => (i % 8 < 4 ? 0 : 100)),
    };
    const { image: small, scale } = shrinkGrey(image, 4);
    expect(scale).toBe(0.5);
    expect([...small.data]).toEqual([0, 0, 100, 100, 0, 0, 100, 100]);
  });
});

describe("RGBA image helpers", () => {
  it("reject a buffer shorter than width*height*4", () => {
    const short = { width: 4, height: 4, data: new Uint8ClampedArray(63) };
    expect(() => convertToGrey(short)).toThrow(/RGBA buffer has 63 bytes, expected 64/);
    expect(() => convertToSaturation(short)).toThrow(/RGBA buffer has 63 bytes, expected 64/);
  });
});

describe("orderQuadCorners", () => {
  const tl = { x: 10, y: 10 };
  const tr = { x: 90, y: 20 };
  const br = { x: 80, y: 90 };
  const bl = { x: 5, y: 80 };

  it("keeps a quad that is already in order", () => {
    expect(orderQuadCorners([tl, tr, br, bl])).toEqual([tl, tr, br, bl]);
  });

  it("puts rotated and mirrored orders of the same corners into TL, TR, BR, BL", () => {
    for (const order of [
      [tr, br, bl, tl],
      [bl, tl, tr, br],
      [tr, tl, bl, br],
      [tl, bl, br, tr],
      [br, bl, tl, tr],
    ]) {
      expect(orderQuadCorners(order)).toEqual([tl, tr, br, bl]);
    }
  });

  it("orders a board tilted past 45 degrees as the detector's orderCorners does", () => {
    const angle = (60 * Math.PI) / 180;
    const corners = [
      [-1, -1],
      [1, -1],
      [1, 1],
      [-1, 1],
    ].map(([x, y]) => ({
      x: 50 + (x ?? 0) * Math.cos(angle) - (y ?? 0) * Math.sin(angle),
      y: 50 + (x ?? 0) * Math.sin(angle) + (y ?? 0) * Math.cos(angle),
    }));
    const shuffled = [corners[3], corners[0], corners[1], corners[2]] as typeof corners;
    expect(orderQuadCorners(shuffled)).toEqual(orderCorners(shuffled));
  });

  it("throws when the corners cross each other or are collinear", () => {
    expect(() => orderQuadCorners([tl, br, tr, bl])).toThrow("the corners cross each other");
    const line = [0, 10, 20, 30].map((x) => ({ x, y: 0 }));
    expect(() => orderQuadCorners(line)).toThrow("the corners cross each other");
  });

  it("throws unless there are four finite corners", () => {
    const message = "a quad needs four finite corners";
    expect(() => orderQuadCorners([tl, tr, br])).toThrow(message);
    expect(() => orderQuadCorners([tl, tr, br, bl, tl])).toThrow(message);
    expect(() => orderQuadCorners([tl, tr, br, { x: Number.NaN, y: 1 }])).toThrow(message);
    expect(() => orderQuadCorners([tl, tr, br, { x: 1, y: Infinity }])).toThrow(message);
  });

  it("isConvexQuad tells convex from folded", () => {
    expect(isConvexQuad([tl, tr, br, bl])).toBe(true);
    expect(isConvexQuad([bl, br, tr, tl])).toBe(true);
    expect(isConvexQuad([tl, br, tr, bl])).toBe(false);
  });
});
