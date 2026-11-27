// Tests the Mahalanobis scorer, `toSurprise` and `refineLayout`.
import { describe, expect, it } from "vitest";
import {
  type EdgeCosts,
  assemblePicture,
  refineLayout,
  toSurprise,
} from "../src/picture/assemble.ts";
import { measureMahalanobisCosts, toLab } from "../src/picture/mahalanobis.ts";
import { TILE_SIDE, type TileCrop } from "../src/picture/types.ts";

/** A smooth, busy picture cut into size x size tiles; photo cell i shows the tile of home `homes[i]`. */
function makeCrops(
  size: 3 | 4,
  homes: number[],
  offsetOf: (cell: number, x: number, y: number) => number = () => 0,
) {
  const side = size * TILE_SIDE;
  return homes.map((tile, cell): TileCrop => {
    const rgb = new Uint8ClampedArray(TILE_SIDE * TILE_SIDE * 3);
    const homeRow = Math.floor(tile / size);
    const homeCol = tile % size;
    for (let y = 0; y < TILE_SIDE; y++)
      for (let x = 0; x < TILE_SIDE; x++) {
        const u = (homeCol * TILE_SIDE + x) / side;
        const v = (homeRow * TILE_SIDE + y) / side;
        const at = (y * TILE_SIDE + x) * 3;
        const offset = offsetOf(cell, x, y);
        rgb[at] = 128 + 100 * Math.sin(7 * u + 3 * v) + offset;
        rgb[at + 1] = 128 + 100 * Math.cos(5 * v - 2 * u + 1) + offset;
        rgb[at + 2] = 128 + 100 * Math.sin(9 * u * v + 4 * v) + offset;
      }
    return { side: TILE_SIDE, rgb };
  });
}

// Pixels along the first lines of the crop edges.
const OUTLIERS = [
  [10, 1],
  [62, 20],
  [40, 62],
  [1, 45],
];

const HOMES = [3, 0, 6, 1, 7, 2, 4, 5, 8];

/** Cells right of / below each cell's tile in the solved picture, as [cell, neighbourCell]. */
function trueNeighbours(direction: "right" | "down"): [number, number][] {
  const pairs: [number, number][] = [];
  HOMES.forEach((home, cell) => {
    const next = direction === "right" ? (home % 3 < 2 ? home + 1 : -1) : home + 3;
    const neighbour = HOMES.indexOf(next);
    if (next >= 0 && next < 9 && neighbour >= 0) pairs.push([cell, neighbour]);
  });
  return pairs;
}

describe("Mahalanobis compatibility", () => {
  const sides = { right: ["right", "left"], down: ["bottom", "top"] } as const;

  for (const [label, offsetOf] of [
    ["a clean picture", () => 0],
    ["a brightness offset on every other tile", (cell: number) => (cell % 2 === 0 ? 25 : 0)],
  ] as const) {
    for (const direction of ["right", "down"] as const) {
      it(`ranks the true ${direction} neighbour first on ${label}`, () => {
        const lab = makeCrops(3, HOMES, offsetOf).map((crop) => toLab(crop));
        const [outgoing, incoming] = sides[direction];
        const costs = measureMahalanobisCosts(lab, TILE_SIDE, outgoing, incoming);
        for (const [cell, neighbour] of trueNeighbours(direction)) {
          const row = costs[cell] ?? [];
          expect(row.indexOf(Math.min(...row)), `cell ${String(cell)}`).toBe(neighbour);
        }
      });
    }
  }

  // The seam must cost about what the clean one does (within 1.5 of it per pixel; a few outlier pixels
  // still cost about 1), and a wrong partner must still cost more.
  describe("a seam disturbed on one side", () => {
    const clean = makeCrops(3, HOMES);
    const costsOf = (crops: TileCrop[], direction: "right" | "down"): number[][] => {
      const [outgoing, incoming] = sides[direction];
      return measureMahalanobisCosts(
        crops.map((crop) => toLab(crop)),
        TILE_SIDE,
        outgoing,
        incoming,
      );
    };
    const disturbances: [string, (cell: number, x: number, y: number) => number][] = [
      // Glare: a brightness ramp of 30 levels over the tile edge, on every other tile.
      [
        "a linear brightness ramp",
        (cell, x, y) => (cell % 2 === 0 ? (30 * (x + y)) / TILE_SIDE : 0),
      ],
      // A few pixels (dust, a hot pixel): one per tile edge, 60 levels off, on every other tile.
      [
        "a few outlier pixels",
        (cell, x, y) =>
          cell % 2 === 0 && OUTLIERS.some(([ox, oy]) => ox === x && oy === y) ? 60 : 0,
      ],
    ];
    for (const [label, disturbance] of disturbances) {
      for (const direction of ["right", "down"] as const) {
        it(`scores the true ${direction} neighbour close to the clean seam with ${label}, and below a wrong partner`, () => {
          const cleanCosts = costsOf(clean, direction);
          const costs = costsOf(makeCrops(3, HOMES, disturbance), direction);
          for (const [cell, neighbour] of trueNeighbours(direction)) {
            const row = costs[cell] ?? [];
            const truth = row[neighbour] ?? Number.NaN;
            expect(truth, `cell ${String(cell)}`).toBeLessThan(
              (cleanCosts[cell]?.[neighbour] ?? 0) + 1.5,
            );
            const wrong = Math.min(...row.filter((_, other) => other !== neighbour));
            expect(truth, `cell ${String(cell)}`).toBeLessThan(wrong);
          }
        });
      }
    }
  });

  it("is infinite on the diagonal", () => {
    const lab = makeCrops(3, HOMES).map((crop) => toLab(crop));
    const costs = measureMahalanobisCosts(lab, TILE_SIDE, "right", "left");
    expect(costs[2]?.[2]).toBe(Number.POSITIVE_INFINITY);
  });

  it("lets assemblePicture restore the picture despite the offsets", () => {
    const crops = makeCrops(3, HOMES, (cell) => (cell % 2 === 0 ? 25 : 0));
    const expected = HOMES.map((_, home) => HOMES.indexOf(home));
    expected[8] = -1;
    // The slot is the cell showing home 8, which is cell 8.
    expect(assemblePicture(crops, 3, 8).solved).toEqual(expected);
  });
});

describe("toSurprise", () => {
  // Tile i fits tile i + 1 well (cost 0.4) and everything else as a wrong neighbour (1).
  const costs = Array.from({ length: 5 }, (_, i) =>
    Array.from({ length: 5 }, (_, j) =>
      i === j ? Number.POSITIVE_INFINITY : j === i + 1 ? 0.4 : 1,
    ),
  );
  const { seam, outOpen } = toSurprise(costs);

  it("gives probabilities: each edge's neighbours and 'no neighbour' sum to one", () => {
    seam.forEach((row, i) => {
      const probabilities = row.map((surprise) => Math.exp(-surprise));
      for (const p of probabilities) {
        expect(p).toBeGreaterThanOrEqual(0);
        expect(p).toBeLessThanOrEqual(1);
      }
      const total = probabilities.reduce((a, b) => a + b, 0) + Math.exp(-(outOpen[i] ?? 0));
      expect(total).toBeCloseTo(1, 1);
    });
  });

  it("prefers the unique match", () => {
    for (let i = 0; i < 4; i++) {
      const row = seam[i] ?? [];
      expect(row.indexOf(Math.min(...row))).toBe(i + 1);
    }
  });
});

describe("refineLayout", () => {
  // 3x3: tile p belongs at position p (p < 8), the empty slot at 8. A true seam costs 0, others 5.
  const tiles = [0, 1, 2, 3, 4, 5, 6, 7];
  const costs: EdgeCosts = {
    right: tiles.map((a) => tiles.map((b) => (b === a + 1 && a % 3 !== 2 ? 0 : 5))),
    down: tiles.map((a) => tiles.map((b) => (b === a + 3 ? 0 : 5))),
    open: {
      left: tiles.map(() => 0),
      right: tiles.map(() => 0),
      top: tiles.map(() => 0),
      bottom: tiles.map(() => 0),
    },
    missing: 1,
  };
  const truth = [0, 1, 2, 3, 4, 5, 6, 7, -1];
  const exchange = (layout: number[], a: number, b: number): number[] => {
    const next = [...layout];
    next[a] = layout[b] ?? -1;
    next[b] = layout[a] ?? -1;
    return next;
  };
  const inversions = (layout: readonly number[]): number => {
    const placed = layout.filter((tile) => tile >= 0);
    let count = 0;
    for (let i = 0; i < placed.length; i++)
      for (let j = i + 1; j < placed.length; j++) if ((placed[j] ?? 0) < (placed[i] ?? 0)) count++;
    return count;
  };

  it("fixes a planted pair of exchanges", () => {
    const planted = exchange(exchange(truth, 0, 1), 3, 7);
    const refined = refineLayout(costs, 3, { tileAt: planted, cost: 1e9, parity: 0 }, () => true);
    expect(refined.tileAt).toEqual(truth);
  });

  it("only returns layouts the caller allows, here those of even parity", () => {
    const isAllowed = (layout: readonly number[]): boolean => inversions(layout) % 2 === 0;
    const start = exchange(exchange(truth, 0, 1), 3, 7);
    expect(isAllowed(start)).toBe(true);
    const refined = refineLayout(costs, 3, { tileAt: start, cost: 1e9, parity: 0 }, isAllowed);
    expect(isAllowed(refined.tileAt)).toBe(true);
    // One exchange alone is not allowed, so it cannot be undone alone.
    const oneOff = exchange(truth, 0, 1);
    expect(isAllowed(oneOff)).toBe(false);
  });
});
