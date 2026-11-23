// Snaps a tile-area quad onto the tile seams. The comb fit in `findGridWithStrength` works in the view
// rectified from a quad that is itself a few percent off (a frame's outer edge, a shadow, perspective the
// quad does not follow), so its tile area can sit tens of pixels off at one corner. The seams themselves
// are the real evidence: this moves the four corners until the N+1 lines in each direction, as a
// homography draws them, lie on the strongest edges across their length. Corners rather than
// per-axis offsets, so perspective is kept.

import { type GridSize } from "../puzzle/board.ts";
import { type Point, type Quad, applyHomography, computeHomography } from "../vision/geometry.ts";
import { type GreyImage, sampleBilinear } from "../vision/image.ts";

/** First and last search step per corner coordinate, as fractions of the board side. */
const START_STEP = 0.03;
const END_STEP = 0.0005;
/** A corner never moves further than this from where it started (a whole pitch would be a different grid). */
const MAX_MOVE = 0.06;
/** Samples along each seam. */
const SAMPLES = 48;
/**
 * Edge baselines (half-widths across the seam, as fractions of the board side), coarse to fine. The
 * coarse pass sees a blurred seam, so it has a smooth hill to climb; picture content makes the fine
 * score full of small peaks that trap a search started far away.
 */
const BASELINES = [0.012, 0.006, 0.003];

/** Corners moved together: each alone, and each side's pair (a lone corner is easily trapped by one strong picture edge). */
const MOVES = [[0], [1], [2], [3], [0, 1], [1, 2], [2, 3], [3, 0]];

const UNIT: Quad = [
  { x: 0, y: 0 },
  { x: 1, y: 0 },
  { x: 1, y: 1 },
  { x: 0, y: 1 },
];

/** Mean edge strength across all 2 * (size + 1) seams of the grid that `quad` bounds. */
function scoreSeams(
  grey: GreyImage,
  quad: Quad,
  size: GridSize,
  side: number,
  baseline: number,
): number {
  const h = computeHomography(UNIT, quad);
  let total = 0;
  for (const vertical of [false, true]) {
    for (let k = 0; k <= size; k++) {
      const s = k / size;
      const from = applyHomography(h, vertical ? { x: s, y: 0 } : { x: 0, y: s });
      const to = applyHomography(h, vertical ? { x: s, y: 1 } : { x: 1, y: s });
      const length = Math.hypot(to.x - from.x, to.y - from.y);
      const nx = -(to.y - from.y) / length;
      const ny = (to.x - from.x) / length;
      for (let i = 0; i < SAMPLES; i++) {
        const t = 0.04 + (0.92 * (i + 0.5)) / SAMPLES;
        const x = from.x + (to.x - from.x) * t;
        const y = from.y + (to.y - from.y) * t;
        const d = baseline * side;
        total += Math.sqrt(
          Math.abs(
            sampleBilinear(grey, x + nx * d, y + ny * d) -
              sampleBilinear(grey, x - nx * d, y - ny * d),
          ),
        );
      }
    }
  }
  return total / (2 * (size + 1) * SAMPLES);
}

/**
 * Coordinate search over the eight corner coordinates, halving the step each time no move helps,
 * once per baseline from coarse to fine. Returns `quad` unchanged if nothing scores better.
 * `grey` and `quad` share pixel coordinates.
 */
export function snapQuadToSeams(grey: GreyImage, quad: Quad, size: GridSize): Quad {
  const side = Math.hypot(quad[1].x - quad[0].x, quad[1].y - quad[0].y);
  const start = quad.map((p): Point => ({ ...p }));
  const current = quad.map((p): Point => ({ ...p }));
  const asQuad = (points: Point[]): Quad => points as unknown as Quad;
  BASELINES.forEach((baseline, stage) => {
    let best = scoreSeams(grey, asQuad(current), size, side, baseline);
    // Later stages only polish what the coarser one found.
    const first = stage === 0 ? START_STEP : START_STEP / 4;
    for (let step = first * side; step >= END_STEP * side; step /= 2) {
      let moved = true;
      while (moved) {
        moved = false;
        for (const corners of MOVES) {
          for (const axis of ["x", "y"] as const) {
            for (const sign of [-1, 1]) {
              const before = corners.map((c) => (current[c] as Point)[axis]);
              const inRange = corners.every((c, i) => {
                const next = (before[i] as number) + sign * step;
                return Math.abs(next - (start[c] as Point)[axis]) <= MAX_MOVE * side;
              });
              if (!inRange) continue;
              corners.forEach(
                (c, i) => ((current[c] as Point)[axis] = (before[i] as number) + sign * step),
              );
              const score = scoreSeams(grey, asQuad(current), size, side, baseline);
              if (score > best) {
                best = score;
                moved = true;
              } else {
                corners.forEach((c, i) => ((current[c] as Point)[axis] = before[i] as number));
              }
            }
          }
        }
      }
    }
  });
  return asQuad(current);
}
