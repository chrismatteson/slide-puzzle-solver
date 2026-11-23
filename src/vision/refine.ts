import { type Point, type Quad, measureDistance } from "./geometry.ts";
import { type GreyImage, sampleBilinear } from "./image.ts";

// Candidate quads from thresholding are a few percent off (shadows, blur, threshold bias). Each
// side is re-fitted to the strongest edge near it: sample edge positions along the side, fit a
// line through the pair of samples with the most inliers (every pair is tried; clutter edges are outliers), intersect neighbouring lines for new corners.

const SAMPLES_PER_SIDE = 24;
const SEARCH_FRACTION = 0.08; // search window half-width, as a fraction of the side length
const SEARCH_STEP = 0.5; // pixels
const INLIER_TOLERANCE = 1.5; // pixels
const MIN_INLIERS = 10;

interface Line {
  /** A point on the line and its unit direction. */
  origin: Point;
  dx: number;
  dy: number;
}

function findStrongestEdgeOffset(
  views: readonly GreyImage[],
  at: Point,
  normal: Point,
  halfWindow: number,
): { offset: number; strength: number } {
  let best = { offset: 0, strength: 0 };
  for (let offset = -halfWindow; offset <= halfWindow; offset += SEARCH_STEP) {
    let strength = 0;
    for (const view of views) {
      const x = at.x + normal.x * offset;
      const y = at.y + normal.y * offset;
      const difference =
        sampleBilinear(view, x + normal.x * 1.5, y + normal.y * 1.5) -
        sampleBilinear(view, x - normal.x * 1.5, y - normal.y * 1.5);
      strength = Math.max(strength, Math.abs(difference));
    }
    // Prefer edges near the candidate line over equally strong ones further away.
    const weighted = strength * (1 - 0.5 * Math.abs(offset / halfWindow));
    if (weighted > best.strength) best = { offset, strength: weighted };
  }
  return best;
}

function fitLine(points: readonly Point[]): Line | undefined {
  let best: Point[] = [];
  for (let i = 0; i < points.length; i++) {
    for (let j = i + 1; j < points.length; j++) {
      const a = points[i] as Point;
      const b = points[j] as Point;
      const length = measureDistance(a, b);
      if (length < 1e-6) continue;
      const dx = (b.x - a.x) / length;
      const dy = (b.y - a.y) / length;
      const inliers = points.filter(
        (p) => Math.abs((p.x - a.x) * dy - (p.y - a.y) * dx) <= INLIER_TOLERANCE,
      );
      if (inliers.length > best.length) best = inliers;
    }
  }
  if (best.length < MIN_INLIERS) return undefined;
  // Total-least-squares line through the inliers.
  const mx = best.reduce((s, p) => s + p.x, 0) / best.length;
  const my = best.reduce((s, p) => s + p.y, 0) / best.length;
  let sxx = 0;
  let sxy = 0;
  let syy = 0;
  for (const p of best) {
    sxx += (p.x - mx) ** 2;
    sxy += (p.x - mx) * (p.y - my);
    syy += (p.y - my) ** 2;
  }
  const angle = 0.5 * Math.atan2(2 * sxy, sxx - syy);
  return { origin: { x: mx, y: my }, dx: Math.cos(angle), dy: Math.sin(angle) };
}

function intersect(a: Line, b: Line): Point | undefined {
  const denom = a.dx * b.dy - a.dy * b.dx;
  if (Math.abs(denom) < 1e-6) return undefined;
  const t = ((b.origin.x - a.origin.x) * b.dy - (b.origin.y - a.origin.y) * b.dx) / denom;
  return { x: a.origin.x + t * a.dx, y: a.origin.y + t * a.dy };
}

/** Returns the quad with each side snapped to the nearest strong straight edge; sides with no clear edge stay as they were. */
export function refineQuad(views: readonly GreyImage[], quad: Quad): Quad {
  const lines: Line[] = [];
  for (let side = 0; side < 4; side++) {
    const a = quad[side] as Point;
    const b = quad[(side + 1) % 4] as Point;
    const length = measureDistance(a, b);
    const dx = (b.x - a.x) / length;
    const dy = (b.y - a.y) / length;
    const normal = { x: -dy, y: dx };
    const halfWindow = SEARCH_FRACTION * length;
    const found: Point[] = [];
    for (let k = 0; k < SAMPLES_PER_SIDE; k++) {
      const t = 0.1 + (0.8 * k) / (SAMPLES_PER_SIDE - 1);
      const at = { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
      const { offset, strength } = findStrongestEdgeOffset(views, at, normal, halfWindow);
      if (strength > 6) found.push({ x: at.x + normal.x * offset, y: at.y + normal.y * offset });
    }
    lines.push(fitLine(found) ?? { origin: a, dx, dy });
  }
  const corners: Point[] = [];
  for (let i = 0; i < 4; i++) {
    const corner = intersect(lines[(i + 3) % 4] as Line, lines[i] as Line);
    if (!corner) return quad;
    corners.push(corner);
  }
  // The loop above pushed exactly four corners, which TypeScript cannot track into a tuple.
  return corners as unknown as Quad;
}
