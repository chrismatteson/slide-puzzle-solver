export interface Point {
  x: number;
  y: number;
}

/** Four corners in the order top-left, top-right, bottom-right, bottom-left. */
export type Quad = readonly [Point, Point, Point, Point];

/** 3x3 homography, row-major, mapping source to destination coordinates. */
export type Homography = readonly [
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
];

/** Solves an n x n linear system in place by Gaussian elimination with partial pivoting. */
function solveLinear(matrix: number[][], rhs: number[]): number[] {
  const n = rhs.length;
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(matrix[row]?.[col] ?? 0) > Math.abs(matrix[pivot]?.[col] ?? 0)) pivot = row;
    }
    const pivotRow = matrix[pivot];
    const topRow = matrix[col];
    if (!pivotRow || !topRow || Math.abs(pivotRow[col] ?? 0) < 1e-12)
      throw new Error("degenerate point set");
    matrix[pivot] = topRow;
    matrix[col] = pivotRow;
    [rhs[pivot], rhs[col]] = [rhs[col] ?? 0, rhs[pivot] ?? 0];
    for (let row = col + 1; row < n; row++) {
      const target = matrix[row] ?? [];
      const factor = (target[col] ?? 0) / (pivotRow[col] ?? 1);
      for (let k = col; k < n; k++) target[k] = (target[k] ?? 0) - factor * (pivotRow[k] ?? 0);
      rhs[row] = (rhs[row] ?? 0) - factor * (rhs[col] ?? 0);
    }
  }
  const solution = new Array<number>(n).fill(0);
  for (let row = n - 1; row >= 0; row--) {
    let sum = rhs[row] ?? 0;
    for (let k = row + 1; k < n; k++) sum -= (matrix[row]?.[k] ?? 0) * (solution[k] ?? 0);
    solution[row] = sum / (matrix[row]?.[row] ?? 1);
  }
  return solution;
}

/** Homography taking the four `from` points to the four `to` points (direct linear transform, h33 = 1). */
export function computeHomography(from: Quad, to: Quad): Homography {
  const matrix: number[][] = [];
  const rhs: number[] = [];
  for (let i = 0; i < 4; i++) {
    const { x, y } = from[i] as Point;
    const { x: u, y: v } = to[i] as Point;
    matrix.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
    rhs.push(u);
    matrix.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
    rhs.push(v);
  }
  const h = solveLinear(matrix, rhs);
  return [
    h[0] ?? 0,
    h[1] ?? 0,
    h[2] ?? 0,
    h[3] ?? 0,
    h[4] ?? 0,
    h[5] ?? 0,
    h[6] ?? 0,
    h[7] ?? 0,
    1,
  ];
}

export function applyHomography(h: Homography, p: Point): Point {
  const w = h[6] * p.x + h[7] * p.y + h[8];
  return { x: (h[0] * p.x + h[1] * p.y + h[2]) / w, y: (h[3] * p.x + h[4] * p.y + h[5]) / w };
}

export function computePolygonArea(points: readonly Point[]): number {
  let sum = 0;
  for (let i = 0; i < points.length; i++) {
    const a = points[i] as Point;
    const b = points[(i + 1) % points.length] as Point;
    sum += a.x * b.y - b.x * a.y;
  }
  return Math.abs(sum) / 2;
}

function computeCross(o: Point, a: Point, b: Point): number {
  return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

/** Convex hull (Andrew's monotone chain), counter-clockwise in a y-up frame (clockwise on screen). */
export function computeConvexHull(points: readonly Point[]): Point[] {
  const sorted = [...points].sort((a, b) => a.x - b.x || a.y - b.y);
  if (sorted.length < 3) return sorted;
  const lower: Point[] = [];
  for (const p of sorted) {
    while (
      lower.length >= 2 &&
      computeCross(lower[lower.length - 2] as Point, lower[lower.length - 1] as Point, p) <= 0
    )
      lower.pop();
    lower.push(p);
  }
  const upper: Point[] = [];
  for (const p of sorted.reverse()) {
    while (
      upper.length >= 2 &&
      computeCross(upper[upper.length - 2] as Point, upper[upper.length - 1] as Point, p) <= 0
    )
      upper.pop();
    upper.push(p);
  }
  lower.pop();
  upper.pop();
  return [...lower, ...upper];
}

function intersectLines(a1: Point, a2: Point, b1: Point, b2: Point): Point | undefined {
  const d1x = a2.x - a1.x;
  const d1y = a2.y - a1.y;
  const d2x = b2.x - b1.x;
  const d2y = b2.y - b1.y;
  const denom = d1x * d2y - d1y * d2x;
  if (Math.abs(denom) < 1e-9 * (Math.hypot(d1x, d1y) * Math.hypot(d2x, d2y) + 1e-12))
    return undefined;
  const t = ((b1.x - a1.x) * d2y - (b1.y - a1.y) * d2x) / denom;
  return { x: a1.x + t * d1x, y: a1.y + t * d1y };
}

/**
 * Reduces a convex polygon to 4 corners by repeatedly deleting the edge whose removal adds the
 * least area (its neighbours are extended until they meet). The result circumscribes the polygon,
 * which suits a board photographed with rounded corners and blur.
 */
export function reduceToQuad(hull: readonly Point[]): Point[] | undefined {
  let polygon = [...hull];
  while (polygon.length > 4) {
    let best = -1;
    let bestPoint: Point | undefined;
    let bestCost = Infinity;
    const n = polygon.length;
    for (let i = 0; i < n; i++) {
      // Edge i runs from polygon[i] to polygon[i+1]; extend edges i-1 and i+1 to meet.
      const prev = polygon[(i + n - 1) % n] as Point;
      const a = polygon[i] as Point;
      const b = polygon[(i + 1) % n] as Point;
      const next = polygon[(i + 2) % n] as Point;
      const meet = intersectLines(prev, a, b, next);
      if (!meet) continue;
      const added = computePolygonArea([a, meet, b]);
      // meet on the same side as the rest of the polygon means the neighbours diverge: skip.
      if (computeCross(a, b, meet) * computeCross(a, b, polygon[(i + 3) % n] as Point) > 0)
        continue;
      if (added < bestCost) {
        bestCost = added;
        best = i;
        bestPoint = meet;
      }
    }
    if (best < 0 || !bestPoint) return undefined;
    const next: Point[] = [];
    for (let k = 0; k < n; k++) {
      if (k === best) next.push(bestPoint);
      else if (k !== (best + 1) % n) next.push(polygon[k] as Point);
    }
    polygon = next;
  }
  return polygon.length === 4 ? polygon : undefined;
}

/**
 * Orders four corners as top-left, top-right, bottom-right, bottom-left for a board that is at
 * most ~45 degrees from upright: walk them clockwise on screen and start so the first edge points right.
 */
export function orderCorners(corners: readonly Point[]): Quad {
  const cx = corners.reduce((s, p) => s + p.x, 0) / corners.length;
  const cy = corners.reduce((s, p) => s + p.y, 0) / corners.length;
  const clockwise = [...corners].sort(
    (a, b) => Math.atan2(a.y - cy, a.x - cx) - Math.atan2(b.y - cy, b.x - cx),
  );
  let bestStart = 0;
  let bestTilt = Infinity;
  for (let start = 0; start < 4; start++) {
    const a = clockwise[start] as Point;
    const b = clockwise[(start + 1) % 4] as Point;
    const tilt = Math.abs(Math.atan2(b.y - a.y, b.x - a.x));
    if (tilt < bestTilt) {
      bestTilt = tilt;
      bestStart = start;
    }
  }
  const pickCorner = (k: number): Point => clockwise[(bestStart + k) % 4] as Point;
  return [pickCorner(0), pickCorner(1), pickCorner(2), pickCorner(3)];
}

export function measureDistance(a: Point, b: Point): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}
