import { type GreyImage, blurGrey } from "./image.ts";
import {
  type Point,
  type Quad,
  computeConvexHull,
  measureDistance,
  orderCorners,
  computePolygonArea,
  reduceToQuad,
} from "./geometry.ts";

// Board finding without a vision library (OpenCV.js is ~8-10 MB, too heavy for a phone page).
// Two kinds of candidate come from thresholding several views of the photo (plain and lighting-flattened) at many levels:
//  - silhouette: a blob (holes filled) whose convex hull reduces to a convex quad. It is the board
//    outline when the board contrasts with its surroundings.
//  - tiles: a cluster of similar, adjacent, square-ish blobs (the tiles themselves). Tiles are
//    printed to be legible, so they contrast with the frame even when the frame blends into the table.
// The caller picks the right candidate by actually reading it, which is far more reliable than any
// shape score here.

export interface BoardCandidate {
  /** Corners in the coordinates of the views passed to findBoardCandidates. */
  quad: Quad;
  kind: "silhouette" | "tiles";
  /** Blob area divided by quad area (silhouettes only; tiles report their mean). */
  fill: number;
}

const LEVELS = 12;
const MIN_AREA_FRACTION = 0.05;
const MAX_AREA_FRACTION = 0.9;
const MIN_FILL = 0.8;
const HULL_VERTICES = 12;
/** Tile blobs: area bounds as a fraction of the image, and the tile count that makes a cluster. */
const TILE_MIN_AREA_FRACTION = 0.0012;
const TILE_MAX_AREA_FRACTION = 0.09;
const TILE_MIN_PIXELS = 40;
const TILE_MIN_FILL = 0.8;
const MIN_TILES_IN_CLUSTER = 5;
const MAX_TILES_IN_CLUSTER = 16;
/** Holes up to this fraction of the image are filled in tile masks (digits), larger ones are gaps. */
const TILE_MAX_HOLE_FRACTION = 0.01;

interface Labels {
  labels: Int32Array;
  areas: number[];
  /** True for components that touch the image border. */
  touchesBorder: boolean[];
  /** Bounding box per label, inclusive pixel indices. */
  boxes: { x0: number; y0: number; x1: number; y1: number }[];
}

/** 4-connected component labelling of pixels equal to `value`; label 0 is unused. */
function labelComponents(mask: Uint8Array, width: number, height: number, value: number): Labels {
  const labels = new Int32Array(width * height);
  const areas: number[] = [0];
  const touchesBorder: boolean[] = [false];
  const boxes = [{ x0: 0, y0: 0, x1: 0, y1: 0 }];
  const stack = new Int32Array(width * height);
  for (let start = 0; start < mask.length; start++) {
    if (mask[start] !== value || labels[start] !== 0) continue;
    const label = areas.length;
    const box = { x0: width, y0: height, x1: -1, y1: -1 };
    let area = 0;
    let top = 0;
    stack[top++] = start;
    labels[start] = label;
    while (top > 0) {
      const index = stack[--top] as number;
      area++;
      const x = index % width;
      const y = (index - x) / width;
      if (x < box.x0) box.x0 = x;
      if (x > box.x1) box.x1 = x;
      if (y < box.y0) box.y0 = y;
      if (y > box.y1) box.y1 = y;
      if (x > 0 && mask[index - 1] === value && labels[index - 1] === 0) {
        labels[index - 1] = label;
        stack[top++] = index - 1;
      }
      if (x < width - 1 && mask[index + 1] === value && labels[index + 1] === 0) {
        labels[index + 1] = label;
        stack[top++] = index + 1;
      }
      if (y > 0 && mask[index - width] === value && labels[index - width] === 0) {
        labels[index - width] = label;
        stack[top++] = index - width;
      }
      if (y < height - 1 && mask[index + width] === value && labels[index + width] === 0) {
        labels[index + width] = label;
        stack[top++] = index + width;
      }
    }
    areas.push(area);
    boxes.push(box);
    touchesBorder.push(
      box.x0 === 0 || box.y0 === 0 || box.x1 === width - 1 || box.y1 === height - 1,
    );
  }
  return { labels, areas, touchesBorder, boxes };
}

/** Removes the vertex whose triangle with its neighbours is smallest until `count` remain. */
function simplifyHull(hull: Point[], count: number): Point[] {
  const points = [...hull];
  while (points.length > count) {
    let best = 0;
    let bestArea = Infinity;
    for (let i = 0; i < points.length; i++) {
      const area = computePolygonArea([
        points[(i + points.length - 1) % points.length] as Point,
        points[i] as Point,
        points[(i + 1) % points.length] as Point,
      ]);
      if (area < bestArea) {
        bestArea = area;
        best = i;
      }
    }
    points.splice(best, 1);
  }
  return points;
}

/** Convex hull of one labelled component reduced to a quad; undefined when it is not quad-like. */
function buildQuadFromComponent(
  labels: Labels,
  label: number,
  width: number,
): { quad: Quad; fill: number } | undefined {
  const box = labels.boxes[label] as Labels["boxes"][number];
  // Extreme pixels of each row and column are enough to build the hull.
  const points: Point[] = [];
  const colMin = new Int32Array(box.x1 - box.x0 + 1).fill(box.y1 + 1);
  const colMax = new Int32Array(box.x1 - box.x0 + 1).fill(-1);
  for (let y = box.y0; y <= box.y1; y++) {
    let left = -1;
    let right = -1;
    for (let x = box.x0; x <= box.x1; x++) {
      if (labels.labels[y * width + x] !== label) continue;
      if (left < 0) left = x;
      right = x;
      const col = x - box.x0;
      if (y < (colMin[col] as number)) colMin[col] = y;
      colMax[col] = y;
    }
    if (left >= 0) points.push({ x: left, y: y + 0.5 }, { x: right + 1, y: y + 0.5 });
  }
  for (let col = 0; col < colMax.length; col++) {
    if ((colMax[col] as number) >= 0) {
      points.push(
        { x: box.x0 + col + 0.5, y: colMin[col] as number },
        { x: box.x0 + col + 0.5, y: (colMax[col] as number) + 1 },
      );
    }
  }
  const hull = computeConvexHull(points);
  if (hull.length < 4) return undefined;
  const corners = reduceToQuad(simplifyHull(hull, HULL_VERTICES));
  if (!corners) return undefined;
  const quad = orderCorners(corners);
  const quadArea = computePolygonArea(quad);
  if (quadArea <= 0) return undefined;
  return { quad, fill: (labels.areas[label] as number) / quadArea };
}

/** Loose shape test: roughly square under perspective (opposite sides alike, angles not extreme). */
export function isPlausibleQuad(quad: Quad): boolean {
  const measureSide = (i: number): number =>
    measureDistance(quad[i] as Point, quad[(i + 1) % 4] as Point);
  const computeRatio = (a: number, b: number): number =>
    Math.max(a, b) / Math.max(1e-6, Math.min(a, b));
  if (
    computeRatio(measureSide(0), measureSide(2)) > 1.8 ||
    computeRatio(measureSide(1), measureSide(3)) > 1.8
  )
    return false;
  const avgH = (measureSide(0) + measureSide(2)) / 2;
  const avgV = (measureSide(1) + measureSide(3)) / 2;
  if (computeRatio(avgH, avgV) > 1.6) return false;
  for (let i = 0; i < 4; i++) {
    const a = quad[(i + 3) % 4] as Point;
    const b = quad[i] as Point;
    const c = quad[(i + 1) % 4] as Point;
    const cos =
      ((a.x - b.x) * (c.x - b.x) + (a.y - b.y) * (c.y - b.y)) /
      (measureDistance(a, b) * measureDistance(c, b));
    if (cos > 0.6 || cos < -0.6) return false; // angle outside roughly 53..127 degrees
  }
  return true;
}

function computePercentileLevels(view: GreyImage): number[] {
  const sorted = Float32Array.from(view.data).sort();
  const levels: number[] = [];
  for (let k = 1; k <= LEVELS; k++) {
    const level = sorted[Math.floor((sorted.length - 1) * (k / (LEVELS + 1)))] as number;
    if (levels.length === 0 || level - (levels[levels.length - 1] as number) >= 3)
      levels.push(level);
  }
  return levels;
}

function findCentre(quad: Quad): Point {
  return { x: quad.reduce((s, p) => s + p.x, 0) / 4, y: quad.reduce((s, p) => s + p.y, 0) / 4 };
}

function doQuadsOverlap(a: Quad, b: Quad): boolean {
  const areaA = computePolygonArea(a);
  const areaB = computePolygonArea(b);
  return (
    measureDistance(findCentre(a), findCentre(b)) < 0.03 * Math.sqrt(areaA) &&
    Math.abs(areaA - areaB) < 0.08 * areaA
  );
}

interface TileBlob {
  centre: Point;
  size: number;
  quad: Quad;
}

/** Groups tile-like blobs of similar size that touch or nearly touch; returns each group as a candidate. */
function clusterTiles(blobs: readonly TileBlob[]): BoardCandidate[] {
  const parent = blobs.map((_, i) => i);
  const findRoot = (i: number): number => {
    let root = i;
    while (parent[root] !== root) root = parent[root] as number;
    parent[i] = root;
    return root;
  };
  for (let i = 0; i < blobs.length; i++) {
    for (let j = i + 1; j < blobs.length; j++) {
      const a = blobs[i] as TileBlob;
      const b = blobs[j] as TileBlob;
      const similar = Math.max(a.size, b.size) / Math.min(a.size, b.size) < 1.35;
      if (similar && measureDistance(a.centre, b.centre) < 1.7 * Math.max(a.size, b.size)) {
        parent[findRoot(i)] = findRoot(j);
      }
    }
  }
  const groups = new Map<number, TileBlob[]>();
  blobs.forEach((blob, i) => {
    const root = findRoot(i);
    groups.set(root, [...(groups.get(root) ?? []), blob]);
  });
  const candidates: BoardCandidate[] = [];
  for (const group of groups.values()) {
    if (group.length < MIN_TILES_IN_CLUSTER || group.length > MAX_TILES_IN_CLUSTER) continue;
    const corners = group.flatMap((blob) => [...blob.quad]);
    const quad = reduceToQuad(simplifyHull(computeConvexHull(corners), HULL_VERTICES));
    if (!quad) continue;
    const ordered = orderCorners(quad);
    if (!isPlausibleQuad(ordered)) continue;
    const tileArea = group.reduce((sum, blob) => sum + blob.size ** 2, 0);
    candidates.push({ quad: ordered, kind: "tiles", fill: tileArea / computePolygonArea(ordered) });
  }
  return candidates;
}

function addCandidate(found: BoardCandidate[], candidate: BoardCandidate): void {
  const duplicate = found.findIndex(
    (other) => other.kind === candidate.kind && doQuadsOverlap(other.quad, candidate.quad),
  );
  if (duplicate < 0) found.push(candidate);
  else if (candidate.fill > (found[duplicate] as BoardCandidate).fill) found[duplicate] = candidate;
}

/**
 * Candidate board quads in `views` (e.g. brightness and colourfulness of the same photo), silhouettes
 * first by fill, then tile clusters. Coordinates are in the views' pixel space.
 */
export function findBoardCandidates(views: readonly GreyImage[]): BoardCandidate[] {
  const found: BoardCandidate[] = [];
  for (const raw of views) {
    const view = blurGrey(raw, 1);
    const { width, height } = view;
    const imageArea = width * height;
    for (const level of computePercentileLevels(view)) {
      for (const above of [true, false]) {
        const mask = new Uint8Array(width * height);
        for (let i = 0; i < mask.length; i++)
          mask[i] = (view.data[i] as number) > level === above ? 1 : 0;
        // Holes (tiles inside the frame, digits inside tiles) are filled, but tile masks only fill small
        // ones: a large hole is the frame and gaps, and filling it would merge all tiles into one blob.
        const background = labelComponents(mask, width, height, 0);
        const silhouetteMask = Uint8Array.from(mask);
        const tileMask = Uint8Array.from(mask);
        for (let i = 0; i < mask.length; i++) {
          const label = background.labels[i] ?? 0;
          if (label === 0 || background.touchesBorder[label]) continue;
          silhouetteMask[i] = 1;
          if ((background.areas[label] as number) <= TILE_MAX_HOLE_FRACTION * imageArea)
            tileMask[i] = 1;
        }

        const silhouettes = labelComponents(silhouetteMask, width, height, 1);
        for (let label = 1; label < silhouettes.areas.length; label++) {
          const area = silhouettes.areas[label] as number;
          if (
            silhouettes.touchesBorder[label] ||
            area < MIN_AREA_FRACTION * imageArea ||
            area > MAX_AREA_FRACTION * imageArea
          ) {
            continue;
          }
          const hull = buildQuadFromComponent(silhouettes, label, width);
          if (!hull || hull.fill < MIN_FILL || !isPlausibleQuad(hull.quad)) continue;
          addCandidate(found, { quad: hull.quad, kind: "silhouette", fill: hull.fill });
        }

        const tiles = labelComponents(tileMask, width, height, 1);
        const blobs: TileBlob[] = [];
        for (let label = 1; label < tiles.areas.length; label++) {
          const area = tiles.areas[label] as number;
          const box = tiles.boxes[label] as Labels["boxes"][number];
          const boxW = box.x1 - box.x0 + 1;
          const boxH = box.y1 - box.y0 + 1;
          if (
            tiles.touchesBorder[label] ||
            area < Math.max(TILE_MIN_PIXELS, TILE_MIN_AREA_FRACTION * imageArea) ||
            area > TILE_MAX_AREA_FRACTION * imageArea ||
            area < 0.4 * boxW * boxH || // a rotated square still fills half its bounding box
            Math.max(boxW, boxH) > 1.6 * Math.min(boxW, boxH)
          ) {
            continue;
          }
          const hull = buildQuadFromComponent(tiles, label, width);
          if (!hull || hull.fill < TILE_MIN_FILL || !isPlausibleQuad(hull.quad)) continue;
          blobs.push({ centre: findCentre(hull.quad), size: Math.sqrt(area), quad: hull.quad });
        }
        for (const candidate of clusterTiles(blobs)) addCandidate(found, candidate);
      }
    }
  }
  const order = { silhouette: 0, tiles: 1 };
  return found.sort((a, b) => order[a.kind] - order[b.kind] || b.fill - a.fill);
}
