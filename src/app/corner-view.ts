// Which part of the photo the corner editor shows. A small board in a big photo is shown cropped to the
// board, so a finger's 3 px are 3 px of the photo, not 10. Pure, so the mapping is unit-tested.

import { type GridSize } from "../puzzle/board.ts";
import {
  type Point,
  type Quad,
  applyHomography,
  computeHomography,
  computePolygonArea,
  isConvexQuad,
} from "../vision/geometry.ts";

/** A part of the photo, in photo pixels. */
export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** A board whose bounding box is under this share of the photo's width or height is shown zoomed. */
const ZOOM_BELOW = 0.6;
/** The zoomed view is the bounding box plus this share of it on each side. */
const ZOOM_MARGIN = 0.35;

/** Smallest tile area worth reading, as a share of the photo's area. */
const MIN_AREA_SHARE = 0.02;

/** Share of the shown view's width (height for vertical moves) one arrow key press moves a dot. */
const KEY_STEP = 0.005;
const KEY_STEP_SHIFT = 0.03;

/** Smallest bounding box worth zooming to, as a share of the photo's width or height. */
const MIN_BOX = 0.02;

function measureBoundingBox(quad: readonly Point[]): Rect {
  const xs = quad.map((p) => p.x);
  const ys = quad.map((p) => p.y);
  const x = Math.min(...xs);
  const y = Math.min(...ys);
  return { x, y, width: Math.max(...xs) - x, height: Math.max(...ys) - y };
}

export function createWholePhotoRect(width: number, height: number): Rect {
  return { x: 0, y: 0, width, height };
}

/** Whether the editor opens zoomed: the quad's bounding box is small in the photo. */
export function opensZoomed(quad: readonly Point[], width: number, height: number): boolean {
  const box = measureBoundingBox(quad);
  return box.width < ZOOM_BELOW * width || box.height < ZOOM_BELOW * height;
}

/**
 * The quad's bounding box plus the margin on each side, cut off at the photo's edges. A quad that is
 * (nearly) a line has no sensible view, so the whole photo is returned.
 */
export function computeZoomRect(quad: readonly Point[], width: number, height: number): Rect {
  const box = measureBoundingBox(quad);
  if (box.width < MIN_BOX * width || box.height < MIN_BOX * height)
    return createWholePhotoRect(width, height);
  const x0 = Math.max(box.x - ZOOM_MARGIN * box.width, 0);
  const y0 = Math.max(box.y - ZOOM_MARGIN * box.height, 0);
  const x1 = Math.min(box.x + (1 + ZOOM_MARGIN) * box.width, width);
  const y1 = Math.min(box.y + (1 + ZOOM_MARGIN) * box.height, height);
  return { x: x0, y: y0, width: x1 - x0, height: y1 - y0 };
}

/** The photo point at `fx`, `fy` (0..1 across and down) of the shown view. */
export function viewToPhoto(view: Rect, fx: number, fy: number): Point {
  return { x: view.x + fx * view.width, y: view.y + fy * view.height };
}

/** Where a photo point is in the shown view, as 0..1 across and down (outside that range if it is off the view). */
export function photoToView(view: Rect, p: Point): Point {
  return { x: (p.x - view.x) / view.width, y: (p.y - view.y) / view.height };
}

export function clampToRect(p: Point, rect: Rect): Point {
  return {
    x: Math.min(Math.max(p.x, rect.x), rect.x + rect.width),
    y: Math.min(Math.max(p.y, rect.y), rect.y + rect.height),
  };
}

/** The arrow key's move in photo pixels, or undefined for any other key. */
export function measureKeyMove(
  key: string,
  shift: boolean,
  width: number,
  height: number,
): Point | undefined {
  const step = shift ? KEY_STEP_SHIFT : KEY_STEP;
  switch (key) {
    case "ArrowLeft":
      return { x: -step * width, y: 0 };
    case "ArrowRight":
      return { x: step * width, y: 0 };
    case "ArrowUp":
      return { x: 0, y: -step * height };
    case "ArrowDown":
      return { x: 0, y: step * height };
    default:
      return undefined;
  }
}

/**
 * Inside the photo (edges included, as the readers require), convex, not folded over itself, and big
 * enough: the homography of anything else is garbage.
 */
export function isReadableQuad(quad: readonly Point[], width: number, height: number): boolean {
  return (
    quad.every((p) => p.x >= 0 && p.y >= 0 && p.x <= width && p.y <= height) &&
    isConvexQuad(quad) &&
    computePolygonArea(quad) >= MIN_AREA_SHARE * width * height
  );
}

/** Interior grid lines of `size` cells inside `quad`, as SVG path data; straight lines stay straight under perspective. */
export function buildGridLinePath(quad: Quad, size: GridSize): string {
  const unit: Quad = [
    { x: 0, y: 0 },
    { x: 1, y: 0 },
    { x: 1, y: 1 },
    { x: 0, y: 1 },
  ];
  const h = computeHomography(unit, quad);
  const formatPointAt = (u: number, v: number): string => {
    const p = applyHomography(h, { x: u, y: v });
    return `${p.x.toFixed(1)} ${p.y.toFixed(1)}`;
  };
  const parts: string[] = [];
  for (let k = 1; k < size; k++) {
    const t = k / size;
    parts.push(
      `M${formatPointAt(t, 0)}L${formatPointAt(t, 1)}`,
      `M${formatPointAt(0, t)}L${formatPointAt(1, t)}`,
    );
  }
  return parts.join("");
}
