import { describe, expect, it } from "vitest";
import { createInsetRectangle, createOpeningState } from "../src/app/corner-editor.ts";
import {
  buildGridLinePath,
  clampToRect,
  computeZoomRect,
  isReadableQuad,
  measureKeyMove,
  opensZoomed,
  photoToView,
  viewToPhoto,
  createWholePhotoRect,
} from "../src/app/corner-view.ts";
import { type Quad } from "../src/vision/geometry.ts";

const rectangle = (x0: number, y0: number, x1: number, y1: number): Quad => [
  { x: x0, y: y0 },
  { x: x1, y: y0 },
  { x: x1, y: y1 },
  { x: x0, y: y1 },
];

describe("zoomed view", () => {
  it("is the bounding box plus 35% on each side", () => {
    const view = computeZoomRect(rectangle(400, 500, 600, 700), 1000, 1000);
    expect(view.x).toBeCloseTo(330);
    expect(view.y).toBeCloseTo(430);
    expect(view.width).toBeCloseTo(340);
    expect(view.height).toBeCloseTo(340);
  });

  it("is cut off at the photo's edges", () => {
    expect(computeZoomRect(rectangle(20, 30, 220, 330), 1000, 400)).toEqual({
      x: 0,
      y: 0,
      width: 220 + 0.35 * 200,
      height: 400,
    });
  });

  it("falls back to the whole photo for a quad with no extent", () => {
    expect(computeZoomRect(rectangle(500, 500, 500, 500), 1000, 800)).toEqual(
      createWholePhotoRect(1000, 800),
    );
  });

  it("opens zoomed when the board is under 60% of the width or the height", () => {
    expect(opensZoomed(rectangle(0, 0, 590, 1000), 1000, 1000)).toBe(true);
    expect(opensZoomed(rectangle(0, 0, 1000, 590), 1000, 1000)).toBe(true);
    expect(opensZoomed(rectangle(0, 0, 600, 600), 1000, 1000)).toBe(false);
  });

  it("maps view positions to photo pixels and back", () => {
    const view = { x: 330, y: 430, width: 340, height: 170 };
    expect(viewToPhoto(view, 0, 0)).toEqual({ x: 330, y: 430 });
    expect(viewToPhoto(view, 1, 1)).toEqual({ x: 670, y: 600 });
    const p = viewToPhoto(view, 0.25, 0.75);
    expect(p).toEqual({ x: 415, y: 557.5 });
    expect(photoToView(view, p)).toEqual({ x: 0.25, y: 0.75 });
  });

  it("clamps dots to the shown view", () => {
    const view = { x: 100, y: 200, width: 300, height: 150 };
    expect(clampToRect({ x: 0, y: 900 }, view)).toEqual({ x: 100, y: 350 });
    expect(clampToRect({ x: 250, y: 250 }, view)).toEqual({ x: 250, y: 250 });
  });
});

describe("arrow key moves", () => {
  it("move by half a percent of the view, six times that with shift", () => {
    expect(measureKeyMove("ArrowRight", false, 1000, 500)).toEqual({ x: 5, y: 0 });
    expect(measureKeyMove("ArrowLeft", false, 1000, 500)).toEqual({ x: -5, y: 0 });
    expect(measureKeyMove("ArrowDown", false, 1000, 500)).toEqual({ x: 0, y: 2.5 });
    expect(measureKeyMove("ArrowUp", true, 1000, 500)).toEqual({ x: 0, y: -15 });
  });

  it("are undefined for any other key", () => {
    expect(measureKeyMove("a", false, 1000, 500)).toBeUndefined();
    expect(measureKeyMove("Enter", true, 1000, 500)).toBeUndefined();
  });
});

describe("readable quad", () => {
  it("is a convex quad of at least 2% of the photo", () => {
    expect(isReadableQuad(rectangle(100, 100, 500, 500), 1000, 1000)).toBe(true);
  });

  it("is not a quad folded over itself", () => {
    const [a, b, c, d] = rectangle(100, 100, 500, 500);
    expect(isReadableQuad([a, c, b, d], 1000, 1000)).toBe(false);
  });

  it("is not a quad under 2% of the photo", () => {
    expect(isReadableQuad(rectangle(0, 0, 100, 100), 1000, 1000)).toBe(false);
    expect(isReadableQuad(rectangle(0, 0, 150, 150), 1000, 1000)).toBe(true);
  });

  it("is not a quad with a corner outside the photo, which the readers refuse", () => {
    expect(isReadableQuad(rectangle(-10, 100, 500, 500), 1000, 1000)).toBe(false);
    expect(isReadableQuad(rectangle(100, 100, 500, 1001), 1000, 1000)).toBe(false);
  });

  it("is a quad with its corners on the photo's edge", () => {
    expect(isReadableQuad(rectangle(0, 0, 1000, 1000), 1000, 1000)).toBe(true);
  });
});

describe("grid line path", () => {
  it("draws the interior lines of a square quad: two vertical and two horizontal for 3 cells", () => {
    expect(buildGridLinePath(rectangle(0, 0, 90, 90), 3)).toBe(
      "M30.0 0.0L30.0 90.0M0.0 30.0L90.0 30.0M60.0 0.0L60.0 90.0M0.0 60.0L90.0 60.0",
    );
  });

  it("draws size - 1 vertical and horizontal lines", () => {
    expect(buildGridLinePath(rectangle(0, 0, 80, 80), 4).match(/M/g)).toHaveLength(6);
  });

  it("follows the quad's corners, not an axis-aligned box", () => {
    const skewed: Quad = [
      { x: 0, y: 0 },
      { x: 90, y: 30 },
      { x: 90, y: 120 },
      { x: 0, y: 90 },
    ];
    expect(buildGridLinePath(skewed, 3)).toBe(
      "M30.0 10.0L30.0 100.0M0.0 30.0L90.0 60.0M60.0 20.0L60.0 110.0M0.0 60.0L90.0 90.0",
    );
  });
});

describe("opening state of the corner editor", () => {
  it("puts a detected corner that is past the photo's edge on the edge, so Read can send it", () => {
    const state = createOpeningState(rectangle(-40, -40, 1100, 1300), 3, 1000, 1200);
    expect(state.quad).toEqual(rectangle(0, 0, 1000, 1200));
    expect(isReadableQuad(state.quad, 1000, 1200)).toBe(true);
  });

  it("keeps a detected quad that is inside the photo as it is", () => {
    expect(createOpeningState(rectangle(200, 150, 800, 900), 4, 1000, 1000).quad).toEqual(
      rectangle(200, 150, 800, 900),
    );
  });

  it("is Auto whatever size the reading had, and keeps that size as the guide", () => {
    const state = createOpeningState(rectangle(200, 200, 800, 800), 3, 1000, 1000);
    expect(state.size).toBe("auto");
    expect(state.guide).toBe(3);
  });

  it("opens zoomed on a small board and on the whole photo for a large one or none", () => {
    expect(createOpeningState(rectangle(400, 400, 600, 600), 4, 1000, 1000).view).toEqual(
      computeZoomRect(rectangle(400, 400, 600, 600), 1000, 1000),
    );
    expect(createOpeningState(rectangle(100, 100, 900, 900), 4, 1000, 1000).view).toEqual(
      createWholePhotoRect(1000, 1000),
    );
    const none = createOpeningState(undefined, undefined, 1000, 800);
    expect(none.view).toEqual(createWholePhotoRect(1000, 800));
    expect(none.guide).toBeUndefined();
    expect(none.quad).toEqual(createInsetRectangle(1000, 800));
  });
});
