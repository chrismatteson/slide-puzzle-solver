// The "Adjust corners" editor: four draggable dots over the photo that the user puts on the outer
// corners of the tiles, with the grid they would make drawn inside. The state lives in a plain object
// that main.ts keeps, so a re-render of the page does not lose the corners; dragging updates the
// DOM in place (a rebuild mid-drag would drop the pointer capture).

import { GRID_SIZES, type GridSize } from "../puzzle/board.ts";
import { type Point, type Quad } from "../vision/geometry.ts";
import {
  type Rect,
  buildGridLinePath,
  clampToRect,
  computeZoomRect,
  createWholePhotoRect,
  isReadableQuad,
  measureKeyMove,
  opensZoomed,
  photoToView,
  viewToPhoto,
} from "./corner-view.ts";
import { buildElement } from "./view.ts";

/** What the user is adjusting; mutated by the editor as the dots move. */
export interface CornerEditState {
  /** Corners in photo pixels, in any order; `prepareUserQuad` orders them. */
  quad: [Point, Point, Point, Point];
  size: GridSize | "auto";
  /** The part of the photo shown; the whole photo or a zoom on the board. Kept here so a re-render keeps it. */
  view: Rect;
  /** Size of the reading the editor was opened from: faint guide lines while the size is Auto. */
  guide?: GridSize | undefined;
}

export interface CornerEditorOptions {
  photo: { url: string; width: number; height: number };
  state: CornerEditState;
  /** `size` is undefined for Auto. */
  onRead: (quad: Quad, size: GridSize | undefined) => void;
  onCancel: () => void;
}

/** The default rectangle is the photo inset by this much on each side. */
const DEFAULT_INSET = 0.15;

/** Magnifier shown while a dot is dragged: a circle of this many CSS px showing the photo this many times larger. */
const LOUPE_SIZE = 110;
const LOUPE_ZOOM = 2.5;
/** How far above the dot's centre the loupe's bottom edge sits, so a finger on the dot does not cover it. */
const LOUPE_GAP = 40;
/** How far above the editor's top edge the loupe may reach before it is put below the dot instead. */
const LOUPE_OVERHANG = 22;

const CORNER_NAMES = [
  "Top-left corner",
  "Top-right corner",
  "Bottom-right corner",
  "Bottom-left corner",
];
const UNREADABLE_QUAD_MESSAGE =
  "The corners cross over or are too close together. Move them onto the four corners of the tiles.";
const HINT = "Drag the four dots onto the outer corners of the tiles (not the frame).";
const SVG_NS = "http://www.w3.org/2000/svg";

/** The starting quad when nothing was detected. */
export function createInsetRectangle(width: number, height: number): [Point, Point, Point, Point] {
  const x0 = width * DEFAULT_INSET;
  const y0 = height * DEFAULT_INSET;
  return [
    { x: x0, y: y0 },
    { x: width - x0, y: y0 },
    { x: width - x0, y: height - y0 },
    { x: x0, y: height - y0 },
  ];
}

/**
 * The editor's starting state for a review: at the detected tile area, clamped to the photo (the detector
 * may put a corner past the edge), or a rectangle inside the photo if there is none. The size is Auto,
 * not the reading's: when the reading is a wrong guess (a frame taken for tiles) keeping its size would
 * carry the mistake into the re-read. The reading's size is only drawn as a guide. A small board opens
 * zoomed.
 */
export function createOpeningState(
  detected: Quad | undefined,
  guide: GridSize | undefined,
  width: number,
  height: number,
): CornerEditState {
  // The detector may put a corner past the photo's edge; the dots never leave the photo.
  const photo = createWholePhotoRect(width, height);
  const quad: [Point, Point, Point, Point] = detected
    ? [
        clampToRect(detected[0], photo),
        clampToRect(detected[1], photo),
        clampToRect(detected[2], photo),
        clampToRect(detected[3], photo),
      ]
    : createInsetRectangle(width, height);
  const view =
    detected && opensZoomed(quad, width, height)
      ? computeZoomRect(quad, width, height)
      : createWholePhotoRect(width, height);
  return { quad, size: "auto", view, guide };
}

/** What Auto means, said under the hint: the reader tries both sizes inside the dots; the faint lines are only a guide. */
function describeAutoSize(guide: GridSize | undefined): string {
  return guide
    ? `Grid: auto (faint lines show the ${String(guide)}×${String(guide)} of the current reading).`
    : "Grid: auto.";
}

function createSizeChoice(state: CornerEditState, onChange: () => void): HTMLElement {
  const choices: (GridSize | "auto")[] = ["auto", ...GRID_SIZES];
  return buildElement(
    "fieldset",
    { class: "corner-size" },
    buildElement("legend", {}, "Grid size"),
    ...choices.map((choice) => {
      const input = buildElement("input", {
        type: "radio",
        name: "corner-size",
        id: `corner-size-${String(choice)}`,
        value: String(choice),
      });
      input.checked = state.size === choice;
      input.addEventListener("change", () => {
        state.size = choice;
        onChange();
      });
      return buildElement(
        "label",
        { class: "corner-size-choice" },
        input,
        choice === "auto" ? "Auto" : `${String(choice)}×${String(choice)}`,
      );
    }),
  );
}

export function createCornerEditor(options: CornerEditorOptions): HTMLElement {
  const { photo, state, onRead, onCancel } = options;
  const { width, height } = photo;

  const wrap = buildElement("div", { class: "photo-wrap corner-wrap" });
  const image = buildElement("img", { src: photo.url, alt: "Your photo of the puzzle" });
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("aria-hidden", "true");
  const outline = document.createElementNS(SVG_NS, "polygon");
  const lines = document.createElementNS(SVG_NS, "path");
  svg.append(outline, lines);
  // The clip cuts the zoomed photo off at the wrap's edge; the dots and the loupe must not be cut off.
  wrap.append(buildElement("div", { class: "corner-clip" }, image, svg));

  const hint = buildElement("p", { class: "hint", id: "corner-hint" });
  const zoomButton = buildElement("button", {
    type: "button",
    id: "editor-zoom",
    class: "secondary",
  });
  zoomButton.addEventListener("click", () => {
    const zoomed = state.view.width < width || state.view.height < height;
    state.view = zoomed
      ? createWholePhotoRect(width, height)
      : computeZoomRect(state.quad, width, height);
    applyView();
  });

  const readButton = buildElement(
    "button",
    { type: "button", id: "read-corners" },
    "Read with these corners",
  );
  readButton.addEventListener("click", () => {
    onRead(state.quad, state.size === "auto" ? undefined : state.size);
  });

  // Always in the DOM so that aria-describedby has something to point at; shown only while Read is disabled.
  const problem = buildElement(
    "p",
    { class: "notice", id: "corner-problem" },
    UNREADABLE_QUAD_MESSAGE,
  );

  const handles = CORNER_NAMES.map((name, i) =>
    buildElement("button", {
      type: "button",
      id: `corner-${String(i)}`,
      class: "corner-handle",
      "aria-label": `${name}: drag, or use the arrow keys`,
    }),
  );
  /** The dot touched last is drawn on top, so of two stacked dots the one just moved can be grabbed again. */
  let raised = -1;
  function raise(index: number): void {
    raised = index;
    handles.forEach((handle, i) => {
      handle.style.zIndex = i === raised ? "2" : "1";
    });
  }

  const loupe = buildElement("canvas", {
    class: "corner-loupe",
    id: "corner-loupe",
    width: String(LOUPE_SIZE),
    height: String(LOUPE_SIZE),
    "aria-hidden": "true",
  });
  loupe.hidden = true;

  /** Shows the photo around dot `index` magnified, above the dot (below it when there is no room above). */
  function showLoupe(index: number): void {
    const p = state.quad[index] as Point;
    const box = wrap.getBoundingClientRect();
    const scale = box.width / state.view.width;
    const reach = LOUPE_SIZE / 2 / (LOUPE_ZOOM * scale);
    const natural = image.naturalWidth > 0 ? image.naturalWidth / width : 1;
    const context = loupe.getContext("2d");
    if (context) {
      context.fillStyle = "#000";
      context.fillRect(0, 0, LOUPE_SIZE, LOUPE_SIZE);
      context.drawImage(
        image,
        (p.x - reach) * natural,
        (p.y - reach) * natural,
        2 * reach * natural,
        2 * reach * natural,
        0,
        0,
        LOUPE_SIZE,
        LOUPE_SIZE,
      );
      context.strokeStyle = "#fff";
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(LOUPE_SIZE / 2, 0);
      context.lineTo(LOUPE_SIZE / 2, LOUPE_SIZE);
      context.moveTo(0, LOUPE_SIZE / 2);
      context.lineTo(LOUPE_SIZE, LOUPE_SIZE / 2);
      context.stroke();
    }
    // Kept within the photo's width so it never makes the page scroll sideways.
    const shown = photoToView(state.view, p);
    const x = Math.min(
      Math.max(shown.x * box.width, LOUPE_SIZE / 2),
      Math.max(box.width - LOUPE_SIZE / 2, LOUPE_SIZE / 2),
    );
    const y = shown.y * box.height;
    const above = y - LOUPE_GAP - LOUPE_SIZE >= -LOUPE_OVERHANG;
    loupe.style.left = `${String(x)}px`;
    loupe.style.top = `${String(above ? y - LOUPE_GAP - LOUPE_SIZE : y + LOUPE_GAP)}px`;
    loupe.hidden = false;
  }

  function redraw(): void {
    outline.setAttribute(
      "points",
      state.quad.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" "),
    );
    const readable = isReadableQuad(state.quad, width, height);
    const guideSize = state.size === "auto" ? state.guide : state.size;
    lines.setAttribute("d", readable && guideSize ? buildGridLinePath(state.quad, guideSize) : "");
    lines.classList.toggle("guide", state.size === "auto");
    hint.textContent = state.size === "auto" ? `${HINT} ${describeAutoSize(state.guide)}` : HINT;
    readButton.disabled = !readable;
    problem.hidden = readable;
    if (readable) readButton.removeAttribute("aria-describedby");
    else readButton.setAttribute("aria-describedby", "corner-problem");
    handles.forEach((handle, i) => {
      const shown = photoToView(state.view, state.quad[i] as Point);
      handle.style.left = `${String(shown.x * 100)}%`;
      handle.style.top = `${String(shown.y * 100)}%`;
    });
  }

  /** Shows `state.view`: the wrap takes its shape, the photo is placed so that the view fills the wrap. */
  function applyView(): void {
    const { x, y, width: w, height: h } = state.view;
    wrap.style.aspectRatio = `${String(w)} / ${String(h)}`;
    image.style.cssText = `left: ${String((-x / w) * 100)}%; top: ${String((-y / h) * 100)}%; width: ${String((width / w) * 100)}%; height: ${String((height / h) * 100)}%`;
    svg.setAttribute("viewBox", `${String(x)} ${String(y)} ${String(w)} ${String(h)}`);
    const zoomed = w < width || h < height;
    zoomButton.textContent = zoomed ? "Show whole photo" : "Zoom to board";
    redraw();
  }

  // A dot can only be dragged within the view, so it never leaves the screen; show the whole photo to move it further.
  function moveTo(index: number, p: Point): void {
    state.quad[index] = clampToRect(p, state.view);
    redraw();
  }

  const toPhoto = (event: PointerEvent): Point => {
    const box = wrap.getBoundingClientRect();
    return viewToPhoto(
      state.view,
      (event.clientX - box.left) / box.width,
      (event.clientY - box.top) / box.height,
    );
  };
  /** The dot being dragged, and the point under the finger minus its centre, so grabbing by the edge does not make it jump. */
  let drag: { index: number; grab: Point } | undefined;
  /** The dot nearest `at`; of equally near ones (stacked) the raised one, which then moves away from the rest. */
  const findNearestHandle = (at: Point): number => {
    let best = raised >= 0 ? raised : 0;
    let bestDistance = Infinity;
    state.quad.forEach((p, i) => {
      const distance = Math.hypot(p.x - at.x, p.y - at.y);
      if (distance < bestDistance || (distance === bestDistance && i === raised)) {
        best = i;
        bestDistance = distance;
      }
    });
    return best;
  };
  handles.forEach((handle, own) => {
    handle.addEventListener("pointerdown", (event) => {
      // The dot under the pointer is only the topmost of the dots in reach; the nearest one is meant.
      const at = toPhoto(event);
      const index = findNearestHandle(at);
      const chosen = handles[index] as HTMLElement;
      chosen.setPointerCapture(event.pointerId);
      const centre = state.quad[index] as Point;
      drag = { index, grab: { x: at.x - centre.x, y: at.y - centre.y } };
      raise(index);
      chosen.focus();
      showLoupe(index);
      event.preventDefault();
    });
    handle.addEventListener("pointermove", (event) => {
      if (!drag || drag.index !== own) return;
      const at = toPhoto(event);
      moveTo(own, { x: at.x - drag.grab.x, y: at.y - drag.grab.y });
      showLoupe(own);
    });
    const release = (): void => {
      if (drag?.index !== own) return;
      drag = undefined;
      loupe.hidden = true;
    };
    handle.addEventListener("pointerup", release);
    handle.addEventListener("pointercancel", release);
    handle.addEventListener("keydown", (event) => {
      const move = measureKeyMove(event.key, event.shiftKey, state.view.width, state.view.height);
      if (!move) return;
      const p = state.quad[own] as Point;
      raise(own);
      moveTo(own, { x: p.x + move.x, y: p.y + move.y });
      event.preventDefault();
    });
  });
  wrap.append(...handles, loupe);
  applyView();

  return buildElement(
    "div",
    { class: "corner-editor" },
    hint,
    buildElement("div", { class: "corner-stage" }, wrap),
    zoomButton,
    createSizeChoice(state, redraw),
    problem,
    buildElement(
      "div",
      { class: "actions" },
      readButton,
      buildElement(
        "button",
        { type: "button", id: "cancel-corners", class: "secondary", onClick: onCancel },
        "Cancel",
      ),
    ),
  );
}
