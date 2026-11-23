import { type Board, type Direction, type GridSize, type Move } from "../puzzle/board.ts";
import { type TileCrop } from "../picture/types.ts";
import { type Quad } from "../vision/geometry.ts";
import { describeMove } from "./review.ts";

// DOM builders. Every interactive element gets a stable id so main.ts can restore focus after it
// rebuilds a section.

type Attributes = Record<string, string | boolean | (() => void) | undefined>;
type Child = Node | string | false | undefined;

/** Builds an element; a function attribute is a click handler, `true` is a bare attribute. */
export function buildElement<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attributes: Attributes,
  ...children: Child[]
): HTMLElementTagNameMap[K] {
  const element = document.createElement(tag);
  for (const [name, value] of Object.entries(attributes)) {
    if (typeof value === "function") element.addEventListener("click", value);
    else if (value === true) element.setAttribute(name, "");
    else if (value !== undefined && value !== false) element.setAttribute(name, value);
  }
  for (const child of children) {
    if (child !== false && child !== undefined) element.append(child);
  }
  return element;
}

const ARROWS: Record<Direction, string> = { up: "↑", down: "↓", left: "←", right: "→" };

export interface GridOptions {
  size: GridSize;
  values: readonly number[];
  /** Cells to flag (misreads, duplicates); drawn with a dashed outline and announced as needing a check. */
  flagged?: ReadonlySet<number>;
  /** The tile that moves next, drawn filled with its direction. */
  moving?: { index: number; direction: Direction };
  /** Makes cells buttons; called with the row-major index. */
  onCell?: (index: number) => void;
  selected?: number | undefined;
  idPrefix: string;
  /** Picture puzzles: the crop to draw under each tile number; the number stays on top of it. */
  crops?: ReadonlyMap<number, TileCrop>;
  /** Spoken name of the grid, and what a tap does; both default to the numbered review's wording. */
  title?: string;
  action?: string;
}

/** A canvas showing `crop`; the page scales it with CSS. */
function createTileCanvas(crop: TileCrop): HTMLCanvasElement {
  const canvas = buildElement("canvas", { class: "tile-image", "aria-hidden": "true" });
  canvas.width = crop.side;
  canvas.height = crop.side;
  const rgba = new Uint8ClampedArray(crop.side * crop.side * 4);
  for (let i = 0; i < crop.side * crop.side; i++) {
    rgba.set(crop.rgb.subarray(i * 3, i * 3 + 3), i * 4);
    rgba[i * 4 + 3] = 255;
  }
  canvas.getContext("2d")?.putImageData(new ImageData(rgba, crop.side, crop.side), 0, 0);
  return canvas;
}

export function createGrid(options: GridOptions): HTMLElement {
  const { size, values, flagged, moving, onCell, selected, idPrefix, crops } = options;
  const grid = buildElement("div", {
    class: "grid",
    style: `--size: ${String(size)}`,
    role: "group",
    "aria-label": options.title ?? `${String(size)} by ${String(size)} board`,
  });
  values.forEach((value, index) => {
    const row = Math.floor(index / size) + 1;
    const col = (index % size) + 1;
    const crop = crops?.get(value);
    const isFlagged = flagged?.has(index) === true;
    const isMoving = moving?.index === index;
    const name = value === 0 ? "empty" : `tile ${String(value)}`;
    const classes = [
      "cell",
      value === 0 ? "blank" : "",
      crop ? "picture" : "",
      isFlagged ? "flagged" : "",
      isMoving ? "moving" : "",
      selected === index ? "selected" : "",
    ]
      .filter(Boolean)
      .join(" ");
    const label = `Row ${String(row)}, column ${String(col)}: ${name}${isFlagged ? ", needs checking" : ""}`;
    const text = value === 0 ? "" : String(value);
    const content: Child[] = crop
      ? [createTileCanvas(crop), buildElement("span", { class: "tile-number" }, text)]
      : [text];
    const arrow = isMoving
      ? buildElement("span", { class: "arrow", "aria-hidden": "true" }, ARROWS[moving.direction])
      : false;
    grid.append(
      onCell
        ? buildElement(
            "button",
            {
              type: "button",
              id: `${idPrefix}-${String(index)}`,
              class: classes,
              "aria-label": `${label}. ${options.action ?? "Tap to change."}`,
              onClick: () => {
                onCell(index);
              },
            },
            ...content,
          )
        : buildElement(
            "div",
            { class: classes, role: "img", "aria-label": label },
            ...content,
            arrow,
          ),
    );
  });
  return grid;
}

/** The photo with the detected board outlined; the SVG shares the photo's pixel coordinates. */
export function createPhotoFigure(
  url: string,
  width: number,
  height: number,
  quad: Quad | undefined,
): HTMLElement {
  const wrap = buildElement("div", {
    class: "photo-wrap",
    style: `aspect-ratio: ${String(width)} / ${String(height)}`,
  });
  wrap.append(buildElement("img", { src: url, alt: "Your photo of the puzzle" }));
  if (quad) {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", `0 0 ${String(width)} ${String(height)}`);
    svg.setAttribute("aria-hidden", "true");
    const outline = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    outline.setAttribute(
      "points",
      quad.map((p) => `${String(Math.round(p.x))},${String(Math.round(p.y))}`).join(" "),
    );
    svg.append(outline);
    wrap.append(svg);
  }
  return buildElement(
    "figure",
    { class: "photo" },
    wrap,
    buildElement(
      "figcaption",
      {},
      quad ? "Your photo, with the board outline we found." : "Your photo.",
    ),
  );
}

/** Buttons for the values a cell can take: blank, then 1..N²-1. */
export function createNumberPad(
  size: GridSize,
  index: number,
  current: number,
  onPick: (value: number) => void,
  onCancel: () => void,
): HTMLElement {
  const row = Math.floor(index / size) + 1;
  const col = (index % size) + 1;
  const buttons = Array.from({ length: size * size }, (_, i) => {
    // Blank first, so tile numbers follow in reading order.
    const value = i;
    return buildElement(
      "button",
      {
        type: "button",
        id: `pad-${String(value)}`,
        class: "pad-button",
        "aria-label": value === 0 ? "Set to empty" : `Set to ${String(value)}`,
        "aria-pressed": value === current ? "true" : "false",
        onClick: () => {
          onPick(value);
        },
      },
      value === 0 ? "Empty" : String(value),
    );
  });
  return buildElement(
    "div",
    { class: "pad", role: "group", "aria-label": `Set row ${String(row)}, column ${String(col)}` },
    buildElement("p", { class: "pad-title" }, `Row ${String(row)}, column ${String(col)} is…`),
    buildElement("div", { class: "pad-buttons" }, ...buttons),
    buildElement(
      "button",
      { type: "button", id: "pad-cancel", class: "secondary", onClick: onCancel },
      "Cancel",
    ),
  );
}

export interface ConfirmListOptions {
  size: GridSize;
  /** Flagged cells, row-major index, with the reader's best guess for each. */
  cells: readonly { index: number; value: number }[];
  onConfirm: (index: number) => void;
  onChange: (index: number) => void;
}

/** One line per low-confidence cell: the best guess, one tap to confirm it, or change it. */
export function createConfirmList(options: ConfirmListOptions): HTMLElement {
  const { size, cells, onConfirm, onChange } = options;
  return buildElement(
    "ul",
    { class: "confirm-list", "aria-label": "Cells to check" },
    ...cells.map(({ index, value }) => {
      const row = Math.floor(index / size) + 1;
      const col = (index % size) + 1;
      const guess = value === 0 ? "empty" : String(value);
      return buildElement(
        "li",
        { class: "confirm-item", id: `confirm-item-${String(index)}` },
        buildElement(
          "span",
          { class: "confirm-text" },
          `Check this cell: row ${String(row)}, column ${String(col)}. Best guess: `,
          buildElement("strong", {}, guess),
        ),
        buildElement(
          "button",
          {
            type: "button",
            id: `confirm-${String(index)}`,
            "aria-label": `Confirm row ${String(row)}, column ${String(col)} is ${guess}`,
            onClick: () => {
              onConfirm(index);
            },
          },
          `Yes, it is ${guess}`,
        ),
        buildElement(
          "button",
          {
            type: "button",
            id: `change-${String(index)}`,
            class: "secondary",
            "aria-label": `Change row ${String(row)}, column ${String(col)}`,
            onClick: () => {
              onChange(index);
            },
          },
          "Change",
        ),
      );
    }),
  );
}

export interface SolutionViewOptions {
  boards: readonly Board[];
  moves: readonly Move[];
  step: number;
  notGuaranteedShortest: boolean;
  millis: number;
  onStep: (step: number) => void;
  crops?: ReadonlyMap<number, TileCrop>;
}

export function createSolutionView(options: SolutionViewOptions): HTMLElement {
  const { boards, moves, step, notGuaranteedShortest, millis, onStep, crops } = options;
  const board = boards[step];
  if (!board) throw new Error(`no board for step ${String(step)}`);
  const total = moves.length;
  const next = moves[step];
  const moving = next
    ? { index: board.cells.indexOf(next.tile), direction: next.direction }
    : undefined;
  const label =
    next === undefined ? (total === 0 ? "Already solved." : "Solved.") : describeMove(next);
  const heading = `${String(total)} ${total === 1 ? "move" : "moves"}`;
  return buildElement(
    "div",
    { class: "solution" },
    buildElement("h2", { id: "solution-heading" }, `Solution: ${heading}`),
    buildElement(
      "p",
      { class: notGuaranteedShortest ? "note warn" : "note" },
      notGuaranteedShortest
        ? "Not guaranteed shortest: the search for the best solution ran out of time, so this is a fast one."
        : `Shortest possible solution, found in ${formatSeconds(millis)}.`,
    ),
    buildElement(
      "p",
      { class: "move-label", id: "move-label", role: "status" },
      label,
      ` `,
      buildElement("span", { class: "step" }, `(step ${String(step)} of ${String(total)})`),
    ),
    createGrid({
      size: board.size,
      values: board.cells,
      ...(moving ? { moving } : {}),
      ...(crops ? { crops } : {}),
      idPrefix: "solution-cell",
    }),
    buildElement(
      "div",
      { class: "actions" },
      buildElement(
        "button",
        {
          type: "button",
          id: "previous",
          class: "secondary",
          disabled: step === 0,
          onClick: () => {
            onStep(step - 1);
          },
        },
        "Previous",
      ),
      buildElement(
        "button",
        {
          type: "button",
          id: "next",
          disabled: step >= total,
          onClick: () => {
            onStep(step + 1);
          },
        },
        "Next",
      ),
    ),
    buildElement("p", { class: "hint" }, "Arrow keys step through the moves."),
    buildElement(
      "ol",
      { class: "move-list", "aria-label": "All moves" },
      ...moves.map((move, index) =>
        buildElement(
          "li",
          {},
          buildElement(
            "button",
            {
              type: "button",
              id: `move-${String(index)}`,
              class: index === step ? "chip current" : index < step ? "chip done" : "chip",
              "aria-label": `Move ${String(index + 1)}: ${describeMove(move)}`,
              "aria-current": index === step ? "step" : undefined,
              onClick: () => {
                onStep(index);
              },
            },
            `${String(move.tile)}${ARROWS[move.direction]}`,
          ),
        ),
      ),
    ),
  );
}

function formatSeconds(millis: number): string {
  return millis < 100 ? "under 0.1 s" : `${(millis / 1000).toFixed(1)} s`;
}
