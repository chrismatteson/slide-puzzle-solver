import "./styles.css";
import patternDbUrl from "../puzzle/pattern-db.bin.gz?url";
import { type Board, type GridSize, type Move, applyMove } from "../puzzle/board.ts";
import { type CellReading } from "../puzzle/consistency.ts";
import { DEFAULT_BUDGET_MS } from "../puzzle/solver.ts";
import { type Quad } from "../vision/geometry.ts";
import { describeCameraError, startCamera, stopCamera } from "./camera.ts";
import { type Photo, captureVideoFrame, loadPhotoFile } from "./photo.ts";
import { checkBoard, confirmCell, setCell } from "./review.ts";
import {
  createGrid,
  createConfirmList,
  createNumberPad,
  createPhotoFigure,
  createSolutionView,
  buildElement,
} from "./view.ts";
import { CancelledError, createSolverClient, createVisionClient } from "./workers.ts";

interface ReviewState {
  photo: Photo;
  quad?: Quad;
  /** Absent when no board was found in the photo. */
  size?: GridSize;
  cells?: CellReading[];
  /** Why there is no grid, when there is none. */
  problem?: string;
  /** Cell open in the number pad. */
  editing?: number | undefined;
}

interface SolutionState {
  boards: Board[];
  moves: Move[];
  notGuaranteedShortest: boolean;
  millis: number;
  step: number;
}

function getElement(id: string): HTMLElement {
  return getTypedElement(id, HTMLElement);
}

function getTypedElement<T extends HTMLElement>(id: string, type: abstract new () => T): T {
  const element = document.getElementById(id);
  if (!(element instanceof type)) throw new Error(`index.html has no #${id} of the expected kind`);
  return element;
}

const statusLine = getElement("status");
const notice = getElement("notice");
const cameraPanel = getElement("camera");
const preview = getTypedElement("preview", HTMLVideoElement);
const reviewSection = getElement("review");
const solutionSection = getElement("solution");
const useCameraButton = getTypedElement("use-camera", HTMLButtonElement);
const fileInput = getTypedElement("photo-file", HTMLInputElement);

// The page fetches the solver tables once, at load, so nothing is fetched while reading or solving.
const patternDb = fetch(patternDbUrl).then(async (response) => {
  if (!response.ok)
    throw new Error(`could not load the 4x4 solver tables (${String(response.status)})`);
  return response.arrayBuffer();
});
const vision = createVisionClient();
const solver = createSolverClient(patternDb);

// A solve or read that has been superseded must not touch the screen when it finally settles. Each
// takes `mine` from supersedeSolve() and returns as soon as `mine !== generation`, so only the latest
// one writes `review`, `solution`, `busy` and the notice. A new photo, a corrected cell and a confirmed
// guess all change the board, so each goes through supersedeSolve(), which cancels a running solve.
let generation = 0;
let review: ReviewState | undefined;
let solution: SolutionState | undefined;
let busy: "reading" | "solving" | undefined;
let solveStartedAt = 0;
let solveTimer: number | undefined;
let stream: MediaStream | undefined;
let focusId: string | undefined;
let openingCamera = false;

function showNotice(text: string | undefined): void {
  notice.hidden = text === undefined;
  notice.textContent = text ?? "";
}

/** Last resort for a background task whose own handling failed: say so instead of going quiet. */
function reportFailure(error: unknown): void {
  showNotice(`Something went wrong: ${error instanceof Error ? error.message : String(error)}`);
}

function formatElapsed(): string {
  return `${((performance.now() - solveStartedAt) / 1000).toFixed(1)} s`;
}

function render(): void {
  const activeId = focusId ?? document.activeElement?.id;
  focusId = undefined;
  renderReview();
  renderSolution();
  if (activeId) document.getElementById(activeId)?.focus();
}

function renderReview(): void {
  reviewSection.hidden = review === undefined && busy !== "reading";
  reviewSection.replaceChildren();
  if (busy === "reading") {
    reviewSection.append(
      buildElement("h2", { id: "review-heading" }, "Reading"),
      buildElement("p", { role: "status" }, "Reading the photo…"),
    );
    return;
  }
  if (!review) return;
  const state = review;
  reviewSection.append(buildElement("h2", { id: "review-heading" }, "Detected board"));
  const figure = createPhotoFigure(
    state.photo.url,
    state.photo.image.width,
    state.photo.image.height,
    state.quad,
  );
  if (!state.cells || !state.size) {
    reviewSection.append(
      figure,
      buildElement(
        "p",
        { class: "notice", role: "alert" },
        `${state.problem ?? "No board found."} Retake the photo, or upload another.`,
      ),
    );
    return;
  }
  const { size, cells } = state;
  const check = checkBoard(size, cells);
  const flagged = new Set(check.accepted ? [] : check.offending);
  const panel = buildElement("div", { class: "board-panel" });
  panel.append(
    createGrid({
      size,
      values: cells.map((cell) => cell.value),
      flagged,
      selected: state.editing,
      onCell: (index) => {
        state.editing = state.editing === index ? undefined : index;
        focusId =
          state.editing === undefined
            ? `cell-${String(index)}`
            : `pad-${String(cells[index]?.value ?? 0)}`;
        render();
      },
      idPrefix: "cell",
    }),
  );
  if (state.editing !== undefined) {
    const index = state.editing;
    panel.append(
      createNumberPad(
        size,
        index,
        cells[index]?.value ?? 0,
        (value) => {
          state.cells = setCell(cells, index, value);
          state.editing = undefined;
          supersedeSolve();
          focusId = `cell-${String(index)}`;
          render();
        },
        () => {
          state.editing = undefined;
          focusId = `cell-${String(index)}`;
          render();
        },
      ),
    );
  }
  if (!check.accepted && check.confirmable) {
    panel.append(
      createConfirmList({
        size,
        cells: check.offending.map((index) => ({ index, value: cells[index]?.value ?? 0 })),
        onConfirm: (index) => {
          state.cells = confirmCell(cells, index);
          supersedeSolve();
          // The confirmed row disappears, so land on the next one, or on Solve when none is left.
          const next = check.offending.find((other) => other !== index);
          focusId = next === undefined ? "solve" : `confirm-${String(next)}`;
          render();
        },
        onChange: (index) => {
          state.editing = index;
          focusId = `pad-${String(cells[index]?.value ?? 0)}`;
          render();
        },
      }),
    );
  }
  panel.append(
    check.accepted
      ? buildElement(
          "p",
          { class: "check ok", id: "check", role: "status" },
          "This board is consistent and can be solved. Compare it with the photo; tap a cell to correct it.",
        )
      : buildElement(
          "p",
          { class: "check bad", id: "check", role: "alert" },
          check.confirmable
            ? `Not sure about ${String(check.offending.length)} ${check.offending.length === 1 ? "cell" : "cells"}. Check each highlighted cell against your photo and confirm or change it.`
            : `Cannot solve yet: ${check.message}. Tap a highlighted cell to fix it, or retake the photo.`,
        ),
  );
  panel.append(createSolveControls(size, check.accepted ? check.board : undefined));
  reviewSection.append(buildElement("div", { class: "review-body" }, figure, panel));
}

function createSolveControls(size: GridSize, board: Board | undefined): HTMLElement {
  if (busy === "solving") {
    return buildElement(
      "div",
      { class: "solving" },
      buildElement("p", { id: "solving-status", role: "status" }, `Solving… ${formatElapsed()}`),
      buildElement(
        "p",
        { class: "hint" },
        size === 4
          ? `Looking for the shortest solution. After ${String(DEFAULT_BUDGET_MS / 1000)} s it settles for a fast one.`
          : "Looking for the shortest solution.",
      ),
      buildElement(
        "button",
        { type: "button", id: "cancel-solve", class: "secondary", onClick: cancelSolve },
        "Cancel",
      ),
    );
  }
  return buildElement(
    "div",
    { class: "actions" },
    buildElement(
      "button",
      {
        type: "button",
        id: "solve",
        disabled: board === undefined || busy !== undefined,
        onClick: () => {
          if (board) startSolve(board).catch(reportFailure);
        },
      },
      "Solve",
    ),
  );
}

function renderSolution(): void {
  solutionSection.hidden = solution === undefined;
  solutionSection.replaceChildren();
  if (!solution) return;
  const state = solution;
  solutionSection.append(
    createSolutionView({
      boards: state.boards,
      moves: state.moves,
      step: state.step,
      notGuaranteedShortest: state.notGuaranteedShortest,
      millis: state.millis,
      onStep: setStep,
    }),
  );
}

function setStep(step: number): void {
  if (!solution) return;
  solution.step = Math.min(Math.max(step, 0), solution.moves.length);
  render();
}

function startSolveTimer(): void {
  solveStartedAt = performance.now();
  solveTimer = window.setInterval(() => {
    const line = document.getElementById("solving-status");
    if (line) line.textContent = `Solving… ${formatElapsed()}`;
  }, 250);
}

function stopSolveTimer(): void {
  window.clearInterval(solveTimer);
  solveTimer = undefined;
}

async function startSolve(board: Board): Promise<void> {
  const mine = ++generation;
  busy = "solving";
  solution = undefined;
  showNotice(undefined);
  startSolveTimer();
  render();
  try {
    const outcome = await solver.solve(board);
    if (mine !== generation) return;
    if (outcome.status === "unsolvable") {
      showNotice("This board cannot be solved.");
    } else {
      const boards = [board];
      for (const move of outcome.moves)
        boards.push(applyMove(boards[boards.length - 1] ?? board, move));
      solution = {
        boards,
        moves: outcome.moves,
        notGuaranteedShortest: outcome.notGuaranteedShortest,
        millis: outcome.millis,
        step: 0,
      };
    }
  } catch (error) {
    if (mine !== generation) return;
    showNotice(
      error instanceof CancelledError
        ? "Solve cancelled."
        : `Solving failed: ${error instanceof Error ? error.message : String(error)}`,
    );
  } finally {
    // A superseded solve leaves the timer and `busy` to whoever superseded it.
    if (mine === generation) finishSolve();
  }
}

function finishSolve(): void {
  stopSolveTimer();
  busy = undefined;
  focusId = solution ? "next" : "solve";
  render();
}

function cancelSolve(): void {
  solver.cancel();
}

// The board on screen changes in three ways: a new photo, a corrected cell, a confirmed guess. A solve
// that was running belongs to the old board, so each of them cancels it and takes the generation, and
// the old solve's result never reaches the screen.
function supersedeSolve(): number {
  const mine = ++generation;
  if (busy === "solving") {
    solver.cancel();
    stopSolveTimer();
    busy = undefined;
  }
  solution = undefined;
  return mine;
}

async function readPhoto(photoPromise: Promise<Photo>): Promise<void> {
  const mine = supersedeSolve();
  busy = "reading";
  review = undefined;
  solution = undefined;
  showNotice(undefined);
  render();
  try {
    const photo = await photoPromise;
    if (mine !== generation) return;
    const result = await vision.read(photo.image);
    if (mine !== generation) return;
    if (result.status === "no-board") {
      review = { photo, problem: result.message };
    } else if (result.status === "rejected") {
      review = {
        photo,
        quad: result.attempt.quad,
        size: result.attempt.size,
        cells: result.attempt.cells,
      };
    } else {
      review = { photo, quad: result.quad, size: result.board.size, cells: result.cells };
    }
  } catch (error) {
    if (mine !== generation) return;
    showNotice(error instanceof Error ? error.message : String(error));
  } finally {
    if (mine === generation) {
      busy = undefined;
      render();
    }
  }
}

function closeCamera(): void {
  if (stream) stopCamera(preview, stream);
  stream = undefined;
  cameraPanel.hidden = true;
}

async function openCamera(): Promise<void> {
  // A second tap while the first camera is still starting would leave the first stream live forever.
  if (openingCamera) return;
  openingCamera = true;
  closeCamera();
  showNotice(undefined);
  try {
    stream = await startCamera(preview);
    cameraPanel.hidden = false;
    getElement("capture").focus();
  } catch (error) {
    showNotice(describeCameraError(error));
  } finally {
    openingCamera = false;
  }
}

function captureFrame(): void {
  let photo: Photo;
  try {
    photo = captureVideoFrame(preview);
  } catch (error) {
    showNotice(error instanceof Error ? error.message : String(error));
    return;
  }
  closeCamera();
  readPhoto(Promise.resolve(photo)).catch(reportFailure);
}

useCameraButton.addEventListener("click", () => {
  openCamera().catch(reportFailure);
});
getElement("close-camera").addEventListener("click", closeCamera);
getElement("capture").addEventListener("click", captureFrame);
fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  // Cleared so choosing the same file again (a retake of the same shot) still fires `change`.
  fileInput.value = "";
  if (!file) return;
  closeCamera();
  readPhoto(loadPhotoFile(file)).catch(reportFailure);
});

document.addEventListener("keydown", (event) => {
  if (!solution || event.target instanceof HTMLInputElement) return;
  if (event.key === "ArrowRight" || event.key === "ArrowDown") setStep(solution.step + 1);
  else if (event.key === "ArrowLeft" || event.key === "ArrowUp") setStep(solution.step - 1);
  else return;
  event.preventDefault();
});

Promise.all([vision.ready, solver.ready]).then(
  () => {
    statusLine.textContent = "Ready. Everything runs on this device.";
    document.body.dataset["ready"] = "true";
  },
  (error: unknown) => {
    statusLine.textContent = `Could not start: ${error instanceof Error ? error.message : String(error)}`;
  },
);
