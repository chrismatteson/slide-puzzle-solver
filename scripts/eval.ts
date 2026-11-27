// Runs the real pipeline over a labelled photo set (generated, or real: see `npm run real:eval`) and
// reports detection, grid-size, per-cell and whole-board accuracy, false accepts and timing.
// Usage: tsx scripts/eval.ts [dir] [out.json] [every-nth: evaluate a subset while iterating]
// Each `name.jpg|jpeg|png` needs a `name.json` label: {"size", "state"} (blank = 0), optionally
// "corners" (outer frame, 4 [x, y]), "font" and "solvable" (computed from the state when absent).
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { dirname, extname, join } from "node:path";
import { pathToFileURL } from "node:url";
import jpeg from "jpeg-js";
import { PNG } from "pngjs";
import { isSolvable } from "../src/puzzle/board.ts";
import { MAX_PHOTO_SIDE, type RgbaImage, shrinkRgba } from "../src/vision/image.ts";
import { type Point } from "../src/vision/geometry.ts";
import { defaultWeights } from "../src/vision/default-weights.ts";
import { type ReadResult, readBoard } from "../src/vision/read-board.ts";
import { type BoardLabel, parseBoardLabel } from "./labels.ts";

const IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"];

interface Outcome {
  name: string;
  font: string;
  size: number;
  solvable: boolean;
  hasCorners: boolean;
  status: ReadResult["status"];
  detectedSize?: number;
  cornerError?: number; // worst corner, fraction of side
  cellsCorrect: number;
  cells: number;
  exact: boolean;
  falseAccept: boolean;
  message?: string;
  /** Lowest cell confidence of the reading that was returned (accepted or best rejected). */
  minConfidence?: number;
  millis: number;
}

function loadImage(path: string): RgbaImage {
  const bytes = readFileSync(path);
  const decoded =
    extname(path).toLowerCase() === ".png"
      ? PNG.sync.read(bytes)
      : jpeg.decode(bytes, { useTArray: true, formatAsRGBA: true });
  // Same downscale as the app applies to uploads, so large phone photos are read as users would.
  return shrinkRgba(
    { width: decoded.width, height: decoded.height, data: new Uint8ClampedArray(decoded.data) },
    MAX_PHOTO_SIDE,
  );
}

/**
 * A found quad counts as a detection if every corner is within this fraction of the board side of
 * the labelled outer frame corner. The generator's frame margin is at most 8% of the edge, so a quad
 * that hugs the tile grid (inside the frame) has corners at most 0.08 * sqrt(2) = 0.113 of the side
 * from the frame's; the tolerance admits that and nothing looser, so a quad that is too big, too
 * small or on the wrong object fails.
 */
export const DETECTION_TOLERANCE = 0.12;
/** Frame-accurate quads, for the second figure reported. */
const TIGHT_TOLERANCE = 0.05;

/** Largest distance between a found and a labelled corner, as a fraction of the labelled board's side. */
function computeWorstCornerError(found: readonly Point[], truth: [number, number][]): number {
  const side = Math.hypot(
    (truth[1]?.[0] ?? 0) - (truth[0]?.[0] ?? 0),
    (truth[1]?.[1] ?? 0) - (truth[0]?.[1] ?? 0),
  );
  let worst = 0;
  for (let i = 0; i < 4; i++) {
    const distance = Math.hypot(
      (found[i]?.x ?? 0) - (truth[i]?.[0] ?? 0),
      (found[i]?.y ?? 0) - (truth[i]?.[1] ?? 0),
    );
    worst = Math.max(worst, distance / side);
  }
  return worst;
}

function formatPercent(part: number, whole: number): string {
  return whole === 0 ? "n/a" : `${((100 * part) / whole).toFixed(1)}%`;
}

function readLabel(dir: string, file: string): BoardLabel {
  const path = join(dir, `${file.slice(0, file.length - extname(file).length)}.json`);
  if (!existsSync(path)) throw new Error(`${path}: missing label for ${file}`);
  let parsed: unknown;
  try {
    parsed = JSON.parse(readFileSync(path, "utf8"));
  } catch (error) {
    throw new Error(`${path}: ${error instanceof Error ? error.message : String(error)}`);
  }
  return parseBoardLabel(parsed, path);
}

export function evaluateDirectory(dir: string, every = 1): Outcome[] {
  if (!Number.isInteger(every) || every < 1)
    throw new Error(`every-nth must be a whole number >= 1, got ${String(every)}`);
  const files = readdirSync(dir)
    .filter((f) => IMAGE_EXTENSIONS.includes(extname(f).toLowerCase()))
    .sort()
    .filter((_, i) => i % every === 0);
  if (files.length === 0) throw new Error(`${dir}: no .jpg or .png photos to evaluate`);
  // Check every label before reading any photo, so a bad label fails in seconds, not after the set.
  const labels = files.map((file) => readLabel(dir, file));
  const outcomes: Outcome[] = [];
  for (const [index, file] of files.entries()) {
    const name = file.slice(0, file.length - extname(file).length);
    const label = labels[index];
    if (!label) throw new Error(`${file}: no label`);
    const image = loadImage(join(dir, file));
    const start = performance.now();
    const result = readBoard(image, defaultWeights);
    const millis = performance.now() - start;

    const attempt =
      result.status === "accepted"
        ? { size: result.board.size, cells: result.cells, quad: result.quad }
        : result.status === "rejected"
          ? result.attempt
          : undefined;
    const outcome: Outcome = {
      name,
      font: label.font ?? "unknown",
      size: label.size,
      solvable: label.solvable ?? isSolvable({ size: label.size, cells: label.state }),
      hasCorners: label.corners !== undefined,
      status: result.status,
      cellsCorrect: 0,
      cells: label.state.length,
      exact: false,
      falseAccept: false,
      millis,
    };
    if (result.status !== "accepted") outcome.message = result.message;
    if (attempt) {
      outcome.detectedSize = attempt.size;
      outcome.minConfidence = Math.min(...attempt.cells.map((c) => c.confidence));
      if (label.corners) outcome.cornerError = computeWorstCornerError(attempt.quad, label.corners);
      if (attempt.size === label.size) {
        outcome.cellsCorrect = attempt.cells.filter((c, i) => c.value === label.state[i]).length;
      }
    }
    if (result.status === "accepted") {
      const same =
        result.board.size === label.size &&
        result.board.cells.every((v, i) => v === label.state[i]);
      outcome.exact = same;
      outcome.falseAccept = !same;
    }
    outcomes.push(outcome);
  }
  return outcomes;
}

function main(): void {
  const dir = process.argv[2] ?? "data/eval";
  const outPath = process.argv[3] ?? "data/eval-results.json";
  const outcomes = evaluateDirectory(dir, Number(process.argv[4] ?? 1));
  const total = outcomes.length;
  const solvable = outcomes.filter((o) => o.solvable);
  const unsolvable = outcomes.filter((o) => !o.solvable);
  // Detection needs a labelled outer frame; real-photo labels usually have none.
  const cornerLabelled = outcomes.filter((o) => o.hasCorners);
  const formatDetection = (part: number): string =>
    cornerLabelled.length === 0
      ? "n/a (no corner labels)"
      : formatPercent(part, cornerLabelled.length);
  const detected = cornerLabelled.filter(
    (o) => o.cornerError !== undefined && o.cornerError <= DETECTION_TOLERANCE,
  );
  const detectedTight = cornerLabelled.filter(
    (o) => o.cornerError !== undefined && o.cornerError <= TIGHT_TOLERANCE,
  );
  const withSize = outcomes.filter((o) => o.detectedSize !== undefined);
  const cellsTotal = outcomes.reduce((s, o) => s + o.cells, 0);
  const times = outcomes.map((o) => o.millis).sort((a, b) => a - b);
  const summary = {
    images: total,
    cornerLabelledImages: cornerLabelled.length,
    detectionRate: cornerLabelled.length === 0 ? null : detected.length / cornerLabelled.length,
    tightDetectionRate:
      cornerLabelled.length === 0 ? null : detectedTight.length / cornerLabelled.length,
    gridSizeAccuracy: withSize.filter((o) => o.detectedSize === o.size).length / total,
    perCellAccuracy: outcomes.reduce((s, o) => s + o.cellsCorrect, 0) / cellsTotal,
    exactReadAccuracy:
      solvable.length === 0 ? null : solvable.filter((o) => o.exact).length / solvable.length,
    acceptedRate: outcomes.filter((o) => o.status === "accepted").length / total,
    falseAcceptRate: outcomes.filter((o) => o.falseAccept).length / total,
    unsolvableCorrectlyRejected: unsolvable.filter((o) => o.status !== "accepted").length,
    unsolvableImages: unsolvable.length,
    meanMillis: times.reduce((s, t) => s + t, 0) / total,
    p95Millis: times[Math.floor(0.95 * (total - 1))] ?? 0,
  };
  const byFont = Object.fromEntries(
    [...new Set(outcomes.map((o) => o.font))].map((font) => {
      const group = solvable.filter((o) => o.font === font);
      return [font, `${String(group.filter((o) => o.exact).length)}/${String(group.length)}`];
    }),
  );

  console.log(
    `images                   ${String(total)} (${String(unsolvable.length)} unsolvable)`,
  );
  console.log(
    `detection rate           ${formatDetection(detected.length)}  (every corner within ${String(100 * DETECTION_TOLERANCE)}% of the board side from the frame's)`,
  );
  console.log(
    `  of which frame-tight   ${formatDetection(detectedTight.length)}  (within ${String(100 * TIGHT_TOLERANCE)}%)`,
  );
  console.log(
    `grid-size accuracy       ${formatPercent(withSize.filter((o) => o.detectedSize === o.size).length, total)}`,
  );
  console.log(`per-cell accuracy        ${(100 * summary.perCellAccuracy).toFixed(1)}%`);
  console.log(
    `exact-read accuracy      ${formatPercent(solvable.filter((o) => o.exact).length, solvable.length)}  (solvable boards read exactly and accepted)`,
  );
  console.log(
    `false-accept rate        ${formatPercent(outcomes.filter((o) => o.falseAccept).length, total)}  (accepted but wrong)`,
  );
  console.log(
    `unsolvable rejected      ${String(summary.unsolvableCorrectlyRejected)}/${String(unsolvable.length)}`,
  );
  console.log(
    `time per image           mean ${summary.meanMillis.toFixed(0)} ms, p95 ${summary.p95Millis.toFixed(0)} ms`,
  );
  console.log(`exact by font            ${JSON.stringify(byFont)}`);

  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, JSON.stringify({ summary, byFont, outcomes }, null, 2));
  console.log(`wrote ${outPath}`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) main();
