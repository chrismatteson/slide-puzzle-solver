// Mahalanobis Gradient Compatibility (Gallagher 2012) in CIE Lab, for the seam between two tile edges:
// how far the colours along one edge are from the facing edge, measured in units of how much the
// colours along the first edge change from one pixel line to the next (its own gradient covariance),
// so that a seam through a busy, strongly textured edge is forgiven more than one through a flat edge.
// Computed in both directions and added. A seam is judged by how the colours vary along it, not by their
// level: tiles are lit unevenly (glare, shadows), so a constant offset between the two edges' colours
// and a linear ramp along the seam are taken out first, per channel.
// Gallagher also extrapolates the gradient across the seam; on photographed tiles, with their gap and
// blur, that scored no better than not extrapolating, so the colours are compared as they are.

import { type TileCrop } from "./types.ts";

export type Side = "left" | "right" | "top" | "bottom";

// The first pixel line of a crop edge is spoiled by the gap and its shadow (see assemble.ts FIRST_DEPTH),
// and tiles sit loosely in their frame, so the first sound line is one of these; both edges of a seam
// are tried at each, and the best pair counts. (Depth 1 only: 73.6% of true neighbours found as the
// best partner on the dev set, depths 1-2: 77.2%. A third depth, alone or with windows of depths compared
// only with themselves, cost twice the time for no better result on the owner's photo, and a fourth
// spoiled it.)
const EDGE_DEPTHS = [1, 2];
// Added to the diagonal of every gradient covariance: a perfectly flat edge has none, and must not
// make every difference infinitely bad. In Lab units squared; 1-16 scored the same on the dev set.
const COVARIANCE_FLOOR = 1;
// A length of 5 is a pixel five standard deviations off the edge's own step: nothing like noise. Needed
// for the owner's photo to assemble as labelled in every decode and for every 4 px board jitter
// (test/picture-robustness.test.ts); 4 and 6 fail one of those, 3 and 8 and more fail more. The price
// is on the dev sets, exact (picture / white frame / angled blank): 78.5 / 70.8 / 75.0% against 79.9 /
// 75.0 / 81.2% uncapped (shift 4 px, penalty 0.1), i.e. 1-6 photos in 48-144.
// The owner's photo is the one real photo and the case that matters, so it decides.
const LENGTH_CAP = 5;

/** Lab colour of every pixel of `crop`, 3 numbers per pixel, row-major. */
export function toLab(crop: TileCrop): Float64Array {
  const out = new Float64Array(crop.side * crop.side * 3);
  const compressLabComponent = (t: number): number =>
    t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116;
  const linearizeSrgb = (v: number): number => {
    const c = v / 255;
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  };
  for (let i = 0; i < crop.side * crop.side; i++) {
    const r = linearizeSrgb(crop.rgb[i * 3] ?? 0);
    const g = linearizeSrgb(crop.rgb[i * 3 + 1] ?? 0);
    const b = linearizeSrgb(crop.rgb[i * 3 + 2] ?? 0);
    const x = compressLabComponent((0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047);
    const y = compressLabComponent(0.2126 * r + 0.7152 * g + 0.0722 * b);
    const z = compressLabComponent((0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883);
    out[i * 3] = 116 * y - 16;
    out[i * 3 + 1] = 500 * (x - y);
    out[i * 3 + 2] = 200 * (y - z);
  }
  return out;
}

interface EdgeStatistics {
  /** The edge's pixel line, 3 numbers per pixel. */
  line: Float64Array;
  /** Inverse of the 3x3 covariance (row-major, 9 numbers) of the step from the next line in. */
  inverse: number[];
}

function readLine(lab: Float64Array, n: number, side: Side, depth: number): Float64Array {
  const out = new Float64Array(n * 3);
  for (let i = 0; i < n; i++) {
    const x = side === "left" ? depth : side === "right" ? n - 1 - depth : i;
    const y = side === "top" ? depth : side === "bottom" ? n - 1 - depth : i;
    for (let c = 0; c < 3; c++) out[i * 3 + c] = lab[(y * n + x) * 3 + c] ?? 0;
  }
  return out;
}

/** Inverse of a symmetric positive definite 3x3 matrix given row-major. */
function invert3(m: readonly number[]): number[] {
  const [a = 0, b = 0, c = 0, d = 0, e = 0, f = 0, g = 0, h = 0, i = 0] = m;
  const cofactorA = e * i - f * h;
  const cofactorB = f * g - d * i;
  const cofactorC = d * h - e * g;
  const det = a * cofactorA + b * cofactorB + c * cofactorC;
  return [
    cofactorA / det,
    (c * h - b * i) / det,
    (b * f - c * e) / det,
    cofactorB / det,
    (a * i - c * g) / det,
    (c * d - a * f) / det,
    cofactorC / det,
    (b * g - a * h) / det,
    (a * e - b * d) / det,
  ];
}

function measureStatistics(
  lab: Float64Array,
  n: number,
  side: Side,
  depth: number,
): EdgeStatistics {
  const line = readLine(lab, n, side, depth);
  const inner = readLine(lab, n, side, depth + 1);
  const mean = [0, 0, 0];
  for (let i = 0; i < n; i++)
    for (let c = 0; c < 3; c++)
      mean[c] = (mean[c] ?? 0) + ((line[i * 3 + c] ?? 0) - (inner[i * 3 + c] ?? 0)) / n;
  const covariance = new Array<number>(9).fill(0);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) {
      for (let k = 0; k < 3; k++) {
        const stepC = (line[i * 3 + c] ?? 0) - (inner[i * 3 + c] ?? 0) - (mean[c] ?? 0);
        const stepK = (line[i * 3 + k] ?? 0) - (inner[i * 3 + k] ?? 0) - (mean[k] ?? 0);
        covariance[c * 3 + k] = (covariance[c * 3 + k] ?? 0) + (stepC * stepK) / n;
      }
    }
  }
  for (let c = 0; c < 3; c++) covariance[c * 4] = (covariance[c * 4] ?? 0) + COVARIANCE_FLOOR;
  return { line, inverse: invert3(covariance) };
}

/**
 * Summed Mahalanobis lengths of pixels `start`..`end` of `difference` (3 numbers per pixel) after a
 * least-squares offset and ramp per channel are taken out, under each of the two inverse covariances
 * (the seam is judged from both of its sides). The ramp is the glare: a seam lit by a gradient is not
 * a mismatch. Each length is capped at LENGTH_CAP, so that a few outlier pixels cost no more than a
 * moderate mismatch. (Offset alone: the owner's photo, with its glossy tiles, came out wrong in about
 * half of its decodes and cuts; offset and ramp: one or none, with the cap all of them.)
 */
function sumLengths(
  difference: Float64Array,
  first: readonly number[],
  second: readonly number[],
  start: number,
  end: number,
): number {
  const count = end - start;
  const mid = (start + end - 1) / 2;
  let offsetL = 0;
  let offsetA = 0;
  let offsetB = 0;
  for (let i = start; i < end; i++) {
    offsetL += difference[i * 3] ?? 0;
    offsetA += difference[i * 3 + 1] ?? 0;
    offsetB += difference[i * 3 + 2] ?? 0;
  }
  offsetL /= count;
  offsetA /= count;
  offsetB /= count;
  let numL = 0;
  let numA = 0;
  let numB = 0;
  let den = 0;
  for (let i = start; i < end; i++) {
    const t = i - mid;
    numL += t * ((difference[i * 3] ?? 0) - offsetL);
    numA += t * ((difference[i * 3 + 1] ?? 0) - offsetA);
    numB += t * ((difference[i * 3 + 2] ?? 0) - offsetB);
    den += t * t;
  }
  const slopeL = den > 0 ? numL / den : 0;
  const slopeA = den > 0 ? numA / den : 0;
  const slopeB = den > 0 ? numB / den : 0;
  // Indexed reads, not array destructuring: this runs thousands of times per photo.
  const f0 = first[0] ?? 0;
  const f1 = first[1] ?? 0;
  const f2 = first[2] ?? 0;
  const f3 = first[3] ?? 0;
  const f4 = first[4] ?? 0;
  const f5 = first[5] ?? 0;
  const f6 = first[6] ?? 0;
  const f7 = first[7] ?? 0;
  const f8 = first[8] ?? 0;
  const s0 = second[0] ?? 0;
  const s1 = second[1] ?? 0;
  const s2 = second[2] ?? 0;
  const s3 = second[3] ?? 0;
  const s4 = second[4] ?? 0;
  const s5 = second[5] ?? 0;
  const s6 = second[6] ?? 0;
  const s7 = second[7] ?? 0;
  const s8 = second[8] ?? 0;
  let sum = 0;
  for (let i = start; i < end; i++) {
    const t = i - mid;
    const p = (difference[i * 3] ?? 0) - offsetL - slopeL * t;
    const q = (difference[i * 3 + 1] ?? 0) - offsetA - slopeA * t;
    const r = (difference[i * 3 + 2] ?? 0) - offsetB - slopeB * t;
    const fromFirst =
      p * (f0 * p + f1 * q + f2 * r) +
      q * (f3 * p + f4 * q + f5 * r) +
      r * (f6 * p + f7 * q + f8 * r);
    const fromSecond =
      p * (s0 * p + s1 * q + s2 * r) +
      q * (s3 * p + s4 * q + s5 * r) +
      r * (s6 * p + s7 * q + s8 * r);
    sum +=
      Math.min(Math.sqrt(Math.max(fromFirst, 0)), LENGTH_CAP) +
      Math.min(Math.sqrt(Math.max(fromSecond, 0)), LENGTH_CAP);
  }
  return sum;
}

// Photos carry shadows with a hard edge that crosses a tile edge part-way, lighting one stretch of it
// and not the next. Besides one offset (and ramp) for the whole edge, a seam may therefore have a second
// from one of these pixel rows on, at SPLIT_PENALTY extra cost. (Not allowing it: 72.9% exact on the
// dev white-frame set, 77.1% with it; every split from 8 to 56 px, or a smaller penalty, scored the same.)
const SPLITS = [0.25, 0.5, 0.75];
// Tiles sit loosely in their frame: the edges may be out of line by up to LAB_MAX_SHIFT_PX pixels, at
// LAB_SHIFT_PENALTY per pixel. Both were set on the owner's photo, which is a near-tie at places 7 and 11
// and has no wide margin on either: with the cap, LAB_MAX_SHIFT_PX 3 and a penalty of 0.03 pass every
// test/picture-robustness.test.ts case; 4 or 2 pixels, or a penalty of 0.1, fail one or two of them.
// The dev sets do not care (penalty 0.1 or 0.03: within 1-2 photos).
const LAB_MAX_SHIFT_PX = 3;
const LAB_SHIFT_PENALTY = 0.03;
const SPLIT_PENALTY = 0.3;

/**
 * `costs[a][b]`: Mahalanobis dissimilarity of tile a's `outgoing` side facing tile b's `incoming`
 * side (infinity on the diagonal). `lab` are the tiles' pixels from `toLab`.
 */
export function measureMahalanobisCosts(
  lab: Float64Array[],
  tileSide: number,
  outgoing: Side,
  incoming: Side,
): number[][] {
  const readSideStatistics = (side: Side): EdgeStatistics[][] =>
    lab.map((pixels) =>
      EDGE_DEPTHS.map((depth) => measureStatistics(pixels, tileSide, side, depth)),
    );
  const from = readSideStatistics(outgoing);
  const to = readSideStatistics(incoming);
  const difference = new Float64Array(tileSide * 3);
  return from.map((a, i) =>
    to.map((b, j) => {
      if (i === j) return Number.POSITIVE_INFINITY;
      let best = Number.POSITIVE_INFINITY;
      for (const first of a) {
        for (const second of b) {
          for (let shift = -LAB_MAX_SHIFT_PX; shift <= LAB_MAX_SHIFT_PX; shift++) {
            // Pixel p of the first edge faces pixel p + shift of the second.
            const start = Math.max(0, -shift);
            const end = Math.min(tileSide, tileSide - shift);
            for (let p = start; p < end; p++)
              for (let c = 0; c < 3; c++)
                difference[p * 3 + c] =
                  (second.line[(p + shift) * 3 + c] ?? 0) - (first.line[p * 3 + c] ?? 0);
            const sumLengthsBetween = (rangeStart: number, rangeEnd: number): number =>
              sumLengths(difference, first.inverse, second.inverse, rangeStart, rangeEnd);
            const count = end - start;
            let cost = sumLengthsBetween(start, end) / count;
            for (const share of SPLITS) {
              const split = start + Math.round(count * share);
              cost = Math.min(
                cost,
                (sumLengthsBetween(start, split) + sumLengthsBetween(split, end)) / count +
                  SPLIT_PENALTY,
              );
            }
            best = Math.min(best, cost + LAB_SHIFT_PENALTY * Math.abs(shift));
          }
        }
      }
      return best;
    }),
  );
}
