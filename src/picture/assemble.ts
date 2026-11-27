// Solves the picture of a picture-tile puzzle from its tile crops: which tile belongs at which solved
// position, and where the empty slot belongs. Jigsaw-style: every pair of tile edges is scored, and
// whole layouts are searched with one beam search per candidate home of the empty slot. Two scorers
// price the seams, each with its own search:
//  - "plain": a prediction-free dissimilarity of the edge lines (Pomeranz 2011 / Gallagher 2012, see
//    measureEdgeCost) divided by the runner-up of its edge, so plain tiles score as "unsure";
//  - "surprise": the geometric mean of that and a ramp-removing Mahalanobis gradient compatibility in
//    Lab (mahalanobis.ts), turned into -log probabilities by Sinkhorn scaling (toSurprise), so that a
//    tile that two others fit equally well is told apart from one that only one fits.
// The surprise layout is the better one almost always, but on the owner's real photo (blurred, glossy,
// large flat blue areas) it is wrong while the plain layout is right, by a hair. See chooseLayout for
// how the two are reconciled.
//
// Measured (npm run eval:picture / eval:corrections, held-out; figures in README.md): see README
// "Measured accuracy". The beam keys on the permutation parity (without it the reachable layouts were
// pruned in favour of unreachable ones), and sides left open are priced by how well they would have
// matched.

import { type GridSize, createGoalBoard, isSolvable } from "../puzzle/board.ts";
import { type Side, measureMahalanobisCosts, toLab } from "./mahalanobis.ts";
import { type Arrangement, type TileCrop } from "./types.ts";

// The gap between tiles, its shadow and the blur spoil the first pixel of every crop edge, but skipping
// more loses the information. Lines 1-4 tolerate crops that are cut a few pixels out of register, as
// those of a photo whose board was fitted a little off are: test/picture-robustness.test.ts needs 8 of
// 12 jittered cuts of the owner's photo (corners off by up to 8 px) assembled exactly. On the synthetic
// set lines 1-4 give 44.4% exact, lines 1-3 (DEPTH_COUNT 3) 42.4%.
const FIRST_DEPTH = 1;
const DEPTH_COUNT = 4;
const MAX_SHIFT = 4; // tiles sit loosely in their frame: neighbours are up to this many pixels out of line
const SHIFT_PENALTY = 0.5;
// Tiles are lit and printed unevenly, so a constant per-channel brightness offset between two edges is
// discounted by this share of itself (1 would ignore offsets, which makes sky and sea match).
const OFFSET_DISCOUNT = 0.5;
// Colours are compared in a log scale, where lighting gains become offsets.
const LOG_SCALE = 40;
const FLOOR = 6; // added to every edge cost before normalising, so near-identical plain edges stay ~1
// The beam is not converged at 200 on 4x4: of the 96 dev photos (corrections, seed 20261123, white-frame
// and angled-blank) 4 have a cheaper best layout at 1600 than at 200; at 400 three remain, with
// differences of 0.25 seam units or less. Held-out white-frame 0025 is missed only for this reason at
// 200. Dev, white-frame / angled-blank exact with the slot found by the app (labelled corners, then
// jittered 3%): 32 and 32 / 22 and 21; slot given: 35 and 34 / 36 and 35. Time at 400: the owner's 4x4
// photo (Chromium decode) takes 0.37-0.45 s of CPU per assembly in Node 22, 0.9-1.0 s on the first, cold
// call, which is what test/picture-chromium.test.ts times against its 2 s budget; the same as width 200
// took before the beam step moved to typed arrays.
const BEAM_WIDTH = 400;
// A missing edge (next to the empty slot) counts as this many seam units in each scorer. Above 1 in the
// plain scorer it favours homes with fewer neighbours (corners), as on bought puzzles. The surprise
// scorer already prices a side left open (OPEN_SHARE).
const PLAIN_MISSING_EDGE_COST = 1.3;
const SURPRISE_MISSING_EDGE_COST = 1;
// How many of the cheapest reachable layouts get a local search, see refineLayout.
const REFINED_CANDIDATES = 5;
// What a doubt about a tile is worth, see measureConfidence: swapping it with another tile making the
// layout this many seam units (-log probability) worse counts as certain. Set so that the owner's photo,
// whose places 7 and 11 are a near-tied pair (swapping them costs about 2 units in every decode, the
// next most doubtful tile 11 or more), has exactly those two below LOW_TILE_CONFIDENCE (0.15); that
// holds for 14-70. On the picture dev set (every other photo, 1656 placed tiles) the swap margin
// separates wrong from right tiles only weakly: at 40, 61% of the wrong tiles and 18% of the right ones
// fall below 0.15; 30: 46% / 9%, 60: 85% / 41%.
const CERTAIN_SWAP_COST = 40;

type Direction = "right" | "down";

/** One sample line per colour channel interleaved: RGB of the pixels `depth` inside one side of a crop. */
function readEdgeLine(crop: TileCrop, side: Side, depth: number): Float64Array {
  const n = crop.side;
  const out = new Float64Array(n * 3);
  for (let i = 0; i < n; i++) {
    const x = side === "left" ? depth : side === "right" ? n - 1 - depth : i;
    const y = side === "top" ? depth : side === "bottom" ? n - 1 - depth : i;
    for (let c = 0; c < 3; c++)
      out[i * 3 + c] = LOG_SCALE * Math.log(1 + (crop.rgb[(y * n + x) * 3 + c] ?? 0) / 16);
  }
  return out;
}

/**
 * A sample line with, for each slide of the edge it may be compared at, the mean and spread (per
 * channel, 6 numbers) of the stretch that slide overlaps, as the line facing right/down (`asFirst`) or
 * left/up (`asSecond`) of a pair. They only depend on the line, so are measured once, not once per pair.
 */
interface EdgeLine {
  values: Float64Array;
  asFirst: Float64Array;
  asSecond: Float64Array;
}

/** `[mean x3, spread x3]` of the samples `start` to `end` of `values` (3 interleaved channels). */
function measureWindow(
  values: Float64Array,
  start: number,
  end: number,
  out: Float64Array,
  at: number,
): void {
  const count = (end - start) / 3;
  const sum = [0, 0, 0];
  const square = [0, 0, 0];
  for (let i = start; i < end; i++) {
    const v = values[i] ?? 0;
    const c = i % 3;
    sum[c] = (sum[c] ?? 0) + v;
    square[c] = (square[c] ?? 0) + v * v;
  }
  for (let c = 0; c < 3; c++) {
    const mean = (sum[c] ?? 0) / count;
    out[at + c] = mean;
    out[at + 3 + c] =
      Math.sqrt(Math.max((square[c] ?? 0) / count - mean ** 2, 0)) + STRUCTURE_FLOOR;
  }
}

function toEdgeLine(values: Float64Array): EdgeLine {
  const slides = 2 * MAX_SHIFT + 1;
  const asFirst = new Float64Array(slides * 6);
  const asSecond = new Float64Array(slides * 6);
  for (let shift = -MAX_SHIFT; shift <= MAX_SHIFT; shift++) {
    const start = Math.max(0, -shift * 3);
    const end = Math.min(values.length, values.length - shift * 3);
    const at = (shift + MAX_SHIFT) * 6;
    measureWindow(values, start, end, asFirst, at);
    measureWindow(values, start + shift * 3, end + shift * 3, asSecond, at);
  }
  return { values, asFirst, asSecond };
}

/** The lines of one tile side at depths FIRST_DEPTH .. FIRST_DEPTH + DEPTH_COUNT - 1. */
function readEdges(crop: TileCrop, side: Side): EdgeLine[] {
  return Array.from({ length: DEPTH_COUNT }, (_, i) =>
    toEdgeLine(readEdgeLine(crop, side, FIRST_DEPTH + i)),
  );
}

// Glare and gloss lift one side of a seam by a different amount in flat blue than in green, which no
// single offset undoes (the owner's photo in Chromium's 960x1280 decode: the true seam between two
// right-column tiles cost more than the wrong one). So each line pair is also compared with both lines
// z-normalised per channel (mean removed, divided by the spread plus this floor): what is left is where
// the line steps and by how much relative to itself. The floor, in LOG_SCALE units, is above the
// texture noise of a plain edge so that two flat edges stay neutral instead of matching on noise. Floor
// and weight were scanned against the six jpeg-js variants of test/picture-corners.test.ts and the
// corner jitter test; 24/45 sits in the middle of the range that passes them. On the synthetic set the
// term takes exact from 41.0% (weight 0) to 44.4%.
const STRUCTURE_FLOOR = 24;
const STRUCTURE_WEIGHT = 45;

/**
 * Mean absolute difference of two lines, the second slid by `shift` pixels, minus a discounted offset,
 * plus the weighted mean difference of their per-channel z-normalised profiles.
 */
function measureShiftedCost(first: EdgeLine, second: EdgeLine, shift: number): number {
  const a = first.values;
  const b = second.values;
  const start = Math.max(0, -shift * 3);
  const end = Math.min(a.length, a.length - shift * 3);
  const at = (shift + MAX_SHIFT) * 6;
  const [mean1, mean2] = [first.asFirst, second.asSecond];
  let level = 0;
  let structure = 0;
  for (let i = start; i < end; i += 3) {
    for (let c = 0; c < 3; c++) {
      const x = (a[i + c] ?? 0) - (mean1[at + c] ?? 0);
      const y = (b[i + c + shift * 3] ?? 0) - (mean2[at + c] ?? 0);
      level += Math.abs(
        x - y + (1 - OFFSET_DISCOUNT) * ((mean1[at + c] ?? 0) - (mean2[at + c] ?? 0)),
      );
      structure += Math.abs(x / (mean1[at + 3 + c] ?? 1) - y / (mean2[at + 3 + c] ?? 1));
    }
  }
  return (level + STRUCTURE_WEIGHT * structure) / (end - start) + SHIFT_PENALTY * Math.abs(shift);
}

/**
 * Dissimilarity of `first`'s edge against `second`'s facing edge: the best match over every pair of
 * depths and slides along the edge. Extrapolating the gradient across the gap (prediction based
 * dissimilarity proper) scored no better than comparing the lines directly on blurred photos.
 */
function measureEdgeCost(first: EdgeLine[], second: EdgeLine[]): number {
  let best = Number.POSITIVE_INFINITY;
  for (const a of first)
    for (const b of second)
      for (let shift = -MAX_SHIFT; shift <= MAX_SHIFT; shift++)
        best = Math.min(best, measureShiftedCost(a, b, shift));
  return best;
}

/** Second smallest value of `values` (the smallest when there is only one). */
function pickRunnerUp(values: number[]): number {
  const sorted = [...values].sort((x, y) => x - y);
  return sorted[1] ?? sorted[0] ?? 1;
}

/**
 * `costs[direction][a][b]` per scorer. Plain: how badly tile b fits next to tile a (b right of / below
 * a), divided by the runner-up cost of a's edge and of b's edge; about 1 for a wrong or ambiguous
 * neighbour, below 1 for a matching one. Surprise: how surprising it is that b is there (see
 * toSurprise), from the geometric mean of the plain and the Mahalanobis dissimilarity, each divided by
 * the runner-up costs.
 */
function computeEdgeCosts(tiles: TileCrop[]): { plain: EdgeCosts; surprise: EdgeCosts } {
  const makeEdgeCosts = (missing: number): EdgeCosts => ({
    right: [],
    down: [],
    open: { left: [], right: [], top: [], bottom: [] },
    missing,
  });
  const plainCosts = makeEdgeCosts(PLAIN_MISSING_EDGE_COST);
  const surpriseCosts = makeEdgeCosts(SURPRISE_MISSING_EDGE_COST);
  const lab = tiles.map((tile) => toLab(tile));
  const side = tiles[0]?.side ?? 0;
  for (const direction of ["right", "down"] as const) {
    const [outgoing, incoming] =
      direction === "right" ? (["right", "left"] as const) : (["bottom", "top"] as const);
    const from = tiles.map((tile) => readEdges(tile, outgoing));
    const to = tiles.map((tile) => readEdges(tile, incoming));
    const mahalanobis = normaliseByRunnerUp(measureMahalanobisCosts(lab, side, outgoing, incoming));
    const plain = normaliseByRunnerUp(
      from.map((a, i) =>
        to.map((b, j) => (i === j ? Number.POSITIVE_INFINITY : measureEdgeCost(a, b) + FLOOR)),
      ),
    );
    const { seam, outOpen, inOpen } = toSurprise(
      plain.map((row, i) =>
        row.map(
          (cost, j) => cost ** PLAIN_WEIGHT * (mahalanobis[i]?.[j] ?? 1) ** (1 - PLAIN_WEIGHT),
        ),
      ),
    );
    // Leaving a side open costs what is missing to a perfect match, the same for both ends of an edge.
    const priceOpenEdge = (costsOfEdge: number[]) =>
      Math.max(1 - Math.min(...costsOfEdge.filter(Number.isFinite)), 0) * PLAIN_OPEN_WEIGHT;
    plainCosts[direction] = plain;
    plainCosts.open[outgoing] = plain.map((row) => priceOpenEdge(row));
    plainCosts.open[incoming] = plain.map((_, j) => priceOpenEdge(plain.map((row) => row[j] ?? 1)));
    surpriseCosts[direction] = seam;
    surpriseCosts.open[outgoing] = outOpen;
    surpriseCosts.open[incoming] = inOpen;
  }
  return { plain: plainCosts, surprise: surpriseCosts };
}

const MIN_RUNNER_UP = 1e-9;

/** Each cost divided by the mean of the runner-up costs of its row (edge of a) and column (edge of b). */
function normaliseByRunnerUp(raw: number[][]): number[][] {
  const rowRunnerUp = raw.map((row) => pickRunnerUp(row));
  const columnRunnerUp = (raw[0] ?? []).map((_, j) => pickRunnerUp(raw.map((row) => row[j] ?? 0)));
  // Three or more identical flat tiles (a uniform background) cost 0 against each other, so both
  // runner-ups are 0 and 0 / 0 would turn the whole matrix into NaN, which leaves the beam search empty.
  return raw.map((row, i) =>
    row.map(
      (cost, j) =>
        cost / Math.max(((rowRunnerUp[i] ?? 1) + (columnRunnerUp[j] ?? 1)) / 2, MIN_RUNNER_UP),
    ),
  );
}

// Softness of the seam probabilities: a normalised cost this far below 1 (a wrong or ambiguous
// neighbour) is e times as likely. 0.05-0.2 scored the same on the dev set.
const TAU = 0.1;
// Price of an open side in the plain scorer, per unit of how much better than 1 its best neighbour fits.
const PLAIN_OPEN_WEIGHT = 1.5;
// Weight of the plain scorer in the geometric mean with the Mahalanobis one (dev set, exact: the plain
// scorer alone 53%, Mahalanobis alone 76%, the mean 78%).
const PLAIN_WEIGHT = 0.5;
// Share of an edge's probability that it has no neighbour at all (picture border or empty slot), next
// to 1 for all the tiles together. 0.2-0.5 scored the same on the dev set, 1.5 lost 5 points.
const OPEN_SHARE = 0.5;
/**
 * Turns normalised costs into -log probabilities that b is a's neighbour: exp(-cost/TAU) scaled
 * alternately per row and column (Sinkhorn) until each edge's probabilities, with a share OPEN_SHARE for "no
 * neighbour at all", sum to one. A tile that two others fit equally well is thus told apart from one
 * that only one fits, which a single runner-up cannot. `outOpen` / `inOpen` are the same for having no
 * neighbour: -log of that share for the row's / column's edge.
 */
export function toSurprise(costs: number[][]): {
  seam: number[][];
  outOpen: number[];
  inOpen: number[];
} {
  const kernel = costs.map((row) => row.map((cost) => Math.exp((1 - cost) / TAU)));
  let rows = costs.map(() => 1);
  let columns = costs.map(() => 1);
  for (let round = 0; round < 30; round++) {
    rows = kernel.map(
      (row) => 1 / (row.reduce((s, k, j) => s + k * (columns[j] ?? 1), 0) + OPEN_SHARE),
    );
    columns = costs.map(
      (_, j) =>
        1 / (kernel.reduce((s, row, i) => s + (row[j] ?? 0) * (rows[i] ?? 1), 0) + OPEN_SHARE),
    );
  }
  return {
    seam: kernel.map((row, i) =>
      row.map((k, j) => -Math.log(k * (rows[i] ?? 1) * (columns[j] ?? 1))),
    ),
    outOpen: rows.map((r) => -Math.log(OPEN_SHARE * r)),
    inOpen: columns.map((c) => -Math.log(OPEN_SHARE * c)),
  };
}

export interface EdgeCosts extends Record<Direction, number[][]> {
  missing: number;
  /** `open[side][tile]`: extra cost of leaving that side of the tile without a neighbour. */
  open: Record<Side, number[]>;
}

export interface Layout {
  /** Index into the tile list per position, -1 at the home of the empty slot. */
  tileAt: number[];
  cost: number;
  /** Parity of the inversions among the placed tiles: layouts that differ in it can reach different sets. */
  parity: number;
}

/** Cost of putting `tile` at `position` given what the layout already holds at the left and above. */
function priceTile(
  costs: EdgeCosts,
  tileAt: readonly number[],
  position: number,
  tile: number,
  width: number,
): number {
  let cost = 0;
  const column = position % width;
  const left = column > 0 ? (tileAt[position - 1] ?? -1) : -2;
  const above = position >= width ? (tileAt[position - width] ?? -1) : -2;
  if (left === -1) cost += costs.missing + (costs.open.left[tile] ?? 0);
  else if (left >= 0) cost += costs.right[left]?.[tile] ?? 0;
  else cost += costs.open.left[tile] ?? 0;
  if (above === -1) cost += costs.missing + (costs.open.top[tile] ?? 0);
  else if (above >= 0) cost += costs.down[above]?.[tile] ?? 0;
  else cost += costs.open.top[tile] ?? 0;
  // Right and bottom sides are only known open once the layout's last column or row is reached.
  if (column === width - 1) cost += costs.open.right[tile] ?? 0;
  if (position >= width * (width - 1)) cost += costs.open.bottom[tile] ?? 0;
  return cost;
}

/** Cost of the empty slot at `position`: its placed neighbours (left, above) have a side open to it. */
function priceHome(
  costs: EdgeCosts,
  tileAt: readonly number[],
  position: number,
  width: number,
): number {
  let cost = 0;
  const left = position % width > 0 ? (tileAt[position - 1] ?? -1) : -1;
  const above = position >= width ? (tileAt[position - width] ?? -1) : -1;
  if (position % width > 0) cost += costs.missing + (costs.open.right[left] ?? 0);
  if (position >= width) cost += costs.missing + (costs.open.bottom[above] ?? 0);
  return cost;
}

/** A partial layout of the beam search. */
interface BeamEntry {
  /** Tile per position so far, -1 at the home of the empty slot. */
  tileAt: number[];
  cost: number;
  parity: number;
  used: number;
}

/** Number of bits set in `value` (a bit mask of at most 31 tiles). */
function countBits(value: number): number {
  let count = 0;
  for (let rest = value; rest !== 0; rest &= rest - 1) count++;
  return count;
}

/**
 * Indices of the `count` cheapest of `costs[0..length)`, cheapest first, ties in index order: the same as
 * a stable sort of the indices by cost followed by `slice(0, count)`, without sorting the thousands that
 * are cut.
 */
export function selectCheapest(costs: Float64Array, length: number, count: number): number[] {
  const kept: number[] = [];
  const byCost = (a: number, b: number): number => (costs[a] ?? 0) - (costs[b] ?? 0);
  if (length <= count) {
    for (let i = 0; i < length; i++) kept.push(i);
    return kept.sort(byCost);
  }
  const threshold = selectSmallest(costs.slice(0, length), count - 1);
  for (let i = 0; i < length; i++) if ((costs[i] ?? 0) < threshold) kept.push(i);
  for (let i = 0; i < length && kept.length < count; i++) if (costs[i] === threshold) kept.push(i);
  return kept.sort(byCost);
}

/** The value that would be at index `rank` if `values` were sorted ascending (reorders `values`). */
function selectSmallest(values: Float64Array, rank: number): number {
  let low = 0;
  let high = values.length - 1;
  while (low < high) {
    const pivot = values[(low + high) >> 1] ?? 0;
    let i = low;
    let j = high;
    while (i <= j) {
      while ((values[i] ?? 0) < pivot) i++;
      while ((values[j] ?? 0) > pivot) j--;
      if (i <= j) {
        const swap = values[i] ?? 0;
        values[i] = values[j] ?? 0;
        values[j] = swap;
        i++;
        j--;
      }
    }
    if (rank <= j) high = j;
    else if (rank >= i) low = i;
    else return values[rank] ?? 0;
  }
  return values[rank] ?? 0;
}

/**
 * Buffers for the candidates of one beam step, reused by every step of a search: allocating them per
 * step made garbage collection a large share of a search. The table has 2^`bits` slots, all -1 between
 * steps.
 */
interface BeamScratch {
  bits: number;
  slots: Int32Array;
  slotOf: Int32Array;
  highKeys: Int32Array;
  lowKeys: Int32Array;
  from: Int32Array;
  placed: Int32Array;
  total: Float64Array;
  parities: Uint8Array;
  useds: Int32Array;
}

/** Buffers for steps of at most `most` candidates. */
function createBeamScratch(most: number): BeamScratch {
  let bits = 4;
  while (2 ** bits < 2 * most) bits++;
  return {
    bits,
    slots: new Int32Array(2 ** bits).fill(-1),
    slotOf: new Int32Array(most),
    highKeys: new Int32Array(most),
    lowKeys: new Int32Array(most),
    from: new Int32Array(most),
    placed: new Int32Array(most),
    total: new Float64Array(most),
    parities: new Uint8Array(most),
    useds: new Int32Array(most),
  };
}

/**
 * One step of the beam search: every way to fill `position` (the empty slot's home if `isHome`, else an
 * unused tile), the cheapest `beamWidth` kept.
 */
function extendBeam(
  costs: EdgeCosts,
  tiles: readonly number[],
  width: number,
  beam: readonly BeamEntry[],
  position: number,
  isHome: boolean,
  beamWidth: number,
  scratch: BeamScratch,
): BeamEntry[] {
  // Two partial layouts that use the same tiles and agree on the last `width` positions have the same
  // future costs, so only the cheaper one is kept. The key is two small integers: tiles used and parity,
  // and the last `width` tiles at 5 bits each. Candidates live in typed arrays and an open-addressing
  // table that keeps them in the order of first insertion, as a Map would, which is what makes ties come
  // out the same every time; one object per candidate and a Map of keys too large for small integers
  // were the search's main cost.
  const { bits, slots, slotOf, highKeys, lowKeys, from, placed, total, parities, useds } = scratch;
  const mask = 2 ** bits - 1;
  const choices = isHome ? [-1] : tiles;
  let count = 0;
  for (let index = 0; index < beam.length; index++) {
    const entry = beam[index] as BeamEntry;
    const { used, tileAt } = entry;
    let recent = 0;
    for (let k = Math.max(0, position - width + 1); k < position; k++)
      recent = recent * 32 + (tileAt[k] ?? -1) + 1;
    for (const tile of choices) {
      if (tile !== -1 && (used & (1 << tile)) !== 0) continue;
      const cost =
        entry.cost +
        (tile === -1
          ? priceHome(costs, tileAt, position, width)
          : priceTile(costs, tileAt, position, tile, width));
      // Placed tiles with a higher index than this one, which all sit before it.
      const parity =
        tile === -1 ? entry.parity : (entry.parity + countBits(used >> (tile + 1))) % 2;
      const nextUsed = tile === -1 ? used : used | (1 << tile);
      const high = nextUsed * 2 + parity;
      const low = recent * 32 + tile + 1;
      let slot = Math.imul(low ^ Math.imul(high, 0x85ebca6b), 0x9e3779b1) >>> (32 - bits);
      let at = slots[slot] ?? -1;
      while (at !== -1 && (highKeys[at] !== high || lowKeys[at] !== low)) {
        slot = (slot + 1) & mask;
        at = slots[slot] ?? -1;
      }
      if (at === -1) {
        at = count++;
        slots[slot] = at;
        slotOf[at] = slot;
        highKeys[at] = high;
        lowKeys[at] = low;
      } else if ((total[at] ?? 0) <= cost) continue;
      from[at] = index;
      placed[at] = tile;
      total[at] = cost;
      parities[at] = parity;
      useds[at] = nextUsed;
    }
  }
  const next = selectCheapest(total, count, beamWidth).map((at) => ({
    tileAt: [...(beam[from[at] ?? 0]?.tileAt ?? []), placed[at] ?? -1],
    cost: total[at] ?? 0,
    parity: parities[at] ?? 0,
    used: useds[at] ?? 0,
  }));
  // Empty the table for the next step: only the slots this step filled.
  for (let at = 0; at < count; at++) slots[slotOf[at] ?? 0] = -1;
  return next;
}

/**
 * Cheapest layouts for every home of the empty slot (index = home): beam search in raster order. Up to
 * its home's position, the search for a home places only tiles, exactly as for any later home, so those
 * steps are run once and shared.
 */
export function searchLayoutsPerHome(
  costs: EdgeCosts,
  width: number,
  beamWidth = BEAM_WIDTH,
): Layout[][] {
  const total = width * width;
  const tiles = [...costs.right.keys()];
  const perHome: Layout[][] = [];
  let shared: BeamEntry[] = [{ tileAt: [], cost: 0, parity: 0, used: 0 }];
  const scratch = createBeamScratch(Math.max(beamWidth, 1) * tiles.length);
  for (let home = 0; home < total; home++) {
    let beam = extendBeam(costs, tiles, width, shared, home, true, beamWidth, scratch);
    for (let position = home + 1; position < total; position++)
      beam = extendBeam(costs, tiles, width, beam, position, false, beamWidth, scratch);
    perHome.push(
      beam.map((entry) => ({ tileAt: entry.tileAt, cost: entry.cost, parity: entry.parity })),
    );
    if (home + 1 < total)
      shared = extendBeam(costs, tiles, width, shared, home, false, beamWidth, scratch);
  }
  return perHome;
}

/** Cost of a whole layout, priced position by position as the beam search does. */
function priceLayout(costs: EdgeCosts, tileAt: readonly number[], width: number): number {
  let cost = 0;
  tileAt.forEach((tile, position) => {
    cost +=
      tile === -1
        ? priceHome(costs, tileAt, position, width)
        : priceTile(costs, tileAt, position, tile, width);
  });
  return cost;
}

/** The layouts one or two exchanges of positions (the empty slot's home included) away from `tileAt`. */
function* listNearLayouts(tileAt: readonly number[]): Generator<number[]> {
  const swapPositions = (layout: readonly number[], a: number, b: number): number[] => {
    const next = [...layout];
    next[a] = layout[b] ?? -1;
    next[b] = layout[a] ?? -1;
    return next;
  };
  for (let a = 0; a < tileAt.length; a++) {
    for (let b = a + 1; b < tileAt.length; b++) {
      const once = swapPositions(tileAt, a, b);
      yield once;
      for (let c = a; c < tileAt.length; c++)
        for (let d = c + 1; d < tileAt.length; d++) yield swapPositions(once, c, d);
    }
  }
}

/**
 * Local search from `start`: while some layout one or two exchanges away is cheaper and `isAllowed`,
 * move to the cheapest such. Two exchanges, not one, because one exchange of two tiles changes the
 * parity, so from a layout sliding can reach every single exchange leaves the reachable layouts, and
 * the beam search's pruning has already looked at those.
 */
export function refineLayout(
  costs: EdgeCosts,
  width: number,
  start: Layout,
  isAllowed: (tileAt: readonly number[]) => boolean,
): Layout {
  let current = start;
  for (;;) {
    let best = current;
    for (const tileAt of listNearLayouts(current.tileAt)) {
      const cost = priceLayout(costs, tileAt, width);
      if (cost < best.cost - 1e-9 && isAllowed(tileAt)) best = { ...current, tileAt, cost };
    }
    if (best === current) return current;
    current = best;
  }
}

/**
 * Whether sliding tiles can turn the photo into `solved` (cell index per position, -1 for the slot's
 * home): the photo, numbered by the goal position of each tile, must be a solvable board.
 */
function isReachable(solved: readonly number[], blank: number, size: GridSize): boolean {
  const home = solved.indexOf(-1);
  const goal = createGoalBoard(size, home).cells;
  const cells = new Array<number>(solved.length).fill(0);
  solved.forEach((cell, position) => {
    if (cell !== -1) cells[cell] = goal[position] ?? 0;
  });
  return isSolvable({ size, cells, blankHome: home });
}

/**
 * 0..1 per position: how much worse the layout gets if the tile there is exchanged with the tile of
 * another position, taking the exchange that hurts least (seams of the two positions only). A tile that
 * could swap places with a look-alike at no cost, or at a gain, scores 0, which is what the review
 * screen flags. (The price of the cheapest *replacement* by any other tile would not single out
 * near-tied pairs, which each look fine against the other's neighbours.)
 */
function measureConfidence(costs: EdgeCosts, tileAt: readonly number[], width: number): number[] {
  /** Cost of the seams between the tile at `position` and its placed neighbours in `layout`. */
  const priceSeams = (layout: readonly number[], position: number): number => {
    const tile = layout[position] ?? -1;
    if (tile === -1) return 0;
    const column = position % width;
    const neighbours: [number | undefined, "right" | "down", boolean][] = [
      [column > 0 ? layout[position - 1] : undefined, "right", true],
      [position >= width ? layout[position - width] : undefined, "down", true],
      [column < width - 1 ? layout[position + 1] : undefined, "right", false],
      [position + width < layout.length ? layout[position + width] : undefined, "down", false],
    ];
    let cost = 0;
    for (const [neighbour, direction, before] of neighbours) {
      if (neighbour === undefined || neighbour === -1) continue;
      cost +=
        (before ? costs[direction][neighbour]?.[tile] : costs[direction][tile]?.[neighbour]) ?? 0;
    }
    return cost;
  };
  return tileAt.map((tile, position) => {
    if (tile === -1) return 1;
    let margin = Number.POSITIVE_INFINITY;
    tileAt.forEach((other, otherPosition) => {
      if (other === -1 || otherPosition === position) return;
      const swapped = [...tileAt];
      swapped[position] = other;
      swapped[otherPosition] = tile;
      const change =
        priceSeams(swapped, position) +
        priceSeams(swapped, otherPosition) -
        priceSeams(tileAt, position) -
        priceSeams(tileAt, otherPosition);
      margin = Math.min(margin, change);
    });
    return Math.min(Math.max(margin / CERTAIN_SWAP_COST, 0), 1);
  });
}

// The plain layout stands unless the surprise scorer rejects it by more than this: the excess cost of
// the plain layout under the surprise scorer, as a share of how much worse than the surprise scorer's
// best a layout of the tiles in photo order is (a stand-in for a random layout, so that the share does
// not depend on the scorer's units). The floor is set by the owner's photo, the only real one: the
// surprise scorer alone (margin 0) assembles it wrongly, and the plain layout is right for every margin
// of 0.04 and more. Dev exact falls as the margin rises, so this is the lowest margin the photo allows.
// Dev exact (picture / angled-blank / white-frame / line-art, of 144 / 48 / 48 / 48, labelled corners, slot given):
// margin 0: 113 / 36 / 36 / 5; 0.04: 114 / 35 / 35 / 4; 0.05: 112 / 35 / 35 / 4; 0.06: 110 / 35 / 34 / 4; 0.07: 107 / 34 / 33 / 4;
// 0.08: 106 / 34 / 33 / 4; 0.09: 106 / 32 / 33 / 4.
const REJECT_MARGIN = 0.04;

/** Best reachable layouts of one scorer, locally refined, cheapest first (the cheapest overall if none is reachable). */
function findLayouts(
  costs: EdgeCosts,
  size: GridSize,
  isAllowed: (tileAt: readonly number[]) => boolean,
): Layout[] {
  const candidates = searchLayoutsPerHome(costs, size).flat();
  candidates.sort((a, b) => a.cost - b.cost);
  // The beam search keeps the cheapest layouts of every home, but only locally cheapest in raster order.
  const refined = candidates
    .filter((layout) => isAllowed(layout.tileAt))
    .slice(0, REFINED_CANDIDATES)
    .map((layout) => refineLayout(costs, size, layout, isAllowed))
    .sort((a, b) => a.cost - b.cost);
  return refined.length > 0 ? refined : candidates.slice(0, 1);
}

/**
 * Picks the plain scorer's layout unless the surprise scorer finds it clearly worse than its own, see
 * REJECT_MARGIN. Both scorers' best layouts are usually the same one.
 */
function chooseLayout(
  models: { plain: EdgeCosts; surprise: EdgeCosts },
  size: GridSize,
  isAllowed: (tileAt: readonly number[]) => boolean,
): Layout {
  const plain = findLayouts(models.plain, size, isAllowed)[0];
  const surprise = findLayouts(models.surprise, size, isAllowed)[0];
  if (!plain || !surprise) throw new Error("no layout found");
  // The tiles in photo order, with the empty slot first.
  const photoOrder = [-1, ...models.plain.right.keys()];
  const range = priceLayout(models.surprise, photoOrder, size) - surprise.cost;
  const rejection = (priceLayout(models.surprise, plain.tileAt, size) - surprise.cost) / range;
  return rejection > REJECT_MARGIN ? surprise : plain;
}

/**
 * Assembles the picture from `cells` (row-major photo cells, `blank` = the empty slot's cell index).
 * The empty slot's home is searched like everything else; among the layouts a slide sequence can reach,
 * the cheapest wins. If none of the best layouts is reachable the cheapest overall is returned.
 */
export function assemblePicture(cells: TileCrop[], size: GridSize, blank: number): Arrangement {
  if (cells.length !== size * size) throw new Error(`expected ${String(size * size)} cells`);
  if (!Number.isInteger(blank) || blank < 0 || blank >= cells.length)
    throw new Error(`empty slot ${String(blank)} is not a cell of the photo`);
  const cellOfTile = cells.map((_, cell) => cell).filter((cell) => cell !== blank);
  const models = computeEdgeCosts(cellOfTile.map((cell) => cells[cell] as TileCrop));
  const toSolved = (tileAt: readonly number[]): number[] =>
    tileAt.map((tile) => (tile === -1 ? -1 : (cellOfTile[tile] ?? -1)));
  const chosen = chooseLayout(models, size, (tileAt) => isReachable(toSolved(tileAt), blank, size));
  return {
    size,
    solved: toSolved(chosen.tileAt),
    confidence: measureConfidence(models.surprise, chosen.tileAt, size),
  };
}
