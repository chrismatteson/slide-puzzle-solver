// Solves the picture of a picture-tile puzzle from its tile crops: which tile belongs at which solved
// position, and where the empty slot belongs. Jigsaw-style: score every pair of tile edges (a
// prediction-free variant of Pomeranz 2011 / Gallagher 2012, see measureEdgeCost), normalise each score
// by the runner-up of its edge so that plain tiles score as "unsure", then search whole layouts with one
// beam search per candidate home of the empty slot.
//
// Synthetic picture eval (npm run gen:picture && npm run eval:picture, 144 photos), assembly alone
// (crops from the labelled tile area, labelled empty slot), exact / neighbours / home: 44.4% / 72.9% /
// 54.2% (one photo in 144 is 0.7%). The per-constant figures below are the same measure with only that
// constant changed. The beam keys on the permutation parity (without it the reachable layouts were
// pruned in favour of unreachable ones) and sides left open are priced by how well they would have
// matched (see OPEN_WEIGHT).

import { type GridSize, createGoalBoard, isSolvable } from "../puzzle/board.ts";
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
// Narrow enough to stay under 2 s for 4x4 in Node; wider beams found no better layouts on the dev set.
const BEAM_WIDTH = 300;
// A missing edge (next to the empty slot) counts as this many normalised units. Above 1 it favours homes
// with fewer neighbours (corners), as on bought puzzles; 1.3-2 scored the same on the dev set, 0.8 lost.
const MISSING_EDGE_COST = 1.3;
// Leaving a side open (picture border or next to the empty slot) costs this much times how well the
// side's best partner matches, so sides that obviously continue into another tile resist being borders.
// Synthetic set, exact: 44.4% at 1.5, 45.8% at 1 (two photos, within noise), 43.1% at 2.
const OPEN_WEIGHT = 1.5;
// What a doubt about a tile is worth, see measureConfidence: swapping it with another tile making the
// layout this many seam units worse counts as certain.
const CERTAIN_SWAP_COST = 4;

type Direction = "right" | "down";
type Side = "left" | "right" | "top" | "bottom";

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

/** The lines of one tile side at depths FIRST_DEPTH .. FIRST_DEPTH + DEPTH_COUNT - 1. */
function readEdges(crop: TileCrop, side: Side): Float64Array[] {
  return Array.from({ length: DEPTH_COUNT }, (_, i) => readEdgeLine(crop, side, FIRST_DEPTH + i));
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
function measureShiftedCost(first: Float64Array, second: Float64Array, shift: number): number {
  const start = Math.max(0, -shift * 3);
  const end = Math.min(first.length, first.length - shift * 3);
  const count = (end - start) / 3;
  const sum1 = [0, 0, 0];
  const sum2 = [0, 0, 0];
  const square1 = [0, 0, 0];
  const square2 = [0, 0, 0];
  for (let i = start; i < end; i++) {
    const a = first[i] ?? 0;
    const b = second[i + shift * 3] ?? 0;
    const c = i % 3;
    sum1[c] = (sum1[c] ?? 0) + a;
    sum2[c] = (sum2[c] ?? 0) + b;
    square1[c] = (square1[c] ?? 0) + a * a;
    square2[c] = (square2[c] ?? 0) + b * b;
  }
  const mean1 = sum1.map((v) => v / count);
  const mean2 = sum2.map((v) => v / count);
  const spread1 = square1.map(
    (v, c) => Math.sqrt(Math.max(v / count - (mean1[c] ?? 0) ** 2, 0)) + STRUCTURE_FLOOR,
  );
  const spread2 = square2.map(
    (v, c) => Math.sqrt(Math.max(v / count - (mean2[c] ?? 0) ** 2, 0)) + STRUCTURE_FLOOR,
  );
  let level = 0;
  let structure = 0;
  for (let i = start; i < end; i++) {
    const c = i % 3;
    const a = (first[i] ?? 0) - (mean1[c] ?? 0);
    const b = (second[i + shift * 3] ?? 0) - (mean2[c] ?? 0);
    level += Math.abs(a - b + (1 - OFFSET_DISCOUNT) * ((mean1[c] ?? 0) - (mean2[c] ?? 0)));
    structure += Math.abs(a / (spread1[c] ?? 1) - b / (spread2[c] ?? 1));
  }
  return (level + STRUCTURE_WEIGHT * structure) / (end - start) + SHIFT_PENALTY * Math.abs(shift);
}

/**
 * Dissimilarity of `first`'s edge against `second`'s facing edge: the best match over every pair of
 * depths and slides along the edge. Extrapolating the gradient across the gap (prediction based
 * dissimilarity proper) scored no better than comparing the lines directly on blurred photos.
 */
function measureEdgeCost(first: Float64Array[], second: Float64Array[]): number {
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

/** 0 for a side with no better than average partner, up to 1 for a perfect one. */
function openness(costs: number[]): number {
  return Math.max(1 - Math.min(...costs.filter(Number.isFinite)), 0);
}

/**
 * `costs[direction][a][b]`: how badly tile b fits next to tile a (b right of / below a), divided by the
 * runner-up cost of a's edge and of b's edge. About 1 for a wrong or ambiguous neighbour, below 1 for a
 * matching one.
 */
function computeEdgeCosts(tiles: TileCrop[]): EdgeCosts {
  const result = {
    right: [],
    down: [],
    open: { left: [], right: [], top: [], bottom: [] },
  } as EdgeCosts;
  for (const direction of ["right", "down"] as const) {
    const [outgoing, incoming] =
      direction === "right" ? (["right", "left"] as const) : (["bottom", "top"] as const);
    const from = tiles.map((tile) => readEdges(tile, outgoing));
    const to = tiles.map((tile) => readEdges(tile, incoming));
    const raw = from.map((a, i) =>
      to.map((b, j) => (i === j ? Number.POSITIVE_INFINITY : measureEdgeCost(a, b) + FLOOR)),
    );
    const rowRunnerUp = raw.map((row) => pickRunnerUp(row));
    const columnRunnerUp = to.map((_, j) => pickRunnerUp(raw.map((row) => row[j] ?? 0)));
    result[direction] = raw.map((row, i) =>
      row.map((cost, j) => cost / (((rowRunnerUp[i] ?? 1) + (columnRunnerUp[j] ?? 1)) / 2)),
    );
    // A side whose best partner matches well should have a neighbour: leaving it open is penalised.
    const [outSide, inSide] =
      direction === "right" ? (["right", "left"] as const) : (["bottom", "top"] as const);
    result.open[outSide] = result[direction].map((row) => openness(row));
    result.open[inSide] = to.map((_, j) => openness(result[direction].map((row) => row[j] ?? 1)));
  }
  return result;
}

interface EdgeCosts extends Record<Direction, number[][]> {
  /** `open[side][tile]`: extra cost of leaving that side of the tile without a neighbour. */
  open: Record<Side, number[]>;
}

interface Layout {
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
  if (left === -1) cost += MISSING_EDGE_COST + OPEN_WEIGHT * (costs.open.left[tile] ?? 0);
  else if (left >= 0) cost += costs.right[left]?.[tile] ?? 0;
  else cost += OPEN_WEIGHT * (costs.open.left[tile] ?? 0);
  if (above === -1) cost += MISSING_EDGE_COST + OPEN_WEIGHT * (costs.open.top[tile] ?? 0);
  else if (above >= 0) cost += costs.down[above]?.[tile] ?? 0;
  else cost += OPEN_WEIGHT * (costs.open.top[tile] ?? 0);
  // Right and bottom sides are only known open once the layout's last column or row is reached.
  if (column === width - 1) cost += OPEN_WEIGHT * (costs.open.right[tile] ?? 0);
  if (position >= width * (width - 1)) cost += OPEN_WEIGHT * (costs.open.bottom[tile] ?? 0);
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
  if (position % width > 0) cost += MISSING_EDGE_COST + OPEN_WEIGHT * (costs.open.right[left] ?? 0);
  if (position >= width) cost += MISSING_EDGE_COST + OPEN_WEIGHT * (costs.open.bottom[above] ?? 0);
  return cost;
}

/** Cheapest layouts with the empty slot's home at `home`: beam search in raster order. */
function searchLayouts(costs: EdgeCosts, width: number, home: number): Layout[] {
  const total = width * width;
  let beam: { layout: Layout; used: number }[] = [
    { layout: { tileAt: [], cost: 0, parity: 0 }, used: 0 },
  ];
  for (let position = 0; position < total; position++) {
    // Two partial layouts that use the same tiles and agree on the last `width` positions have the same
    // future costs, so only the cheaper one is kept.
    const best = new Map<string, { layout: Layout; used: number }>();
    for (const { layout, used } of beam) {
      const candidates =
        position === home ? [-1] : [...costs.right.keys()].filter((t) => (used & (1 << t)) === 0);
      for (const tile of candidates) {
        const cost =
          layout.cost +
          (tile === -1
            ? priceHome(costs, layout.tileAt, position, width)
            : priceTile(costs, layout.tileAt, position, tile, width));
        const tileAt = [...layout.tileAt, tile];
        const parity =
          tile === -1
            ? layout.parity
            : (layout.parity + layout.tileAt.filter((other) => other > tile).length) % 2;
        const key = `${String(used | (tile === -1 ? 0 : 1 << tile))}:${String(parity)}:${tileAt.slice(-width).join(",")}`;
        const known = best.get(key);
        if (known && known.layout.cost <= cost) continue;
        best.set(key, {
          layout: { tileAt, cost, parity },
          used: tile === -1 ? used : used | (1 << tile),
        });
      }
    }
    beam = [...best.values()].sort((a, b) => a.layout.cost - b.layout.cost).slice(0, BEAM_WIDTH);
  }
  return beam.map((entry) => entry.layout);
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
  const costs = computeEdgeCosts(cellOfTile.map((cell) => cells[cell] as TileCrop));
  const candidates: Layout[] = [];
  for (let home = 0; home < size * size; home++)
    candidates.push(...searchLayouts(costs, size, home));
  candidates.sort((a, b) => a.cost - b.cost);
  const toSolved = (layout: Layout): number[] =>
    layout.tileAt.map((tile) => (tile === -1 ? -1 : (cellOfTile[tile] ?? -1)));
  const chosen =
    candidates.find((layout) => isReachable(toSolved(layout), blank, size)) ?? candidates[0];
  if (!chosen) throw new Error("no layout found");
  return {
    size,
    solved: toSolved(chosen),
    confidence: measureConfidence(costs, chosen.tileAt, size),
  };
}
