import { type Board } from "./board.ts";

// Additive 6-6-3 pattern database for the 4x4 puzzle (Korf & Felner 2002). Each table holds, for
// every placement of a group of tiles, the fewest moves of *those tiles* that bring them home;
// groups are disjoint, so their sum never overestimates the true distance.
//
// The groups are chosen to be mirror images across the main diagonal: the six tiles above it
// (UPPER_TILES), the six below it (their transposes) and the diagonal three (DIAGONAL_TILES).
// Mirroring means one 6-tile table serves both six-tile groups, so only one file is shipped.
//
// Tables are indexed directly by packing each tile's position into 4 bits (tile i of the group in
// bits 4i..4i+3). That wastes the entries where two tiles share a cell (a third of 16^6), but it
// lets the search update the index with one addition per move instead of ranking a permutation.
// Building the 6-tile table takes ~8 s and ~100 MB, too much for a phone page, so
// `scripts/build-pattern-db.ts` builds it once and `pattern-db.bin.gz` (2.4 MB) is shipped.

export const UPPER_TILES: readonly number[] = [2, 3, 4, 7, 8, 12];
export const DIAGONAL_TILES: readonly number[] = [1, 6, 11];

const NOT_REACHED = 255;
const CELLS = 16;

export interface PatternTables {
  /** Indexed by the packed positions of UPPER_TILES (or of their mirror images). */
  corner: Uint8Array;
  /** Indexed by the packed positions of DIAGONAL_TILES. */
  diagonal: Uint8Array;
  /** Lower bound on the moves needed to solve `board`; a 4x4 board only. */
  estimateDistance(board: Board): number;
}

/** Position of the cell mirrored across the main diagonal. */
export function mirrorPosition(position: number): number {
  return (position & 3) * 4 + (position >> 2);
}

/** The tile that takes `tile`'s place when the whole board is mirrored. */
export function mirrorTile(tile: number): number {
  return mirrorPosition(tile - 1) + 1;
}

function listNeighbours(position: number): number[] {
  const row = position >> 2;
  const col = position & 3;
  const neighbours: number[] = [];
  if (row > 0) neighbours.push(position - 4);
  if (row < 3) neighbours.push(position + 4);
  if (col > 0) neighbours.push(position - 1);
  if (col < 3) neighbours.push(position + 1);
  return neighbours;
}

function growQueue(queue: Int32Array): Int32Array {
  const bigger = new Int32Array(queue.length * 2);
  bigger.set(queue);
  return bigger;
}

/**
 * Breadth-first search backwards from the goal over (tile placement, blank position) states,
 * where only a move of one of `tiles` costs 1 (a 0-1 BFS, run level by level). The result is
 * indexed by packed tile positions and holds the least cost over all blank positions.
 */
export function buildPatternTable(tiles: readonly number[]): Uint8Array {
  const count = tiles.length;
  const table = new Uint8Array(1 << (4 * count)).fill(NOT_REACHED);
  // One bit per (placement, blank) state; a state is 4 bits of blank under the packed placement.
  const visited = new Uint32Array((table.length * CELLS) >>> 5);
  const neighbours = Array.from({ length: CELLS }, (_, p) => listNeighbours(p));
  const claim = (state: number): boolean => {
    const word = state >>> 5;
    const bit = 1 << (state & 31);
    if (((visited[word] ?? 0) & bit) !== 0) return false;
    visited[word] = (visited[word] ?? 0) | bit;
    return true;
  };

  let goal = 0;
  tiles.forEach((tile, i) => {
    goal |= (tile - 1) << (4 * i);
  });
  let current: Int32Array = new Int32Array(1 << 16);
  let currentLength = 0;
  let next: Int32Array = new Int32Array(1 << 16);
  let nextLength = 0;
  const startState = (goal << 4) | (CELLS - 1);
  claim(startState);
  current[currentLength++] = startState;
  table[goal] = 0;

  const slotAt = new Int8Array(CELLS);
  for (let level = 0; currentLength > 0; level++) {
    // `current` grows while scanned: free moves stay on this level.
    for (let i = 0; i < currentLength; i++) {
      const state = current[i] ?? 0;
      const blank = state & 15;
      const placement = state >>> 4;
      slotAt.fill(-1);
      for (let slot = 0; slot < count; slot++) slotAt[(placement >>> (4 * slot)) & 15] = slot;
      for (const to of neighbours[blank] ?? []) {
        const slot = slotAt[to] ?? -1;
        if (slot < 0) {
          const moved = (placement << 4) | to;
          if (claim(moved)) {
            if (currentLength === current.length) current = growQueue(current);
            current[currentLength++] = moved;
          }
        } else {
          const placed = (placement & ~(15 << (4 * slot))) | (blank << (4 * slot));
          const moved = (placed << 4) | to;
          // Not claimed yet: a free move may still reach this state at the current level.
          if (((visited[moved >>> 5] ?? 0) & (1 << (moved & 31))) === 0) {
            if (nextLength === next.length) next = growQueue(next);
            next[nextLength++] = moved;
          }
        }
      }
    }
    let kept = 0;
    for (let i = 0; i < nextLength; i++) {
      const state = next[i] ?? 0;
      if (!claim(state)) continue;
      const placement = state >>> 4;
      if (table[placement] === NOT_REACHED) table[placement] = level + 1;
      next[kept++] = state;
    }
    [current, next] = [next, current];
    currentLength = kept;
    nextLength = 0;
  }
  return table;
}

/** Packs positions of `tiles` as they sit on `board`, optionally as seen on the mirrored board. */
function packPlacement(board: Board, tiles: readonly number[], mirrored: boolean): number {
  let packed = 0;
  tiles.forEach((tile, slot) => {
    const sought = mirrored ? mirrorTile(tile) : tile;
    const position = board.cells.indexOf(sought);
    packed |= (mirrored ? mirrorPosition(position) : position) << (4 * slot);
  });
  return packed;
}

/** Builds the full set from the shipped 6-tile table; the 3-tile table is small enough to build here. */
export function createPatternTables(corner: Uint8Array): PatternTables {
  if (corner.length !== 1 << 24)
    throw new Error(`corner table has ${String(corner.length)} entries`);
  const diagonal = buildPatternTable(DIAGONAL_TILES);
  return {
    corner,
    diagonal,
    estimateDistance(board) {
      if (board.size !== 4) throw new Error("pattern database is for 4x4 boards");
      return (
        (corner[packPlacement(board, UPPER_TILES, false)] ?? 0) +
        (corner[packPlacement(board, UPPER_TILES, true)] ?? 0) +
        (diagonal[packPlacement(board, DIAGONAL_TILES, false)] ?? 0)
      );
    },
  };
}

/**
 * Decodes the shipped `pattern-db.bin.gz` as fetched or read from disk. Some static hosts serve
 * `.gz` files with `Content-Encoding: gzip`, so the browser may already have unzipped them; the
 * gzip magic bytes say which case this is.
 */
export async function decodePatternTables(bytes: Uint8Array<ArrayBuffer>): Promise<PatternTables> {
  const isGzip = bytes[0] === 0x1f && bytes[1] === 0x8b;
  if (!isGzip) return createPatternTables(bytes);
  const unzipped = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  return createPatternTables(new Uint8Array(await new Response(unzipped).arrayBuffer()));
}
