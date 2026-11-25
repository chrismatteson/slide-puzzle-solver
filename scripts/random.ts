// Seeded randomness shared by the correction eval and the tests that move corners the way a finger would.

import { type Point, type Quad } from "../src/vision/geometry.ts";

/** Deterministic uniform [0, 1) numbers (mulberry32), so a rerun jitters the same corners the same way. */
export function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 2 ** 32;
  };
}

/** Mean length of the quad's four edges. */
export function measureSide(quad: Quad): number {
  let total = 0;
  for (let i = 0; i < 4; i++) {
    const a = quad[i] as Point;
    const b = quad[(i + 1) % 4] as Point;
    total += Math.hypot(a.x - b.x, a.y - b.y);
  }
  return total / 4;
}

/** Moves every corner by a random distance up to `share` of the quad's side, in a random direction. */
export function jitterQuad(quad: Quad, share: number, random: () => number): Quad {
  const reach = share * measureSide(quad);
  const nudgePoint = (p: Point): Point => {
    const angle = 2 * Math.PI * random();
    const distance = reach * Math.sqrt(random());
    return { x: p.x + distance * Math.cos(angle), y: p.y + distance * Math.sin(angle) };
  };
  return [nudgePoint(quad[0]), nudgePoint(quad[1]), nudgePoint(quad[2]), nudgePoint(quad[3])];
}
