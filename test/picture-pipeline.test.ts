import { readFileSync } from "node:fs";
import jpeg from "jpeg-js";
import { describe, expect, it } from "vitest";
import { assemblePicture } from "../src/picture/assemble.ts";
import { toBoard } from "../src/picture/arrangement.ts";
import { readPictureBoard } from "../src/picture/read.ts";
import { applyMoves, isGoalBoard } from "../src/puzzle/board.ts";
import { loadPatternTables } from "../src/puzzle/pattern-db-node.ts";
import { solve } from "../src/puzzle/solver.ts";

const tables = await loadPatternTables();
const label = JSON.parse(
  readFileSync(new URL("fixtures/owner-picture/picture-4x4-01.json", import.meta.url), "utf8"),
) as { blank: number; solved: number[] };

describe("the owner's photo, fully automatic (no hand label)", () => {
  it("reads, assembles and solves in the optimal 24 moves", () => {
    const decoded = jpeg.decode(
      readFileSync(new URL("fixtures/owner-picture/picture-4x4-01.jpg", import.meta.url)),
      { useTArray: true, formatAsRGBA: true },
    );
    const reading = readPictureBoard({
      width: decoded.width,
      height: decoded.height,
      data: new Uint8ClampedArray(decoded.data),
    });
    if ("status" in reading) throw new Error(`no board: ${reading.message}`);

    const arrangement = assemblePicture(reading.cells, reading.size, reading.blank);
    expect(reading.blank).toBe(label.blank);
    expect(arrangement.solved).toEqual(label.solved);
    const board = toBoard(reading, arrangement);
    const result = solve(board, { tables });

    if (result.status !== "solved") throw new Error(`not solved: ${result.status}`);
    expect(result.notGuaranteedShortest).toBeUndefined();
    expect(result.moves).toHaveLength(24);
    expect(isGoalBoard(applyMoves(board, result.moves))).toBe(true);
  }, 30_000);
});
