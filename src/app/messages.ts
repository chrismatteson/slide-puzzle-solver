import { type Board, type Move } from "../puzzle/board.ts";
import { type ReadResult } from "../vision/read-board.ts";
import { type RgbaImage } from "../vision/image.ts";

// Messages between the page and its two workers. Vision and solving have separate workers so that
// cancelling a long solve (terminating its worker) leaves the loaded classifier alone.

export type VisionRequest = { type: "read"; id: number; image: RgbaImage };
export type VisionResponse =
  | { type: "ready" }
  | { type: "read"; id: number; result: ReadResult }
  | { type: "error"; id: number; message: string };

export type SolveRequest =
  /** The still-compressed pattern database, fetched once by the page so a restarted worker needs no network. */
  { type: "init"; patternDb: ArrayBuffer } | { type: "solve"; id: number; board: Board };

export type SolveOutcome =
  | { status: "solved"; moves: Move[]; notGuaranteedShortest: boolean; millis: number }
  | { status: "unsolvable" };
export type SolveResponse =
  | { type: "ready" }
  | { type: "solve"; id: number; outcome: SolveOutcome }
  | { type: "error"; id?: number; message: string };
