import { type Board, type Move } from "../puzzle/board.ts";
import { type PuzzleProfile } from "../picture/profile.ts";
import { type RgbaImage } from "../vision/image.ts";
import { type PictureReading } from "../picture/types.ts";
import { type ReadOptions } from "../vision/read-board.ts";
import { type PhotoReading, type PictureArranged, type ReadMode } from "./routing.ts";

// Messages between the page and its two workers. Vision and solving have separate workers so that
// cancelling a long solve (terminating its worker) leaves the loaded classifier alone.

export type VisionRequest =
  | {
      type: "read";
      id: number;
      image: RgbaImage;
      mode: ReadMode;
      /** Saved picture puzzles to match a picture photo against first. */
      profiles: PuzzleProfile[];
      /** Corners and size the user set; absent for a first read. */
      options?: ReadOptions;
    }
  /** Arranges an already-read picture with the empty slot at the photo cell the user chose. */
  | { type: "reassemble"; id: number; reading: PictureReading; blank: number };
export type VisionResponse =
  | { type: "ready" }
  | { type: "read"; id: number; result: PhotoReading }
  | { type: "reassemble"; id: number; result: PictureArranged }
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
