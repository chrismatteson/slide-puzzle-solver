// Decides, for a photo, whether it is a numbered puzzle or a picture puzzle, and reads it that way.
// Runs in the vision worker (both readers take 1-3 s).

import { MIN_CELL_CONFIDENCE } from "../puzzle/consistency.ts";
import { assemblePicture } from "../picture/assemble.ts";
import { readPictureBoard, setBlank } from "../picture/read.ts";
import { findBestProfileMatch, type PuzzleProfile } from "../picture/profile.ts";
import { type Arrangement, type PictureReading } from "../picture/types.ts";
import { defaultWeights } from "../vision/default-weights.ts";
import { type RgbaImage } from "../vision/image.ts";
import { type ReadOptions, type ReadResult, readBoard } from "../vision/read-board.ts";

export { type ReadOptions };

/** What the user asked for: `auto` lets the numbered reader's result decide. */
export type ReadMode = "auto" | "numbers" | "picture";

export type PhotoReading =
  | { kind: "numbers"; result: ReadResult }
  | {
      kind: "picture";
      reading: PictureReading;
      arrangement: Arrangement;
      /** Set when the arrangement comes from a saved puzzle instead of being assembled. */
      recognised: boolean;
    }
  | { kind: "no-picture"; message: string };

/** A reading known to be a picture one, such as what re-arranging a picture reading gives. */
export type PictureArranged = Extract<PhotoReading, { kind: "picture" }>;

/**
 * Smallest share of cells the numbered reader must be sure of (MIN_CELL_CONFIDENCE) for a rejected
 * numbered reading to still be treated as a numbered board. Measured on this repo's fixtures
 * (2026-11-23): the owner's picture puzzle reads 1 of 16 cells confidently (0.06), mean confidence
 * 0.77; the numbered photo that is rejected only for low confidence (photo-lowconf) reads 4 of 9
 * (0.44), mean 0.79. The means are too alike to tell them apart, the share is not; one third sits
 * between. A wrong call is one tap to undo: both review screens have a switch to the other reading.
 */
export const MIN_CONFIDENT_SHARE = 1 / 3;

/**
 * Is `result` a board of numbers? An accepted reading always is. A rejected one is when the reader
 * was sure of enough cells that the rejection is about a few doubtful digits; if it was unsure of
 * nearly all of them there were probably no digits to read. No board at all is not numbered.
 */
export function isNumberedReading(result: ReadResult): boolean {
  if (result.status === "accepted") return true;
  if (result.status === "no-board") return false;
  const { cells } = result.attempt;
  const sure = cells.filter((cell) => cell.confidence >= MIN_CELL_CONFIDENCE).length;
  return sure >= cells.length * MIN_CONFIDENT_SHARE;
}

function readAsPicture(
  image: RgbaImage,
  profiles: readonly PuzzleProfile[],
  options: ReadOptions,
): PhotoReading | undefined {
  const reading = readPictureBoard(image, options);
  if ("status" in reading) return undefined;
  const saved = findBestProfileMatch(reading, profiles);
  // The saved puzzle knows which cell is empty (the user may have chosen it); the reader's guess may not.
  return saved
    ? {
        kind: "picture",
        reading: saved.blank === reading.blank ? reading : setBlank(reading, saved.blank),
        arrangement: saved.arrangement,
        recognised: true,
      }
    : {
        kind: "picture",
        reading,
        arrangement: assemblePicture(reading.cells, reading.size, reading.blank),
        recognised: false,
      };
}

/**
 * Arranges `reading`'s cells with the empty slot at photo cell `blank`, as the user chose. Saved puzzles
 * are not consulted: the user has just said what the photo shows.
 */
export function reassemblePicture(reading: PictureReading, blank: number): PictureArranged {
  const chosen = setBlank(reading, blank);
  return {
    kind: "picture",
    reading: chosen,
    arrangement: assemblePicture(chosen.cells, chosen.size, blank),
    recognised: false,
  };
}

export function readPhoto(
  image: RgbaImage,
  mode: ReadMode,
  profiles: readonly PuzzleProfile[],
  options: ReadOptions = {},
): PhotoReading {
  if (mode === "picture") {
    return (
      readAsPicture(image, profiles, options) ?? {
        kind: "no-picture",
        message: "No board-like shape was found in the photo.",
      }
    );
  }
  const numbers = readBoard(image, defaultWeights, options);
  if (mode === "numbers" || isNumberedReading(numbers)) return { kind: "numbers", result: numbers };
  // Not numbered and no picture board either: the numbered reader's message is the better one.
  return readAsPicture(image, profiles, options) ?? { kind: "numbers", result: numbers };
}

/**
 * What a review remembers of the read that made it: the corners the user placed, which a re-read of
 * the photo keeps. A read the app did alone leaves nothing to keep.
 */
export function keepUserCorners(options: ReadOptions): ReadOptions | undefined {
  return options.quad ? options : undefined;
}
