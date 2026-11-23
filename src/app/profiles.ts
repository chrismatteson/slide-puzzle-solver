// Saved picture puzzles in localStorage (or anything shaped like it). Everything here tolerates a
// store that is missing, full or holding junk: saving a puzzle is a convenience and must never stop
// the app.

import { GRID_SIZES } from "../puzzle/board.ts";
import { type PuzzleProfile } from "../picture/profile.ts";

export const STORAGE_KEY = "slide-puzzle-solver.profiles.v1";
/** A 4x4 profile is about 15 KB of JSON; five fit well inside the ~5 MB a browser allows. */
export const MAX_PROFILES = 5;
/** Base64 of a 16x16 RGB thumbnail; profile.ts rejects any other length when it matches. */
const THUMBNAIL_CHARS = 1024;

export type ProfileStore = Pick<Storage, "getItem" | "setItem" | "removeItem">;

function isProfile(value: unknown): value is PuzzleProfile {
  if (typeof value !== "object" || value === null) return false;
  const profile = value as Partial<PuzzleProfile>;
  const { size, blankHome, tiles } = profile;
  return (
    profile.version === 1 &&
    (GRID_SIZES as readonly unknown[]).includes(size) &&
    Number.isInteger(blankHome) &&
    typeof size === "number" &&
    typeof blankHome === "number" &&
    blankHome >= 0 &&
    blankHome < size * size &&
    Array.isArray(tiles) &&
    tiles.length === size * size - 1 &&
    tiles.every((tile) => typeof tile === "string" && tile.length === THUMBNAIL_CHARS)
  );
}

/** Saved puzzles, newest first; entries that do not look like profiles are dropped. */
export function loadProfiles(store: ProfileStore | undefined): PuzzleProfile[] {
  if (!store) return [];
  try {
    const parsed: unknown = JSON.parse(store.getItem(STORAGE_KEY) ?? "[]");
    return Array.isArray(parsed) ? parsed.filter(isProfile) : [];
  } catch {
    return [];
  }
}

/**
 * Keeps `profile` as the newest saved puzzle (dropping `replaces`, the profile it supersedes, and the
 * oldest beyond MAX_PROFILES). If the store refuses the write (quota), older puzzles are dropped one
 * by one until it fits. Returns whether the puzzle was saved.
 */
export function saveProfile(
  store: ProfileStore | undefined,
  profile: PuzzleProfile,
  replaces?: PuzzleProfile,
): boolean {
  if (!store) return false;
  const others = loadProfiles(store).filter(
    (saved) => saved.tiles.join() !== replaces?.tiles.join(),
  );
  let kept = [profile, ...others].slice(0, MAX_PROFILES);
  while (kept.length > 0) {
    try {
      store.setItem(STORAGE_KEY, JSON.stringify(kept));
      return true;
    } catch {
      kept = kept.slice(0, -1);
    }
  }
  return false;
}

export function forgetProfiles(store: ProfileStore | undefined): void {
  try {
    store?.removeItem(STORAGE_KEY);
  } catch {
    // Nothing more can be done about a store that will not even delete.
  }
}

/** The browser's localStorage, or undefined where it is blocked (some private modes throw on access). */
export function openBrowserStore(): ProfileStore | undefined {
  try {
    return window.localStorage;
  } catch {
    return undefined;
  }
}
