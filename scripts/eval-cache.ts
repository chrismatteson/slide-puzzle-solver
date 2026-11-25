import { createHash } from "node:crypto";
import { readdirSync } from "node:fs";

/**
 * The file name of an eval-assembly crop cache. It hashes the folder, the --corners flag and the text
 * of every source in `sources` (in the order given, which is sorted; see listCropSources), so a changed
 * constant or algorithm cannot be answered from crops cut by the old code.
 */
export function computeCacheKey(dir: string, corners: boolean, sources: string[]): string {
  const hash = createHash("sha256");
  // Length-prefixed, so ["ab", "c"] and ["a", "bc"] hash differently.
  for (const part of [dir, corners ? "corners" : "plain", ...sources])
    hash.update(`${String(part.length)}:${part}`);
  return hash.digest("hex");
}

/**
 * Every file the crops depend on, as sorted paths relative to the project root: all of src/ (the picture
 * reader and the vision code under it) and scripts/read-photo.ts, which decodes the photos. Hashing only
 * src/picture would answer from stale crops after a change to src/vision.
 */
export function listCropSources(): string[] {
  const inSrc = readdirSync("src", { recursive: true, encoding: "utf8" })
    .filter((file) => file.endsWith(".ts"))
    .map((file) => `src/${file.split("\\").join("/")}`);
  return [...inSrc, "scripts/read-photo.ts"].sort();
}
