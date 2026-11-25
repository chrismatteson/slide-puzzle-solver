// Dev eval of assembly alone, for tuning: crops come from a labelled photo set, assemblePicture runs on
// them, and each photo prints one line "name exact|miss ms". Tile crops are cached in data/crop-cache
// (one file per set and per --corners, which cuts different crops), since decoding the photos costs more
// than a short assembly.
//  - picture sets (scripts/gen_picture_photos.py): crops cut from the labelled tile area, labelled slot;
//  - correction sets (scripts/gen_correction_photos.py): the app's corner path
//    (readPictureBoard(image, { quad, size })), and the slot is the labelled one with `--given`, else
//    the one the app finds. The arrangement then goes through checkArrangement, as in the app.
// Usage: tsx scripts/eval-assembly.ts <dir> [--given] [--corners] [--cache-only] [--shard i/n]
//   --corners: cut the crops through the app's corner path (as for correction sets) instead of straight from
//   the labelled quad; --cache-only: only build the crop cache, print nothing.
//   bash: for i in 0 1 2 3; do tsx scripts/eval-assembly.ts data/picture-dev --shard $i/4 & done | sort
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { basename, join } from "node:path";
import { computeCacheKey, listCropSources } from "./eval-cache.ts";
import { checkArrangement } from "../src/picture/arrangement.ts";
import { assemblePicture } from "../src/picture/assemble.ts";
import { cutTileCrops, readPictureBoard } from "../src/picture/read.ts";
import { type Quad } from "../src/vision/geometry.ts";
import { type PictureReading, TILE_SIDE, type TileCrop } from "../src/picture/types.ts";
import { parsePictureLabel } from "./labels.ts";
import { readPhoto } from "./read-photo.ts";

interface Case {
  size: 3 | 4;
  blank: number;
  foundBlank: number;
  solved: number[];
  cells: string[]; // base64 of each crop's RGB
}

// checkArrangement reads cells and blank only, and the crop cache does not keep the quad.
const ORIGIN = { x: 0, y: 0 };
const NO_QUAD: Quad = [ORIGIN, ORIGIN, ORIGIN, ORIGIN];

function findPhotos(dir: string): string[] {
  const names: string[] = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    if (entry.isDirectory()) names.push(...findPhotos(join(dir, entry.name)));
    else if (entry.name.endsWith(".json")) names.push(join(dir, entry.name.slice(0, -5)));
  }
  return names.sort();
}

function buildCase(path: string, corners: boolean): Case {
  const label = parsePictureLabel(JSON.parse(readFileSync(`${path}.json`, "utf8")), path);
  const image = readPhoto(`${path}.jpg`);
  if (!corners) {
    const crops = cutTileCrops(image, label.tilesQuad, label.size);
    return {
      size: label.size,
      blank: label.blank,
      foundBlank: label.blank,
      solved: label.solved,
      cells: crops.map((c) => Buffer.from(c.rgb).toString("base64")),
    };
  }
  const reading = readPictureBoard(image, { quad: label.tilesQuad, size: label.size });
  if ("status" in reading) throw new Error(`${path}: no board`);
  return {
    size: label.size,
    blank: label.blank,
    foundBlank: reading.blank,
    solved: label.solved,
    cells: reading.cells.map((c) => Buffer.from(c.rgb).toString("base64")),
  };
}

function main(): void {
  const args = process.argv.slice(2);
  const dir = args.find((a) => !a.startsWith("--") && !/^\d+\/\d+$/.test(a)) ?? "data/picture-dev";
  const given = args.includes("--given");
  const shardArg = args[args.indexOf("--shard") + 1];
  const [shard, shards] = args.includes("--shard")
    ? (shardArg ?? "0/1").split("/").map(Number)
    : [0, 1];
  const corners = args.includes("--corners");
  const cacheDir = "data/crop-cache";
  mkdirSync(cacheDir, { recursive: true });
  const sources = listCropSources().map((file) => readFileSync(file, "utf8"));
  const cacheFile = join(
    cacheDir,
    `${basename(dir)}-${computeCacheKey(dir, corners, sources)}.json`,
  );
  let cases: Record<string, Case> = {};
  if (existsSync(cacheFile))
    cases = JSON.parse(readFileSync(cacheFile, "utf8")) as Record<string, Case>;
  else {
    for (const path of findPhotos(dir)) cases[path] = buildCase(path, corners);
    writeFileSync(cacheFile, JSON.stringify(cases));
  }
  if (args.includes("--cache-only")) return;
  Object.entries(cases).forEach(([path, c], index) => {
    if (index % (shards ?? 1) !== (shard ?? 0)) return;
    const cells: TileCrop[] = c.cells.map((b) => ({
      side: TILE_SIDE,
      rgb: new Uint8ClampedArray(Buffer.from(b, "base64")),
    }));
    const blank = given ? c.blank : c.foundBlank;
    const started = performance.now();
    const found = assemblePicture(cells, c.size, blank);
    const millis = performance.now() - started;
    const reading: PictureReading = {
      size: c.size,
      quad: NO_QUAD,
      cells,
      blank,
      blankConfidence: 1,
    };
    const checked = corners ? checkArrangement(reading, found) : undefined;
    const solved =
      checked && (checked.status === "ok" || checked.status === "fixed")
        ? checked.arrangement.solved
        : found.solved;
    const exact = solved.every((t, p) => t === c.solved[p]);
    console.log(
      `${path.replace(`${dir}/`, "")} ${String(c.size)} ${exact ? "exact" : "miss"} ${millis.toFixed(0)}`,
    );
  });
}

main();
