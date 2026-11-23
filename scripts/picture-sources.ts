// Downloads the pictures listed in picture-sources.toml into data/pictures/ (idempotent, checks
// sha256). They feed scripts/gen_picture_photos.py. Usage: tsx scripts/picture-sources.ts fetch
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { parse } from "smol-toml";
import { download } from "./real-photos.ts";

const MANIFEST = "picture-sources.toml";
const OUT_DIR = "data/pictures";
const PAUSE_MS = 3000; // Wikimedia answers bursts with 429

interface Entry {
  name: string;
  url: string;
  sha256: string;
}

function readText(raw: Record<string, unknown>, key: string, where: string): string {
  const value = raw[key];
  if (typeof value !== "string" || value === "") throw new Error(`${where} has no ${key}`);
  return value;
}

function loadManifest(): Entry[] {
  const pictures: unknown = parse(readFileSync(MANIFEST, "utf8")).picture ?? [];
  if (!Array.isArray(pictures)) throw new Error(`${MANIFEST}: picture must be an array of tables`);
  return pictures.map((raw: unknown, index): Entry => {
    const where = `${MANIFEST}: picture #${String(index + 1)}`;
    if (typeof raw !== "object" || raw === null) throw new Error(`${where} is not a table`);
    const fields = raw as Record<string, unknown>; // checked as an object just above
    for (const key of ["page", "author", "license"]) readText(fields, key, where);
    if (!["dev", "eval"].includes(readText(fields, "split", where)))
      throw new Error(`${where}: split must be dev or eval`);
    const sha256 = readText(fields, "sha256", where);
    if (!/^[0-9a-f]{64}$/.test(sha256)) throw new Error(`${where}: sha256 must be 64 hex digits`);
    return { name: readText(fields, "name", where), url: readText(fields, "url", where), sha256 };
  });
}

const hashSha256 = (bytes: Uint8Array): string => createHash("sha256").update(bytes).digest("hex");

async function fetchPictures(entries: Entry[]): Promise<void> {
  mkdirSync(OUT_DIR, { recursive: true });
  let failed = 0;
  for (const entry of entries) {
    const path = join(OUT_DIR, `${entry.name}.jpg`);
    if (existsSync(path)) {
      if (hashSha256(readFileSync(path)) !== entry.sha256) {
        console.error(`${entry.name}: existing file does not match sha256 in ${MANIFEST}`);
        failed++;
      }
      continue;
    }
    try {
      const bytes = await download(entry.url);
      const actual = hashSha256(bytes);
      if (actual !== entry.sha256)
        throw new Error(`sha256 ${actual}, manifest says ${entry.sha256}`);
      writeFileSync(path, bytes);
      console.log(`${entry.name}: downloaded ${String(bytes.length)} bytes`);
    } catch (error) {
      console.error(`${entry.name}: ${error instanceof Error ? error.message : String(error)}`);
      failed++;
    }
    await new Promise((resolve) => setTimeout(resolve, PAUSE_MS));
  }
  if (failed > 0) throw new Error(`${String(failed)} picture(s) missing or wrong; see above`);
}

async function main(): Promise<void> {
  if (process.argv[2] !== "fetch") throw new Error("usage: tsx scripts/picture-sources.ts fetch");
  await fetchPictures(loadManifest());
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) await main();
