// Real-photo evaluation set: `fetch` downloads the photos listed in real-photos.toml into data/real/
// (idempotent, checks sha256), `labels` writes a `name.json` label next to each downloaded photo.
// Usage: tsx scripts/real-photos.ts fetch|labels   (see `npm run real:fetch`, `npm run real:eval`)
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { parse } from "smol-toml";
import { parseBoardLabel } from "./labels.ts";

const MANIFEST = "real-photos.toml";
const OUT_DIR = "data/real";
// Wikimedia asks for an identifying User-Agent and answers 429 to bursts, so go slowly and back off.
const USER_AGENT =
  "slide-puzzle-solver-eval/1.0 (held-out real photo test set; see real-photos.toml)";
const PAUSE_MS = 3000;
const MAX_ATTEMPTS = 6;
const DOWNLOAD_TIMEOUT_MS = 60_000; // per request; a stalled host must not hang the command

interface Entry {
  name: string;
  page: string;
  url: string;
  author: string;
  license: string;
  sha256: string;
  size: 3 | 4;
  state: number[];
}

function readText(raw: Record<string, unknown>, key: string, where: string): string {
  const value = raw[key];
  if (typeof value !== "string" || value === "") throw new Error(`${where} has no ${key}`);
  return value;
}

function loadManifest(): Entry[] {
  // Hand-edited file: every field is checked before use.
  const photos: unknown = parse(readFileSync(MANIFEST, "utf8")).photo ?? [];
  if (!Array.isArray(photos)) throw new Error(`${MANIFEST}: photo must be an array of tables`);
  return photos.map((raw: unknown, index): Entry => {
    const where = `${MANIFEST}: photo #${String(index + 1)}`;
    if (typeof raw !== "object" || raw === null) throw new Error(`${where} is not a table`);
    const fields = raw as Record<string, unknown>; // checked as an object just above
    const sha256 = readText(fields, "sha256", where);
    if (!/^[0-9a-f]{64}$/.test(sha256)) throw new Error(`${where}: sha256 must be 64 hex digits`);
    const { size, state } = parseBoardLabel({ size: fields.size, state: fields.state }, where);
    return {
      name: readText(fields, "name", where),
      page: readText(fields, "page", where),
      url: readText(fields, "url", where),
      author: readText(fields, "author", where),
      license: readText(fields, "license", where),
      sha256,
      size,
      state,
    };
  });
}

function findExtension(url: string): string {
  const match = /\.(jpe?g|png)$/i.exec(new URL(url).pathname);
  if (!match) throw new Error(`${url}: not a .jpg/.jpeg/.png URL`);
  return (match[1] ?? "").toLowerCase();
}

const buildPhotoPath = (e: Entry): string => join(OUT_DIR, `${e.name}.${findExtension(e.url)}`);
const hashSha256 = (bytes: Uint8Array): string => createHash("sha256").update(bytes).digest("hex");
const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

/** Downloads `url`, retrying on 429; each request is aborted after `timeoutMs`. */
export async function download(
  url: string,
  timeoutMs = DOWNLOAD_TIMEOUT_MS,
  fetchImpl: typeof fetch = fetch,
): Promise<Uint8Array> {
  for (let attempt = 1; ; attempt++) {
    let response: Response;
    try {
      // The signal also covers reading the body, so a stalled transfer is caught too.
      response = await fetchImpl(url, {
        headers: { "User-Agent": USER_AGENT },
        signal: AbortSignal.timeout(timeoutMs),
      });
      if (response.ok) return new Uint8Array(await response.arrayBuffer());
    } catch (error) {
      if (error instanceof DOMException && error.name === "TimeoutError")
        throw new Error(`${url}: no response within ${String(timeoutMs)} ms`);
      throw error;
    }
    if (response.status !== 429 || attempt === MAX_ATTEMPTS) {
      throw new Error(`${url}: HTTP ${String(response.status)}`);
    }
    await sleep(30_000 * attempt);
  }
}

async function fetchPhotos(entries: Entry[]): Promise<void> {
  mkdirSync(OUT_DIR, { recursive: true });
  let failed = 0;
  for (const e of entries) {
    const path = buildPhotoPath(e);
    if (existsSync(path)) {
      if (hashSha256(readFileSync(path)) !== e.sha256) {
        console.error(`${e.name}: existing file does not match sha256 in ${MANIFEST}`);
        failed++;
      }
      continue;
    }
    try {
      const bytes = await download(e.url);
      const actual = hashSha256(bytes);
      if (actual !== e.sha256) throw new Error(`sha256 ${actual}, manifest says ${e.sha256}`);
      writeFileSync(path, bytes);
      console.log(`${e.name}: downloaded ${String(bytes.length)} bytes`);
    } catch (error) {
      console.error(`${e.name}: ${error instanceof Error ? error.message : String(error)}`);
      failed++;
    }
    await sleep(PAUSE_MS);
  }
  if (failed > 0) throw new Error(`${String(failed)} photo(s) missing or wrong; see above`);
}

function writeLabels(entries: Entry[]): void {
  for (const e of entries) {
    if (!existsSync(buildPhotoPath(e)))
      throw new Error(`${e.name}: not downloaded; run npm run real:fetch`);
    writeFileSync(
      join(OUT_DIR, `${e.name}.json`),
      JSON.stringify({ size: e.size, state: e.state }),
    );
  }
  console.log(`wrote ${String(entries.length)} labels to ${OUT_DIR}`);
}

async function main(): Promise<void> {
  const command = process.argv[2];
  const entries = loadManifest();
  if (command === "fetch") await fetchPhotos(entries);
  else if (command === "labels") writeLabels(entries);
  else throw new Error("usage: tsx scripts/real-photos.ts fetch|labels");
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) await main();
