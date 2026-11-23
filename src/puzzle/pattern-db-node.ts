import { readFile } from "node:fs/promises";
import { type PatternTables, decodePatternTables } from "./pattern-db.ts";

/** Reads the shipped 6-tile table from disk (Node: tests, eval). The browser fetches it instead. */
export async function loadPatternTables(): Promise<PatternTables> {
  return decodePatternTables(await readFile(new URL("./pattern-db.bin.gz", import.meta.url)));
}
