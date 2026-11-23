// Builds src/puzzle/pattern-db.bin.gz, the 6-tile pattern table (~10 s, ~100 MB of memory).
// Usage: tsx scripts/build-pattern-db.ts
import { writeFileSync } from "node:fs";
import { gzipSync } from "node:zlib";
import { UPPER_TILES, buildPatternTable } from "../src/puzzle/pattern-db.ts";

const start = performance.now();
const table = buildPatternTable(UPPER_TILES);
const out = new URL("../src/puzzle/pattern-db.bin.gz", import.meta.url);
writeFileSync(out, gzipSync(table, { level: 9 }));
console.log(`wrote ${out.pathname} in ${((performance.now() - start) / 1000).toFixed(1)} s`);
