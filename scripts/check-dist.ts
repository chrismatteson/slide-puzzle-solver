// Post-build check (run by checks.sh after `npm run build`): dist/ must carry what the app needs at
// runtime and nothing from test/. The pattern table and classifier weights are committed, so the
// deploy host needs no Python and no generated data; the owner's photos under test/fixtures must
// never be published.
import { createHash } from "node:crypto";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";

function files(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) =>
    entry.isDirectory() ? files(join(dir, entry.name)) : [join(dir, entry.name)],
  );
}
const sha = (path: string) => createHash("sha256").update(readFileSync(path)).digest("hex");

const dist = files("dist");
const problems: string[] = [];

// Vite hashes the name (pattern-db.bin-<hash>.gz); the table is ~2.4 MB, so a stub would be caught too.
const tables = dist.filter((f) => /pattern-db\.bin-.*\.gz$/.test(f));
if (tables.length !== 1 || statSync(tables[0] ?? "").size < 2_000_000) {
  problems.push("dist/ has no pattern table (pattern-db.bin-*.gz, at least 2 MB)");
}

// The classifier weights are inlined in the vision worker bundle as an object literal.
const workers = dist.filter((f) => /vision\.worker-.*\.js$/.test(f));
if (workers.length !== 1 || !readFileSync(workers[0] ?? "", "utf8").includes("convs:[")) {
  problems.push("dist/ has no vision worker carrying the classifier weights");
}

// Match by content as well as by name, so a renamed or hashed copy of a fixture is still caught.
const fixtureHashes = new Set(files("test").map(sha));
for (const f of dist) {
  if (/(^|[\\/])(test|fixtures|owner-picture)([\\/]|$)/.test(f) || fixtureHashes.has(sha(f))) {
    problems.push(`${f} comes from test/`);
  }
}

if (problems.length > 0) {
  console.error(`check-dist: FAILED\n  ${problems.join("\n  ")}`);
  process.exit(1);
}
console.log(`check-dist: ok (${String(dist.length)} files, table ${tables.join(", ")})`);
