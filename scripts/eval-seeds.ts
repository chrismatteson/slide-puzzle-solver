// Generates (if missing) and evaluates the held-out photo set for several seeds, one at a time so the
// timings are not distorted, and prints a per-seed table. Usage: tsx scripts/eval-seeds.ts [seed...]
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync, readdirSync } from "node:fs";

const DEFAULT_SEEDS = [20240601, 7, 99];
const COUNT = 240;

interface Summary {
  images: number;
  exactReadAccuracy: number | null; // null when the set has no solvable board
  falseAcceptRate: number;
  unsolvableCorrectlyRejected: number;
  unsolvableImages: number;
  detectionRate: number | null;
  tightDetectionRate: number | null;
  gridSizeAccuracy: number;
  perCellAccuracy: number;
  meanMillis: number;
}

function countPhotos(dir: string): number {
  return existsSync(dir) ? readdirSync(dir).filter((f) => f.endsWith(".jpg")).length : 0;
}

function evaluateSeed(seed: number): Summary {
  const dir = `data/eval-seed${String(seed)}`;
  const results = `data/eval-seed${String(seed)}-results.json`;
  if (countPhotos(dir) !== COUNT) {
    execFileSync(
      "bash",
      [
        "scripts/python-env.sh",
        "scripts/gen_photos.py",
        "--split",
        "eval",
        "--count",
        String(COUNT),
        "--seed",
        String(seed),
        "--out",
        dir,
      ],
      { stdio: "inherit" },
    );
  }
  execFileSync("npx", ["tsx", "scripts/eval.ts", dir, results], {
    stdio: ["ignore", "ignore", "inherit"], // stderr carries a bad label's file name and reason
  });
  const parsed = JSON.parse(readFileSync(results, "utf8")) as { summary: Summary };
  return parsed.summary;
}

function formatPercent(fraction: number | null): string {
  return fraction === null ? "n/a" : `${(100 * fraction).toFixed(1)}%`;
}

function main(): void {
  const seeds = process.argv.slice(2).map(Number);
  const rows = (seeds.length > 0 ? seeds : DEFAULT_SEEDS).map((seed) => ({
    seed,
    summary: evaluateSeed(seed),
  }));
  console.log(
    "| seed | images | exact-read | false-accept | unsolvable rejected | detection | frame-tight | grid-size | per-cell | mean ms/image |",
  );
  console.log(
    "| ---- | ------ | ---------- | ------------ | ------------------- | --------- | ----------- | --------- | -------- | ------------- |",
  );
  for (const { seed, summary: s } of rows) {
    console.log(
      `| ${String(seed)} | ${String(s.images)} | ${formatPercent(s.exactReadAccuracy)} | ${formatPercent(s.falseAcceptRate)} | ${String(s.unsolvableCorrectlyRejected)}/${String(s.unsolvableImages)} | ${formatPercent(s.detectionRate)} | ${formatPercent(s.tightDetectionRate)} | ${formatPercent(s.gridSizeAccuracy)} | ${formatPercent(s.perCellAccuracy)} | ${s.meanMillis.toFixed(0)} |`,
    );
  }
}

main();
