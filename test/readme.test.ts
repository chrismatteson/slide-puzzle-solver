import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const readme = readFileSync("README.md", "utf8");
const scripts = (JSON.parse(readFileSync("package.json", "utf8")) as { scripts: object }).scripts;

/** Folders and files the README names that only exist after a command it describes has run. */
const GENERATED = [/^data\//, /^dist\//, /^e2e-artifacts\//, /^\/tmp\//];

const FILE_EXTENSIONS = /\.(ts|md|json|toml|py|sh|jpg|png|gz|mjs|html|txt|npz)$/;

function listBackticked(text: string): string[] {
  return [...text.matchAll(/`([^`\n]+)`/g)].map((m) => m[1] as string);
}

function looksLikePath(span: string): boolean {
  if (/[\s*<>${}()=:,]|:\/\//.test(span)) return false; // glob, placeholder, URL, code
  if (/^[-~.]/.test(span)) return false; // flag, home dir, or a bare extension like .npz
  return span.includes("/") || FILE_EXTENSIONS.test(span);
}

describe("README.md", () => {
  it("names only files that exist", () => {
    const missing = listBackticked(readme)
      .filter(looksLikePath)
      .filter((path) => !GENERATED.some((pattern) => pattern.test(path)))
      .filter((path) => !existsSync(path));
    expect(missing).toEqual([]);
  });

  it("names only npm scripts that exist", () => {
    const named = [...readme.matchAll(/npm run ([\w:-]+)/g)].map((m) => m[1] as string);
    expect(named.length).toBeGreaterThan(0);
    expect(named.filter((name) => !(name in scripts))).toEqual([]);
  });

  it("does not claim the shipped changes were chosen away from the held-out seed", () => {
    expect(readme.replace(/\s+/g, " ")).not.toMatch(
      /chosen[\s\S]{0,200}on dev numbers, not on held-out/,
    );
  });
});
