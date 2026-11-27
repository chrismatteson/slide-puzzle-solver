import { describe, expect, it } from "vitest";
import { computeCacheKey, listCropSources } from "../scripts/eval-cache.ts";

describe("computeCacheKey", () => {
  const sources = ["export const A = 1;", "export const B = 2;"];

  it("is the same for the same inputs", () => {
    expect(computeCacheKey("data/picture-dev", false, sources)).toBe(
      computeCacheKey("data/picture-dev", false, [...sources]),
    );
  });

  it("changes when a source file changes", () => {
    const changed = ["export const A = 1;", "export const B = 3;"];
    expect(computeCacheKey("data/picture-dev", false, changed)).not.toBe(
      computeCacheKey("data/picture-dev", false, sources),
    );
  });

  it("changes with the corners flag and with the folder", () => {
    const base = computeCacheKey("data/picture-dev", false, sources);
    expect(computeCacheKey("data/picture-dev", true, sources)).not.toBe(base);
    expect(computeCacheKey("data/other", false, sources)).not.toBe(base);
  });

  it("does not confuse where one source ends and the next begins", () => {
    expect(computeCacheKey("d", false, ["ab", "c"])).not.toBe(
      computeCacheKey("d", false, ["a", "bc"]),
    );
  });
});

describe("listCropSources", () => {
  it("lists everything the crops depend on, sorted", () => {
    const sources = listCropSources();
    expect(sources).toContain("src/vision/grid.ts");
    expect(sources).toContain("src/picture/read.ts");
    expect(sources).toContain("scripts/read-photo.ts");
    expect(sources).toEqual([...sources].sort());
  });
});
