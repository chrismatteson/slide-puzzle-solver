import { describe, expect, it } from "vitest";
import { formatBars, type Outcome } from "../scripts/eval-corrections.ts";

/** An outcome whose corner paths are right or wrong as given; the other fields do not enter the bars. */
function createOutcome(
  set: string,
  corners: boolean,
  cornersJittered: boolean,
  autoExact = false,
): Outcome {
  return {
    set,
    name: "photo",
    size: 3,
    autoExact,
    detectionRight: true,
    blankRight: true,
    blankOnly: false,
    corners: { foundBlank: corners, givenBlank: corners },
    cornersJittered: { foundBlank: cornersJittered, givenBlank: cornersJittered },
    rawJittered: { foundBlank: false, givenBlank: false },
  };
}

/** `right` of 100 photos exact with the labelled corners and `rightJittered` of 100 with jittered ones. */
function createLine(set: string, right: number, rightJittered: number): Outcome[] {
  return Array.from({ length: 100 }, (_, i) => createOutcome(set, i < right, i < rightJittered));
}

const bar3 = (outcomes: Outcome[], set: string): string =>
  formatBars(outcomes)
    .split("\n")
    .find((line) => line.startsWith(`3. ${set} `)) ?? "";

describe("formatBars, bar 3 (jitter gap)", () => {
  it("passes when the jittered rate is a few points below the labelled one", () => {
    expect(bar3(createLine("line-art", 80, 75), "line-art")).toMatch(/PASS$/);
  });

  it("fails when the jittered rate is more than 10 points below the labelled one", () => {
    expect(bar3(createLine("line-art", 80, 60), "line-art")).toMatch(/FAIL$/);
  });

  it("fails when the jittered rate is 12 points above the labelled one: not within 10 points", () => {
    const line = bar3(createLine("line-art", 60, 72), "line-art");
    expect(line).toContain("jitter gap -12.0 points");
    expect(line).toMatch(/FAIL$/);
  });
});
