import { describe, expect, it } from "vitest";
import { describeRun } from "../scripts/eval-corrections.ts";

describe("describeRun", () => {
  it("records the arguments the run was given and its stride", () => {
    expect(
      describeRun(["node", "scripts/eval-corrections.ts", "data/corrections", "out.json", "4"], 4),
    ).toEqual({ command: "scripts/eval-corrections.ts data/corrections out.json 4", stride: 4 });
  });

  it("records the script relative to the working directory when it is given as an absolute path", () => {
    expect(
      describeRun(
        ["/usr/bin/node", "/home/someone/repo/scripts/eval-corrections.ts", "data/corrections"],
        1,
        "/home/someone/repo",
      ),
    ).toEqual({ command: "scripts/eval-corrections.ts data/corrections", stride: 1 });
  });

  it("records a full run as stride 1", () => {
    expect(describeRun(["node", "scripts/eval-corrections.ts"], 1)).toEqual({
      command: "scripts/eval-corrections.ts",
      stride: 1,
    });
  });
});
