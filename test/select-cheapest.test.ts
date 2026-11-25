import { describe, expect, it } from "vitest";
import { selectCheapest } from "../src/picture/assemble.ts";
import { createRandom } from "../scripts/random.ts";

describe("selectCheapest", () => {
  it("equals a stable sort of the indices by cost cut to the count, ties included", () => {
    const random = createRandom(11);
    for (let round = 0; round < 200; round++) {
      // Few distinct costs, so many ties fall on the cut.
      const length = 1 + Math.floor(random() * 400);
      const costs = Float64Array.from({ length }, () => Math.floor(random() * 12));
      const count = 1 + Math.floor(random() * 250);
      const expected = Array.from({ length }, (_, index) => index)
        .sort((a, b) => (costs[a] ?? 0) - (costs[b] ?? 0))
        .slice(0, count);
      expect(selectCheapest(costs, length, count)).toEqual(expected);
    }
  });

  it("looks only at the first `length` costs and leaves them as they were", () => {
    const costs = Float64Array.from([5, 1, 4, 1, 0, 0]);
    expect(selectCheapest(costs, 4, 2)).toEqual([1, 3]);
    expect([...costs]).toEqual([5, 1, 4, 1, 0, 0]);
  });
});
