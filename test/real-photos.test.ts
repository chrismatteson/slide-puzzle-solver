import { describe, expect, it } from "vitest";
import { download } from "../scripts/real-photos.ts";

describe("download", () => {
  it("gives up on a host that never answers", async () => {
    // Never resolves on its own: only the abort signal can end it, as with a stalled server.
    const hangUntilAborted: typeof fetch = (_input, init) =>
      new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => {
          reject(init.signal?.reason instanceof Error ? init.signal.reason : new Error("aborted"));
        });
      });
    await expect(download("http://example.invalid/x.jpg", 50, hangUntilAborted)).rejects.toThrow(
      /http:\/\/example\.invalid\/x\.jpg: no response within 50 ms/,
    );
  });

  it("returns the body of a successful response", async () => {
    const answer: typeof fetch = () => Promise.resolve(new Response(new Uint8Array([1, 2, 3])));
    expect([...(await download("http://example.invalid/y.jpg", 50, answer))]).toEqual([1, 2, 3]);
  });
});
