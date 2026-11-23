import { defaultWeights } from "../vision/default-weights.ts";
import { readBoard } from "../vision/read-board.ts";
import { type VisionRequest, type VisionResponse } from "./messages.ts";

function reply(message: VisionResponse): void {
  postMessage(message);
}

addEventListener("message", (event: MessageEvent<VisionRequest>) => {
  const { id, image } = event.data;
  // The page must hear about a failure, or it would wait on this photo forever.
  try {
    reply({ type: "read", id, result: readBoard(image, defaultWeights) });
  } catch (error) {
    reply({ type: "error", id, message: error instanceof Error ? error.message : String(error) });
  }
});

reply({ type: "ready" });
