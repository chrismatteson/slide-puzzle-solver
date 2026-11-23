import { type VisionRequest, type VisionResponse } from "./messages.ts";
import { readPhoto } from "./routing.ts";

function reply(message: VisionResponse): void {
  postMessage(message);
}

addEventListener("message", (event: MessageEvent<VisionRequest>) => {
  const { id, image, mode, profiles } = event.data;
  // The page must hear about a failure, or it would wait on this photo forever.
  try {
    reply({ type: "read", id, result: readPhoto(image, mode, profiles) });
  } catch (error) {
    reply({ type: "error", id, message: error instanceof Error ? error.message : String(error) });
  }
});

reply({ type: "ready" });
