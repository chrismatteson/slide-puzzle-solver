import { type VisionRequest, type VisionResponse } from "./messages.ts";
import { readPhoto, reassemblePicture } from "./routing.ts";

function reply(message: VisionResponse): void {
  postMessage(message);
}

addEventListener("message", (event: MessageEvent<VisionRequest>) => {
  const request = event.data;
  const { id } = request;
  // The page must hear about a failure, or it would wait on this photo forever.
  try {
    switch (request.type) {
      case "read":
        reply({
          type: "read",
          id,
          result: readPhoto(request.image, request.mode, request.profiles, request.options),
        });
        break;
      case "reassemble":
        reply({
          type: "reassemble",
          id,
          result: reassemblePicture(request.reading, request.blank),
        });
        break;
    }
  } catch (error) {
    reply({ type: "error", id, message: error instanceof Error ? error.message : String(error) });
  }
});

reply({ type: "ready" });
