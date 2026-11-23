/** Maps getUserMedia failures to something a person can act on. */
export function describeCameraError(error: unknown): string {
  const name = error instanceof DOMException ? error.name : "";
  if (name === "NotAllowedError" || name === "SecurityError") {
    return "Camera access was denied. Allow it in your browser's site settings, or upload a photo instead.";
  }
  if (name === "NotFoundError" || name === "OverconstrainedError") {
    return "No camera was found on this device. Upload a photo instead.";
  }
  if (name === "NotReadableError") {
    return "The camera is in use by another app. Close it and try again, or upload a photo instead.";
  }
  return "The camera could not be started. Upload a photo instead.";
}

/** Starts the rear camera in `video`. Resolves with the stream so the caller can stop it. */
export async function startCamera(video: HTMLVideoElement): Promise<MediaStream> {
  // Browsers expose mediaDevices only on HTTPS or localhost; without it there is no camera to ask for.
  if (!("mediaDevices" in navigator)) {
    throw new DOMException("camera API unavailable (page is not secure)", "NotFoundError");
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "environment" },
  });
  video.srcObject = stream;
  await video.play();
  return stream;
}

export function stopCamera(video: HTMLVideoElement, stream: MediaStream): void {
  for (const track of stream.getTracks()) track.stop();
  video.srcObject = null;
}
