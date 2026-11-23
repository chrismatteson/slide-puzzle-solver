import { MAX_PHOTO_SIDE, type RgbaImage, fitWithin } from "../vision/image.ts";

export interface Photo {
  image: RgbaImage;
  /** The same pixels as a JPEG data URL for display (no network involved). */
  url: string;
}

function drawToCanvas(source: CanvasImageSource, width: number, height: number): Photo {
  const size = fitWithin(width, height, MAX_PHOTO_SIDE);
  const canvas = document.createElement("canvas");
  canvas.width = size.width;
  canvas.height = size.height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("this browser cannot read pixels from a canvas");
  context.drawImage(source, 0, 0, canvas.width, canvas.height);
  const { data } = context.getImageData(0, 0, canvas.width, canvas.height);
  return {
    image: { width: canvas.width, height: canvas.height, data },
    url: canvas.toDataURL("image/jpeg", 0.85),
  };
}

/** Decodes an uploaded file; the browser applies the photo's EXIF rotation. */
export async function loadPhotoFile(file: Blob): Promise<Photo> {
  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(file);
  } catch (error) {
    throw new Error("That file could not be opened as an image. Try a JPEG or PNG photo.", {
      cause: error,
    });
  }
  try {
    return drawToCanvas(bitmap, bitmap.width, bitmap.height);
  } finally {
    bitmap.close();
  }
}

export function captureVideoFrame(video: HTMLVideoElement): Photo {
  if (video.videoWidth === 0)
    throw new Error("The camera has not produced a picture yet. Try again.");
  return drawToCanvas(video, video.videoWidth, video.videoHeight);
}
