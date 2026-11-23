// Minimal image helpers on plain RGBA buffers, so the same code runs in Node and in the browser
// (where the buffer comes from a canvas). No DOM, no native modules.

export interface RgbaImage {
  width: number;
  height: number;
  /** RGBA, row-major, 4 bytes per pixel. */
  data: Uint8ClampedArray;
}

/** Longest side of the image handed to the reader; phone cameras give 4000+ px, the reader needs about 1000. */
export const MAX_PHOTO_SIDE = 1280;

/** Size after scaling down (never up) so that the longer side is at most `maxSide`. */
export function fitWithin(
  width: number,
  height: number,
  maxSide: number,
): { width: number; height: number } {
  const scale = Math.min(1, maxSide / Math.max(width, height));
  return { width: Math.round(width * scale), height: Math.round(height * scale) };
}

/**
 * Box-filter downscale of an RGBA image to `fitWithin` size. The browser does this with a canvas
 * (`src/app/photo.ts`); this is the Node equivalent for `scripts/eval.ts`, which has no canvas.
 */
export function shrinkRgba(image: RgbaImage, maxSide: number): RgbaImage {
  checkRgbaSize(image);
  const target = fitWithin(image.width, image.height, maxSide);
  if (target.width === image.width && target.height === image.height) return image;
  const data = new Uint8ClampedArray(target.width * target.height * 4);
  const xScale = image.width / target.width;
  const yScale = image.height / target.height;
  for (let y = 0; y < target.height; y++) {
    const y0 = Math.floor(y * yScale);
    const y1 = Math.max(y0 + 1, Math.min(image.height, Math.floor((y + 1) * yScale)));
    for (let x = 0; x < target.width; x++) {
      const x0 = Math.floor(x * xScale);
      const x1 = Math.max(x0 + 1, Math.min(image.width, Math.floor((x + 1) * xScale)));
      for (let c = 0; c < 4; c++) {
        let sum = 0;
        for (let yy = y0; yy < y1; yy++) {
          for (let xx = x0; xx < x1; xx++) sum += image.data[(yy * image.width + xx) * 4 + c] ?? 0;
        }
        data[(y * target.width + x) * 4 + c] = sum / ((y1 - y0) * (x1 - x0));
      }
    }
  }
  return { width: target.width, height: target.height, data };
}

/** Single-channel float image, values nominally in 0..255. */
export interface GreyImage {
  width: number;
  height: number;
  data: Float32Array;
}

/** A short buffer would otherwise read as black pixels (`?? 0`) and fail later as "no board". */
export function checkRgbaSize(image: RgbaImage): void {
  const expected = image.width * image.height * 4;
  if (image.data.length < expected) {
    throw new Error(
      `RGBA buffer has ${String(image.data.length)} bytes, expected ${String(expected)} for ${String(image.width)}x${String(image.height)}`,
    );
  }
}

export function convertToGrey(image: RgbaImage): GreyImage {
  checkRgbaSize(image);
  const data = new Float32Array(image.width * image.height);
  const src = image.data;
  for (let i = 0; i < data.length; i++) {
    data[i] =
      0.299 * (src[i * 4] ?? 0) + 0.587 * (src[i * 4 + 1] ?? 0) + 0.114 * (src[i * 4 + 2] ?? 0);
  }
  return { width: image.width, height: image.height, data };
}

/** Per-pixel max(R,G,B) - min(R,G,B): a lighting-insensitive second view that separates coloured boards from grey tables. */
export function convertToSaturation(image: RgbaImage): GreyImage {
  checkRgbaSize(image);
  const data = new Float32Array(image.width * image.height);
  const src = image.data;
  for (let i = 0; i < data.length; i++) {
    const r = src[i * 4] ?? 0;
    const g = src[i * 4 + 1] ?? 0;
    const b = src[i * 4 + 2] ?? 0;
    data[i] = Math.max(r, g, b) - Math.min(r, g, b);
  }
  return { width: image.width, height: image.height, data };
}

/** Box-filter downscale so that the longer side is at most `maxSide`; returns the factor applied (<= 1). */
export function shrinkGrey(image: GreyImage, maxSide: number): { image: GreyImage; scale: number } {
  const scale = Math.min(1, maxSide / Math.max(image.width, image.height));
  if (scale === 1) return { image, scale };
  const width = Math.max(1, Math.round(image.width * scale));
  const height = Math.max(1, Math.round(image.height * scale));
  const data = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    const y0 = Math.floor(y / scale);
    const y1 = Math.max(y0 + 1, Math.min(image.height, Math.floor((y + 1) / scale)));
    for (let x = 0; x < width; x++) {
      const x0 = Math.floor(x / scale);
      const x1 = Math.max(x0 + 1, Math.min(image.width, Math.floor((x + 1) / scale)));
      let sum = 0;
      for (let yy = y0; yy < y1; yy++) {
        for (let xx = x0; xx < x1; xx++) sum += image.data[yy * image.width + xx] ?? 0;
      }
      data[y * width + x] = sum / ((y1 - y0) * (x1 - x0));
    }
  }
  return { image: { width, height, data }, scale: width / image.width };
}

/** Separable box blur with the given radius (edge pixels clamp). */
export function blurGrey(image: GreyImage, radius: number): GreyImage {
  if (radius <= 0) return image;
  const { width, height } = image;
  const tmp = new Float32Array(width * height);
  const out = new Float32Array(width * height);
  const span = 2 * radius + 1;
  for (let y = 0; y < height; y++) {
    const row = y * width;
    let sum = 0;
    for (let k = -radius; k <= radius; k++)
      sum += image.data[row + Math.min(width - 1, Math.max(0, k))] ?? 0;
    for (let x = 0; x < width; x++) {
      tmp[row + x] = sum / span;
      sum +=
        (image.data[row + Math.min(width - 1, x + radius + 1)] ?? 0) -
        (image.data[row + Math.max(0, x - radius)] ?? 0);
    }
  }
  for (let x = 0; x < width; x++) {
    let sum = 0;
    for (let k = -radius; k <= radius; k++)
      sum += tmp[Math.min(height - 1, Math.max(0, k)) * width + x] ?? 0;
    for (let y = 0; y < height; y++) {
      out[y * width + x] = sum / span;
      sum +=
        (tmp[Math.min(height - 1, y + radius + 1) * width + x] ?? 0) -
        (tmp[Math.max(0, y - radius) * width + x] ?? 0);
    }
  }
  return { width, height, data: out };
}

/** Bilinear sample; coordinates are pixel-index space (pixel centres at integers), clamped at the border. */
export function sampleBilinear(image: GreyImage, x: number, y: number): number {
  const { width, height, data } = image;
  const cx = Math.min(width - 1, Math.max(0, x));
  const cy = Math.min(height - 1, Math.max(0, y));
  const x0 = Math.min(width - 2, Math.floor(cx));
  const y0 = Math.min(height - 2, Math.floor(cy));
  const fx = cx - x0;
  const fy = cy - y0;
  const i = y0 * width + x0;
  const top = (data[i] ?? 0) * (1 - fx) + (data[i + 1] ?? 0) * fx;
  const bottom = (data[i + width] ?? 0) * (1 - fx) + (data[i + width + 1] ?? 0) * fx;
  return top * (1 - fy) + bottom * fy;
}

/**
 * The image minus its local mean (box blur of `radius`): what is left is local contrast, so tiles
 * stay separable from their frame under a lighting gradient or a hard shadow edge, where a single
 * global brightness threshold merges or splits them. Values are centred on 0, not 0..255.
 */
export function flattenGrey(image: GreyImage, radius: number): GreyImage {
  const local = blurGrey(image, radius).data;
  const data = new Float32Array(image.data.length);
  for (let i = 0; i < data.length; i++) data[i] = (image.data[i] ?? 0) - (local[i] ?? 0);
  return { width: image.width, height: image.height, data };
}
