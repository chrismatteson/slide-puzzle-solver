import { type GreyImage } from "./image.ts";

// These constants and sampleCrop mirror scripts/synth.py, which makes the training crops;
// test/vision.test.ts compares the two on a fixture so they cannot drift apart.
export const CROP_PIXELS = 32;
/** Crop side as a fraction of the cell pitch: the tile's digits with a little surround. */
export const CROP_WINDOW = 0.8;
const CROP_SUBSAMPLES = 3;
/** Floor on the contrast normaliser, so a blank cell stays low-amplitude instead of amplifying noise. */
const CROP_MIN_SPREAD = 0.08;

/** CROP_PIXELS² crop centred at (cx, cy) in pixel-index coordinates, contrast-normalised. */
export function sampleCrop(grey: GreyImage, cx: number, cy: number, side: number): Float32Array {
  const samples = CROP_PIXELS * CROP_SUBSAMPLES;
  const { width, height, data } = grey;
  const xs = new Float64Array(samples);
  const ys = new Float64Array(samples);
  for (let i = 0; i < samples; i++) {
    const offset = (i + 0.5) / samples - 0.5;
    xs[i] = cx + offset * side;
    ys[i] = cy + offset * side;
  }
  const crop = new Float32Array(CROP_PIXELS * CROP_PIXELS);
  for (let sy = 0; sy < samples; sy++) {
    const y = ys[sy] ?? 0;
    const y0 = Math.min(height - 2, Math.max(0, Math.floor(y)));
    const fy = Math.min(1, Math.max(0, y - y0));
    const outRow = Math.floor(sy / CROP_SUBSAMPLES) * CROP_PIXELS;
    for (let sx = 0; sx < samples; sx++) {
      const x = xs[sx] ?? 0;
      const x0 = Math.min(width - 2, Math.max(0, Math.floor(x)));
      const fx = Math.min(1, Math.max(0, x - x0));
      const i = y0 * width + x0;
      const top = (data[i] ?? 0) * (1 - fx) + (data[i + 1] ?? 0) * fx;
      const bottom = (data[i + width] ?? 0) * (1 - fx) + (data[i + width + 1] ?? 0) * fx;
      const index = outRow + Math.floor(sx / CROP_SUBSAMPLES);
      crop[index] = (crop[index] ?? 0) + (top * (1 - fy) + bottom * fy) / 255;
    }
  }
  const scale = 1 / (CROP_SUBSAMPLES * CROP_SUBSAMPLES);
  let mean = 0;
  for (let i = 0; i < crop.length; i++) {
    crop[i] = (crop[i] ?? 0) * scale;
    mean += crop[i] ?? 0;
  }
  mean /= crop.length;
  let variance = 0;
  for (const v of crop) variance += (v - mean) ** 2;
  const spread = Math.max(Math.sqrt(variance / crop.length), CROP_MIN_SPREAD);
  for (let i = 0; i < crop.length; i++) crop[i] = ((crop[i] ?? 0) - mean) / spread;
  return crop;
}
