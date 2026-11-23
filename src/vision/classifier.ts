// Pure-TS inference for the small tile CNN trained by scripts/train_classifier.py:
// three (3x3 conv, ReLU, 2x2 max-pool) blocks, then two dense layers. Classes: 0 = blank, 1..15 = tile.

export interface ConvWeights {
  in: number;
  out: number;
  weights: number[];
  bias: number[];
}
export interface DenseWeights {
  in: number;
  out: number;
  weights: number[];
  bias: number[];
}
export interface ClassifierWeights {
  input: number;
  convs: ConvWeights[];
  fcs: DenseWeights[];
}

// The weights arrive as plain number[] from JSON; typed arrays are much faster to index in the hot loops.
const typedWeights = new WeakMap<object, Float32Array>();
function getTypedWeights(layer: ConvWeights | DenseWeights): Float32Array {
  let typed = typedWeights.get(layer);
  if (!typed) {
    typed = Float32Array.from(layer.weights);
    typedWeights.set(layer, typed);
  }
  return typed;
}

/**
 * 3x3 convolution (zero padding) then ReLU. This is the pipeline's hot spot (every cell of every
 * candidate is classified), so it works on zero-padded planes with the nine taps unrolled: no bounds
 * checks in the inner loop, and one store per output pixel and input channel. The plain per-pixel
 * version with bounds checks was about twice as slow.
 */
function convolveRelu(input: Float32Array, size: number, layer: ConvWeights): Float32Array {
  const plane = size * size;
  const stride = size + 2;
  const padded = new Float32Array(layer.in * stride * stride);
  for (let c = 0; c < layer.in; c++) {
    for (let y = 0; y < size; y++) {
      const from = c * plane + y * size;
      padded.set(input.subarray(from, from + size), c * stride * stride + (y + 1) * stride + 1);
    }
  }
  const out = new Float32Array(layer.out * plane);
  const kernels = getTypedWeights(layer);
  for (let o = 0; o < layer.out; o++) {
    const outBase = o * plane;
    out.fill(layer.bias[o] ?? 0, outBase, outBase + plane);
    for (let c = 0; c < layer.in; c++) {
      const k = (o * layer.in + c) * 9;
      const w0 = kernels[k] as number;
      const w1 = kernels[k + 1] as number;
      const w2 = kernels[k + 2] as number;
      const w3 = kernels[k + 3] as number;
      const w4 = kernels[k + 4] as number;
      const w5 = kernels[k + 5] as number;
      const w6 = kernels[k + 6] as number;
      const w7 = kernels[k + 7] as number;
      const w8 = kernels[k + 8] as number;
      const inBase = c * stride * stride;
      for (let y = 0; y < size; y++) {
        const r0 = inBase + y * stride;
        const r1 = r0 + stride;
        const r2 = r1 + stride;
        const row = outBase + y * size;
        for (let x = 0; x < size; x++) {
          out[row + x] =
            (out[row + x] as number) +
            w0 * (padded[r0 + x] as number) +
            w1 * (padded[r0 + x + 1] as number) +
            w2 * (padded[r0 + x + 2] as number) +
            w3 * (padded[r1 + x] as number) +
            w4 * (padded[r1 + x + 1] as number) +
            w5 * (padded[r1 + x + 2] as number) +
            w6 * (padded[r2 + x] as number) +
            w7 * (padded[r2 + x + 1] as number) +
            w8 * (padded[r2 + x + 2] as number);
        }
      }
    }
    for (let i = outBase; i < outBase + plane; i++) if ((out[i] as number) < 0) out[i] = 0;
  }
  return out;
}

function applyMaxPool(input: Float32Array, channels: number, size: number): Float32Array {
  const half = size / 2;
  const out = new Float32Array(channels * half * half);
  for (let c = 0; c < channels; c++) {
    for (let y = 0; y < half; y++) {
      for (let x = 0; x < half; x++) {
        const base = c * size * size + 2 * y * size + 2 * x;
        out[c * half * half + y * half + x] = Math.max(
          input[base] ?? 0,
          input[base + 1] ?? 0,
          input[base + size] ?? 0,
          input[base + size + 1] ?? 0,
        );
      }
    }
  }
  return out;
}

function applyDense(input: Float32Array, layer: DenseWeights, relu: boolean): Float32Array {
  if (input.length !== layer.in)
    throw new Error(`dense layer expects ${String(layer.in)} inputs, got ${String(input.length)}`);
  const out = new Float32Array(layer.out);
  const weights = getTypedWeights(layer);
  for (let o = 0; o < layer.out; o++) {
    let sum = layer.bias[o] ?? 0;
    const row = o * layer.in;
    for (let i = 0; i < layer.in; i++) sum += (input[i] ?? 0) * (weights[row + i] ?? 0);
    out[o] = relu && sum < 0 ? 0 : sum;
  }
  return out;
}

/** Logits for one normalised crop of `weights.input` x `weights.input` pixels. */
export function computeLogits(weights: ClassifierWeights, crop: Float32Array): Float32Array {
  let size = weights.input;
  let activations = crop;
  let channels = 1;
  for (const layer of weights.convs) {
    activations = applyMaxPool(convolveRelu(activations, size, layer), layer.out, size);
    channels = layer.out;
    size /= 2;
  }
  if (activations.length !== channels * size * size)
    throw new Error("classifier weights do not match the crop size");
  const last = weights.fcs.length - 1;
  weights.fcs.forEach((layer, i) => {
    activations = applyDense(activations, layer, i < last);
  });
  return activations;
}

/**
 * Class probabilities for a crop knowing the grid size: only values 0..valueCount-1 are possible,
 * so the softmax runs over those. A 3x3 board can never contain a 12, and reading it as one is just noise.
 */
export function classifyCropProbabilities(
  weights: ClassifierWeights,
  crop: Float32Array,
  valueCount: number,
): number[] {
  const logits = computeLogits(weights, crop);
  let max = -Infinity;
  for (let i = 0; i < valueCount; i++) max = Math.max(max, logits[i] ?? -Infinity);
  const exps = Array.from({ length: valueCount }, (_, i) =>
    Math.exp((logits[i] ?? -Infinity) - max),
  );
  const total = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / total);
}
