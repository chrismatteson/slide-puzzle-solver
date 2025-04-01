import * as ort from 'onnxruntime-web';

// Configure ONNX Runtime Web
ort.env.wasm.wasmPaths = {
  'ort-wasm-simd-threaded.wasm': './assets/ort-wasm-simd-threaded.wasm'
};

// Export the configured ONNX Runtime
export { ort }; 