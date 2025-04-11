#!/bin/bash

# Create necessary directories
mkdir -p web/public/models
mkdir -p web/public/assets

# Copy ONNX Runtime WebAssembly files
cp web/node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm web/public/assets/
cp web/node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.wasm web/public/assets/
cp web/node_modules/onnxruntime-web/dist/ort.wasm.js web/public/assets/
cp web/node_modules/onnxruntime-web/dist/ort.wasm.min.js web/public/assets/

echo "ONNX Runtime WebAssembly files copied successfully" 