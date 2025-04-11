#!/bin/bash

# Exit on error
set -e

# Install required packages
pip install torch torchvision onnx timm

# Clone MobileSAM repository if it doesn't exist
if [ ! -d "MobileSAM" ]; then
    git clone https://github.com/ChaoningZhang/MobileSAM.git
fi

# Install MobileSAM
pip install -e MobileSAM

# Create directories if they don't exist
mkdir -p MobileSAM/weights
mkdir -p web/public/models

# Download the model if it doesn't exist
if [ ! -f "MobileSAM/weights/mobile_sam.pt" ]; then
    wget -O MobileSAM/weights/mobile_sam.pt https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
fi

# Convert model to ONNX
cd MobileSAM
python convert_model.py
cd ..

# Check if the ONNX file was created
if [ ! -f "MobileSAM/weights/mobile_sam.onnx" ]; then
    echo "Error: Failed to create ONNX file"
    exit 1
fi

# Move the ONNX model to the web directory
mv MobileSAM/weights/mobile_sam.onnx web/public/models/

echo "MobileSAM model downloaded and converted to ONNX format successfully"
