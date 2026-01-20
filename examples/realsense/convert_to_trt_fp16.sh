#!/bin/bash
# File: convert_to_trt_fp16.sh
# Run this script on your Jetson Orin NX

H=240
W=320
ONNX_DIR="."
OUTPUT_DIR="trt_engines"

mkdir -p ${OUTPUT_DIR}

echo "=========================================="
echo "Converting DROID-SLAM to TensorRT FP16"
echo "Resolution: ${H}x${W}"
echo "=========================================="
echo ""

# ============================================================
# Convert Feature Network to TensorRT FP16
# ============================================================
echo "Converting Feature Network..."
/usr/src/tensorrt/bin/trtexec \
    --onnx=${ONNX_DIR}/droid_fnet_${H}x${W}.onnx \
    --saveEngine=${OUTPUT_DIR}/droid_fnet_${H}x${W}_fp16.trt \
    --fp16 \
    --workspace=4096 \
    --verbose \
    --dumpLayerInfo \
    --dumpProfile \
    --exportLayerInfo=${OUTPUT_DIR}/fnet_layer_info.json \
    --exportProfile=${OUTPUT_DIR}/fnet_profile.json

if [ $? -eq 0 ]; then
    echo "✓ Feature network converted successfully"
else
    echo "✗ Feature network conversion failed"
    exit 1
fi

echo ""

# ============================================================
# Convert Context Network to TensorRT FP16
# ============================================================
echo "Converting Context Network..."
/usr/src/tensorrt/bin/trtexec \
    --onnx=${ONNX_DIR}/droid_cnet_${H}x${W}.onnx \
    --saveEngine=${OUTPUT_DIR}/droid_cnet_${H}x${W}_fp16.trt \
    --fp16 \
    --workspace=4096 \
    --verbose \
    --dumpLayerInfo \
    --dumpProfile \
    --exportLayerInfo=${OUTPUT_DIR}/cnet_layer_info.json \
    --exportProfile=${OUTPUT_DIR}/cnet_profile.json

if [ $? -eq 0 ]; then
    echo "✓ Context network converted successfully"
else
    echo "✗ Context network conversion failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Conversion Complete!"
echo "=========================================="
echo ""
echo "TensorRT engines saved in: ${OUTPUT_DIR}/"
ls -lh ${OUTPUT_DIR}/*.trt

echo ""
echo "=========================================="
echo "Benchmarking engines..."
echo "=========================================="

# ============================================================
# Benchmark Feature Network
# ============================================================
echo ""
echo "Benchmarking Feature Network..."
/usr/src/tensorrt/bin/trtexec \
    --loadEngine=${OUTPUT_DIR}/droid_fnet_${H}x${W}_fp16.trt \
    --warmUp=500 \
    --duration=10 \
    --iterations=100

# ============================================================
# Benchmark Context Network
# ============================================================
echo ""
echo "Benchmarking Context Network..."
/usr/src/tensorrt/bin/trtexec \
    --loadEngine=${OUTPUT_DIR}/droid_cnet_${H}x${W}_fp16.trt \
    --warmUp=500 \
    --duration=10 \
    --iterations=100

echo ""
echo "=========================================="
echo "All Done!"
echo "=========================================="