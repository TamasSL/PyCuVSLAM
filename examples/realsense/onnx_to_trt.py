#!/usr/bin/env python3
"""
Convert ONNX models to TensorRT engines on Jetson Orin NX
"""

import tensorrt as trt
import os
import subprocess

def print_system_info():
    """Print system information"""
    print("="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    
    # Get GPU info
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,compute_cap', '--format=csv,noheader'], 
                              capture_output=True, text=True)
        print(f"GPU: {result.stdout.strip()}")
    except:
        print("Could not get GPU info")
    
    # Get TensorRT version
    print(f"TensorRT version: {trt.__version__}")
    
    # Get hostname
    try:
        hostname = subprocess.run(['hostname'], capture_output=True, text=True)
        print(f"Hostname: {hostname.stdout.strip()}")
    except:
        pass
    
    print("="*60 + "\n")

def build_engine(onnx_path, engine_path, fp16=True):
    """Build TensorRT engine from ONNX model"""
    
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX
    print(f"\n{'='*60}")
    print(f"Building: {os.path.basename(engine_path)}")
    print(f"{'='*60}")
    print(f"ONNX file: {onnx_path}")
    print(f"Engine file: {engine_path}")
    
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse ONNX file")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
    
    print(f"✓ ONNX parsed successfully")
    
    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB workspace
    
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("✓ FP16 mode enabled")
    
    # Build engine
    print("Building TensorRT engine (this may take a few minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("ERROR: Failed to build engine")
        return False
    
    # Save engine
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    
    engine_size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"✓ Engine saved: {engine_size_mb:.2f} MB")
    print(f"{'='*60}\n")
    
    return True


if __name__ == "__main__":
    # Print system info first
    print_system_info()
    
    H, W = 240, 320
    
    # Paths
    onnx_dir = "trt_models"
    engine_dir = "trt_engines"
    os.makedirs(engine_dir, exist_ok=True)
    
    models = [
        ("droid_fnet", f"{H}x{W}"),
        ("droid_cnet", f"{H}x{W}"),
    ]
    
    print("="*60)
    print("BUILDING TENSORRT ENGINES")
    print("="*60)
    print(f"ONNX directory: {onnx_dir}")
    print(f"Engine directory: {engine_dir}")
    print("="*60)
    
    success_count = 0
    for model_name, size in models:
        onnx_path = f"{onnx_dir}/{model_name}_{size}.onnx"
        engine_path = f"{engine_dir}/{model_name}_{size}_fp16.trt"
        
        if not os.path.exists(onnx_path):
            print(f"\nERROR: ONNX file not found: {onnx_path}")
            print("Available files in trt_models/:")
            try:
                for f in os.listdir(onnx_dir):
                    print(f"  - {f}")
            except:
                pass
            continue
        
        success = build_engine(onnx_path, engine_path, fp16=True)
        
        if success:
            success_count += 1
        else:
            print(f"Failed to build {model_name}")
            break
    
    print("\n" + "="*60)
    print("BUILD SUMMARY")
    print("="*60)
    print(f"Successfully built: {success_count}/{len(models)} engines")
    
    if success_count == len(models):
        print("\n✓ All engines built successfully!")
        print("\nYou can now run: python3 trt_inference.py")
    else:
        print("\n✗ Some engines failed to build")
    
    print("="*60)