# File: convert_onnx_to_trt.py
import tensorrt as trt
import os

def build_engine(onnx_path, engine_path, fp16=True):
    """Convert ONNX to TensorRT engine"""
    
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # Parse ONNX
    print(f"Loading ONNX: {onnx_path}")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # Configure
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB
    
    # FP16
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("Using FP16 precision")
    
    # Disable DLA (use GPU only)
    # No need to explicitly disable - just don't set DLA device
    
    # Build
    print(f"Building engine (this may take a few minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("ERROR: Failed to build engine")
        return None
    
    # Save
    print(f"Saving engine: {engine_path}")
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    
    print(f"✓ Success! Size: {os.path.getsize(engine_path) / (1024**2):.2f} MB")
    return engine_path


if __name__ == "__main__":
    H, W = 240, 320
    
    os.makedirs("trt_engines", exist_ok=True)
    
    print("="*60)
    print("Converting DROID-SLAM ONNX to TensorRT FP16")
    print("="*60)
    
    # Feature network
    print("\n1. Feature Network")
    build_engine(
        f"droid_fnet_{H}x{W}.onnx",
        f"trt_engines/droid_fnet_{H}x{W}_fp16.trt",
        fp16=True
    )
    
    # Context network
    print("\n2. Context Network")
    build_engine(
        f"droid_cnet_{H}x{W}.onnx",
        f"trt_engines/droid_cnet_{H}x{W}_fp16.trt",
        fp16=True
    )
    
    print("\n" + "="*60)
    print("Conversion Complete!")
    print("="*60)