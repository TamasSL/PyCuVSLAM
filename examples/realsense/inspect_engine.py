#!/usr/bin/env python3
"""Check TensorRT engine build info"""

import tensorrt as trt

def inspect_engine(engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    with open(engine_path, 'rb') as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())
    
    if engine is None:
        print(f"Failed to load engine from {engine_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"Engine: {engine_path}")
    print(f"{'='*60}")
    print(f"Device type: {engine.device_type}")
    print(f"Num layers: {engine.num_layers}")
    print(f"Num IO tensors: {engine.num_io_tensors}")
    
    # Get platform info
    import subprocess
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                              capture_output=True, text=True)
        print(f"Current GPU: {result.stdout.strip()}")
    except:
        pass
    
    print(f"{'='*60}\n")

if __name__ == "__main__":
    inspect_engine('trt_engines/droid_fnet_240x320_fp16.trt')

"""
nvidia-smi --query-gpu=name,compute_cap --format=csv

# What TensorRT version?
python3 -c "import tensorrt as trt; print(f'TensorRT: {trt.__version__}')"

# List your engine files
ls -lh trt_engines/
"""