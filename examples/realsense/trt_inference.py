#!/usr/bin/env python3
"""
TensorRT Inference Wrapper for DROID-SLAM Networks
Uses PyTorch CUDA interface instead of PyCUDA to avoid context conflicts
"""

import numpy as np
import torch
import tensorrt as trt
import ctypes

class TRTDroidNetwork:
    """TensorRT wrapper for DROID-SLAM fnet/cnet with FP16 support"""
    
    def __init__(self, engine_path, input_shape, output_shape, use_fp16=True):
        """
        Args:
            engine_path: Path to .trt engine file
            input_shape: (C, H, W) tuple for input
            output_shape: (C, H, W) tuple for output  
            use_fp16: Whether engine uses FP16 precision
        """
        self.engine_path = engine_path
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.use_fp16 = use_fp16
        
        # Determine dtype
        self.dtype = torch.float16 if use_fp16 else torch.float32
        
        print(f"\n{'='*60}")
        print(f"Loading TensorRT Engine: {engine_path}")
        print(f"{'='*60}")
        print(f"  Input shape:  {input_shape}")
        print(f"  Output shape: {output_shape}")
        print(f"  Precision:    {'FP16' if use_fp16 else 'FP32'}")
        
        # Load engine
        self._load_engine()
        
        # Allocate buffers
        self._allocate_buffers()
        
        print(f"  Status:       ✓ Ready")
        print(f"{'='*60}")
    
    def _load_engine(self):
        """Load TensorRT engine from file"""
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        with open(self.engine_path, 'rb') as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine from {self.engine_path}")
        
        self.context = self.engine.create_execution_context()
        
        if self.context is None:
            raise RuntimeError("Failed to create execution context")
    
    def _allocate_buffers(self):
        """Allocate GPU buffers using PyTorch"""
        # Allocate device memory using PyTorch
        self.d_input = torch.zeros(
            (1, *self.input_shape), 
            dtype=self.dtype, 
            device='cuda'
        )
        
        self.d_output = torch.zeros(
            (1, *self.output_shape), 
            dtype=self.dtype, 
            device='cuda'
        )
        
        # Get tensor names for execute_async_v3
        self.input_name = None
        self.output_name = None
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name
        
        if self.input_name is None or self.output_name is None:
            raise RuntimeError("Failed to identify input/output tensor names")
        
        print(f"  Input tensor:  {self.input_name}")
        print(f"  Output tensor: {self.output_name}")
        
        # Create CUDA stream using PyTorch
        self.stream = torch.cuda.Stream()
    
    def __call__(self, image_tensor):
        """
        Run inference on input image tensor
        
        Args:
            image_tensor: PyTorch tensor of shape [1, C, H, W] on GPU
            
        Returns:
            PyTorch tensor of shape [1, C_out, H_out, W_out] on GPU
        """
        with torch.cuda.stream(self.stream):
            # Ensure input is contiguous and correct dtype
            if self.use_fp16:
                image_tensor = image_tensor.half().contiguous()
            else:
                image_tensor = image_tensor.float().contiguous()
            
            # Copy to input buffer
            self.d_input.copy_(image_tensor)
            
            # Set tensor addresses for execute_async_v3
            self.context.set_tensor_address(
                self.input_name, 
                self.d_input.data_ptr()
            )
            self.context.set_tensor_address(
                self.output_name, 
                self.d_output.data_ptr()
            )
            
            # Execute inference
            # Get CUDA stream handle as ctypes pointer
            stream_ptr = ctypes.c_void_p(self.stream.cuda_stream)
            self.context.execute_async_v3(stream_handle=stream_ptr.value)
            
            # Synchronize stream
            self.stream.synchronize()
        
        # Return output (convert back to FP32 if needed)
        if self.use_fp16:
            return self.d_output.float()
        else:
            return self.d_output.clone()


# ============================================================
# Test Script
# ============================================================
if __name__ == "__main__":
    import time
    
    H, W = 240, 320
    
    print("\n" + "="*60)
    print("Loading TensorRT Engines")
    print("="*60)
    
    # Load feature network
    fnet_trt = TRTDroidNetwork(
        engine_path=f'trt_engines/droid_fnet_{H}x{W}_fp16.trt',
        input_shape=(3, H, W),
        output_shape=(128, H//8, W//8)
    )
    
    print("")
    
    # Load context network
    cnet_trt = TRTDroidNetwork(
        engine_path=f'trt_engines/droid_cnet_{H}x{W}_fp16.trt',
        input_shape=(3, H, W),
        output_shape=(256, H//8, W//8)
    )
    
    print("\n" + "="*60)
    print("Running Inference Test")
    print("="*60)
    
    # Create test image
    test_image = torch.randn(1, 3, H, W).cuda()
    
    # Warmup
    print("\nWarming up...")
    for _ in range(10):
        _ = fnet_trt(test_image)
        _ = cnet_trt(test_image)
    
    # Benchmark
    N = 100
    print(f"\nBenchmarking ({N} iterations)...")
    
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(N):
        features = fnet_trt(test_image)
        context = cnet_trt(test_image)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print("\n" + "="*60)
    print("Benchmark Results")
    print("="*60)
    print(f"Total time:           {elapsed:.3f}s")
    print(f"Avg time per frame:   {elapsed/N*1000:.2f}ms")
    print(f"Throughput:           {N/elapsed:.1f} FPS")
    print(f"\nFeature output shape: {features.shape}")
    print(f"Context output shape: {context.shape}")
    
    # Sanity check - outputs shouldn't be all zeros
    print(f"\nSanity checks:")
    print(f"  Feature mean: {features.mean().item():.6f}")
    print(f"  Feature std:  {features.std().item():.6f}")
    print(f"  Context mean: {context.mean().item():.6f}")
    print(f"  Context std:  {context.std().item():.6f}")
    print("="*60)