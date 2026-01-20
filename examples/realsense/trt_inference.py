import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import torch

class TRTDroidNetwork:
    """Wrapper for TensorRT DROID network inference"""
    
    def __init__(self, engine_path, input_shape, output_shape):
        """
        Args:
            engine_path: Path to .trt engine file
            input_shape: (C, H, W) e.g., (3, 240, 320)
            output_shape: (C, H, W) e.g., (128, 30, 40)
        """
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        print(f"Loading TensorRT engine: {engine_path}")
        
        # Load engine
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine from {engine_path}")
        
        self.context = self.engine.create_execution_context()
        
        # Setup buffers
        self.input_shape = (1,) + tuple(input_shape)
        self.output_shape = (1,) + tuple(output_shape)
        
        # Use FP16 for Jetson
        dtype = np.float16
        
        # Allocate host (CPU) memory
        self.h_input = cuda.pagelocked_empty(self.input_shape, dtype=dtype)
        self.h_output = cuda.pagelocked_empty(self.output_shape, dtype=dtype)
        
        # Allocate device (GPU) memory
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        
        # Create CUDA stream
        self.stream = cuda.Stream()
        
        print(f"  ✓ Engine loaded")
        print(f"  Input shape:  {self.input_shape}")
        print(f"  Output shape: {self.output_shape}")
        print(f"  Data type:    {dtype}")
    
    def infer(self, image_tensor):
        """
        Run inference on RGB image
        
        Args:
            image_tensor: PyTorch tensor [1, 3, H, W] (can be on CPU or GPU)
        
        Returns:
            PyTorch tensor [1, C_out, H_out, W_out] on GPU
        """
        # Convert to numpy and correct dtype
        if image_tensor.is_cuda:
            img_np = image_tensor.cpu().numpy()
        else:
            img_np = image_tensor.numpy()
        
        img_np = img_np.astype(np.float16)
        
        # Copy to page-locked host memory
        np.copyto(self.h_input, img_np)
        
        # Transfer input to GPU
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        
        # Run inference
        self.context.execute_async_v2(
            bindings=[int(self.d_input), int(self.d_output)],
            stream_handle=self.stream.handle
        )
        
        # Transfer output to CPU
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        
        # Synchronize stream
        self.stream.synchronize()
        
        # Convert to PyTorch tensor on GPU
        output_torch = torch.from_numpy(self.h_output.copy()).cuda()
        
        return output_torch
    
    def __call__(self, image_tensor):
        """Shorthand for infer()"""
        return self.infer(image_tensor)
    
    def __del__(self):
        """Cleanup CUDA memory"""
        try:
            del self.d_input
            del self.d_output
            del self.stream
        except:
            pass


# ============================================================
# Example Usage and Benchmark
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
    print("="*60)