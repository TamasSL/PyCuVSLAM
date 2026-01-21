import torch
import torch.nn as nn
from droid_slam.droid import Droid
from trt_inference import TRTDroidNetwork

class TRTNetWrapper(nn.Module):
    """Wrapper to make TRTDroidNetwork compatible with torch.nn.Module"""
    
    def __init__(self, trt_network):
        super().__init__()
        self.trt_network = trt_network
    
    def forward(self, x):
        """Forward pass through TensorRT network"""
        return self.trt_network(x)


class DroidTRT:
    """DROID-SLAM with TensorRT-optimized feature extraction"""
    
    def __init__(self, args, use_trt=True):
        self.use_trt = use_trt
        self.H = args.image_size[0]
        self.W = args.image_size[1]
        
        # Initialize standard DROID
        self.droid = Droid(args)
        
        # Replace feature/context networks with TensorRT if enabled
        if use_trt:
            print("Loading TensorRT engines...")
            
            self.fnet_trt = TRTDroidNetwork(
                engine_path=f'trt_engines/droid_fnet_{self.H}x{self.W}_fp16.trt',
                input_shape=(3, self.H, self.W),
                output_shape=(128, self.H//8, self.W//8)
            )
            
            self.cnet_trt = TRTDroidNetwork(
                engine_path=f'trt_engines/droid_cnet_{self.H}x{self.W}_fp16.trt',
                input_shape=(3, self.H, self.W),
                output_shape=(256, self.H//8, self.W//8)
            )
            
            # Monkey-patch DROID's feature extraction
            self._patch_droid_networks()
            
            print("✓ TensorRT engines loaded and integrated")
    
    def _patch_droid_networks(self):
        """Replace DROID's fnet/cnet with TensorRT versions"""
        
        # Wrap TensorRT networks in nn.Module
        self.droid.net.fnet = TRTNetWrapper(self.fnet_trt)
        self.droid.net.cnet = TRTNetWrapper(self.cnet_trt)
    
    def track(self, tstamp, image, depth=None, intrinsics=None):
        """Forward to DROID's track method"""
        return self.droid.track(tstamp, image, depth, intrinsics)
    
    def terminate(self, stream=None):
        """Forward to DROID's terminate method"""
        return self.droid.terminate(stream)
    
    def __getattr__(self, name):
        """Forward all other attributes to the underlying DROID instance"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.droid, name)


# ============================================================
# Usage Example
# ============================================================
if __name__ == "__main__":
    import argparse
    
    # Your existing DROID arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='pretrained_models/droid.pth')
    parser.add_argument('--buffer', type=int, default=512)
    parser.add_argument('--image_size', default=[240, 320])
    parser.add_argument('--stereo', action='store_true')
    parser.add_argument('--disable_vis', action='store_true')
    # ... add other DROID args as needed
    
    args = parser.parse_args()
    
    # Create TensorRT-accelerated DROID
    droid_trt = DroidTRT(args, use_trt=True)
    
    # Use exactly as before
    timestamp = 0
    rgb_image = ...  # Your RGB image
    disparity = ...  # Your disparity map
    intrinsics = ...  # Your camera intrinsics
    
    # Track frame (TensorRT used internally for feature extraction)
    droid_trt.track(timestamp, rgb_image, depth=disparity, intrinsics=intrinsics)
    
    # Extract points and poses (works exactly as before)
    points, poses = droid_trt.extract_points_and_poses()
    
    print(f"Extracted {len(points)} points")
    print(f"Current pose:\n{poses[-1]}")