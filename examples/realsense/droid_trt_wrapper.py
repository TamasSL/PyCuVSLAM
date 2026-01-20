import torch
import numpy as np
from droid_slam.droid import Droid
from droid_slam.geom import SE3
import droid_backends
from trt_inference import TRTDroidNetwork

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
        
        # Store original networks (in case needed)
        self._original_fnet = self.droid.net.fnet
        self._original_cnet = self.droid.net.cnet
        
        # Create wrapper that matches DROID's expected interface
        class TRTNetWrapper:
            def __init__(self, trt_net):
                self.trt_net = trt_net
            
            def __call__(self, images):
                # images: [batch, num_frames, 3, H, W]
                b, n, c, h, w = images.shape
                
                # Process each frame
                features_list = []
                for i in range(n):
                    # Extract single frame: [b, 3, h, w]
                    frame = images[:, i, :, :, :]
                    # Run TensorRT inference
                    feat = self.trt_net(frame)  # [b, C, h/8, w/8]
                    features_list.append(feat)
                
                # Stack back to [batch, num_frames, C, h/8, w/8]
                features = torch.stack(features_list, dim=1)
                return features
        
        # Replace networks
        self.droid.net.fnet = TRTNetWrapper(self.fnet_trt)
        self.droid.net.cnet = TRTNetWrapper(self.cnet_trt)
    
    def track(self, timestamp, image, depth=None, intrinsics=None):
        """Track frame - same interface as original DROID"""
        return self.droid.track(timestamp, image, depth=depth, intrinsics=intrinsics)
    
    def terminate(self):
        """Terminate DROID"""
        return self.droid.terminate()
    
    @property
    def video(self):
        """Access to DROID's video buffer"""
        return self.droid.video
    
    # ============================================================
    # Your existing point extraction code - UNCHANGED
    # ============================================================
    @torch.no_grad()
    def extract_points_and_poses(self):
        """Extract 3D points and camera poses from DROID"""
        video = self.droid.video
        t = video.counter.value
        
        if t == 0:
            return np.zeros((0, 3)), np.eye(4)[None, ...]
        
        # Get data from video buffer (all PyTorch, no TensorRT involved)
        intrinsics = video.intrinsics[0]
        poses = video.poses[:t]
        disps = video.disps[:t]
        
        # Filtering parameters
        filter_thresh = 0.02
        filter_count = 2
        
        index = torch.arange(t, device="cuda")
        thresh = filter_thresh * torch.ones_like(disps.mean(dim=[1, 2]))
        
        # Back-project to 3D (uses CUDA kernels, not neural networks)
        points = droid_backends.iproj(SE3(poses).inv().data, disps, intrinsics)
        
        # Filter points by depth consistency
        counts = droid_backends.depth_filter(poses, disps, intrinsics, index, thresh)
        
        mask = (counts >= filter_count) & (disps > 0.25 * disps.mean())
        mask = mask.cpu().numpy()
        
        # Extract and filter points
        points = points.cpu().numpy()
        points = points.reshape(-1, 3)[mask.reshape(-1)]
        
        # Height filtering
        points = points[points[:, 1] < 0.5]
        points = points[points[:, 1] > -0.5]
        
        # Get poses
        camera_poses = SE3(poses).inv().data.cpu().numpy()
        
        # Scale points to centimeters
        return points * 1e2, camera_poses


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