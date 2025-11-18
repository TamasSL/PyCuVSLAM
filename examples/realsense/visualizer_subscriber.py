#
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import pyrealsense2 as rs
import time

import torch
import argparse
import cuvslam as vslam
import sys

import threading
import queue

from camera_utils import get_rs_stereo_rig
from publish_subscribe import Publisher

from nvblox_torch.examples.realsense.realsense_utils import rs_intrinsics_to_matrix, rs_extrinsics_to_homogeneous
from nvblox_torch.examples.realsense.realsense_dataloader import RealsenseDataloader
from nvblox_torch.examples.realsense.vslam_utils import get_vslam_stereo_rig, to_homogeneous
from visualizer import RerunVisualizer
from nvblox_torch.projective_integrator_types import ProjectiveIntegratorType
from nvblox_torch.mapper import Mapper
from nvblox_torch.mapper_params import MapperParams, ProjectiveIntegratorParams
from nvblox_torch.timer import Timer, timer_status_string

# pylint: disable=invalid-name

PRINT_TIMING_EVERY_N_SECONDS = 1.0

def quaternion_to_euler(qx, qy, qz, qw):
        """
        Extract yaw (rotation about vertical/y-axis) from quaternion.
        For camera frame where y is up.
        """
        import numpy as np

        bqw = qw
        bqx = qz
        bqy = qx
        bqz = -qy
        
        # Yaw is rotation about the y-axis (vertical in camera frame)
        # Standard yaw extraction from quaternion:
        siny_cosp = 2 * (bqw * bqz + bqx * bqy)
        cosy_cosp = 1 - 2 * (bqy * bqy + bqz * bqz)
        yaw = -np.arctan2(siny_cosp, cosy_cosp)

        roll = np.arctan2(2 * (bqw * bqx + bqy * bqz), 1 - 2 * (bqx**2 + bqy**2))

        sinp = 2 * (bqw * bqy - bqz * bqx)
        pitch = float(np.where(np.abs(sinp) >= 1, np.copysign(np.pi / 2, sinp), np.arcsin(sinp)))
        
        return yaw, roll, pitch

class VisualizerSubscriber:
    def __init__(self, publisher: Publisher, name: str = "Subscriber"):
        self.name = name
        self.queue = publisher.subscribe()
        self._running = False
        self._thread = None
        self._processed_count = 0

        self.voxel_size_m = 0.01
        self.max_frames = 5000
        self.max_integration_distance_m = 5.0
        self.visualize_mesh_hz = 5
        self.last_visualize_mesh_time = time.time()
        self.last_print_time = time.time()
        self._initialize_intrinsics()

        # Create some parameters
        projective_integrator_params = ProjectiveIntegratorParams()
        projective_integrator_params.projective_integrator_max_integration_distance_m = \
            self.max_integration_distance_m
        mapper_params = MapperParams()
        mapper_params.set_projective_integrator_params(projective_integrator_params)

        # Initialize nvblox mapper
        self.nvblox_mapper = Mapper(voxel_sizes_m=self.voxel_size_m,
                            integrator_types=ProjectiveIntegratorType.TSDF,
                            mapper_parameters=mapper_params)

        # Visualize in rerun
        self.visualizer = RerunVisualizer()

    def _initialize_intrinsics(self):
        self.depth_intrinsic = rs.pyrealsense2.intrinsics()
        self.depth_intrinsic.width = 640
        self.depth_intrinsic.height = 480
        self.depth_intrinsic.ppx = 321.09686279296875
        self.depth_intrinsic.ppy = 238.68887329101562
        self.depth_intrinsic.fx = 391.66400146484375
        self.depth_intrinsic.fy = 391.66400146484375
        self.depth_intrinsic.model = rs.pyrealsense2.distortion.brown_conrady
        self.depth_intrinsic.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]

        self.color_intrinsic = rs.pyrealsense2.intrinsics()
        self.color_intrinsic.width = 640
        self.color_intrinsic.height = 480
        self.color_intrinsic.ppx = 327.8804016113281
        self.color_intrinsic.ppy = 241.63455200195312
        self.color_intrinsic.fx = 383.58721923828125
        self.color_intrinsic.fy = 383.0414123535156
        self.color_intrinsic.model = rs.pyrealsense2.distortion.inverse_brown_conrady
        self.color_intrinsic.coeffs = [-0.05629764497280121, 0.0679863691329956, -0.0002407759748166427, 0.0009052956593222916, -0.021819550544023514]

    
    def start(self):
        """Start processing in background thread"""
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        print(f"{self.name} started")
    
    def stop(self):
        """Stop processing"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        print(f"{self.name} stopped (processed {self._processed_count} items)")
    
    def _process_loop(self):
        """Main processing loop (runs in separate thread)"""
        while self._running:
            try:
                # Wait for data with timeout to allow clean shutdown
                data = self.queue.get(timeout=0.1)
                
                # Process data
                self.visualize_callback(data)
                self._processed_count += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"{self.name} error: {e}")
        

    def visualize_callback(self, data) -> int:
        dataload_timer = None
        T_W_C_left_infrared = None

        if dataload_timer is not None:
            dataload_timer.stop()

        # Do reconstruction using the depth
        with Timer('depth'):
            T_W_C_left_infrared = to_homogeneous(data['position'], data['quaternion'])
            T_W_C_left_infrared = torch.from_numpy(T_W_C_left_infrared).float()
            if data['depth'] is not None and \
                T_W_C_left_infrared is not None:     
                depth = torch.from_numpy(data['depth']).float().to('cuda')     
                depth_intrinsics = torch.from_numpy(rs_intrinsics_to_matrix(self.depth_intrinsic)).float()      
                self.nvblox_mapper.add_depth_frame(depth, T_W_C_left_infrared, depth_intrinsics)

        with Timer('color'):
            if T_W_C_left_infrared is not None and \
                data['rgb'] is not None:
                # Convert the left infrared camera pose to the color camera frame
                T_W_C_color = T_W_C_left_infrared
                color = torch.from_numpy(data['rgb']).to('cuda')     
                color_intrinsics = torch.from_numpy(rs_intrinsics_to_matrix(self.color_intrinsic)).float()      
                self.nvblox_mapper.add_color_frame(color, T_W_C_color, color_intrinsics)

        with Timer('visualize_rerun'):
            # Visualize pose. This occurs every time we track.
            #if T_W_C_left_infrared is not None and data['left_infrared_image'] is not None:
            #    self.visualizer.visualize_cuvslam(T_W_C_left_infrared.cpu().numpy(),
            #                                 data['left_infrared_image'],
            #                                 data['last_observation'])

            if T_W_C_left_infrared is not None:
                self.visualizer.visualize_cuvslam(T_W_C_left_infrared.cpu().numpy(), None, None)

            self.visualizer._visualize_map(data['points'])

            drone_pos = [[-data['position'][0] * 10 + 80, data['position'][2] * 10 + 80]] # shifted by map-size for centering
            yaw, roll, pitch = quaternion_to_euler(data['quaternion'][0], data['quaternion'][1], data['quaternion'][2], data['quaternion'][3])
            self.visualizer._visualize_drone(drone_pos, yaw)

            # Visualize mesh. This is performed at an (optionally) reduced rate.
            current_time = time.time()
            if (current_time - self.last_visualize_mesh_time) >= (1.0 / self.visualize_mesh_hz):
                with Timer('mesh/update'):
                    self.nvblox_mapper.update_color_mesh()
                with Timer('mesh/to_cpu'):
                    color_mesh = self.nvblox_mapper.get_color_mesh()
                with Timer('visualize/mesh'):
                    self.visualizer.visualize_nvblox(color_mesh)
                self.last_visualize_mesh_time = current_time

        # Print timing statistics
        current_time = time.time()
        if current_time - self.last_print_time >= PRINT_TIMING_EVERY_N_SECONDS:
            print(timer_status_string())
            self.last_print_time = current_time

        # This timer times how long it takes to get the next frame
        dataload_timer = Timer('dataload')

        # Print final timing statistics
        print(timer_status_string())

        print('Done')

        return 0
