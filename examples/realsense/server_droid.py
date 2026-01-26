#!/usr/bin/env python3
"""
Server - Receives RGB, Depth, and Pose data and runs nvblox reconstruction
"""

import grpc
import asyncio
import numpy as np
import cv2
from concurrent import futures
from typing import Iterator
import time
import pyrealsense2 as rs
import torch
import math
import os

# Import generated protobuf code
import sensor_stream_pb2
import sensor_stream_pb2_grpc

from nvblox_torch.examples.realsense.realsense_utils import rs_intrinsics_to_matrix, rs_extrinsics_to_homogeneous
from nvblox_torch.examples.realsense.realsense_dataloader import RealsenseDataloader
from nvblox_torch.examples.realsense.vslam_utils import get_vslam_stereo_rig, to_homogeneous
from rerun_visualizer import RerunVisualizer
from nvblox_torch.projective_integrator_types import ProjectiveIntegratorType
from nvblox_torch.mapper import Mapper
from nvblox_torch.mapper_params import MapperParams, ProjectiveIntegratorParams
from nvblox_torch.timer import Timer, timer_status_string

PRINT_TIMING_EVERY_N_SECONDS = 1.0


class SensorStreamServicer(sensor_stream_pb2_grpc.SensorStreamServiceServicer):
    def __init__(self):
        self.frames_received = 0
        self.latest_pose = None
        
        self.voxel_size_m = 0.01
        self.max_integration_distance_m = 3.0
        self.visualize_mesh_hz = 0.5
        self.last_visualize_mesh_time = time.time()
        self.last_print_time = time.time()

        self.trajectory: List[np.ndarray] = []

        # Visualize in rerun
        self.visualizer = RerunVisualizer()
        self.last_position = None
        print("Server initialized")
    
    def decompress_image(self, data: bytes, encoding: str, width: int, height: int) -> np.ndarray:
        """Decompress image data"""
        if encoding == "jpeg":
            return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        elif encoding == "png":
            img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            # PNG can be either 8-bit or 16-bit, return as-is
            return img
        elif encoding == "bgr8" or encoding == "rgb8":
            # Raw color image - 3 channels
            img = np.frombuffer(data, dtype=np.uint8)
            expected_size = height * width * 3
            if img.size != expected_size:
                print(f"ERROR: Expected {expected_size} bytes for {encoding}, got {img.size}")
                return None
            return img.reshape((height, width, 3))
        elif encoding == "16UC1":
            # Raw 16-bit depth - 1 channel
            img = np.frombuffer(data, dtype=np.uint16)
            expected_size = height * width
            if img.size != expected_size:
                print(f"ERROR: Expected {expected_size} values for depth, got {img.size}")
                return None
            return img.reshape((height, width))
        elif encoding == "64FC1":
            # Raw 16-bit float depth
            expected_size = height * width
            img = np.frombuffer(data, dtype=np.float32)
            if img.size != expected_size:
                print(f"ERROR: Expected {expected_size} values for depth, got {img.size}")
                return None
            return img.reshape((height, width))
        else:
            print(f"Unknown encoding: {encoding}")
            return None

    async def StreamSensorData(
        self,
        request_iterator: Iterator[sensor_stream_pb2.SensorData],
        context: grpc.aio.ServicerContext
    ):
        """Bidirectional stream - receive sensor data, send commands"""
        
        print("Client connected, starting bidirectional stream...")
        
        # Start receiving data in background task
        receive_task = asyncio.create_task(
            self._receive_sensor_data(request_iterator)
        )
        
        try:
            # Yield commands directly in this method
            while not context.cancelled():
                try:
                    # Wait for commands in queue
                    command = await asyncio.wait_for(
                        self.command_queue.get(),
                        timeout=1.0
                    )
                    
                    print(f"📤 Sending command to drone: {command.command}")
                    yield command
                    
                except asyncio.TimeoutError:
                    # Send heartbeat to keep stream alive
                    yield sensor_stream_pb2.DroneCommand(
                        command=sensor_stream_pb2.DroneCommand.NONE,
                        x=self.stg_x_ned,
                        y=self.stg_y_ned,
                        z=self.stg_relative_angle
                    )
                    
        except asyncio.CancelledError:
            print("Stream cancelled")
        finally:
            # Clean up receive task
            receive_task.cancel()
            try:
                await receive_task
            except asyncio.CancelledError:
                pass

    
    async def _receive_sensor_data(self, request_iterator):
        async for sensor_data in request_iterator:
            # Extract pose
            pose = sensor_data.pose
            position = np.array([pose.x, pose.y, pose.z])
            orientation = np.array([pose.qw, pose.qx, pose.qy, pose.qz])
            
            # Decompress images
            color_image = self.decompress_image(
                sensor_data.color_image.data,
                sensor_data.color_image.encoding,
                sensor_data.color_image.width,
                sensor_data.color_image.height
            )
            
            depth_image = self.decompress_image(
                sensor_data.depth_image.data,
                sensor_data.depth_image.encoding,
                sensor_data.depth_image.width,
                sensor_data.depth_image.height
            )                
            # Convert depth to meters if needed
            # if sensor_data.depth_image.encoding == "16UC1":
            # depth_image = depth_image.astype(np.float32) * sensor_data.depth_image.depth_scale

            points_array = []
            if sensor_data.points_data and sensor_data.num_points > 0:
                points_array = np.frombuffer(
                    sensor_data.points_data, 
                    dtype=np.int16
                ).reshape(sensor_data.num_points, 3)
                
            
            # Process with nvblox
            self.process_frame(
                color_image,
                depth_image,
                position,
                orientation,
                points_array,
                sensor_data.color_image.timestamp_us
            )
            
            self.frames_received += 1

    
    def process_frame(
        self,
        color_image: np.ndarray,
        depth_image: np.ndarray,
        position: np.ndarray,
        orientation: np.ndarray,
        points_array,
        timestamp_us: int
    ):  
        dataload_timer = None
        T_W_C_left_infrared = None

        if dataload_timer is not None:
            dataload_timer.stop()

        self.trajectory.append(translation)

        with Timer('visualize_rerun'):
            # Visualize results for color and depth cameras
            # Same observations for both, since we only have one image
            self.visualizer.visualize_frame(
                frame_id=frame_id,
                images=[color_image, depth_image],
                # pose=odom_pose,
                translation=position,
                quaternion=orientation,
                observations_main_cam=[[], []],
                trajectory=trajectory,
                timestamp=timestamp
            )



async def serve(port: int = 50051):
    """Start the gRPC server"""
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
        ]
    )

    os.makedirs('images', exist_ok=True)
    
    servicer = SensorStreamServicer()
    sensor_stream_pb2_grpc.add_SensorStreamServiceServicer_to_server(
        servicer, server
    )
    
    server.add_insecure_port(f'[::]:{port}')
    
    print(f"Server starting on port {port}...")
    await server.start()
    
    print("Server ready to receive data")
    
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        await server.stop(5)


if __name__ == "__main__":
    asyncio.run(serve(port=50051))
