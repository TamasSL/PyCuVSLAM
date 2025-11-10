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

# Import generated protobuf code
import sensor_stream_pb2
import sensor_stream_pb2_grpc

from nvblox_torch.examples.realsense.realsense_utils import rs_intrinsics_to_matrix, rs_extrinsics_to_homogeneous
from nvblox_torch.examples.realsense.realsense_dataloader import RealsenseDataloader
from nvblox_torch.examples.realsense.vslam_utils import get_vslam_stereo_rig, to_homogeneous
from nvblox_torch.examples.realsense.visualizer import RerunVisualizer
from nvblox_torch.projective_integrator_types import ProjectiveIntegratorType
from nvblox_torch.mapper import Mapper
from nvblox_torch.mapper_params import MapperParams, ProjectiveIntegratorParams
from nvblox_torch.timer import Timer, timer_status_string

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


class SensorStreamServicer(sensor_stream_pb2_grpc.SensorStreamServiceServicer):
    def __init__(self):
        self.frames_received = 0
        self.latest_pose = None
        
        self.voxel_size_m = 0.007
        self.max_integration_distance_m = 2.0
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
        self.last_position = None
        
        print("Server initialized")

    def _initialize_intrinsics(self):
        self.depth_intrinsic = rs.pyrealsense2.intrinsics()
        self.depth_intrinsic.width = 640
        self.depth_intrinsic.height = 480
        self.depth_intrinsic.ppx = 327.8804016113281
        self.depth_intrinsic.ppy = 241.63455200195312
        self.depth_intrinsic.fx = 383.58721923828125
        self.depth_intrinsic.fy = 383.0414123535156
        self.depth_intrinsic.model = rs.pyrealsense2.distortion.inverse_brown_conrady
        self.depth_intrinsic.coeffs = [-0.05629764497280121, 0.0679863691329956, -0.0002407759748166427, 0.0009052956593222916, -0.021819550544023514]

        self.color_intrinsic = rs.pyrealsense2.intrinsics()
        self.color_intrinsic.width = 640
        self.color_intrinsic.height = 480
        self.color_intrinsic.ppx = 327.8804016113281
        self.color_intrinsic.ppy = 241.63455200195312
        self.color_intrinsic.fx = 383.58721923828125
        self.color_intrinsic.fy = 383.0414123535156
        self.color_intrinsic.model = rs.pyrealsense2.distortion.inverse_brown_conrady
        self.color_intrinsic.coeffs = [-0.05629764497280121, 0.0679863691329956, -0.0002407759748166427, 0.0009052956593222916, -0.021819550544023514]
    
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
    ) -> Iterator[sensor_stream_pb2.StreamResponse]:
        """Handle streaming sensor data"""
        
        print("Client connected, starting to receive data...")
        
        try:
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
                    ).reshape(sensor_data.num_points, 2)
                    
                    print(f"Points shape: {points_array.shape}")
                
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
                
                # Send acknowledgment every 10 frames
                if self.frames_received % 10 == 0:
                    response = sensor_stream_pb2.StreamResponse(
                        success=True,
                        message="OK",
                        frames_received=self.frames_received
                    )
                    yield response
                    print(f"Processed {self.frames_received} frames")
                
        except Exception as e:
            print(f"Error processing stream: {e}")
            yield sensor_stream_pb2.StreamResponse(
                success=False,
                message=str(e),
                frames_received=self.frames_received
            )
    
    def process_frame(
        self,
        color_image: np.ndarray,
        depth_image: np.ndarray,
        position: np.ndarray,
        orientation: np.ndarray,
        points_array,
        timestamp_us: int
    ):
        """Process frame with nvblox"""

        #if self.last_position is not None:
        #    distance = np.linalg.norm(position - self.last_position)
        #    print(f"Camera moved: {distance:.3f} meters")
        #self.last_position = position.copy()
        
        dataload_timer = None
        T_W_C_left_infrared = None

        if dataload_timer is not None:
            dataload_timer.stop()

        # Do reconstruction using the depth
        with Timer('depth'):
            T_W_C_left_infrared = to_homogeneous(position, orientation)
            T_W_C_left_infrared = torch.from_numpy(T_W_C_left_infrared).float()
            if depth_image is not None and \
                T_W_C_left_infrared is not None:     
                depth = torch.from_numpy(depth_image).float().to('cuda')     
                depth_intrinsics = torch.from_numpy(rs_intrinsics_to_matrix(self.depth_intrinsic)).float()      
                self.nvblox_mapper.add_depth_frame(depth, T_W_C_left_infrared, depth_intrinsics)

        with Timer('color'):
            if T_W_C_left_infrared is not None and \
                color_image is not None:
                # Convert the left infrared camera pose to the color camera frame
                T_W_C_color = T_W_C_left_infrared
                color = torch.from_numpy(color_image).to('cuda')     
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

            self.visualizer._visualize_map(points_array)

            drone_pos = [[-data['position'][0] * 10 + 40, data['position'][2] * 10 + 40]] # shifted by map-size for centering (400)
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
        
        # For now, just log
        print(f"Frame at position: {position}, orientation: {orientation}")
        
        # Optional: Save frames for debugging
        # cv2.imwrite(f"frame_{self.frames_received}_color.jpg", color_image)

    
    async def SendFrame(
        self,
        request: sensor_stream_pb2.SensorData,
        context: grpc.aio.ServicerContext
    ) -> sensor_stream_pb2.StreamResponse:
        """Handle single frame (unary call)"""
        
        # Process single frame
        pose = request.pose
        position = np.array([pose.x, pose.y, pose.z])
        orientation = np.array([pose.qw, pose.qx, pose.qy, pose.qz])
        
        color_image = self.decompress_image(
            request.color_image.data,
            request.color_image.encoding
        )
        
        depth_image = self.decompress_image(
            request.depth_image.data,
            request.depth_image.encoding
        )
        
        self.process_frame(color_image, depth_image, position, orientation, request.color_image.timestamp_us)
        
        return sensor_stream_pb2.StreamResponse(
            success=True,
            message="Frame processed",
            frames_received=1
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
    
    sensor_stream_pb2_grpc.add_SensorStreamServiceServicer_to_server(
        SensorStreamServicer(), server
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
