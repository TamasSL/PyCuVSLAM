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
from visualizer import RerunVisualizer
from nvblox_torch.projective_integrator_types import ProjectiveIntegratorType
from nvblox_torch.mapper import Mapper
from nvblox_torch.mapper_params import MapperParams, ProjectiveIntegratorParams
from nvblox_torch.timer import Timer, timer_status_string
from fmm_planner import FMMPlanner
from exploration_planner import ExplorationPlanner
from map_builder import MapBuilder

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

        self.command_queue = asyncio.Queue()  # Commands to send to drone
        
        self.voxel_size_m = 0.01
        self.max_integration_distance_m = 3.0
        self.visualize_mesh_hz = 0.5
        self.last_visualize_mesh_time = time.time()
        self.last_print_time = time.time()
        self._initialize_intrinsics()

        # Visualize in rerun
        self.visualizer = RerunVisualizer()
        self.last_position = None

        self.map_size = 160   # the 2d map is of size map_size x map_size where each element represents a 10x10cm square
        self.stg_x_gt = 0
        self.stg_y_gt = 0
        self.stg_h = 0
        self.stg_x_ned = 0
        self.stg_y_ned = 0
        self.stg_z_ned = -0.25
        self.stg_relative_angle = 0
        self.initial_plan = True
        self.long_term_goal_planner = ExplorationPlanner(grid_resolution=0.1, safety_distance=0.3, visualizer=self.visualizer)
        self.frame_id = 0
        self.save_frames_to_disk = False

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

        self.map_builder = MapBuilder()
        
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

    async def send_command_to_drone(self, command_type, **kwargs):
        """Queue a command to send to drone"""
        command = sensor_stream_pb2.DroneCommand(
            command=command_type,
            x=self.stg_x_ned,
            y=self.stg_y_ned,
            z=self.stg_z_ned,
            velocity=self.stg_relative_angle,   # todo: cleanup
        )
        await self.command_queue.put(command)
        print(f"Queued command: {command_type}")

    
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
                self.save_depth(depth_image)

        with Timer('color'):
            if T_W_C_left_infrared is not None and \
                color_image is not None:
                # Convert the left infrared camera pose to the color camera frame
                T_W_C_color = T_W_C_left_infrared
                color = torch.from_numpy(color_image).to('cuda')     
                color_intrinsics = torch.from_numpy(rs_intrinsics_to_matrix(self.color_intrinsic)).float()      
                self.nvblox_mapper.add_color_frame(color, T_W_C_color, color_intrinsics)
                self.save_image(color_image)

        with Timer('visualize_rerun'):
            # Visualize pose. This occurs every time we track.
            #if T_W_C_left_infrared is not None and data['left_infrared_image'] is not None:
            #    self.visualizer.visualize_cuvslam(T_W_C_left_infrared.cpu().numpy(),
            #                                 data['left_infrared_image'],
            #                                 data['last_observation'])

            if T_W_C_left_infrared is not None:
                self.visualizer.visualize_cuvslam(T_W_C_left_infrared.cpu().numpy(), None, None)

            drone_pos = [[-position[0] * 10 + 80, position[2] * 10 + 80, math.floor(-position[1] * 2 + 0.4)]] # shifted by map-size for centering
            yaw, roll, pitch = quaternion_to_euler(orientation[0], orientation[1], orientation[2], orientation[3])
            self.visualizer._visualize_drone(drone_pos, yaw)

            traversible = np.ones(
                (
                    3,
                    self.map_size,
                    self.map_size
                ),
                dtype=np.uint8,
            )
            long_term_grid = np.full(
                (self.map_size, self.map_size),
                -1,
                dtype=np.int8,
            )
            for p in points_array:
                x = p[0]
                y = p[1]
                if p[2] == 0: # obstacle at current level
                    traversible[1][x][y] = 0
                    for i in range(max(x-2,0),min(x+3, self.map_size)):
                        for j in range(max(y-2,0),min(y+3, self.map_size)):
                            traversible[1][i][j] = 0
                    long_term_grid[x][y] = 1
                elif p[2] == 2: #explored, free space
                    long_term_grid[x][y] = 0
                if p[2] == 1: # obstacle at level above
                    traversible[2][x][y] = 0
                    for i in range(max(x-1,0),min(x+2, self.map_size)):
                        for j in range(max(y-1,0),min(y+2, self.map_size)):
                            traversible[2][i][j] = 0
                if p[2] == -1: # obstacle at level below
                    traversible[0][x][y] = 0
                    for i in range(max(x-1,0),min(x+2, self.map_size)):
                        for j in range(max(y-1,0),min(y+2, self.map_size)):
                            traversible[0][i][j] = 0
        
            fmm_planner = FMMPlanner(traversible, "fm2")

            lt_target = self.long_term_goal_planner.get_next_exploration_target(
                long_term_grid, 
                [drone_pos[0][0], drone_pos[0][1]],
                max_distance=50
            )
            
            #if target is None:
            #    print("✅ Exploration complete!")

            ltg = [80, 90] # lt_target or [80, 90]
            reachable = fmm_planner.set_goal((ltg[1], ltg[0]))
            # print(f"drone pos {-position[1]}")
            if self.initial_plan:
                self.stg_x_gt, self.stg_y_gt, diff_z, replan = fmm_planner.get_short_term_goal([drone_pos[0][0], drone_pos[0][1], drone_pos[0][2]])
                self.stg_h += diff_z
                self.initial_plan = False
                print(f"stg: {self.stg_x_gt} {self.stg_y_gt} {self.stg_h} {replan}")
            elif ((abs(drone_pos[0][0] - self.stg_x_gt) < 3) and (abs(drone_pos[0][1] - self.stg_y_gt) < 3)) and drone_pos[0][2] == self.stg_h:
                self.stg_x_gt, self.stg_y_gt, diff_z, replan = fmm_planner.get_short_term_goal([self.stg_x_gt, self.stg_y_gt, self.stg_h])
                self.stg_h += diff_z
                print(f"stg: {self.stg_x_gt} {self.stg_y_gt} {self.stg_h} {replan}")
            
            self.visualizer._visualize_goal([[ltg[0], ltg[1]]])
            self.visualizer._visualize_map(points_array)

            angle_st_goal = math.degrees(
                math.atan2(self.stg_x_gt - drone_pos[0][0], self.stg_y_gt - drone_pos[0][1])
            )
            yaw_degrees = math.degrees(yaw)
            angle_agent = (yaw_degrees) % 360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = angle_st_goal % 360 # (angle_agent + angle_st_goal) % 360.0
            if relative_angle > 180:
                relative_angle -= 360

            self.stg_x_ned = (self.stg_y_gt - self.map_size / 2) / 10
            self.stg_y_ned = -(self.stg_x_gt - self.map_size / 2) / 10
            self.stg_z_ned = -self.stg_h * 0.5 - 0.25 # - 0.25
            #print(f"Target height: {self.stg_z_ned}")
            self.stg_relative_angle = -relative_angle


            # print(f"angle_agent: {angle_agent}, st_angle: {-angle_st_goal}, rel_angle: {relative_angle}")

            self.visualizer._visualize_stg([self.stg_x_gt, self.stg_y_gt], self.stg_h)
            
            # Visualize mesh. This is performed at an (optionally) reduced rate.
            current_time = time.time()
            if (current_time - self.last_visualize_mesh_time) >= (1.0 / self.visualize_mesh_hz):
                self.nvblox_mapper.update_color_mesh()
                color_mesh = self.nvblox_mapper.get_color_mesh()
                with Timer('visualize/mesh'):
                    self.visualizer.visualize_nvblox(color_mesh)
                self.last_visualize_mesh_time = current_time

    def export_recording(self):
        self.nvblox_mapper.update_color_mesh()
        color_mesh = self.nvblox_mapper.get_color_mesh()
        print("writing to file")
        success = color_mesh.save('/home/tamas/Documents/code/fork/PyCuVSLAM/examples/realsense/output_mesh.glb')
        print("written")
        print(success)

    def save_image(self, image: np.ndarray, quality: int = 80):
        """Compress image to JPEG"""
        if self.save_frames_to_disk:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            cv2.imwrite(f'images/rgb_{self.frame_id}.jpg', image, encode_param)

    def save_depth(self, depth: np.ndarray):
        """Compress depth using PNG (lossless)"""
        if self.save_frames_to_disk:
            self.frame_id += 1
            cv2.imwrite(f'images/depth_{self.frame_id}.png', depth * 100)

    def enable_saving_frames(self):
        self.save_frames_to_disk = True

    def disable_saving_frames(self):
        self.save_frames_to_disk = False

async def manual_control(servicer):
        """Example: Send commands from server keyboard"""
        while True:
            print("\nCommands: a=arm, t=takeoff, l=land, f=follow once, c=follow continously, v=unfollow, r=record 3d mesh, s=start saving RGB/depth frames, d=stop saving frames, q=quit")
            cmd = await asyncio.to_thread(input, "Command: ")
            
            if cmd == 'a':
                await servicer.send_command_to_drone(
                    sensor_stream_pb2.DroneCommand.ARM
                )
            elif cmd == 't':
                await servicer.send_command_to_drone(
                    sensor_stream_pb2.DroneCommand.TAKEOFF
                )
            elif cmd == 'l':
                await servicer.send_command_to_drone(
                    sensor_stream_pb2.DroneCommand.LAND
                )
            elif cmd == 'f':
                await servicer.send_command_to_drone(
                    sensor_stream_pb2.DroneCommand.FOLLOW_ONCE
                )
            elif cmd == 'c':
                await servicer.send_command_to_drone(
                    sensor_stream_pb2.DroneCommand.FOLLOW
                )
            elif cmd == 'v':
                await servicer.send_command_to_drone(
                    sensor_stream_pb2.DroneCommand.UNFOLLOW
                )
            
            elif cmd == 'r':
                servicer.export_recording()

            elif cmd == 's':
                servicer.enable_saving_frames()

            elif cmd == 'd':
                servicer.disable_saving_frames()

            elif cmd == 'q':
                break

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

    # Start manual control task
    control_task = asyncio.create_task(manual_control(servicer))
    
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        await server.stop(5)


if __name__ == "__main__":
    asyncio.run(serve(port=50051))
