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

from dpvo_slam_module import DpvoSLAM
from sensor import Sensor

from rerun_visualizer import RerunVisualizer

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

        self.target_x = 0
        self.target_y = 0
        self.target_z = 0
        self.target_angle = 0

        self.trajectory: List[np.ndarray] = []

        # Visualize in rerun
        self.visualizer = RerunVisualizer()
        self.last_position = None
        print("Server initialized")

        # self.droid_slam = DpvoSLAM(intrinsics=None, H=480, W=640)
    
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
        
        try:
            # Process incoming data
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
                
                points_array = []
                points_3d = []
                points_2d = []
                if sensor_data.points_data and sensor_data.num_points > 0:
                    points_array = np.frombuffer(
                        sensor_data.points_data, 
                        dtype=np.float16
                    ).reshape(sensor_data.num_points, 3)
                    for p in points_array:
                        if p[2] in [-1000, -1001, -2000, -2001]:  # obstacle map, explored area, ltg, stg
                            # 2d map
                            points_2d.append(p)
                        else:
                            points_3d.append(p)
                    # points_array = points_array / 100
                
                # Process frame
                self.process_frame(
                    color_image,
                    depth_image,
                    position,
                    orientation,
                    points_3d,
                    points_2d,
                    sensor_data.color_image.timestamp_us
                )
                
                self.frames_received += 1
                
                # Optionally send commands back to client
                # If you want to send commands, yield them here:
                # command = sensor_stream_pb2.Command(command="some_command")
                # yield command
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
                        x=0,
                        y=0,
                        z=0
                    )
                
        except Exception as e:
            print(f"Error in stream: {e}")
            import traceback
            traceback.print_exc()

    
    def process_frame(
        self,
        color_image: np.ndarray,
        depth_image: np.ndarray,
        position: np.ndarray,
        orientation: np.ndarray,
        points_3d,
        points_2d,
        timestamp_us: int
    ):  
        dataload_timer = None
        T_W_C_left_infrared = None

        filtered_points = []
        for point in points_3d:
            if point[0] < -15 or point[0] > 10:
                continue
            if point[1] < -15 or point[1] > 10:
                continue
            if point[2] < -15 or point[2] > 10:
                continue
            filtered_points.append(point)
        points_3d = filtered_points

        if dataload_timer is not None:
            dataload_timer.stop()

        self.trajectory.append(position)

        yaw, _, _ = quaternion_to_euler(orientation[0], orientation[1], orientation[2], orientation[3])

        # Visualize results for color and depth cameras
        # Same observations for both, since we only have one image
        self.visualizer.visualize_frame(
            frame_id=self.frames_received,
            images=[color_image, depth_image],
            map_3d=points_3d,
            map_2d=points_2d,
            # pose=odom_pose,
            translation=position,
            quaternion=orientation,
            yaw=yaw,
            observations_main_cam=[[], []],
            trajectory=self.trajectory,
            timestamp=timestamp_us
        )


    def save_image(self, image: np.ndarray, quality: int = 100):
        """Compress image to JPEG"""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        cv2.imwrite(f'images/rgb/rgb_{self.frames_received:05}.jpg', image, encode_param)

    def save_depth(self, depth: np.ndarray):
        """Compress depth using PNG (lossless)"""
        cv2.imwrite(f'images/depth/depth_{self.frames_received:05}.png', depth * 100)

    async def send_command_to_drone(self, command_type, **kwargs):
        """Queue a command to send to drone"""
        # Extract values from kwargs with defaults
        delta_x = kwargs.get('delta_x', 0)
        delta_y = kwargs.get('delta_y', 0)
        delta_z = kwargs.get('delta_z', 0)
        delta_angle = kwargs.get('delta_angle', 0)

        self.target_x += delta_x
        self.target_y += delta_y
        self.target_z += delta_z
        self.target_angle += delta_angle

        command = sensor_stream_pb2.DroneCommand(
            command=command_type,
            x=self.target_x,
            y=self.target_y,
            z=self.target_z,
            velocity=self.target_angle,   # todo: cleanup
        )
        await self.command_queue.put(command)
        print(f"Queued command: {command_type}")


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
                    sensor_stream_pb2.DroneCommand.FOLLOW_ONCE, delta_x=0.1
                )
            elif cmd == 'c':
                await servicer.send_command_to_drone(
                    sensor_stream_pb2.DroneCommand.FOLLOW_ONCE, delta_angle=-10
                )
            elif cmd == 'v':
                await servicer.send_command_to_drone(
                    sensor_stream_pb2.DroneCommand.FOLLOW_ONCE, delta_angle=10
                )
            elif cmd == 'q':
                break
            """
            elif cmd == 'r':
                servicer.export_recording()

            elif cmd == 's':
                servicer.enable_saving_frames()

            elif cmd == 'd':
                servicer.disable_saving_frames()
            """
            


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
