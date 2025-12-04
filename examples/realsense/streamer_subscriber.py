import grpc
import asyncio
import numpy as np
import cv2
import time
from typing import Optional

# Import generated protobuf code
# Run: python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. sensor_stream.proto
import sensor_stream_pb2
import sensor_stream_pb2_grpc

import threading
import queue

import pyrealsense2 as rs
from publish_subscribe import Publisher

SERVER_ADDRESS = "192.168.0.103:50051"


class StreamerSubscriber:
    def __init__(self, slam_publisher: Publisher, command_publisher: Publisher, name: str = "Subscriber", server_address: str = SERVER_ADDRESS, compress_images: bool = False):
        self.name = name
        self.queue = slam_publisher.subscribe()
        self._command_publisher = command_publisher
        self._running = False
        self._thread = None
        self._processed_count = 0
        self.server_address = server_address
        self.compress_images = compress_images
        self.frame_id = 0
        
        # Stream state
        self.channel = None
        self.stub = None
        self._data_queue = asyncio.Queue(maxsize=20)  # Internal async queue for generator
    
    def start(self):
        """Start processing in background thread"""
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        print(f"{self.name} started")
    
    def _process_loop(self):
        """Main processing loop"""
        self.channel = grpc.insecure_channel(
            self.server_address,
            options=[
                ('grpc.max_send_message_length', 50 * 1024 * 1024),
                ('grpc.max_receive_message_length', 50 * 1024 * 1024),
                ('grpc.default_compression_algorithm', grpc.Compression.Gzip),
            ]
        )
        self.stub = sensor_stream_pb2_grpc.SensorStreamServiceStub(self.channel)
        
        try:
            def generate_sensor_data():
                while self._running:
                    try:
                        data = self.queue.get(timeout=0.1)
                        sensor_data = self._to_protobuf(data)
                        del data
                        yield sensor_data
                        del sensor_data
                        self._processed_count += 1
                    except queue.Empty:
                        continue

            # Create stream
            print(f"{self.name}: Starting stream to {self.server_address}")
            command_stream = self.stub.StreamSensorData(generate_sensor_data())
            
            for command in command_stream:
                print(f"📨 Received command from server: {command.command}")
                
                # Schedule the command to be executed by offboard controller
                self._command_publisher.publish(command)

        except grpc.RpcError as e:
            print(f"gRPC error: {e.code()}: {e.details()}")
        except Exception as e:
            print(f"Streaming error: {e}")
        finally:
            if self.channel:
                self.channel.close()
    
    def _to_protobuf(self, data) -> sensor_stream_pb2.SensorData:
        """Convert queue data to protobuf message"""
        # Extract pose
        pose_msg = sensor_stream_pb2.Pose()
        pose = data['position']
        quaternion = data['quaternion']
        pose_msg.x = pose[0]
        pose_msg.y = pose[1]
        pose_msg.z = pose[2]
        pose_msg.qw = quaternion[0]
        pose_msg.qx = quaternion[1]
        pose_msg.qy = quaternion[2]
        pose_msg.qz = quaternion[3]
        
        # Compress images if enabled
        if self.compress_images:
            color_data = self.compress_image(data['rgb'])
            depth_data = self.compress_depth(data['depth'])
            color_encoding = "jpeg"
            depth_encoding = "png"
        else:
            color_data = data['rgb'].tobytes()
            depth_data = data['depth'].tobytes()
            color_encoding = "bgr8"
            depth_encoding = "64FC1"
        
        # Create color image message
        color_msg = sensor_stream_pb2.ImageFrame(
            data=color_data,
            width=data['rgb'].shape[1],
            height=data['rgb'].shape[0],
            encoding=color_encoding,
            timestamp_us=0
        )
        
        # Create depth image message
        depth_msg = sensor_stream_pb2.DepthFrame(
            data=depth_data,
            width=data['depth'].shape[1],
            height=data['depth'].shape[0],
            encoding=depth_encoding,
            depth_scale=0.001,
            timestamp_us=0
        )
        
        # Combine into sensor data message
        sensor_data = sensor_stream_pb2.SensorData(
            pose=pose_msg,
            color_image=color_msg,
            depth_image=depth_msg,
            frame_id=self.frame_id
        )

        # Convert points to numpy array and serialize
        if 'points' in data and len(data['points']) > 0:
            points_array = np.array(data['points'], dtype=np.int16)  # Shape: (N, 3)
        
            sensor_data.points_data = points_array.tobytes()
            sensor_data.num_points = len(points_array)

        self.frame_id += 1
        
        return sensor_data
    
    def compress_image(self, image: np.ndarray, quality: int = 80) -> bytes:
        """Compress image to JPEG"""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', image, encode_param)
        return encoded.tobytes()
    
    def compress_depth(self, depth: np.ndarray) -> bytes:
        """Compress depth using PNG (lossless)"""
        _, encoded = cv2.imencode('.png', depth)
        return encoded.tobytes()
