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
from typing import List, Optional

import asyncio
import numpy as np
import pyrealsense2 as rs
import time

import torch
from mavsdk import System
from mavsdk.mocap import VisionPositionEstimate, Quaternion, PositionBody, AngleBody, Covariance, SpeedBody, AngularVelocityBody, Odometry

from fmm_planner_2d import FMMPlanner
from droid_slam_module import DroidSLAM
from droid_map_builder import DroidMapBuilder
from sensor import Sensor
from streamer_subscriber import StreamerSubscriber
from publish_subscribe import Publisher
from offboard_controller_subscriber import OffboardControllerSubscriber

# Constants
RESOLUTION = (640, 480)
FPS = 30
WARMUP_FRAMES = 60
IMAGE_JITTER_THRESHOLD_MS = 35 * 1e6  # 35ms in nanoseconds
NUM_VIZ_CAMERAS = 2

# DRONE_ADDRESS = "serial:///dev/ttyTHS1:921600"  # JETSON
DRONE_ADDRESS = "serial:///dev/ttyACM0:57600"

def reset_realsense_device():
    """Reset all RealSense devices"""
    ctx = rs.context()
    devices = ctx.query_devices()
    for dev in devices:
        dev.hardware_reset()
        print(f"Reset device: {dev.get_info(rs.camera_info.name)}")
    
    # Wait for device to come back online
    import time
    time.sleep(3)


def transform_to_ned(x, y, z):
    """Transform camera frame pose to PX4 NED frame pose"""
    ned_x = z
    ned_y = x
    ned_z = y
    
    return ned_x, ned_y, ned_z

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


async def print_ned_coordinates(drone):
    async for position_ned in drone.telemetry.position_velocity_ned():
        x = position_ned.position.north_m
        y = position_ned.position.east_m
        z = position_ned.position.down_m
        print(f"Position NED: x={x:.2f}, y={y:.2f}, z={z:.2f}")
        break

    async for attitude in drone.telemetry.attitude_euler():
        r = attitude.roll_deg
        p = attitude.pitch_deg
        y = attitude.yaw_deg
        # print(f"Attitude: roll={r:.1f}, pitch={p:.1f}, yaw={y:.1f}")
        break

async def main() -> None:
    reset_realsense_device()

    """Main function for RGBD tracking."""
    # Initialize RealSense configuration
    config = rs.config()
    pipeline = rs.pipeline()

    # Configure streams
    config.enable_stream(
        rs.stream.color, RESOLUTION[0], RESOLUTION[1], rs.format.bgr8, FPS
    )
    config.enable_stream(
        rs.stream.depth, RESOLUTION[0], RESOLUTION[1], rs.format.z16, FPS
    )

    # Start pipeline to get intrinsics and extrinsics
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    align_to = rs.stream.color
    align = rs.align(align_to)

    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    # Prepare camera parameters
    camera_params = {'left': {}}

    # Get intrinsics
    color_profile = color_frame.profile.as_video_stream_profile()
    camera_params['left']['intrinsics'] = color_profile.intrinsics

    pipeline.stop()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    # Enable IR emitter for depth sensing
    depth_sensor = device.query_sensors()[0]
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 1)

    reset_realsense_device()

    # Start pipeline for tracking
    profile = pipeline.start(config)

    drone = System()
    await drone.connect(system_address=DRONE_ADDRESS)

    event_loop = asyncio.get_running_loop()
    command_publisher = Publisher(maxsize=3)
    offboard_controller = OffboardControllerSubscriber(command_publisher, drone, event_loop)
    offboard_controller.start()

    """
    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("✅ Drone connected!")
            break
    """

    frame_id = 0
    prev_timestamp: Optional[int] = None

    slam_publisher = Publisher(maxsize=5)
    streamer_subscriber = StreamerSubscriber(slam_publisher, command_publisher, "StreamerSubscriber")
    streamer_subscriber.start()

    calib = np.loadtxt("thirdparty/dpvo/calib/d435i.txt", delimiter=" ")
    fx, fy, cx, cy = calib[:4]
    intrinsics = torch.as_tensor([fx, fy, cx, cy])
    droid_slam = DroidSLAM(intrinsics)
    map_builder = DroidMapBuilder()

    try:
        while True:
            # Wait for frames
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                print("Warning: No aligned depth or color frame")
                continue

            timestamp = int(color_frame.timestamp * 1e6)  # Convert to nanoseconds

            # Check timestamp difference with previous frame
            if prev_timestamp is not None:
                timestamp_diff = timestamp - prev_timestamp
                if timestamp_diff > IMAGE_JITTER_THRESHOLD_MS:
                    print(
                        f"Warning: Camera stream message drop: timestamp gap "
                        f"({timestamp_diff/1e6:.2f} ms) exceeds threshold "
                        f"{IMAGE_JITTER_THRESHOLD_MS/1e6:.2f} ms"
                    )

            frame_id += 1

            # Warmup for specified number of frames
            if frame_id > WARMUP_FRAMES:
                images = [
                    np.asanyarray(color_frame.get_data()).copy(),
                    np.asanyarray(aligned_depth_frame.get_data()).copy()
                ]

                obs = {
                    Sensor.RGB: images[0],
                    Sensor.STEREO: images[1] / 1000
                }

                # call droid slam here
                points, poses = await asyncio.get_event_loop().run_in_executor(None, droid_slam.update, obs)
                current_pose = poses[-1]
                translation = [current_pose[0], current_pose[1], current_pose[2]]
                quaternion = [current_pose[3], current_pose[4], current_pose[5], current_pose[6]]

                x_ned, y_ned, z_ned = transform_to_ned(translation[0], translation[1], translation[2])
                yaw, roll, pitch = quaternion_to_euler(quaternion[3], quaternion[0], quaternion[1], quaternion[2])
                offboard_controller.update_estimated_position(x_ned, y_ned, z_ned, yaw)

                # await print_ned_coordinates(drone)

                drone_x_2d = translation[2] * 10 + 80
                drone_y_2d = -translation[0] * 10 + 80
                obstacle_map, explored_area = map_builder.update_map(points, [translation[2] * 100, -translation[0] * 100, -yaw + np.pi])
                points_to_stream = points / 100

                traversible = 1 - obstacle_map
                planner = FMMPlanner(traversible, 360 / 45)
                ltg = [85, 80]
                reachable = planner.set_goal((ltg[1], ltg[0]))
                stg_x, stg_y, _ = planner.get_short_term_goal([drone_x_2d, drone_y_2d])

                H, W = obstacle_map.shape
                points_2d = []
                for i in range(0,H):
                    for j in range (0,W):
                        if obstacle_map[i][j] == 1:
                            points_2d.append([i, j, -1000])
                        elif explored_area[i][j] >= 1:
                            points_2d.append([i, j, -1001])
                points_2d.append([ltg[0], ltg[1], -2000])
                points_2d.append([stg_x, stg_y, -2001])

                points_to_stream = np.concatenate((points_to_stream, np.array(points_2d)))

                # Store current timestamp for next iteration
                prev_timestamp = timestamp

                publish_data = dict(
                        depth = images[1].astype(np.float32), # * np.float32(depth_scale),
                        rgb = images[0],
                        position=translation,
                        quaternion=quaternion,
                        points=points_to_stream
                    )
                slam_publisher.publish(publish_data)
                await asyncio.sleep(0)
                # streamer_subscriber.stream_callback(publish_data)

    finally:
        pipeline.stop()


if __name__ == "__main__":
    asyncio.run(main())