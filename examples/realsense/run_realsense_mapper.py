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
import time
import asyncio

import numpy as np
import torch
import cuvslam as vslam
import pyrealsense2 as rs
import sys

from mavsdk import System
from mavsdk.mocap import VisionPositionEstimate, Quaternion, PositionBody, AngleBody, Covariance, SpeedBody, AngularVelocityBody, Odometry
from map_builder import MapBuilder

from camera_utils import get_rs_stereo_rig
from streamer_subscriber import StreamerSubscriber
from visualizer_subscriber import VisualizerSubscriber
from publish_subscribe import Publisher
from offboard_controller_subscriber import OffboardControllerSubscriber

# pylint: disable=invalid-name

PRINT_TIMING_EVERY_N_SECONDS = 1.0
RESOLUTION = (640, 480)
FPS = 30
WARMUP_FRAMES = 60

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


async def send_vision_position(drone, x, y, z, quaternion, yaw, roll, pitch):
    """Send vision position estimate to PX4"""
    p = PositionBody(x, y, z)
    s = SpeedBody(0.0, 0.0, 0.0)
    av = AngularVelocityBody(0.0, 0.0, 0.0)
    a = AngleBody(roll, pitch, yaw)

    pose_covariance = Covariance([
        0.01, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.01, 0.0, 0.0, 0.0, 0.0,
        0.01, 0.0, 0.0, 0.0,
        0.001, 0.0, 0.0,
        0.001, 0.0,
        0.001
    ])

    vision_position = VisionPositionEstimate(
        time_usec=int(time.time() * 1e6),
        #frame_id=Odometry.MavFrame.LOCAL_FRD,
        position_body=p,
        #q=quaternion,
        #speed_body=s,
        #angular_velocity_body=av,
        angle_body=a,
        pose_covariance=pose_covariance,
        #velocity_covariance=pose_covariance,
    )
    await drone.mocap.set_vision_position_estimate(vision_position)


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


def transform_to_ned(x, y, z):
    """Transform cuVSLAM pose to PX4 NED frame"""
    # Adjust this based on your actual cuVSLAM frame convention
    ned_x = z
    ned_y = x
    ned_z = y
    
    # Transform orientation similarly
    # (May need quaternion transformation)
    
    return ned_x, ned_y, ned_z

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


async def main() -> int:

    reset_realsense_device()

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

    reset_realsense_device()

    # Configure RGBD settings
    rgbd_settings = vslam.Tracker.OdometryRGBDSettings()
    rgbd_settings.depth_scale_factor = 1 / depth_scale
    rgbd_settings.depth_camera_id = 0
    rgbd_settings.enable_depth_stereo_tracking = False

    # Configure tracker
    cfg = vslam.Tracker.OdometryConfig(
        async_sba=False,
        enable_final_landmarks_export=True,
        odometry_mode=vslam.Tracker.OdometryMode.RGBD,
        rgbd_settings=rgbd_settings
    )

    # Create rig using utility function
    rig = get_rs_stereo_rig(camera_params)

    # Initialize tracker and visualizer
    tracker = vslam.Tracker(rig, cfg)

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    # Enable IR emitter for depth sensing
    depth_sensor = device.query_sensors()[0]
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 1)

    # Start pipeline for tracking
    profile = pipeline.start(config)

    frame_id = 0
    prev_timestamp: Optional[int] = None


    drone = System()
    await drone.connect(system_address="serial:///dev/ttyACM0:57600")

    print("Waiting for drone to connect...")
    #async for state in drone.core.connection_state():
    #    if state.is_connected:
    #        print("✅ Drone connected!")
    #        break

    publisher = Publisher(maxsize=5)
    streamer_subscriber = VisualizerSubscriber(publisher, "VisualizerSubscriber")
    streamer_subscriber.start()

    offboard_controller = OffboardControllerSubscriber(drone)
    offboard_controller.start()

    map_builder = MapBuilder()

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
                print(f"({timestamp_diff/1e6:.2f} ms) ")

            frame_id += 1

            # Warmup for specified number of frames
            if frame_id > WARMUP_FRAMES:
                images = [
                    np.asanyarray(color_frame.get_data()),
                    np.asanyarray(aligned_depth_frame.get_data())
                ]

                # Track frame
                odom_pose_estimate, _ = tracker.track(
                    timestamp, images=[images[0]], depths=[images[1]]
                )

                if odom_pose_estimate is not None:
                    # send position to drone
                    odom_pose = odom_pose_estimate.world_from_rig.pose
                    x_ned, y_ned, z_ned = transform_to_ned(odom_pose.translation[0], odom_pose.translation[1], odom_pose.translation[2])

                    orientation=Quaternion(
                        odom_pose.rotation[3],
                        odom_pose.rotation[0],
                        odom_pose.rotation[1],
                        odom_pose.rotation[2]
                    )
                    yaw, roll, pitch = quaternion_to_euler(odom_pose.rotation[0], odom_pose.rotation[1], odom_pose.rotation[2], odom_pose.rotation[3])

                    agent_view_cropped, map_gt, agent_view_explored, explored_gt = map_builder.update_map(images[1] * np.float32(depth_scale) * 100, [odom_pose.translation[2] * 100, -odom_pose.translation[0] * 100, -yaw])

                    map_to_visualize = map_gt
                    H, W = map_to_visualize.shape
                    points = []
                    for i in range(0,H):
                        for j in range (0,W):
                            if map_to_visualize[i][j] == 1:
                                points.append([i, j])

                    # await send_vision_position(drone, x_ned, y_ned, z_ned, orientation, yaw, roll, pitch)
                    # await print_ned_coordinates(drone)

                    publish_data = dict(
                        depth = images[1] * np.float32(depth_scale),
                        rgb = images[0],
                        # left_infrared_image = frame['left_infrared_image'],
                        # last_observation = cuvslam_tracker.get_last_observations(0),
                        position=odom_pose_estimate.world_from_rig.pose.translation,
                        quaternion=odom_pose_estimate.world_from_rig.pose.rotation,
                        points=points
                    )
                    publisher.publish(publish_data)
                    # streamer_subscriber.stream_callback(publish_data)
    finally:
        pipeline.stop()
    print('Done')

    return 0


if __name__ == '__main__':
    asyncio.run(main())