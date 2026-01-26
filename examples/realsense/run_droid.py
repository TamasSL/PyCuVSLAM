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

import numpy as np
import pyrealsense2 as rs

from droid_slam_module import DroidSLAM
from sensor import Sensor
from rerun_visualizer import RerunVisualizer

# Constants
RESOLUTION = (640, 480)
FPS = 30
WARMUP_FRAMES = 60
IMAGE_JITTER_THRESHOLD_MS = 35 * 1e6  # 35ms in nanoseconds
NUM_VIZ_CAMERAS = 2

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

def main() -> None:
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

    visualizer = RerunVisualizer(num_viz_cameras=NUM_VIZ_CAMERAS)

    reset_realsense_device()

    # Start pipeline for tracking
    profile = pipeline.start(config)

    frame_id = 0
    prev_timestamp: Optional[int] = None

    slam_publisher = Publisher(maxsize=5)
    streamer_subscriber = StreamerSubscriber(slam_publisher, command_publisher, "StreamerSubscriber")
    streamer_subscriber.start()

    print("pre init")
    droid_slam = DroidSLAM()
    print("post init")
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
                    np.asanyarray(color_frame.get_data()),
                    np.asanyarray(aligned_depth_frame.get_data())
                ]

                obs = {
                    Sensor.RGB: images[0],
                    Sensor.STEREO: images[1]
                }

                print("droid update")
                # call droid slam here
                points, poses = droid_slam.update(obs)
                current_pose = poses[-1]
                translation = [current_pose[0], current_pose[1], current_pose[2]]
                quaternion = [current_pose[3], current_pose[4], current_pose[5], current_pose[6]]

                # Store current timestamp for next iteration
                prev_timestamp = timestamp

                publish_data = dict(
                        depth = images[1] * np.float32(depth_scale),
                        rgb = images[0],
                        position=translation,
                        quaternion=quaternion,
                        points=points
                    )
                # slam_publisher.publish(publish_data)
                streamer_subscriber.stream_callback(publish_data)

    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()