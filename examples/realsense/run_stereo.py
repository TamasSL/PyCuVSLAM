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

import cuvslam as vslam
from camera_utils import get_rs_stereo_rig
from visualizer import RerunVisualizer
from visualizer_subscriber import VisualizerSubscriber
from publish_subscribe import Publisher

# Constants
RESOLUTION = (640, 360)
FPS = 30
WARMUP_FRAMES = 60
IMAGE_JITTER_THRESHOLD_MS = 35 * 1e6  # 35ms in nanoseconds

def reset_realsense_device():
    """Reset all RealSense devices"""
    print("1")
    ctx = rs.context()
    print("2")
    devices = ctx.query_devices()
    print("3")
    for dev in devices:
        print("5")
        dev.hardware_reset()
        print("5")
        print(f"Reset device: {dev.get_info(rs.camera_info.name)}")
    
    # Wait for device to come back online
    import time
    time.sleep(3)


def main() -> None:
    """Main function for stereo tracking."""
    # Initialize RealSense configuration
    reset_realsense_device()
    config = rs.config()
    pipeline = rs.pipeline()

    # Configure streams
    config.enable_stream(rs.stream.color, RESOLUTION[0], RESOLUTION[1], rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, RESOLUTION[0], RESOLUTION[1], rs.format.z16, FPS)
    config.enable_stream(
        rs.stream.infrared, 1, RESOLUTION[0], RESOLUTION[1], rs.format.y8, FPS
    )
    config.enable_stream(
        rs.stream.infrared, 2, RESOLUTION[0], RESOLUTION[1], rs.format.y8, FPS
    )
    
    # Start pipeline to get intrinsics and extrinsics
    profile = pipeline.start(config)
    frames = pipeline.wait_for_frames()
    # pipeline.stop()

    # Prepare camera parameters
    camera_params = {'left': {}, 'right': {}}

    # Get extrinsics and intrinsics
    left_profile = frames.get_infrared_frame(1).profile.as_video_stream_profile()
    right_profile = frames.get_infrared_frame(2).profile.as_video_stream_profile()
    
    camera_params['left']['intrinsics'] = left_profile.intrinsics
    camera_params['right']['intrinsics'] = right_profile.intrinsics
    camera_params['right']['extrinsics'] = right_profile.get_extrinsics_to(
        left_profile
    )

    # reset_realsense_device()

    # Configure tracker
    cfg = vslam.Tracker.OdometryConfig(
        async_sba=False,
        enable_final_landmarks_export=True,
        enable_observations_export=True,
        horizontal_stereo_camera=True
    )
    slam_cfg = vslam.Tracker.SlamConfig(sync_mode=False)

    # Create rig using utility function
    rig = get_rs_stereo_rig(camera_params)

    # Initialize tracker and visualizer
    tracker = vslam.Tracker(rig, cfg, slam_cfg)

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    # Disable IR emitter if supported
    depth_sensor = device.query_sensors()[0]
    if depth_sensor.supports(rs.option.emitter_enabled):
       depth_sensor.set_option(rs.option.emitter_enabled, 0)

    # Start pipeline for tracking
    # profile = pipeline.start(config)

    frame_id = 0
    prev_timestamp: Optional[int] = None

    publisher = Publisher(maxsize=10)
    vis_subscriber = VisualizerSubscriber(publisher, "VisualizerSubscriber")
    # vis_subscriber.start()

    try:
        while True:
            # Wait for frames
            frames = pipeline.wait_for_frames()

            left_frame = frames.get_infrared_frame(1)
            right_frame = frames.get_infrared_frame(2)
            
            if not left_frame or not right_frame:
                print("Warning: Failed to get frames")
                continue

            frame_id += 1
            timestamp = int(left_frame.timestamp * 1e6)  # Convert to nanoseconds

            # Check timestamp difference with previous frame
            if prev_timestamp is not None:
                timestamp_diff = timestamp - prev_timestamp
                if timestamp_diff > IMAGE_JITTER_THRESHOLD_MS:
                    print(
                        f"Warning: Camera stream message drop: timestamp gap "
                        f"({timestamp_diff/1e6:.2f} ms) exceeds threshold "
                        f"{IMAGE_JITTER_THRESHOLD_MS/1e6:.2f} ms"
                    )

            # Store current timestamp for next iteration
            prev_timestamp = timestamp

            images = (
                np.asanyarray(left_frame.get_data()),
                np.asanyarray(right_frame.get_data())
            )

            # Warmup for specified number of frames
            if frame_id > WARMUP_FRAMES:
                # Track frame
                odom_pose_estimate, _ = tracker.track(timestamp, images)
                
                if odom_pose_estimate.world_from_rig is None:
                    print("Warning: Pose tracking not valid")
                    continue
                
                odom_pose = odom_pose_estimate.world_from_rig.pose

                publish_data = dict(
                    color_image=images[0],
                    depth_image=images[1],
                    position=odom_pose.translation,
                    quaternion=odom_pose.rotation,
                    timestamp=timestamp,
                    depth_instrinsics=depth_instrinsics,
                    color_instrinsics=color_instrinsics
                )
                #publisher.publish(publish_data)
                vis_subscriber.visualize_callback(publish_data)
                

    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
