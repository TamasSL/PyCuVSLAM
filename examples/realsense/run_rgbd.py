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

import os
import numpy as np
import pyrealsense2 as rs
import time

import cuvslam as vslam
from camera_utils import get_rs_stereo_rig
from visualizer import RerunVisualizer
from visualizer_subscriber import VisualizerSubscriber
from publish_subscribe import Publisher

from mavsdk.mocap import PositionBody, Quaternion

# Constants
RESOLUTION = (640, 480)
FPS = 30
WARMUP_FRAMES = 60
IMAGE_JITTER_THRESHOLD_MS = 35 * 1e6  # 35ms in nanoseconds
NUM_VIZ_CAMERAS = 2

sequence_path = os.path.join(
    os.path.dirname(__file__),
    "dataset"
)

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

def save_callback(success):
    global map_saved
    map_saved = success

def main() -> None:
    """Main function for RGBD tracking."""
    # Initialize RealSense configuration
    reset_realsense_device()
    config = rs.config()
    pipeline = rs.pipeline()
    map_saved = False
    max_wait_time = 20.0  # seconds

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

    # pipeline.stop()

    # Configure RGBD settings
    rgbd_settings = vslam.Tracker.OdometryRGBDSettings()
    rgbd_settings.depth_scale_factor = 1 / depth_scale
    rgbd_settings.depth_camera_id = 0
    rgbd_settings.enable_depth_stereo_tracking = False

    # Configure tracker
    cfg = vslam.Tracker.OdometryConfig(
        async_sba=True,
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
    # profile = pipeline.start(config)

    frame_id = 0
    prev_timestamp: Optional[int] = None

    publisher = Publisher(maxsize=10)
    vis_subscriber = VisualizerSubscriber(publisher, "VisualizerSubscriber")
    # vis_subscriber.start()

    try:
        while frame_id < 1000:
            # Wait for frames
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if frame_id == 0:
                depth_instrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
                color_instrinsics = color_frame.profile.as_video_stream_profile().intrinsics

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
                    np.asanyarray(aligned_depth_frame.get_data()) # check if this needs to be multiplied by some depth scale
                ]

                # Track frame
                odom_pose_estimate, slam_pose = tracker.track(
                    timestamp, images=[images[0]], depths=[images[1]]
                )

                odom_pose = odom_pose_estimate.world_from_rig.pose


                # Store current timestamp for next iteration
                prev_timestamp = timestamp

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

    map_path = os.path.join(sequence_path, 'map')
    os.makedirs(map_path, exist_ok=True)
    print(f"Saving mao")
    tracker.save_map(map_path, save_callback)

    start_time = time.time()
    while not map_saved and (time.time() - start_time) < max_wait_time:
        time.sleep(0.1)
        print(f"Waiting for map saving to complete... {time.time() - start_time} seconds")

    if map_saved:
        print("Map saved successfully")
    else:
        print("WARNING: Map saving may not have completed")


if __name__ == "__main__":
    main()
