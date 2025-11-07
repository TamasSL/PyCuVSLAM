from typing import List, Optional

import numpy as np
import pyrealsense2 as rs
import time

import asyncio
import math
import cuvslam as vslam
from mavsdk import System
from mavsdk.mocap import VisionPositionEstimate, Quaternion, PositionBody, AngleBody, Covariance, SpeedBody, AngularVelocityBody, Odometry
from camera_utils import get_rs_stereo_rig
from visualizer import RerunVisualizer

# Constants
RESOLUTION = (640, 360)
FPS = 30
WARMUP_FRAMES = 60
IMAGE_JITTER_THRESHOLD_MS = 35 * 1e6  # 35ms in nanoseconds
NUM_VIZ_CAMERAS = 2

class VisionController:
    def __init__(self, drone):
        self.drone = drone

    def _transform_to_ned(self, x, y, z):
        """Transform cuVSLAM pose to PX4 NED frame"""
        ned_x = z
        ned_y = x
        ned_z = y
        return ned_x, ned_y, ned_z

    def _quaternion_to_euler(self, qx, qy, qz, qw):
            """
            Extract yaw (rotation about vertical/y-axis) from quaternion.
            For camera frame where y is up.
            """
            bqw = qw
            bqx = qz
            bqy = qx
            bqz = -qy
            
            # Yaw is rotation about the y-axis (vertical in camera frame)
            # Standard yaw extraction from quaternion:
            siny_cosp = 2 * (bqw * bqz + bqx * bqy)
            cosy_cosp = 1 - 2 * (bqy * bqy + bqz * bqz)
            yaw = np.arctan2(siny_cosp, cosy_cosp)

            roll = np.arctan2(2 * (bqw * bqx + bqy * bqz), 1 - 2 * (bqx**2 + bqy**2))

            sinp = 2 * (bqw * bqy - bqz * bqx)
            pitch = float(np.where(np.abs(sinp) >= 1, np.copysign(np.pi / 2, sinp), np.arcsin(sinp)))
            
            return yaw, roll, pitch

    async def _send_vision_position(self, x, y, z, quaternion, yaw, roll, pitch):
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
            position_body=p,
            angle_body=a,
            pose_covariance=pose_covariance,
        )
        await self.drone.mocap.set_vision_position_estimate(vision_position)


    async def run(self):
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
        profile = pipeline.start(config)

        frame_id = 0
        prev_timestamp: Optional[int] = None
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

                    # Track frame
                    odom_pose_estimate, _ = tracker.track(
                        timestamp, images=[images[0]], depths=[images[1]]
                    )

                    odom_pose = odom_pose_estimate.world_from_rig.pose

                    x_ned, y_ned, z_ned = self._transform_to_ned(odom_pose.translation[0], odom_pose.translation[1], odom_pose.translation[2])
                    orientation=Quaternion(
                        odom_pose.rotation[3],
                        odom_pose.rotation[0],
                        odom_pose.rotation[1],
                        odom_pose.rotation[2]
                    )
                    yaw, roll, pitch = self._quaternion_to_euler(odom_pose.rotation[0], odom_pose.rotation[1], odom_pose.rotation[2], odom_pose.rotation[3])
                    await self._send_vision_position(x_ned, y_ned, z_ned, orientation, yaw, roll, pitch)
                    # Store current timestamp for next iteration
                    prev_timestamp = timestamp

        finally:
            pipeline.stop()

