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
from typing import Tuple
from dataclasses import dataclass
import time

import numpy as np
import numpy.typing as npt
import pyrealsense2 as rs

# pylint: disable=invalid-name

FRAME_QUEUE_SIZE = 50
REALSENSE_IMAGE_WIDTH = 640
REALSENSE_IMAGE_HEIGHT = 480
INFRA_AND_DEPTH_FPS = 60
COLOR_FPS = 5
LEFT_INFRARED_INDEX = 1
RIGHT_INFRARED_INDEX = 2


@dataclass
class RealSenseCameraParameters:
    """Container for all camera parameters from a RealSense camera.

    The parameters are obtained during camera initialization and remain constant
    throughout the camera's operation.
    """
    # Scale factor to convert depth units to meters
    depth_scale_to_m: float
    # Transformation matrix from left to right infrared camera
    T_C_left_infrared_C_right_infrared: rs.pyrealsense2.extrinsics
    # Transformation matrix from left infrared to color camera
    T_C_left_infrared_C_color: rs.pyrealsense2.extrinsics
    # Intrinsic parameters (focal length, principal point) of left infrared camera
    left_infrared_intrinsics: rs.pyrealsense2.intrinsics
    # Intrinsic parameters (focal length, principal point) of right infrared camera
    right_infrared_intrinsics: rs.pyrealsense2.intrinsics
    # Intrinsic parameters (focal length, principal point) of depth camera
    depth_intrinsics: rs.pyrealsense2.intrinsics
    # Intrinsic parameters (focal length, principal point) of color camera
    color_intrinsics: rs.pyrealsense2.intrinsics


def setup_realsense(
    flash_emitter_on_off: bool = False
) -> Tuple[rs.pipeline, RealSenseCameraParameters, rs.frame_queue]:
    """
    Initialize and configure the RealSense camera pipeline.

    This function sets up the RealSense camera with the following streams:
    - Infrared streams (left and right) for stereo tracking
    - Depth stream for 3D reconstruction
    - Color stream for texture mapping

    Args:
        flash_emitter_on_off (bool): Whether to flash the IR emitter

    Returns:
        Tuple containing:
        - rs.pipeline: Configured RealSense pipeline
        - CameraParameters: Dataclass containing camera parameters

    Note:
        The function configures the camera with 640x480 resolution at 30fps for all streams.
        It also disables the IR emitter to improve stereo tracking performance.
    """

    # Reset devices
    devices = rs.context().devices
    for dev in devices:
        dev.hardware_reset()
    time.sleep(1)

    pipeline = rs.pipeline()
    config = rs.config()

    # Configure all camera streams with optimal settings for mapping
    # Using
    # - 640x480 resolution at INFRA_AND_DEPTH_FPS for infra and depth
    #   NOTE: This is effectively halved because of the emitter flashing.
    # - 640x480 resolution at COLOR_FPS for color
    config.enable_stream(rs.stream.infrared, LEFT_INFRARED_INDEX, REALSENSE_IMAGE_WIDTH,
                         REALSENSE_IMAGE_HEIGHT, rs.format.y8, INFRA_AND_DEPTH_FPS)
    config.enable_stream(rs.stream.infrared, RIGHT_INFRARED_INDEX, REALSENSE_IMAGE_WIDTH,
                         REALSENSE_IMAGE_HEIGHT, rs.format.y8, INFRA_AND_DEPTH_FPS)
    config.enable_stream(rs.stream.depth, REALSENSE_IMAGE_WIDTH, REALSENSE_IMAGE_HEIGHT,
                         rs.format.z16, INFRA_AND_DEPTH_FPS)
    config.enable_stream(rs.stream.color, REALSENSE_IMAGE_WIDTH, REALSENSE_IMAGE_HEIGHT,
                         rs.format.rgb8, COLOR_FPS)

    # Start the camera pipeline with the configured streams
    queue = rs.frame_queue(FRAME_QUEUE_SIZE)
    profile = pipeline.start(config, queue)
    device = profile.get_device()

    # Get depth scale factor to convert depth units to meters
    depth_sensor = device.first_depth_sensor()
    depth_scale_to_m = depth_sensor.get_depth_scale()

    # Flash the IR emitter or turn it off
    assert depth_sensor.supports(rs.option.emitter_enabled)
    if flash_emitter_on_off:
        print('IR emitter flashing enabled')
        depth_sensor.set_option(rs.option.emitter_enabled, 1)
        assert depth_sensor.supports(
            rs.option.emitter_on_off
        ), 'You requested emitter flashing but your camera does not support it.'
        depth_sensor.set_option(rs.option.emitter_on_off, 1)
    else:
        depth_sensor.set_option(rs.option.emitter_enabled, 0)
        print('IR emitter disabled')

    # Capture initial frames to extract camera parameters
    left_infrared_frame = None
    right_infrared_frame = None
    color_frame = None
    while left_infrared_frame is None or right_infrared_frame is None or color_frame is None:
        frames = queue.wait_for_frame()
        frames = frames.as_frameset()
        left_infrared_frame_tmp = frames.get_infrared_frame(LEFT_INFRARED_INDEX)
        right_infrared_frame_tmp = frames.get_infrared_frame(RIGHT_INFRARED_INDEX)
        color_frame_tmp = frames.get_color_frame()
        if left_infrared_frame_tmp.is_frame():
            left_infrared_frame = left_infrared_frame_tmp
        if right_infrared_frame_tmp.is_frame():
            right_infrared_frame = right_infrared_frame_tmp
        if color_frame_tmp.is_frame():
            color_frame = color_frame_tmp

    # Transformations
    T_C_left_infrared_C_right_infrared = right_infrared_frame.profile.get_extrinsics_to(
        left_infrared_frame.profile)
    T_C_left_infrared_C_color = color_frame.profile.get_extrinsics_to(left_infrared_frame.profile)

    # Intrinsics
    left_infrared_intrinsics = left_infrared_frame.profile.as_video_stream_profile().intrinsics
    right_infrared_intrinsics = right_infrared_frame.profile.as_video_stream_profile().intrinsics
    depth_instrinsics = frames.get_depth_frame().profile.as_video_stream_profile().intrinsics
    color_instrinsics = color_frame.profile.as_video_stream_profile().intrinsics

    return pipeline, RealSenseCameraParameters(
        depth_scale_to_m=depth_scale_to_m,
        T_C_left_infrared_C_right_infrared=T_C_left_infrared_C_right_infrared,
        T_C_left_infrared_C_color=T_C_left_infrared_C_color,
        left_infrared_intrinsics=left_infrared_intrinsics,
        right_infrared_intrinsics=right_infrared_intrinsics,
        depth_intrinsics=depth_instrinsics,
        color_intrinsics=color_instrinsics,
    ), queue


def rs_extrinsics_to_homogeneous(extrinsics: rs.pyrealsense2.extrinsics) -> npt.NDArray:
    """
    Convert a RealSense extrinsics object to a 4x4 homogeneous transformation matrix.
    """
    rotation = np.array(extrinsics.rotation).reshape(3, 3)
    translation = np.array(extrinsics.translation)
    homogenous_transform = np.eye(4)
    homogenous_transform[:3, :3] = rotation
    homogenous_transform[:3, 3] = translation
    return homogenous_transform


def rs_intrinsics_to_matrix(intrinsics: rs.pyrealsense2.intrinsics) -> npt.NDArray:
    """
    Convert RealSense intrinsics to a 3x3 camera intrinsic matrix.
    """
    intrinsic_matrix = np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1],
    ])
    return intrinsic_matrix