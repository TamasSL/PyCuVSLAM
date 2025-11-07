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
import numpy as np
import pyrealsense2 as rs
import torch

from realsense_utils import setup_realsense, LEFT_INFRARED_INDEX, RIGHT_INFRARED_INDEX

# pylint: disable=invalid-name


class RealsenseDataloader:
    """
    Handles the acquisition and preprocessing of data from a RealSense camera,
    including depth, color, and infrared images.
    """
    # Maximum allowed timestamp difference between synchronized frames (in seconds)
    MAX_TIMESTAMP_DIFF_S = 0.001    # 1ms

    def __init__(self, max_steps: int = 10000) -> None:
        """
        Initialize the RealSense data loader.

        Args:
            max_steps: Maximum number of frames to process
        """
        self.max_steps = max_steps
        self.step = 0
        self.first_frame = True
        # Initialize RealSense camera and get calibration parameters
        self.realsense_pipeline, self.camera_params, self.frame_queue = \
            setup_realsense(flash_emitter_on_off=True)

        # The last published frame indices, to prevent publishing duplicate frames.
        self.last_left_infrared_frame_index = None
        self.last_right_infrared_frame_index = None
        self.last_depth_frame_index = None
        self.last_color_frame_index = None

    def __del__(self) -> None:
        print('Cleaning up the realsense pipeline.')
        self.realsense_pipeline.stop()

    def __len__(self) -> int:
        """Return the total number of frames to process."""
        return self.max_steps

    def __iter__(self) -> 'RealsenseDataloader':
        """
        Initialize the iterator.

        Returns:
            self: The data loader instance

        Raises:
            IndexError: If attempting to iterate more than once
        """
        # Ensure we haven't started iterating yet
        assert self.step == 0, 'RealsenseDataloader can only be iterated once'
        return self

    def __next__(self) -> dict:
        """
        Get the next frame from the camera.

        Raises:
            StopIteration: When max_steps is reached
        """
        if self.step >= self.max_steps:
            raise StopIteration
        self.step += 1
        return self.get_next_frame()

    def _get_infrared_as_array(self, infrared_frame: rs.frame) -> np.ndarray:
        """Convert a realsense infrared frame to a numpy array.

        Args:
            infrared_frame: The realsense infrared frame to convert

        Returns:
            np.ndarray: The infrared numpy array
        """
        return np.asanyarray(infrared_frame.get_data())

    def _get_depth_as_tensor(self, depth_frame: rs.frame) -> torch.Tensor:
        """Convert a realsense depth frame to a tensor.

        Args:
            depth_frame: The realsense depth frame to convert

        Returns:
            torch.Tensor: The depth tensor
        """
        depth_image_m = np.asanyarray(depth_frame.get_data()) * self.camera_params.depth_scale_to_m
        depth = torch.from_numpy(depth_image_m).float().to('cuda')
        return depth

    def _get_color_as_tensor(self, color_frame: rs.frame) -> torch.Tensor:
        """Convert a realsense color frame to a tensor.

        Args:
            color_frame: The realsense color frame to convert

        Returns:
            torch.Tensor: The color tensor
        """
        color_image = np.asanyarray(color_frame.get_data())
        rgb = torch.from_numpy(color_image).to('cuda')
        return rgb

    @staticmethod
    def _is_frame_new(frame_index: int, last_frame_index: int | None) -> bool:
        """Check if a frame is new based on their frame indices."""
        return frame_index != last_frame_index

    @staticmethod
    def _warn_if_frame_skipped(frame_index: int,
                               last_frame_index: int | None,
                               stream_name: str,
                               skip_threshold: int = 1) -> None:
        """Warn if frames are skipped for a given stream."""
        if last_frame_index is not None and frame_index != (last_frame_index + skip_threshold):
            print(f'WARNING: Frames skipped for {stream_name}: '
                  f'{frame_index - last_frame_index} frames')

    def assert_within_timestamp_tolerance(self, frame_one: rs.frame, frame_two: rs.frame,
                                          tolerance: float) -> bool:
        """Assert that two RealSense frames are within a specified timestamp tolerance."""
        # NOTE: Sometimes at start up we get a stray unsynchronized frame.
        # We therefore ignore the check on the first frame.
        if not self.first_frame:
            if abs(frame_one.timestamp - frame_two.timestamp) > tolerance:
                print('Frame timestamps differ by ' \
                f'{abs(frame_one.timestamp - frame_two.timestamp):.6f}s, ' \
                f'exceeding threshold of {tolerance}s')
                return True
        else:
            self.first_frame = False
        return False

    def get_next_frame(self) -> dict:
        """
        Capture and process the next frame from the RealSense camera.

        Returns:
            dict: Dictionary containing:
                - left_infrared_image (np.ndarray): Left IR camera image
                - right_infrared_image (np.ndarray): Right IR camera image
                - depth (torch.Tensor): Depth image in meters
                - rgb (torch.Tensor): Color image
                - timestamp (int): Frame timestamp in nanoseconds
        """
        # Wait for synchronized frames from all streams
        frames = self.frame_queue.wait_for_frame()
        frames = frames.as_frameset()

        # Get frames
        left_infrared_frame = frames.get_infrared_frame(LEFT_INFRARED_INDEX)
        right_infrared_frame = frames.get_infrared_frame(RIGHT_INFRARED_INDEX)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        current_timestamp = int(left_infrared_frame.timestamp * 1e6)    # Convert to ns
        if self.assert_within_timestamp_tolerance(left_infrared_frame, right_infrared_frame, self.MAX_TIMESTAMP_DIFF_S):
            return {
                'left_infrared_image': None,
                'right_infrared_image': None,
                'depth': None,
                'rgb': None,
                'timestamp': current_timestamp
            }

        # Depth and infra frames depend on the emitter state.
        # - Emitter off: reutrn infrared frames for SLAM
        # - Emitter on: return depth frame for mapping
        # Get the frame metadata
        left_emitter_mode = left_infrared_frame.get_frame_metadata(
            rs.frame_metadata_value.frame_emitter_mode)
        right_emitter_mode = right_infrared_frame.get_frame_metadata(
            rs.frame_metadata_value.frame_emitter_mode)
        if left_emitter_mode != right_emitter_mode:
            print('Left and right infrared frames must have the same emitter mode')
            return {
                'left_infrared_image': None,
                'right_infrared_image': None,
                'depth': None,
                'rgb': None,
                'timestamp': current_timestamp
            }
        # Emitter off
        if left_emitter_mode == 0:
            left_frame_index = left_infrared_frame.get_frame_number()
            right_frame_index = right_infrared_frame.get_frame_number()
            if self._is_frame_new(left_frame_index, self.last_left_infrared_frame_index) and \
                self._is_frame_new(right_frame_index, self.last_right_infrared_frame_index):
                self._warn_if_frame_skipped(left_frame_index,
                                            self.last_left_infrared_frame_index,
                                            'left infrared',
                                            skip_threshold=2)
                self._warn_if_frame_skipped(right_frame_index,
                                            self.last_right_infrared_frame_index,
                                            'right infrared',
                                            skip_threshold=2)
                left_infrared_image = self._get_infrared_as_array(left_infrared_frame)
                right_infrared_image = self._get_infrared_as_array(right_infrared_frame)
                self.last_left_infrared_frame_index = left_frame_index
                self.last_right_infrared_frame_index = right_frame_index
            else:
                left_infrared_image = None
                right_infrared_image = None
            # Emitter off - depth not usable
            depth = None
        # Emitter on
        else:
            depth_frame_index = depth_frame.get_frame_number()
            if self._is_frame_new(depth_frame_index, self.last_depth_frame_index):
                self._warn_if_frame_skipped(depth_frame_index,
                                            self.last_depth_frame_index,
                                            'depth',
                                            skip_threshold=2)
                depth = self._get_depth_as_tensor(depth_frame)
                self.last_depth_frame_index = depth_frame_index
            else:
                depth = None
            # Emitter on - infra not usable
            left_infrared_image = None
            right_infrared_image = None

        # Color doesn't depend on the emitter state.
        color_frame_index = color_frame.get_frame_number()
        if self._is_frame_new(color_frame_index, self.last_color_frame_index):
            self._warn_if_frame_skipped(color_frame_index, self.last_color_frame_index, 'color')
            rgb = self._get_color_as_tensor(color_frame)
            self.last_color_frame_index = color_frame_index
        else:
            rgb = None

        return {
            'left_infrared_image': left_infrared_image,
            'right_infrared_image': right_infrared_image,
            'depth': np.asanyarray(depth_frame.get_data()) * self.camera_params.depth_scale_to_m,
            'rgb': np.asanyarray(color_frame.get_data()),
            'timestamp': current_timestamp
        }

    def left_infrared_intrinsics(self) -> rs.pyrealsense2.intrinsics:
        """Return the intrinsic parameters of the left infrared camera."""
        return self.camera_params.left_infrared_intrinsics

    def right_infrared_intrinsics(self) -> rs.pyrealsense2.intrinsics:
        """Return the intrinsic parameters of the right infrared camera."""
        return self.camera_params.right_infrared_intrinsics

    def depth_intrinsics(self) -> rs.pyrealsense2.intrinsics:
        """Return the intrinsic parameters of the depth camera."""
        return self.camera_params.depth_intrinsics

    def color_intrinsics(self) -> rs.pyrealsense2.intrinsics:
        """Return the intrinsic parameters of the color camera."""
        return self.camera_params.color_intrinsics

    def T_C_left_infrared_C_color(self) -> rs.pyrealsense2.extrinsics:
        """Return the extrinsic parameters (transformation) from left infrared to color camera."""
        return self.camera_params.T_C_left_infrared_C_color

    def T_C_left_infrared_C_right_infrared(self) -> rs.pyrealsense2.extrinsics:
        """
        Return the extrinsic parameters (transformation) from left infrared to
        right infrared camera.
        """
        return self.camera_params.T_C_left_infrared_C_right_infrared