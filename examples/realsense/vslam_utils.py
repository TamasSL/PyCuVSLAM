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

import numpy as np
import numpy.typing as npt
import cuvslam as vslam
from scipy.spatial.transform import Rotation
import pyrealsense2 as rs


def rs_transform_to_vslam_pose(rs_transform: rs.pyrealsense2.extrinsics) -> vslam.Pose:
    """
    Convert transformations provided by realsense to a vslam.Pose object.
    """
    rotation_matrix, translation_vec = np.array(rs_transform.rotation).reshape(
        [3, 3]), rs_transform.translation
    rotation_quat = Rotation.from_matrix(rotation_matrix).as_quat()
    return vslam.Pose(rotation=rotation_quat, translation=translation_vec)


def vslam_identity_pose() -> vslam.Pose:
    """
    Return a vslam.Pose object representing the identity transformation.
    """
    rotation_matrix, translation_vec = np.eye(3), [0] * 3
    rotation_quat = Rotation.from_matrix(rotation_matrix).as_quat()
    return vslam.Pose(rotation=rotation_quat, translation=translation_vec)


def to_homogeneous(translation: np.ndarray, rotation: np.ndarray) -> npt.NDArray:
    """
    Convert translation and quaternion rotation to a 4x4 homogeneous transformation matrix.

    This function is used to convert the camera pose from cuvslam's format
    (translation + quaternion) to a homogeneous transformation matrix that can be
    used by nvblox for mapping.

    Args:
        translation (np.ndarray): Shape (3,), translation vector in meters.
        rotation (np.ndarray): Shape (4,), quaternion [x, y, z, w] representing rotation.

    Returns:
        np.ndarray: 4x4 homogeneous transformation matrix representing the camera pose.
    """
    assert translation.shape == (3, )
    assert rotation.shape == (4, )
    # Convert quaternion to rotation matrix
    rot = Rotation.from_quat(rotation)
    rotation_matrix = rot.as_matrix()    # Shape (3, 3)

    # Build the 4x4 transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation
    return transform


def from_homogeneous(transform: npt.NDArray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a 4x4 homogeneous transformation matrix to translation and quaternion rotation.
    """
    assert transform.shape == (4, 4)
    assert transform[3, 3] == 1.0
    translation = transform[:3, 3]
    rotation_matrix = transform[:3, :3]
    rotation_quat = Rotation.from_matrix(rotation_matrix).as_quat()
    return translation, rotation_quat


def get_vslam_camera(rs_intrinsics: rs.pyrealsense2.intrinsics,
                     left_t_right: rs.pyrealsense2.extrinsics = None) -> vslam.Camera:
    """Get a Camera object from RealSense intrinsics."""
    cam = vslam.Camera()
    cam.distortion = vslam.Distortion(vslam.Distortion.Model.Pinhole)
    cam.focal = rs_intrinsics.fx, rs_intrinsics.fy
    cam.principal = rs_intrinsics.ppx, rs_intrinsics.ppy
    cam.size = rs_intrinsics.width, rs_intrinsics.height
    print(f'Camera size: {cam.size}')
    if left_t_right:
        cam.rig_from_camera = rs_transform_to_vslam_pose(left_t_right)
    else:
        cam.rig_from_camera = vslam_identity_pose()
    return cam


def get_vslam_stereo_rig(left_intrinsics: rs.pyrealsense2.intrinsics,
                         right_intrinsics: rs.pyrealsense2.intrinsics,
                         left_t_right: rs.pyrealsense2.extrinsics) -> vslam.Rig:
    """Get a VIO Rig object with cameras from RealSense parameters."""
    rig = vslam.Rig()
    rig.cameras = [
        get_vslam_camera(left_intrinsics),
        get_vslam_camera(right_intrinsics, left_t_right)
    ]
    return rig

def get_vslam_rgbd_rig(left_intrinsics: rs.pyrealsense2.intrinsics) -> vslam.Rig:
    rig = vslam.Rig()
    rig.cameras = [
        get_vslam_camera(left_intrinsics),
    ]
    return rig