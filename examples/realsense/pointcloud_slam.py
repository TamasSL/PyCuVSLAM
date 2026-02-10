from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from sensor import SensorData


@dataclass
class CameraInfo:
    frame_width: int
    frame_height: int
    hfov: float


class PointCloudSLAM(ABC):
    _name = ""

    def __init__(self, enable_rgb: bool = False):
        self.xyz = np.ndarray((0, 3), dtype=np.float32)
        self.rgb = np.ndarray((0, 3), dtype=np.float32)
        self.poses = np.ndarray((0, 7), dtype=np.float32)

        self.enable_rgb = enable_rgb

        self.visualizer = None

    @classmethod
    def _voxelize_points(
        cls,
        points: np.ndarray,
        voxel_size: float = 10,
    ):
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must be (N, 3)")

        if voxel_size <= 0:
            raise ValueError("voxel_size must be > 0")

        if points.size == 0:
            return (
                np.empty((0, 3), dtype=float),
                np.empty((0,), dtype=int),
            )

        vox_idx = np.floor((points) / voxel_size).astype(np.int64)
        voxels, reps_idx = np.unique(vox_idx, axis=0, return_index=True)
        voxels_centers = (voxels.astype(float) + 0.5) * voxel_size

        return voxels_centers, reps_idx

    @abstractmethod
    def update(self, obs: SensorData) -> (np.ndarray, np.ndarray):
        """
        Convention: RUF (right, up, forward)
        Units: meters
        Returns: (3D point cloud, poses history)
        """
        pass

    def visualize(self):
        if self.visualizer is None:
            from gym_src.slam.pointcloud_visualizer import PointCloudVisualizer

            self.visualizer = PointCloudVisualizer(name=self._name)

        self.visualizer.update_points(self.xyz, self.rgb)
        self.visualizer.update_cameras(self.poses)
        self.visualizer.render()
