import math
import multiprocessing
from argparse import Namespace

import cv2
import numpy as np
import torch
from lietorch import SE3
from thirdparty.dpvo.dpvo.config import cfg

from sensor import SensorData, Sensor
from pointcloud_slam import PointCloudSLAM, CameraInfo
from base_settings import droid_slam_settings as settings, env_settings


class DpvoSLAM(PointCloudSLAM):
    _name = "DPVO"

    def __init__(self):
        cfg.merge_from_file('config/default.yaml')
        self.image_size = (240, 320)
        self.slam = DPVO(cfg, 'dpvo.pth', ht=240, wd=320, viz=False)
        self.intrinsics = self._compute_intrinsics(
            87, *self.image_size
        )
        self.tstamp = 0

    @classmethod
    def _compute_intrinsics(cls, hfov: float, H: float, W: float):
        fx = fy = W / (2 * math.tan(math.radians(hfov) / 2))
        cx = W / 2
        cy = H / 2

        return torch.as_tensor([fx, fy, cx, cy])

    def _preprocess_img(self, image: np.ndarray):
        if image.shape[0] == 3:
            image = image.transpose((1, 2, 0))
        image_size = (320, 240)
        image = cv2.resize(image, image_size)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = torch.as_tensor(image).permute(2, 0, 1)
        return image[None]

    @torch.no_grad()
    def _step_slam(self, obs: SensorData):
        raw_rgb = obs[Sensor.RGB]
        rgb = self._preprocess_img(raw_rgb)

        self.slam(self.tstamp, rgb, intrinsics=self.intrinsics)
        self.tstamp += 1

    @torch.no_grad()
    def _extract_points_and_poses(self):
        pose = self.slam.get_pose(self.tstamp - 1)
        points = self.slam.pg.points_.cpu().numpy()[:slam.m]
        return points * 1e2, pose

    def update(self, obs: SensorData):
        self._step_slam(obs)

        points, poses = self._extract_points_and_poses()
        self.xyz = self._voxelize_points(points)[0]
        self.poses = poses

        return self.xyz, self.poses
