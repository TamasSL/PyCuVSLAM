import math
import multiprocessing
from argparse import Namespace

import cv2
import numpy as np
import torch
import lietorch
from lietorch import SE3
from thirdparty.dpvo.dpvo.config import cfg
from thirdparty.dpvo.dpvo.dpvo import DPVO

from sensor import SensorData, Sensor
from pointcloud_slam import PointCloudSLAM, CameraInfo


class DpvoSLAM(PointCloudSLAM):
    _name = "DPVO"

    def __init__(self):
        cfg.merge_from_file('thirdparty/dpvo/config/default.yaml')
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

        image_size = (320, 240)
        image = cv2.resize(image, image_size)

        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16]
    
        
        image = torch.from_numpy(image).permute(2,0,1).cuda()
        return image

    @torch.no_grad()
    def _step_slam(self, obs: SensorData):
        raw_rgb = obs[Sensor.RGB]
        rgb = self._preprocess_img(raw_rgb)

        self.slam(self.tstamp, rgb, intrinsics=self.intrinsics)
        self.tstamp += 1

    @torch.no_grad()
    def _extract_points_and_poses(self):
        poses = [self.slam.get_pose(t) for t in range(self.slam.counter)]
        poses = lietorch.stack(poses, dim=0)
        poses = poses.inv().data.cpu().numpy()

        points = self.slam.pg.points_.cpu().numpy()[:self.slam.m]
        return points, poses

    def update(self, obs: SensorData):
        self._step_slam(obs)

        points, poses = self._extract_points_and_poses()
        self.xyz = points # self._voxelize_points(points)[0]
        self.poses = poses

        return self.xyz, self.poses
