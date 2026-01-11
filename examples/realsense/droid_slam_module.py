import math
import multiprocessing
from argparse import Namespace

import cv2
import numpy as np
import torch
from lietorch import SE3
from droid_slam.droid import Droid
from droid_slam.depth_video import droid_backends

from sensor import SensorData, Sensor
from pointcloud_slam import PointCloudSLAM, CameraInfo
from base_settings import droid_slam_settings as settings, env_settings


class DroidSLAM(PointCloudSLAM):
    _name = "DROID"

    def __init__(self):
        super().__init__()
        print("starting multiprocessing")
        multiprocessing.set_start_method("spawn")
        self.image_size = (env_settings.frame_height, env_settings.frame_width)
        print("initializing droid")
        self.droid = Droid(
            Namespace(**dict(settings.base_config), image_size=self.image_size)
        )
        print("hfov")
        print(env_settings.simulator.camera_hfov)
        self.intrinsics = self._compute_intrinsics(
            env_settings.simulator.camera_hfov, *self.image_size
        )
        self.tstamp = 0
        self.prev_droid_tstamp = 0

    @classmethod
    def _compute_intrinsics(cls, hfov: float, H: float, W: float):
        fx = fy = W / (2 * math.tan(math.radians(hfov) / 2))
        cx = W / 2
        cy = H / 2

        return torch.as_tensor([fx, fy, cx, cy])

    @property
    def K(self):
        K = np.eye(3)
        K[0, 0] = self.intrinsics[0]
        K[1, 1] = self.intrinsics[1]
        K[0, 2] = self.intrinsics[2]
        K[1, 2] = self.intrinsics[3]
        return K

    def _preprocess_img(self, image: np.ndarray, distort=None):
        if image.shape[0] == 3:
            image = image.transpose((1, 2, 0))
        image = cv2.resize(image, self.image_size)
        if distort:
            image = cv2.undistort(image, self.K, distort)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = torch.as_tensor(image).permute(2, 0, 1)
        return image[None]

    def _preprocess_depth(self, depth: np.ndarray | None):
        if depth is None:
            return None
        depth = cv2.resize(depth, self.image_size, interpolation=cv2.INTER_NEAREST)
        depth = torch.as_tensor(depth, dtype=torch.float32) * 1e-2  # cm -> m
        depth[(depth >= 3) | (depth <= 0.3) | depth.isnan()] = 0.0
        return depth

    @torch.no_grad()
    def _step_slam(self, obs: SensorData):
        raw_rgb, raw_stereo = obs[Sensor.RGB], obs[Sensor.STEREO]
        rgb = self._preprocess_img(raw_rgb)
        stereo = self._preprocess_depth(raw_stereo)

        self.droid.track(self.tstamp, rgb, depth=stereo, intrinsics=self.intrinsics)
        self.tstamp += 1

    @torch.no_grad()
    def _extract_points_and_poses(self):
        video = self.droid.video
        t = video.counter.value

        if t == self.prev_droid_tstamp:
            return self.xyz, self.poses

        if t == 1000:
            print("!!!! RUNNING OUT OF STORAGE, PRESS ANY KEY TO CONTINUE")
            input()
        self.prev_droid_tstamp = t

        intrinsics = video.intrinsics[0]
        poses = video.poses[:t]
        disps = video.disps[:t]
        filter_thresh = 0.02
        filter_count = 2

        index = torch.arange(t, device="cuda")
        thresh = filter_thresh * torch.ones_like(disps.mean(dim=[1, 2]))

        points = droid_backends.iproj(SE3(poses).inv().data, disps, intrinsics)
        counts = droid_backends.depth_filter(poses, disps, intrinsics, index, thresh)
        mask = (counts >= filter_count) & (disps > 0.25 * disps.mean())
        mask = mask.cpu().numpy()

        points = points.cpu().numpy()
        points = points.reshape(-1, 3)[mask.reshape(-1)]

        points = points[points[:, 1] < 0.5]
        points = points[points[:, 1] > -0.5]

        return points * 1e2, SE3(poses).inv().data.cpu().numpy()

    def update(self, obs: SensorData):
        self._step_slam(obs)

        points, poses = self._extract_points_and_poses()
        self.xyz = self._voxelize_points(points)[0]
        self.poses = poses

        return self.xyz, self.poses
