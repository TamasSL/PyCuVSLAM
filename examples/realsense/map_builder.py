import numpy as np

import depth_utils as du

params = dict(
    frame_width = 640,
    frame_height = 480,
    hfov = 87,
    vision_range= 20,
    map_size_cm=1600,
    resolution=10,
    agent_min_z=80,
    agent_max_z=100,
    du_scale = 2,
    visualize=True,
    obs_threshold=1.1,
    agent_height=60,
    agent_view_angle=0
)

class MapBuilder(object):
    def __init__(self):
        self.params = params
        frame_width = params["frame_width"]
        frame_height = params["frame_height"]
        hfov = params["hfov"]
        self.camera_matrix = du.get_camera_matrix(frame_width, frame_height, hfov)
        self.vision_range = params["vision_range"]

        self.map_size_cm = params["map_size_cm"]
        self.resolution = params["resolution"]
        agent_min_z = params["agent_min_z"]
        agent_max_z = params["agent_max_z"]
        self.z_bins = [agent_min_z, agent_max_z]
        self.du_scale = params["du_scale"]
        self.visualize = params["visualize"]
        self.obs_threshold = params["obs_threshold"]

        self.map = np.zeros(
            (
                self.map_size_cm // self.resolution,
                self.map_size_cm // self.resolution,
                len(self.z_bins) + 1,
            ),
            dtype=np.float32,
        )

        self.agent_height = params["agent_height"]
        self.agent_view_angle = params["agent_view_angle"]
        return

    def update_map(self, depth: np.ndarray, current_pose: (float, float, float)):
        # print(depth[320][240])
        with np.errstate(invalid="ignore"):
            depth[depth > self.vision_range * self.resolution] = (
                np.nan
            )  # can be adjusted to capture more
        point_cloud = du.get_point_cloud_from_z(
            depth, self.camera_matrix, scale=self.du_scale
        )

        agent_view = du.transform_camera_view(
            point_cloud, self.agent_height, self.agent_view_angle
        )

        shift_loc = [self.vision_range * self.resolution // 2, 0, np.pi / 2.0]
        agent_view_centered = du.transform_pose(agent_view, shift_loc)

        agent_view_flat = du.bin_points(
            agent_view_centered, self.vision_range, self.z_bins, self.resolution
        )

        agent_view_cropped = agent_view_flat[:, :, 1]

        agent_view_cropped = agent_view_cropped / self.obs_threshold
        agent_view_cropped[agent_view_cropped >= 0.5] = 1.0
        agent_view_cropped[agent_view_cropped < 0.5] = 0.0

        agent_view_explored = agent_view_flat.sum(2)
        agent_view_explored[agent_view_explored > 0] = 1.0

        current_pose[0] += self.map_size_cm / 2
        current_pose[1] += self.map_size_cm / 2
        geocentric_pc = du.transform_pose(agent_view, current_pose)

        geocentric_flat = du.bin_points(
            geocentric_pc, self.map.shape[0], self.z_bins, self.resolution
        )

        self.map = self.map + geocentric_flat

        map_gt = self.map[:, :, 1] / self.obs_threshold
        map_gt[map_gt >= 0.5] = 1.0
        map_gt[map_gt < 0.5] = 0.0

        explored_gt = self.map.sum(2)
        explored_gt[explored_gt > 1] = 1.0

        return agent_view_cropped, map_gt, agent_view_explored, explored_gt

    def get_st_pose(self, current_loc: (float, float, float)):
        loc = [
            -(
                current_loc[0] / self.resolution
                - self.map_size_cm // (self.resolution * 2)
            )
            / (self.map_size_cm // (self.resolution * 2)),
            -(
                current_loc[1] / self.resolution
                - self.map_size_cm // (self.resolution * 2)
            )
            / (self.map_size_cm // (self.resolution * 2)),
            90 - np.rad2deg(current_loc[2]),
        ]
        return loc

    def reset_map(self, map_size: int):
        self.map_size_cm = map_size

        self.map = np.zeros(
            (
                self.map_size_cm // self.resolution,
                self.map_size_cm // self.resolution,
                len(self.z_bins) + 1,
            ),
            dtype=np.float32,
        )

    def get_map(self):
        return self.map
