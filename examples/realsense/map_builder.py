import numpy as np

import depth_utils as du

params = dict(
    frame_width = 640,
    frame_height = 480,
    hfov = 87,
    vision_range=20,
    map_size_cm=1600,
    resolution=10,
    agent_min_z=80,
    agent_max_z=100,
    du_scale = 2,
    visualize=True,
    obs_threshold=1.3,
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

        # remove false obstacles
        map_gt[(explored_gt > 1) and (geocentric_flat < 0.5)] = 0

        return agent_view_cropped, map_gt, agent_view_explored, explored_gt


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


    def get_explored_area(x, y, yaw_deg):
        """
        Calculate which grid cells are visible to the camera.
        
        Args:
            x, y: camera's position as grid coordinates
            yaw_deg: camera's orientation
        
        Returns:
            grid: numpy array of shape (grid_size, grid_size)
                0 = not visible
                1 = visible to camera
        """
        
        # Initialize grid (all unexplored)
        grid_size = self.map_size_cm // self.resolution
        grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
        
        # Check if drone is within grid bounds
        if not (0 <= x < grid_size and 0 <= y < grid_size):
            return grid
        
        # Convert yaw to radians
        # Assuming 0° = North (+Y), 90° = East (+X), etc.
        yaw_rad = np.deg2rad(yaw_deg)
        
        # Calculate FOV boundaries
        fov_rad = np.deg2rad(self.hfov)
        half_fov = fov_rad / 2
        
        # Iterate through all grid cells within max range
        for row in range(max(0, x - vision_range), 
                        min(grid_size, x + vision_range + 1)):
            for col in range(max(0, y - vision_range), 
                            min(grid_size, y + vision_range + 1)):
                
                dx = abs(row - x)
                dy = abs(col - y)
                distance = np.sqrt(dx**2 + dy**2)
                
                # Skip if too far
                if distance > self.vision_range or distance < 0.01:  # Skip drone's own cell
                    continue
                
                # Angle to cell
                angle_to_cell = np.arctan2(dx, dy)  # Note: atan2(x, y) for our convention
                
                # Normalize angle difference to [-π, π]
                angle_diff = angle_to_cell - yaw_rad
                angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
                
                # Check if within FOV
                if abs(angle_diff) <= half_fov:
                    grid[row, col] = 1
        
        return grid
