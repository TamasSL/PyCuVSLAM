import numpy as np
import math

import depth_utils as du

# Set to True to enable debug prints (disable for production)
DEBUG_PRINTS = False

params = dict(
    vision_range=20,   # voxels
    map_size_cm=1600,  #cm
    resolution=10,     # voxel size
    agent_min_z=20,    # cm
    agent_max_z=60,    # cm
    agent_height=20,   # cm
    hfov = 90          # degrees
)

class DroidMapBuilder(object):
    def __init__(self):
        self.params = params
        self.vision_range = params["vision_range"]
        self.map_size_cm = params["map_size_cm"]
        self.resolution = params["resolution"]
        self.agent_min_z = params["agent_min_z"]
        self.agent_max_z = params["agent_max_z"]

        self.map_size = self.map_size_cm // self.resolution
        self.map = np.zeros(
            (
                self.map_size,
                self.map_size,
            ),
            dtype=np.uint8,
        )
        self.explored_area = np.zeros(
            (
                self.map_size,
                self.map_size,
            ),
            dtype=np.uint8,
        )

        self.agent_height = params["agent_height"]
        self.hfov = params["hfov"]
        self.current_field_of_view = np.zeros((self.map_size, self.map_size), dtype=np.uint8)

        # Pre-compute static grids for FOV calculation (vectorized)
        self._precompute_fov_grids()

        # Pre-allocate voxel grid to avoid repeated allocation
        map_height = 300
        x_cells = int(np.round(self.map_size_cm / self.resolution))
        y_cells = int(np.round(self.map_size_cm / self.resolution))
        z_cells = int(np.round(map_height / self.resolution))
        self._voxel_grid = np.zeros((x_cells, y_cells, z_cells), dtype=np.uint8)
        self._map_height = map_height
        return

    def _precompute_fov_grids(self):
        """Pre-compute distance and angle grids for FOV calculation."""
        vr = self.vision_range
        size = 2 * vr + 1

        # Create coordinate grids relative to center
        rows, cols = np.ogrid[-vr:vr+1, -vr:vr+1]

        # Distance from center
        self._fov_distances = np.sqrt(rows**2 + cols**2).astype(np.float32)

        # Angle to each cell (used for FOV check)
        self._fov_angles = np.arctan2(-rows, cols).astype(np.float32)

        # Mask for cells within vision range
        self._fov_range_mask = (self._fov_distances <= vr) & (self._fov_distances > 0.01)

    def set_height(self, height):
        self.agent_height = height

    def update_map(self, points: np.ndarray, current_pose: (float, float, float)):
        """ Updates the 2D obstacle and explored area map based on the SLAM
        point-cloud and drone's current position
        """
        current_pose[0] += self.map_size_cm / 2
        current_pose[1] += self.map_size_cm / 2
        x = int(current_pose[1] / self.resolution)
        y = int(current_pose[0] / self.resolution)

        voxel_grid, _ = self._points_to_fixed_voxel_grid(points)
        self.map = self._build_2d_occupancy_map(voxel_grid)
        self._update_explored_area(x, y, -current_pose[2])
        
        return self.map, self.explored_area


    def _points_to_fixed_voxel_grid(self, points, map_height=None):
        """
        Convert 3D point cloud to fixed-size voxel grid centered at origin.
        Reuses pre-allocated voxel grid to avoid repeated memory allocation.

        Args:
            points: Nx3 numpy array of 3D points [x, y, z] in centimeters
            map_height: Height of map in Z direction in centimeters (default 300cm)

        Returns:
            voxel_grid: 3D numpy array (x, y, z) with 1 for occupied, 0 for free
            origin: (x_min, y_min, z_min) - origin of the voxel grid in meters
        """
        if map_height is None:
            map_height = self._map_height

        # Fixed bounds centered at [0, 0, 0] in XY
        x_min = -self.map_size_cm / 2.0
        x_max = self.map_size_cm / 2.0
        y_min = -self.map_size_cm / 2.0
        y_max = self.map_size_cm / 2.0
        z_min = -map_height / 2
        z_max = map_height / 2

        # Reuse pre-allocated voxel grid (clear it)
        self._voxel_grid.fill(0)
        voxel_grid = self._voxel_grid
        x_cells, y_cells, z_cells = voxel_grid.shape

        if len(points) == 0:
            return voxel_grid, (x_min, y_min, z_min)

        # Filter points within bounds (vectorized)
        neg_x = -points[:, 0]
        mask = (
            (neg_x >= x_min) & (neg_x < x_max) &
            (points[:, 2] >= y_min) & (points[:, 2] < y_max) &
            (points[:, 1] >= z_min) & (points[:, 1] < z_max)
        )

        points_in_bounds = points[mask]
        if DEBUG_PRINTS:
            print(f"  Points in bounds: {len(points_in_bounds):,} / {len(points):,} ({len(points_in_bounds)/len(points)*100:.1f}%)")

        if len(points_in_bounds) == 0:
            if DEBUG_PRINTS:
                print("Warning: No points within map bounds!")
            return voxel_grid, (x_min, y_min, z_min)

        # Convert points to voxel indices (vectorized, avoid intermediate arrays)
        inv_res = 1.0 / self.resolution
        x_indices = np.clip(((-points_in_bounds[:, 0] - x_min) * inv_res).astype(np.int32), 0, x_cells - 1)
        y_indices = np.clip(((points_in_bounds[:, 2] - y_min) * inv_res).astype(np.int32), 0, y_cells - 1)
        z_indices = np.clip(((points_in_bounds[:, 1] - z_min) * inv_res).astype(np.int32), 0, z_cells - 1)

        # Mark occupied voxels
        voxel_grid[x_indices, y_indices, z_indices] = 1

        if DEBUG_PRINTS:
            print(f"  Occupied voxels: {np.sum(voxel_grid):,}")

        return voxel_grid, (x_min, y_min, z_min)

    
    def _build_2d_occupancy_map(self, voxel_grid_3d):
        """
        Build a 2D occupancy map from a 3D voxel grid.

        Args:
            voxel_grid_3d: 3D numpy array (x, y, z) where each cell is 10x10x10cm
                        Values: 0 = free, 1 = occupied (or any non-zero = occupied)

        Returns:
            occupancy_map_2d: 2D numpy array (x, y) with values 0 (free) or 1 (occupied)
        """
        z_dim = voxel_grid_3d.shape[2]

        # Convert height range to voxel indices
        z_min_idx = int(self.agent_min_z / self.resolution) + 15   # shift to account for negative height
        z_max_idx = int(self.agent_max_z / self.resolution) + 15

        # Clamp indices to valid range
        z_min_idx = max(0, z_min_idx)
        z_max_idx = min(z_dim, z_max_idx)

        if DEBUG_PRINTS:
            print(f"3D voxel grid shape: {voxel_grid_3d.shape}")
            print(f"Z indices: {z_min_idx} to {z_max_idx}")

        # Extract the height slice and project to 2D
        # Use np.any along z-axis to check if ANY voxel in height range is occupied
        occupancy_map_2d = np.any(voxel_grid_3d[:, :, z_min_idx:z_max_idx] > 0, axis=2).astype(np.uint8)

        if DEBUG_PRINTS:
            print(f"2D occupancy map shape: {occupancy_map_2d.shape}")
            print(f"Occupied cells: {np.sum(occupancy_map_2d)}")

        return occupancy_map_2d


    def _update_explored_area(self, x, y, yaw_rad):
        """
        Vectorized FOV update for explored area.

        Args:
            x, y: camera's position as grid coordinates
            yaw_rad: camera's orientation
        """
        grid_size = self.map_size

        if DEBUG_PRINTS:
            print(f'cam position {x} {y}')

        # Check if drone is within grid bounds
        if not (0 <= x < grid_size and 0 <= y < grid_size):
            return

        # Clear current FOV
        self.current_field_of_view.fill(0)

        # Calculate FOV boundaries
        fov_rad = np.deg2rad(self.hfov - 10)
        half_fov = fov_rad / 2

        vr = self.vision_range

        # Compute window bounds in the full grid
        row_start = max(0, int(x) - vr)
        row_end = min(grid_size, int(x) + vr + 1)
        col_start = max(0, int(y) - vr)
        col_end = min(grid_size, int(y) + vr + 1)

        # Corresponding indices in the pre-computed grids
        local_row_start = row_start - (int(x) - vr)
        local_row_end = local_row_start + (row_end - row_start)
        local_col_start = col_start - (int(y) - vr)
        local_col_end = local_col_start + (col_end - col_start)

        # Extract relevant portions of pre-computed grids
        distances = self._fov_distances[local_row_start:local_row_end, local_col_start:local_col_end]
        angles = self._fov_angles[local_row_start:local_row_end, local_col_start:local_col_end]
        range_mask = self._fov_range_mask[local_row_start:local_row_end, local_col_start:local_col_end]

        # Compute angle difference (vectorized)
        angle_diff = angles - yaw_rad

        # Normalize to [-pi, pi]
        angle_diff = np.mod(angle_diff + np.pi, 2 * np.pi) - np.pi

        # FOV mask: within range and within FOV angle
        fov_mask = range_mask & (np.abs(angle_diff) <= half_fov)

        # For cells in FOV, check line of sight (this is still the expensive part)
        # Get indices of cells that pass the FOV check
        local_rows, local_cols = np.where(fov_mask)

        # Convert to global coordinates
        global_rows = local_rows + row_start
        global_cols = local_cols + col_start

        # Check line of sight for each candidate cell
        for i in range(len(local_rows)):
            gr, gc = global_rows[i], global_cols[i]
            if self._has_line_of_sight_fast(int(x), int(y), gr, gc):
                self.current_field_of_view[gr, gc] = 1

        self.explored_area += self.current_field_of_view

        if DEBUG_PRINTS:
            print(f"Explored cells: {np.sum(self.explored_area)}")

    def _has_line_of_sight_fast(self, x0, y0, x1, y1):
        """
        Optimized Bresenham's line algorithm for line of sight check.
        Avoids creating intermediate list of points.
        """
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1

        err = dx - dy
        x, y = x0, y0

        # Check all points except the last one
        while not (x == x1 and y == y1):
            if self.map[x, y] == 1:
                return False

            e2 = 2 * err

            if e2 > -dy:
                err -= dy
                x += sx

            if e2 < dx:
                err += dx
                y += sy

        return True

