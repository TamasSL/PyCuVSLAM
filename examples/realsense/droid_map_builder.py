import numpy as np
import math

import depth_utils as du

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
        return

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


    def _points_to_fixed_voxel_grid(self, points, map_height=300):
        """
        Convert 3D point cloud to fixed-size voxel grid centered at origin.
        
        Args:
            points: Nx3 numpy array of 3D points [x, y, z] in centimeters
            map_height: Height of map in Z direction in centimeters (default 30cm)
        
        Returns:
            voxel_grid: 3D numpy array (x, y, z) with 1 for occupied, 0 for free
            origin: (x_min, y_min, z_min) - origin of the voxel grid in meters
        """
        # Fixed bounds centered at [0, 0, 0] in XY, starting at z=0
        x_min = -self.map_size_cm / 2.0
        x_max = self.map_size_cm / 2.0
        y_min = -self.map_size_cm / 2.0
        y_max = self.map_size_cm / 2.0
        z_min = -map_height / 2
        z_max = map_height / 2
        
        # Compute grid dimensions
        x_cells = int(np.round(self.map_size_cm / self.resolution))
        y_cells = int(np.round(self.map_size_cm / self.resolution))
        z_cells = int(np.round(map_height / self.resolution))
        
        """
        print(f"Fixed voxel grid:")
        print(f"  Map bounds:")
        print(f"    X: [{x_min:.2f}, {x_max:.2f}] m")
        print(f"    Y: [{y_min:.2f}, {y_max:.2f}] m")
        print(f"    Z: [{z_min:.2f}, {z_max:.2f}] m")
        print(f"  Grid shape: {x_cells} x {y_cells} x {z_cells}")
        print(f"  Total voxels: {x_cells * y_cells * z_cells:,}")
        print(f"  Memory: {(x_cells * y_cells * z_cells) / (1024**2):.2f} MB")
        """

        # Initialize empty voxel grid
        voxel_grid = np.zeros((x_cells, y_cells, z_cells), dtype=np.uint8)
        
        if len(points) == 0:
            # print("Warning: Empty point cloud!")
            return voxel_grid, (x_min, y_min, z_min)
        
        # Filter points within bounds
        mask = (
            (-points[:, 0] >= x_min) & (-points[:, 0] < x_max) &
            (points[:, 2] >= y_min) & (points[:, 2] < y_max) &
            (points[:, 1] >= z_min) & (points[:, 1] < z_max)
        )
        
        points_in_bounds = points[mask]
        print(f"  Points in bounds: {len(points_in_bounds):,} / {len(points):,} ({len(points_in_bounds)/len(points)*100:.1f}%)")
        
        if len(points_in_bounds) == 0:
            print("Warning: No points within map bounds!")
            return voxel_grid, (x_min, y_min, z_min)
        
        # Convert points to voxel indices
        x_indices = ((-points_in_bounds[:, 0] - x_min) / self.resolution).astype(int)
        y_indices = ((points_in_bounds[:, 2] - y_min) / self.resolution).astype(int)
        z_indices = ((points_in_bounds[:, 1] - z_min) / self.resolution).astype(int)
        
        # Clamp indices to valid range (safety check)
        x_indices = np.clip(x_indices, 0, x_cells - 1)
        y_indices = np.clip(y_indices, 0, y_cells - 1)
        z_indices = np.clip(z_indices, 0, z_cells - 1)
        
        # Mark occupied voxels
        voxel_grid[x_indices, y_indices, z_indices] = 1
        
        occupied_voxels = np.sum(voxel_grid)
        print(f"  Occupied voxels: {occupied_voxels:,}")
        
        origin = (x_min, y_min, z_min)
        return voxel_grid, origin

    
    def _build_2d_occupancy_map(self, voxel_grid_3d):
        """
        Build a 2D occupancy map from a 3D voxel grid.
        
        Args:
            voxel_grid_3d: 3D numpy array (x, y, z) where each cell is 10x10x10cm
                        Values: 0 = free, 1 = occupied (or any non-zero = occupied)
            z_resolution: Size of each voxel in z-direction in meters (default 0.1 = 10cm)
        
        Returns:
            occupancy_map_2d: 2D numpy array (x, y) with values 0 (free) or 1 (occupied)
        """
        # Get dimensions
        x_dim, y_dim, z_dim = voxel_grid_3d.shape
        
        # Convert height range to voxel indices
        z_min_idx = int(self.agent_min_z / self.resolution) + 15   # shift to account for negative height
        z_max_idx = int(self.agent_max_z / self.resolution) + 15
        
        # Clamp indices to valid range
        z_min_idx = max(0, z_min_idx)
        z_max_idx = min(z_dim, z_max_idx)
        
        print(f"3D voxel grid shape: {voxel_grid_3d.shape}")
        print(f"Z indices: {z_min_idx} to {z_max_idx}")
        
        # Extract the height slice
        height_slice = voxel_grid_3d[:, :, z_min_idx:z_max_idx]
        
        # Project to 2D: any occupied voxel in the height range -> occupied in 2D
        # Use np.any along z-axis to check if ANY voxel in height range is occupied
        occupancy_map_2d = np.any(height_slice > 0, axis=2).astype(np.uint8)
        
        print(f"2D occupancy map shape: {occupancy_map_2d.shape}")
        print(f"Occupied cells: {np.sum(occupancy_map_2d)}")
        print(f"Free cells: {np.sum(occupancy_map_2d == 0)}")
        
        return occupancy_map_2d


    def _update_explored_area(self, x, y, yaw_rad):
        """
        Args:
            x, y: camera's position as grid coordinates
            yaw_rad: camera's orientation
        """
        
        # Initialize grid (all unexplored)
        grid_size = self.map_size_cm // self.resolution
        del self.current_field_of_view
        self.current_field_of_view = np.zeros((grid_size, grid_size), dtype=np.uint8)
        
        # Check if drone is within grid bounds
        print(f'cam position {x} {y}')
        if not (0 <= x < grid_size and 0 <= y < grid_size):
            return
        
        # Calculate FOV boundaries
        fov_rad = np.deg2rad(self.hfov - 10)  # subtract 10 degrees, it's worst to overestimate the explored area
        half_fov = fov_rad / 2
        
        # Iterate through all grid cells within max range
        for row in range(max(0, int(x) - self.vision_range), 
                        min(grid_size, int(x) + self.vision_range + 1)):
            for col in range(max(0, int(y) - self.vision_range), 
                            min(grid_size, int(y) + self.vision_range + 1)):
                dx = row - x
                dy = col - y
                distance = np.sqrt(dx**2 + dy**2)
                
                # Skip if too far
                if distance > self.vision_range or distance < 0.01:  # Skip drone's own cell
                    continue
                
                # Angle to cell
                angle_to_cell = np.arctan2(-dx, dy)
                
                # Normalize angle difference to [-π, π]
                angle_diff = angle_to_cell - yaw_rad
                
                # Check if within FOV
                if abs(angle_diff) <= half_fov:
                    if self._has_line_of_sight(x, y, row, col):
                        self.current_field_of_view[row, col] = 1

        self.explored_area += self.current_field_of_view
        print(f"Explored cells: {np.sum(self.explored_area)}")

    def _has_line_of_sight(self, x0, y0, x1, y1):
        """
        Bresenham's line algorithm for line of sight check.
        """
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        
        # Get points along the line
        points = self._bresenham_line(x0, y0, x1, y1)
        
        # Check if any point (except the last) is an obstacle
        for i, (px, py) in enumerate(points[:-1]):  # Exclude target cell
            if self.map[px, py] == 1:
                return False
        
        del points
        return True


    def _bresenham_line(self, x0, y0, x1, y1):
        """
        Generate points along a line using Bresenham's algorithm.
        
        Returns:
            List of (x, y) tuples along the line
        """
        points = []
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        
        err = dx - dy
        
        x, y = x0, y0
        
        while True:
            points.append((x, y))
            
            if x == x1 and y == y1:
                break
            
            e2 = 2 * err
            
            if e2 > -dy:
                err -= dy
                x += sx
            
            if e2 < dx:
                err += dx
                y += sy
        
        return points
