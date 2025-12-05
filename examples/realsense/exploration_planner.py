"""
Frontier-Based Exploration Algorithm for Drone Navigation

Given:
- Current drone position
- Occupancy grid (0=free, 1=occupied, -1=unknown)

Returns:
- Next best target position to explore
"""

import numpy as np
from scipy.ndimage import binary_dilation, label, generate_binary_structure
import heapq


class ExplorationPlanner:
    def __init__(self, grid_resolution=0.05, safety_distance=0.3, visualizer = None):
        """
        Args:
            grid_resolution: Meters per grid cell
            safety_distance: Minimum distance from obstacles (meters)
        """
        self.grid_resolution = grid_resolution
        self.safety_margin = int(safety_distance / grid_resolution)
        self.visualizer = visualizer
    
    def get_next_exploration_target(self, occupancy_grid, current_target_pos, 
                                     max_distance=None):
        """
        Find next best exploration target using frontier-based exploration.
        
        Args:
            occupancy_grid: 2D numpy array (H, W)
                           0 = free space
                           1 = occupied
                          -1 = unknown/unexplored
            current_target_pos: (row, col) current target position in grid coordinates
            max_distance: Maximum distance to consider (in grid cells), None = unlimited
        
        Returns:
            target_pos: (row, col) next target position
            or None if no frontiers found
        """
        
        # 1. Find all frontier cells (boundary between free and unknown)
        frontiers = self._find_frontiers(occupancy_grid)
        
        if len(frontiers) == 0:
            print("No frontiers found - exploration complete!")
            return None
        
        # 2. Cluster frontiers into regions
        frontier_clusters = self._cluster_frontiers(frontiers, occupancy_grid.shape)
        
        # 3. Evaluate each frontier cluster
        best_target = None
        best_score = -float('inf')

        valid_frontier_clusters = []
        
        for cluster in frontier_clusters:
            if len(cluster) < 3:  # Ignore tiny frontiers
                continue
            
            # Get centroid of frontier cluster
            centroid = np.mean(cluster, axis=0).astype(int)

            diameter = self._get_cluster_diameter_approx(cluster)
            if diameter < 4:
                continue
            
            # Calculate distance from current target
            distance = np.linalg.norm(np.array(current_target_pos) - centroid)
            
            # Skip if too far
            #if max_distance and distance > max_distance:
            #    continue
            
            # Calculate score (larger frontiers closer to drone are better)
            frontier_size = len(cluster)
            information_gain = frontier_size  # How much new area we'll see
            
            # Score function: balance between information gain and distance
            w_info = 1.0    # Weight for information gain
            w_dist = -1.0    # Weight for distance (negative = prefer closer)
            
            score = w_info * information_gain + w_dist * distance
            valid_frontier_clusters.append(cluster)
            
            if score > best_score:
                best_score = score
                best_target = tuple(centroid)

        self.visualizer.visualize_frontier_clusters(valid_frontier_clusters)
        
        return best_target

    def _get_cluster_diameter_approx(self, cluster):
        """
        Approximate diameter using bounding box diagonal.
        Fast but may overestimate.
        """
        if len(cluster) < 2:
            return 0.0
        
        cluster = np.array(cluster)
        
        # Get bounding box
        min_coords = cluster.min(axis=0)
        max_coords = cluster.max(axis=0)
        
        # Diagonal of bounding box
        return np.linalg.norm(max_coords - min_coords)
    
    def _find_frontiers(self, occupancy_grid):
        """
        Find frontier cells (free cells adjacent to unknown cells).
        
        Returns:
            List of (row, col) frontier positions
        """
        # Create masks
        free_mask = (occupancy_grid == 0)
        unknown_mask = (occupancy_grid == -1)
        
        # Dilate unknown regions to find borders
        structure = generate_binary_structure(2, 2)  # 2D, full connectivity
        unknown_dilated = binary_dilation(unknown_mask, iterations=1, structure=structure)
        
        # Frontiers are free cells adjacent to unknown
        frontier_mask = free_mask & unknown_dilated
        
        # Get frontier coordinates
        frontier_coords = np.argwhere(frontier_mask)
        
        return frontier_coords.tolist()
    
    def _cluster_frontiers(self, frontiers, grid_shape):
        """
        Cluster nearby frontier cells into regions.
        
        Returns:
            List of clusters, where each cluster is a list of (row, col)
        """
        if len(frontiers) == 0:
            return []
        
        # Create frontier mask
        frontier_mask = np.zeros(grid_shape, dtype=bool)
        for r, c in frontiers:
            frontier_mask[r, c] = True
        
        # Use connected component labeling
        labeled, num_clusters = label(frontier_mask)
        
        # Extract clusters
        clusters = []
        for cluster_id in range(1, num_clusters + 1):
            cluster_coords = np.argwhere(labeled == cluster_id)
            clusters.append(cluster_coords.tolist())
        
        return clusters
    
    def _is_reachable(self, occupancy_grid, start, goal):
        """
        Check if goal is reachable from start.
        
        Returns:
            True if reachable, False otherwise
        """
        # Check if goal is in free space
        if occupancy_grid[goal[0], goal[1]] != 0:
            return False
        
        # Quick check: if very close, assume reachable
        distance = np.linalg.norm(np.array(start) - np.array(goal))
        if distance < 5:
            return True
        
        # For longer distances, do quick A* check
        return self._has_path_astar(occupancy_grid, start, goal, max_iterations=500)
    
    def _has_path_astar(self, occupancy_grid, start, goal, max_iterations=1000):
        """
        Quick A* path existence check.
        """
        H, W = occupancy_grid.shape
        start = tuple(start)
        goal = tuple(goal)
        
        if start == goal:
            return True
        
        # Create obstacle mask
        obstacle_mask = (occupancy_grid == 1)
        if self.safety_margin > 0:
            obstacle_mask = binary_dilation(obstacle_mask, iterations=self.safety_margin)
        
        # A* search
        open_set = []
        heapq.heappush(open_set, (0, start))
        g_score = {start: 0}
        
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        iterations = 0
        while open_set and iterations < max_iterations:
            iterations += 1
            _, current = heapq.heappop(open_set)
            
            if current == goal:
                return True
            
            # Check 8-connected neighbors
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                neighbor = (current[0] + dr, current[1] + dc)
                neighbor = [int(x) for x in neighbor]
                
                if not (0 <= neighbor[0] < H and 0 <= neighbor[1] < W):
                    continue
                
                if obstacle_mask[neighbor[0], neighbor[1]]:
                    continue
                
                if occupancy_grid[neighbor[0], neighbor[1]] == -1:
                    continue
                
                tentative_g = g_score[tuple(current)] + 1
                
                if tuple(neighbor) not in g_score or tentative_g < g_score[tuple(neighbor)]:
                    g_score[tuple(neighbor)] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
        
        return False


def grid_to_world(grid_pos, origin, resolution):
    """Convert grid coordinates to world coordinates."""
    row, col = grid_pos
    x = origin[0] + col * resolution
    y = origin[1] + row * resolution
    return (x, y)


def world_to_grid(world_pos, origin, resolution):
    """Convert world coordinates to grid coordinates."""
    x, y = world_pos
    col = int((x - origin[0]) / resolution)
    row = int((y - origin[1]) / resolution)
    return (row, col)
