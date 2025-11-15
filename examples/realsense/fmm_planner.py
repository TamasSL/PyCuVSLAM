from typing import Literal

import cv2
import numpy as np
import skfmm
from numpy import ma


def get_mask(sx: float, sy: float) -> np.ndarray:
    """Create a mask for valid step locations.

    Args:
        sx: X-axis subpixel offset from grid center
        sy: Y-axis subpixel offset from grid center

    Returns:
        Binary mask where 1 indicates locations within step_size radius
    """
    size = 3
    mask = np.zeros((size, size))
    center = size // 2

    for i in range(size):   # this can be simplified the the neihboring cells from the current location
        for j in range(size):
            dist_sq = ((i + 0.5) - (center + sx)) ** 2 + (
                (j + 0.5) - (center + sy)
            ) ** 2
            if dist_sq <= 1:
                mask[i, j] = 1
    return mask


def get_dist(sx: float, sy: float) -> np.ndarray:
    """Compute distance weights for locations within step radius.

    Args:
        sx: X-axis subpixel offset from grid center
        sy: Y-axis subpixel offset from grid center

    Returns:
        Distance matrix with minimum value of 5 for locations within step_size radius,
        1e-10 elsewhere
    """
    size = 3
    dist_matrix = np.full((size, size), 1e-10)
    center = size // 2

    for i in range(size):
        for j in range(size):
            dist_sq = ((i + 0.5) - (center + sx)) ** 2 + (
                (j + 0.5) - (center + sy)
            ) ** 2
            if dist_sq <= 1:
                dist_matrix[i, j] = max(5.0, dist_sq**0.5)
    return dist_matrix


class FMMPlanner:
    """Fast Marching Method planner for computing optimal navigation paths.

    Uses the Fast Marching Method to compute distance fields on a traversability
    map and plan short-term goals toward a long-term target.
    """

    _MIN_LOCAL_DIST = 4.0
    _REPLAN_THRESHOLD = -0.0001

    def __init__(
        self,
        traversible: np.ndarray,
        num_rots: int,
        version: Literal["classic", "fm2"] = "fm2",
    ):
        """Initialize the FMM planner.

        Args:
            traversible: Binary traversability map (1=traversible, 0=obstacle)
            num_rots: Number of discrete rotation angles for the agent
            version: Planning version - "classic" for standard FMM, "fm2" for Fast Marching Square
        """

        self.traversible = traversible

        self.angle_value = [0, 2.0 * np.pi / num_rots, -2.0 * np.pi / num_rots, 0]
        self.du = 1
        self.num_rots = num_rots
        self.version = version
        self.fmm_dist = None
        self.velocity_map = None
        self.viz_dict = {}

        # Precompute velocity map for FM² (version 2)
        if self.version == "fm2":
            self._compute_velocity_map()

    def _compute_velocity_map(self) -> None:
        """Compute velocity map for Fast Marching Square (FM²).

        The velocity at each point is proportional to its distance from the nearest
        obstacle. This encourages paths to stay in open spaces (safer and smoother).
        """
        # Check if there are any obstacles
        if np.all(self.traversible == 1):
            # No obstacles - uniform velocity
            self.velocity_map = np.ones_like(self.traversible, dtype=float)
            self.viz_dict["velocity_map"] = self.velocity_map.copy()
            return

        # Create signed distance function for FMM
        # Obstacles (traversible=0) should be negative, free space (traversible=1) positive
        phi = np.where(self.traversible == 0, -1.0, 1.0)

        # Compute signed distance from obstacle boundaries
        # This gives positive distance in free space, negative inside obstacles
        try:
            dist_to_obstacles = skfmm.distance(phi, dx=1)
        except ValueError:
            # Fallback: uniform velocity if distance computation fails
            self.velocity_map = np.ones_like(self.traversible, dtype=float)
            self.velocity_map[self.traversible == 0] = 1e-6
            self.viz_dict["velocity_map"] = self.velocity_map.copy()
            return

        # Take absolute value and ensure free space has positive distances
        dist_to_obstacles = np.abs(dist_to_obstacles)
        dist_to_obstacles[self.traversible == 1] = np.abs(
            dist_to_obstacles[self.traversible == 1]
        )

        # Velocity is proportional to distance from obstacles
        # Normalize to [0.1, 1.0] range to avoid zero velocities
        max_dist = np.max(dist_to_obstacles[self.traversible == 1])
        if max_dist > 0:
            self.velocity_map = 0.1 + 0.9 * (dist_to_obstacles / max_dist)
        else:
            self.velocity_map = np.ones_like(self.traversible, dtype=float)

        # Set velocity to near-zero at obstacles
        self.velocity_map[self.traversible == 0] = 1e-6

        # Store for visualization
        self.viz_dict["velocity_map"] = self.velocity_map.copy()

    def set_goal(self, goal: tuple[float, float]) -> np.ndarray:
        """Set the goal location and compute FMM distance field.

        For version "1": Uses standard FMM with uniform speed.
        For version "2": Uses Fast Marching Square with velocity field based on
                        distance to obstacles (paths prefer open spaces).

        Args:
            goal: (x, y) coordinates of the goal in the original map frame

        Returns:
            Binary mask indicating reachable locations from the goal
        """
        # Create masked array where 0 = non-traversible
        traversible_ma = ma.masked_values(self.traversible, 0)

        goal_x = int(goal[0])
        goal_y = int(goal[1])
        traversible_ma[goal_y, goal_x] = 0

        if self.version == "fm2" and self.velocity_map is not None:
            # FM^2: Use velocity map (paths prefer high-velocity/open areas)
            dd = skfmm.travel_time(traversible_ma, speed=self.velocity_map, dx=1)
        else:
            # Standard FMM: Uniform speed
            dd = skfmm.distance(traversible_ma, dx=1)

        dd_mask = np.invert(np.isnan(ma.filled(dd, np.nan)))
        dd = ma.filled(dd, np.max(dd) + 1)
        self.fmm_dist = dd

        self.viz_dict["map"] = dd.copy()
        self.viz_dict["goal"] = (goal_y, goal_x)

        return dd_mask

    def _extract_local_window(
        self, array: np.ndarray, center: list[int], window_size: int
    ) -> np.ndarray:
        """Extract a local window around a center position.

        Args:
            array: Array to extract from
            center: Center position [x, y]
            window_size: Size of the window (half-width in each direction)

        Returns:
            Extracted window of shape (2*window_size+1, 2*window_size+1)
        """
        return array[
            center[0] : center[0] + 2 * window_size + 1,
            center[1] : center[1] + 2 * window_size + 1,
        ]

    def _compute_local_fmm_distance(self, traversible_window: np.ndarray) -> np.ndarray:
        """Compute FMM distance from the center of a traversible window.

        Args:
            traversible_window: Local traversibility map

        Returns:
            Distance field from the center position
        """
        traversible_ma = ma.masked_values(traversible_window, 0)
        center = self.du
        traversible_ma[center, center] = 0

        distance_field = skfmm.distance(traversible_ma, dx=1)
        max_dist = np.max(distance_field)
        distance_field = ma.filled(distance_field, max_dist + 1)

        return distance_field

    def _apply_gradient_filter(
        self,
        cost_map: np.ndarray,
        distance_weights: np.ndarray,
        threshold: float = -1.5,
    ) -> np.ndarray:
        """Filter out locations with poor gradient descent.

        Args:
            cost_map: Cost map to filter
            distance_weights: Distance weights for normalization
            threshold: Gradient ratio threshold

        Returns:
            Filtered cost map
        """
        gradient_ratio = cost_map / distance_weights
        cost_map[gradient_ratio < threshold] = 1.0
        return cost_map

    def _compute_path(
        self, start: tuple[int, int], goal: tuple[int, int]
    ) -> np.ndarray:
        """Compute optimal path from start to goal using gradient descent on FMM field.

        Args:
            start: Start position (y, x) in scaled coordinates
            goal: Goal position (y, x) in scaled coordinates

        Returns:
            Array of shape (N, 2) containing path points as (y, x) coordinates
        """
        path = [start]
        # map = np.pad(self.fmm_dist, self.du, mode="edge")
        map = self.fmm_dist
        current = np.array(start, dtype=float)
        max_steps = 5000  # Prevent infinite loops
        step_size = 1

        for _ in range(max_steps):
            # Check if we reached the goal
            dist_to_goal = np.linalg.norm(current - np.array(goal))
            if dist_to_goal <= step_size:
                path.append(goal)
                break

            # Compute gradient of distance field at current position
            y, x = int(current[0]), int(current[1])

            # Check bounds
            if y <= 0 or y >= map.shape[0] - 1 or x <= 0 or x >= map.shape[1] - 1:
                break

            # Compute gradient using central differences
            grad_y = (map[y + 1, x] - map[y - 1, x]) / 2.0
            grad_x = (map[y, x + 1] - map[y, x - 1]) / 2.0

            # Move in direction of negative gradient (toward goal)
            grad_norm = np.sqrt(grad_y**2 + grad_x**2)
            if grad_norm < 1e-6:
                break

            current[0] -= step_size * grad_y / grad_norm
            current[1] -= step_size * grad_x / grad_norm

            # Add point to path (every few steps to avoid too dense)
            path.append(tuple(current.astype(int)))

        return np.array(path)

    def get_short_term_goal(self, state: list[int]) -> tuple[float, float, bool]:
        """Compute the next short-term goal toward the long-term goal.

        This method uses Fast Marching Method to find the locally optimal next
        step that makes progress toward the goal while respecting traversibility
        constraints and avoiding local minima. Prefers forward moves over rotations.

        Args:
            state: Current agent state [x, y, orientation] in pixels

        Returns:
            Tuple of (stg_x, stg_y, replan) where:
                - stg_x: Short-term goal x coordinate
                - stg_y: Short-term goal y coordinate
                - replan: Whether replanning is needed (goal unreachable)
        """
        # Convert to scaled coordinates and extract subpixel offsets
        state_scaled = state
        subpixel_dx = state_scaled[0] - int(state_scaled[0])
        subpixel_dy = state_scaled[1] - int(state_scaled[1])
        state_int = [int(x) for x in state_scaled]

        self.viz_dict["start"] = state_int
        self.viz_dict["path"] = self._compute_path(
            self.viz_dict["start"], self.viz_dict["goal"]
        )

        # Create masks for valid reachable locations
        step_mask = get_mask(subpixel_dx, subpixel_dy)
        distance_weights = get_dist(subpixel_dx, subpixel_dy)

        # Extract local region of global FMM distance field
        pad_value = self.fmm_dist.shape[0] ** 2

        # For FM2, apply light smoothing to reduce noise from velocity field
        fmm_for_planning = self.fmm_dist
        if self.version == "fm2":
            from scipy.ndimage import gaussian_filter
            fmm_for_planning = gaussian_filter(self.fmm_dist, sigma=1.0)

        padded_fmm = np.pad(
            fmm_for_planning, self.du, "constant", constant_values=pad_value
        )

        local_fmm_dist = self._extract_local_window(padded_fmm, state_int, self.du)
        # Validate extraction
        expected_size = 2 * self.du + 1
        assert local_fmm_dist.shape == (expected_size, expected_size), (
            f"Planning error: unexpected local window shape {local_fmm_dist.shape}, "
            f"expected ({expected_size}, {expected_size})"
        )

        # Compute relative cost: distance to goal from each location
        relative_cost = local_fmm_dist * step_mask
        relative_cost += (1 - step_mask) * pad_value  # Penalize unreachable locations
        relative_cost -= relative_cost[
            self.du, self.du
        ]  # Make relative to current position

        # Adjust gradient filtering threshold for FM2
        gradient_threshold = -1.5 if self.version == "classic" else -2.5
        relative_cost = self._apply_gradient_filter(
            relative_cost, distance_weights, threshold=gradient_threshold
        )

        # Compute local FMM distance to avoid local minima
        padded_traversible = np.pad(
            self.traversible, self.du, "constant", constant_values=0
        )
        local_traversible = self._extract_local_window(
            padded_traversible, state_int, self.du
        )
        local_distance = self._compute_local_fmm_distance(local_traversible)

        # Normalize cost by local distance to encourage exploration
        local_distance = np.maximum(local_distance, self._MIN_LOCAL_DIST)
        normalized_cost = relative_cost / local_distance
        normalized_cost = self._apply_gradient_filter(
            normalized_cost, np.ones_like(normalized_cost), threshold=gradient_threshold
        )

        # Favor moves that make direct progress toward goal
        # Compute gradient direction from FMM field
        center = self.du
        center_y, center_x = state_int[0], state_int[1]

        # Check bounds for gradient computation
        if (center_y > 0 and center_y < fmm_for_planning.shape[0] - 1 and
            center_x > 0 and center_x < fmm_for_planning.shape[1] - 1):

            # Gradient points toward goal (negative gradient of distance field)
            grad_y = -(fmm_for_planning[center_y + 1, center_x] - fmm_for_planning[center_y - 1, center_x]) / 2.0
            grad_x = -(fmm_for_planning[center_y, center_x + 1] - fmm_for_planning[center_y, center_x - 1]) / 2.0
            grad_norm = np.sqrt(grad_y**2 + grad_x**2)

            if grad_norm > 1e-6:
                # Normalize gradient
                grad_y /= grad_norm
                grad_x /= grad_norm

                # Compute alignment of each candidate with goal direction
                size = normalized_cost.shape[0]
                y_indices, x_indices = np.meshgrid(
                    np.arange(size) - center, np.arange(size) - center, indexing='ij'
                )

                # Direction vectors to each candidate
                distances = np.sqrt(y_indices**2 + x_indices**2)
                # Avoid division by zero at center
                with np.errstate(divide='ignore', invalid='ignore'):
                    dir_y = y_indices / distances
                    dir_x = x_indices / distances
                    dir_y = np.nan_to_num(dir_y)
                    dir_x = np.nan_to_num(dir_x)

                # Dot product: alignment with goal direction (1 = aligned, -1 = opposite)
                alignment = dir_y * grad_y + dir_x * grad_x

                # Apply penalty for moves not aligned with goal (favor forward progress)
                # Reduce penalty weight for FM2 to account for less direct paths
                penalty_weight = 0.5 if self.version == "classic" else 0.2
                alignment_penalty = (1.0 - alignment) * (distances > 0.5) * penalty_weight
                normalized_cost = normalized_cost + alignment_penalty

        # Find location with minimum cost (steepest descent toward goal)
        min_idx = np.argmin(normalized_cost)
        local_goal_x, local_goal_y = np.unravel_index(min_idx, normalized_cost.shape)

        # Determine if replanning is needed (no valid descent direction)
        replan = normalized_cost[local_goal_x, local_goal_y] > self._REPLAN_THRESHOLD

        # Convert back to original coordinate frame
        goal_x = (local_goal_x + state_int[0] - self.du) + 0.5
        goal_y = (local_goal_y + state_int[1] - self.du) + 0.5

        return goal_x, goal_y, replan
