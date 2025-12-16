#
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from typing import List, Dict, Deque
from collections import deque

import numpy as np
import numpy.typing as npt
import rerun as rr
import rerun.blueprint as rrb
import torch
import cuvslam as vslam

from nvblox_torch.examples.realsense.vslam_utils import from_homogeneous
from nvblox_torch.mesh import Mesh

# NOTE(alexmillane, 2025.05.22): This is a modified version of the RerunVisualizer class
# from the cuvslam examples. Right now the visualizer does not ship with the PyCuVSLAM
# package, so we need to copy it here. If they move it into their package we can import
# it from there and subclass it.

# pylint: disable=invalid-name


class RerunVisualizer:
    """Visualizer for cuvslam and nvblox for the nvblox_torch realsense example."""

    def __init__(self) -> None:
        """Initialize rerun visualizer."""
        self._start_rerun_visualizer()
        # Parameters
        self.camera_pose_axis_scale = 0.1
        self.trajectory_length = 100
        # State
        self.track_colors: Dict[int, npt.NDArray] = {}
        self.t_W_C_history: Deque[npt.NDArray] = deque(maxlen=self.trajectory_length)

    def _start_rerun_visualizer(self) -> None:
        rr.init('cuVSLAM Visualizer', spawn=True)
        rr.log('world', rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        rr.send_blueprint(rrb.Blueprint(
            rrb.TimePanel(state='collapsed'),
            rrb.Horizontal(column_shares=[0.5, 0.5],
                           contents=[
                               rrb.Spatial2DView(origin='world/camera_0'),
                               rrb.Spatial3DView(origin='world'),
                           ])),
                          make_active=True)

    def _log_rig_pose(self, q_W_C: npt.NDArray, t_W_C: npt.NDArray) -> None:
        """Log rig pose to Rerun."""
        rr.log(
            'world/camera_0',
            rr.Transform3D(translation=t_W_C, quaternion=q_W_C),
            rr.Arrows3D(
                vectors=np.eye(3) * self.camera_pose_axis_scale,
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]    # RGB for XYZ axes
            ))

    def _log_observations(self, observations_main_cam: List[vslam.Observation],
                          image: npt.NDArray) -> None:
        """Log 2D observations for a specific camera with consistent colors per track."""
        if not observations_main_cam:
            return

        # Assign random color to new tracks
        for obs in observations_main_cam:
            if obs.id not in self.track_colors:
                self.track_colors[obs.id] = np.random.randint(0, 256, size=3)

        points = np.array([[obs.u, obs.v] for obs in observations_main_cam])
        colors = np.array([self.track_colors[obs.id] for obs in observations_main_cam])

        rr.log('world/camera_0/observations', rr.Points2D(positions=points,
                                                          colors=colors,
                                                          radii=5.0),
               rr.Image(image).compress())

    def _log_trajectory(self) -> None:
        """Log the trajectory to Rerun."""
        rr.log('world/trajectory', rr.LineStrips3D(self.t_W_C_history), static=True)

    def _visualize_map(self, map_uv):
        obstacles = []
        explored = []
        for p in map_uv:
            if p[2] == 0:
                obstacles.append([p[0],p[1]])
            elif p[2] == 2:
                explored.append([p[0],p[1]])
        rr.log('world/camera_0/obstacles', rr.Points2D(
            obstacles, radii=0.5, colors=[255, 255, 255]
        ))
        rr.log('world/camera_0/explored', rr.Points2D(
            explored, radii=0.5, colors=[128, 128, 128]
        ))

    def visualize_frontier_clusters(self, clusters):
        for index in range(10):
            rr.log(f'world/camera_0/clusters_{index}', rr.Points2D(
                    [], radii=0.5, colors=[(index*20 + 55) % 255, (index*30) % 255, (index*40 + 75) % 255], draw_order=50
                ))

        # print(f'num clusters: {len(clusters)}')
        for index, cluster in enumerate(clusters):
            points = []
            for row, col in cluster:
                points.append([row, col])
            rr.log(f'world/camera_0/clusters_{index}', rr.Points2D(
                points, radii=0.5, colors=[(index*20 + 55) % 255, (index*30) % 255, (index*40 + 75) % 255], draw_order=50
            ))

    def _visualize_goal(self, goal):
        rr.log('world/camera_0/ltg', rr.Points2D(
            goal, radii=0.5, colors=[0, 0, 255], draw_order=150
        ))

    def _visualize_stg(self, goal, goal_height):
        if goal_height == 0:
            color = [255, 165, 0]
        elif goal_height == 1:
            color = [255, 0, 0]
        elif goal_height == 2:
            color = [255, 0, 165]
        else:
            color = [0, 255, 0]
        
        rr.log('world/camera_0/stg', rr.Points2D(
            goal, radii=0.5, colors=color,  draw_order=80
        ))

    def _visualize_drone(self, drone_uv, yaw_rad):
        """
        Visualize drone position and orientation
        
        Args:
            drone_uv: (u, v) position in pixels
            yaw_rad: yaw angle in radians
        """
        # Draw the drone position as a point
        #rr.log('world/camera_0/drone', rr.Points2D(
        #    drone_uv, radii=0.5, colors=[128, 55, 60]
        #))
        
        # Draw arrow showing orientation
        arrow_length = 3  # pixels
        
        # Calculate arrow direction from yaw
        yaw_rad += np.pi / 2.0
        dx = arrow_length * np.cos(yaw_rad)
        dy = arrow_length * np.sin(yaw_rad)

        if drone_uv[0][2] == 0:
            color = [255, 165, 0]
        elif drone_uv[0][2] == 1:
            color = [255, 0, 0]
        elif drone_uv[0][2] == 2:
            color = [255, 0, 165]
        else:
            color = [0, 255, 0]
        
        # Arrow origin and vector
        rr.log('world/camera_0/z_drone_orientation', rr.Arrows2D(
            origins=[drone_uv[0][0], drone_uv[0][1]],
            vectors=[[dx, dy]],
            colors=color,  # Red arrow
            radii=0.3,
            draw_order=100
        ))

    def _visualize_nvblox_mesh(self, mesh: Mesh) -> None:
        rr.log(
            'world/mesh',
            rr.Mesh3D(
                vertex_positions=mesh.vertices().cpu().numpy(),
                vertex_colors=mesh.vertex_colors().cpu().numpy(),
                triangle_indices=mesh.triangles().cpu().numpy(),
            ))

    def visualize_cuvslam(self, T_W_C: torch.Tensor, image: npt.NDArray,
                          observations: List[vslam.Observation]) -> None:
        """Visualize the cuvslam outputs.

        We visualize:
        1) The camera pose,
        2) The image used for tracking and the features
        3) The past N camera positions as a trajectory.

        Args:
            T_W_C: The camera pose in the world frame.
            image: The image used for tracking and the features.
            observations: The observations used for tracking.
        """
        t_W_C, q_W_C = from_homogeneous(T_W_C)
        #self.t_W_C_history.append(t_W_C)
        self._log_rig_pose(q_W_C, t_W_C)
        #self._log_observations(observations, image)
        #self._log_trajectory()

    def visualize_nvblox(self, mesh: Mesh) -> None:
        """Visualize the nvblox mesh.

        Args:
            mesh: The nvblox mesh to visualize.
        """
        self._visualize_nvblox_mesh(mesh)