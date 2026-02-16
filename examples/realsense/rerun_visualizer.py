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
from typing import List, Optional, Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

# Constants
DEFAULT_NUM_VIZ_CAMERAS = 2
POINT_RADIUS = 5.0
ARROW_SCALE = 0.1
GRAVITY_ARROW_SCALE = 0.02


class RerunVisualizer:
    """Rerun-based visualizer for cuVSLAM tracking results."""
    
    def __init__(self, num_viz_cameras: int = DEFAULT_NUM_VIZ_CAMERAS) -> None:
        """Initialize rerun visualizer.
        
        Args:
            num_viz_cameras: Number of cameras to visualize
        """
        self.num_viz_cameras = num_viz_cameras
        rr.init("cuVSLAM Visualizer", spawn=True)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        
        # Set up the visualization layout
        self._setup_blueprint()
        self.track_colors = {}

    def _setup_blueprint(self) -> None:
        """Set up the Rerun blueprint for visualization layout."""
        rr.send_blueprint(
            rrb.Blueprint(
                rrb.TimePanel(state="collapsed"),
                rrb.Horizontal(
                    column_shares=[0.5, 0.5],
                    contents=[
                        rrb.Vertical(contents=[
                            rrb.Spatial2DView(origin='world/level'),
                            rrb.Spatial2DView(origin=f'world/camera_0'),
                            rrb.Spatial2DView(origin=f'world/camera_1')
                        ]),
                        rrb.Spatial3DView(origin='world')
                    ]
                )
            ),
            make_active=True
        )

    def _log_rig_pose(
        self, rotation_quat: np.ndarray, translation: np.ndarray
    ) -> None:
        """Log rig pose to Rerun.
        
        Args:
            rotation_quat: Rotation quaternion
            translation: Translation vector
        """
        rr.log(
            "world/camera_0",
            rr.Transform3D(translation=translation, quaternion=rotation_quat),
            rr.Arrows3D(
                vectors=np.eye(3) * ARROW_SCALE,
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]  # RGB for XYZ
            )
        )

    def _log_observations(
        self,
        observations_main_cam: List[Any],
        image: np.ndarray,
        camera_name: str
    ) -> None:
        """Log 2D observations for a specific camera with consistent colors.
        
        Args:
            observations_main_cam: List of observations
            image: Camera image
            camera_name: Name of the camera for logging
        """
        #if not observations_main_cam:
        #    return

        # Assign random color to new tracks
        #for obs in observations_main_cam:
        #    if obs.id not in self.track_colors:
        #        self.track_colors[obs.id] = np.random.randint(0, 256, size=3)

        points = []
        colors = []
        #points = np.array([[obs.u, obs.v] for obs in observations_main_cam])
        #colors = np.array([
        #    self.track_colors[obs.id] for obs in observations_main_cam
        #])

        if image is None:
            return

        # Handle different image datatypes for compression
        if image.dtype == np.uint8:
            image_log = rr.Image(image).compress()
        else:
            # For other datatypes, don't compress to avoid issues
            image_log = rr.Image(image)

        rr.log(
            f"world/{camera_name}/observations",
            rr.Points2D(positions=points, colors=colors, radii=POINT_RADIUS),
            image_log
        )

    def _log_gravity(self, gravity: np.ndarray) -> None:
        """Log gravity vector to Rerun.
        
        Args:
            gravity: Gravity vector
        """
        rr.log(
            "world/camera_0/gravity",
            rr.Arrows3D(
                vectors=gravity,
                colors=[[255, 0, 0]],
                radii=GRAVITY_ARROW_SCALE
            )
        )

    def visualize_frame(
        self,
        frame_id: int,
        images: List[np.ndarray],
        map_3d,
        map_2d,
        clrs,
        #pose,
        translation,
        quaternion,
        yaw,
        observations_main_cam: List[List[Any]],
        trajectory: List[np.ndarray],
        timestamp: int,
        gravity: Optional[np.ndarray] = None
    ) -> None:
        """Visualize current frame state using Rerun.
        
        Args:
            frame_id: Current frame ID
            images: List of camera images
            pose: Current pose estimate
            observations_main_cam: List of observations for each camera
            trajectory: List of trajectory points
            timestamp: Current timestamp
            gravity: Optional gravity vector
        """
        rr.set_time_sequence("frame", frame_id)
        rr.log("world/trajectory", rr.LineStrips3D(trajectory), static=True)
        self._visualize_3d_point_cloud(map_3d, clrs)
        self._visualize_2d_map(map_2d)

        map_2d_height = 160
        map_2d_width = 160
        drone_uv = [-translation[0] * 10 + map_2d_width / 2, translation[2] * 10 + map_2d_height / 2]
        self._visualize_drone(drone_uv, yaw)

        self._log_rig_pose(quaternion, translation)
        
        for i in range(self.num_viz_cameras):
            self._log_observations(
                observations_main_cam[i], images[i], f"camera_{i}"
            )
            
        if gravity is not None:
            self._log_gravity(gravity)
            
        rr.log("world/timestamp", rr.TextLog(str(timestamp)))

    def _visualize_3d_point_cloud(self, points, clrs):
        rr.log('world/points', rr.Points3D(
            points, radii=0.01, colors=clrs or [255, 255, 255]
        ))

    def _visualize_2d_map(self, map_2d):
        if map_2d == None:
            return 
            
        obstacles = []
        explored = []
        for p in map_2d:
            if p[2] == -1000:
                obstacles.append([p[0],p[1]])
            elif p[2] == -1001:
                explored.append([p[0],p[1]])
            elif p[2] == -2000:
                rr.log('world/level/ltg', rr.Points2D(
                    p[0:2], radii=0.5, colors=[0, 0, 255], draw_order=45
                ))
            elif p[2] == -2001:
                rr.log('world/level/stg', rr.Points2D(
                    [p[1], p[0]], radii=0.5, colors=[255, 165, 0], draw_order=50
                ))
        rr.log('world/level/obstacles', rr.Points2D(
            obstacles, radii=0.5, colors=[255, 255, 255],  draw_order=30
        ))
        rr.log('world/level/explored', rr.Points2D(
            explored, radii=0.5, colors=[128, 128, 128],  draw_order=20
        ))

    def _visualize_drone(self, drone_uv, yaw_rad):
        """
        Visualize drone position and orientation
        
        Args:
            drone_uv: (u, v) position in pixels
            yaw_rad: yaw angle in radians
        """
        rr.log(f'world/level/z_drone_orientation', rr.Arrows2D(
            origins=[],
            vectors=[],
            colors=[255, 0, 0],  # Red arrow
            radii=0.3,
            draw_order=100
        ))
        
        # Draw arrow showing orientation
        arrow_length = 3  # pixels
        
        # Calculate arrow direction from yaw
        yaw_rad += np.pi / 2.0
        dx = arrow_length * np.cos(yaw_rad)
        dy = arrow_length * np.sin(yaw_rad)
        
        # Arrow origin and vector
        rr.log(f'world/level/z_drone_orientation', rr.Arrows2D(
            origins=[drone_uv[0], drone_uv[1]],
            vectors=[[dx, dy]],
            colors=[255, 0, 0],  # Red arrow
            radii=0.3,
            draw_order=100
        ))