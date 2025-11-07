from pymavlink import mavutil
import numpy as np
import time

# Connect to PX4
mavlink = mavutil.mavlink_connection(
    'udp:192.168.0.10:14540',  # or '/dev/ttyTHS1' for serial
    source_system=1,
    source_component=191
)

def transform_to_ned(slam_pose):
    """Transform cuVSLAM pose to PX4 NED frame"""
    # Adjust this based on your actual cuVSLAM frame convention
    x_ned = slam_pose['x']
    y_ned = -slam_pose['y']  
    z_ned = -slam_pose['z']
    
    # Transform orientation similarly
    # (May need quaternion transformation)
    
    return x_ned, y_ned, z_ned

def send_vision_position(x, y, z, roll=0, pitch=0, yaw=0):
    """Send vision position estimate to PX4"""
    timestamp_us = int(time.time() * 1e6)
    
    mavlink.mav.vision_position_estimate_send(
        timestamp_us,
        x, y, z,
        roll, pitch, yaw
    )

# Main loop - run at 30-50 Hz
while True:
    # Get pose from cuVSLAM
    slam_pose = get_cuvslam_pose()  # Your cuVSLAM interface
    
    # Transform to NED
    x, y, z = transform_to_ned(slam_pose)
    
    # Send to PX4
    send_vision_position(x, y, z)
    
    time.sleep(0.02)  # 50 Hz