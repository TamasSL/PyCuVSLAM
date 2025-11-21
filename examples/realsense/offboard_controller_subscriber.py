import grpc
import asyncio
import numpy as np
import cv2
import time
from typing import Optional

from mavsdk.offboard import VelocityBodyYawspeed, OffboardError, PositionNedYaw

# Import generated protobuf code
# Run: python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. sensor_stream.proto
import sensor_stream_pb2
import sensor_stream_pb2_grpc

import threading
import queue

import pyrealsense2 as rs
from publish_subscribe import Publisher

USE_POSITION_NED_COMMANDS = True


class OffboardControllerSubscriber:
    def __init__(self, publisher: Publisher, drone, event_loop):
        self.drone = drone
        self.queue = publisher.subscribe()
        self.event_loop = event_loop
        self._running = False
        self._thread = None
        self._offboard_running = False

        self.takeoff_altitude = 0.2

        self.target_north = 0
        self.target_east = 0
        self.target_down = -self.takeoff_altitude
        self.target_yaw = 0

        self.follow_target_position = False

        self.current_velocity = VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
        self._heartbeat_task = None
    
    def start(self):
        """Start processing in background thread"""
        self._running = True
        self._thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self._thread.start()
        print("Offboard controller started")


    def _run_async_loop(self):
        """Synchronous wrapper that runs async code"""
        future = asyncio.run_coroutine_threadsafe(
            self._process_loop(),
            self.event_loop
        )
        future.result()  # Wait for completion

    
    async def _process_loop(self):
        """Main processing loop"""
        while self._running:
            
            try:
                command = await asyncio.to_thread(self.queue.get, timeout=0.1)
                await self._execute_command(command)
            except queue.Empty:
                continue

    async def _execute_command(self, command):
        """Execute command received from the queue"""
        try:
            cmd_type = command.command
            target_x_ned = command.x
            target_y_ned = command.y
            diff_to_target_angle = command.z
            
            if cmd_type == sensor_stream_pb2.DroneCommand.ARM:
                await self._arm()
                
            elif cmd_type == sensor_stream_pb2.DroneCommand.TAKEOFF:
                await self._takeoff()
                
            elif cmd_type == sensor_stream_pb2.DroneCommand.LAND:
                await self._land()

            elif cmd_type == sensor_stream_pb2.DroneCommand.FORWARD:
                print(f"✅ Command executed: follow target position once")
                await self.move_to_position(target_x_ned, target_y_ned, diff_to_target_angle)

            elif cmd_type == sensor_stream_pb2.DroneCommand.LEFT:
                print(f"✅ Command executed: enable continously following target position")
                self.follow_target_position = True

            elif cmd_type == sensor_stream_pb2.DroneCommand.RIGHT:
                print(f"✅ Command executed: disable continously following target position")
                self.follow_target_position = False

            if self.follow_target_position:
                await self.move_to_position(target_x_ned, target_y_ned, diff_to_target_angle)
                
            print(f"current [{self.target_north}, {self.target_east}], target: [{target_x_ned}, {target_y_ned}, {diff_to_target_angle}]")
            
        except Exception as e:
            print(f"❌ Failed to execute command: {e}")

    async def _arm(self):
        print("Arming...")
        await self.drone.action.arm()
        
    async def _takeoff(self):
        print("Taking off...")
        await self.drone.action.set_takeoff_altitude(self.takeoff_altitude)
        await self.drone.action.takeoff()
        await asyncio.sleep(5)  # Wait for takeoff

        if USE_POSITION_NED_COMMANDS:
            # Get current position
            async for position in self.drone.telemetry.position_velocity_ned():
                self.target_north = position.position.north_m
                self.target_east = position.position.east_m
                self.target_down = position.position.down_m
                break
            
            # Get current yaw
            async for attitude in self.drone.telemetry.attitude_euler():
                self.target_yaw = attitude.yaw_deg
                break
            
            await self.drone.offboard.set_position_ned(
                PositionNedYaw(self.target_north, self.target_east, self.target_down, self.target_yaw)
            )
            await asyncio.sleep(2)  # Wait for takeoff
        else:
            # Must send setpoint before starting
            await self.drone.offboard.set_velocity_body(self.current_velocity)
        
        try:
            await self.drone.offboard.start()
            self._offboard_running = True

            for i in range(36):
                self.target_yaw += 10
                await asyncio.sleep(0.5)
            
            # Start heartbeat task
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            print("✅ Offboard mode active")
        except OffboardError as e:
            print(f"❌ Failed to start offboard: {e}")
            raise

    async def _land(self):
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        await self.drone.offboard.stop()
        print("✅ Offboard mode stopped")
        self._offboard_running = False

        print("Landing...")
        await self.drone.action.land()

    async def _left(self):
        if USE_POSITION_NED_COMMANDS:
            await self.move_left_position()
        else:
            # rotate left 45 degrees
            await self.set_yaw(-15)
            await asyncio.sleep(3)

            await self.hover()

    async def _right(self):
        if USE_POSITION_NED_COMMANDS:
            await self.move_right_position()
        else:
            # rotate right 45 degrees
            await self.set_yaw(15)
            await asyncio.sleep(3)

            await self.hover()
    
    async def _heartbeat_loop(self):
        """Continuously send setpoints to keep offboard mode alive"""
        """Send position setpoints"""
        try:
            while self._offboard_running:

                if USE_POSITION_NED_COMMANDS:
                    await self.drone.offboard.set_position_ned(
                        PositionNedYaw(self.target_north, self.target_east, self.target_down, self.target_yaw)
                    )
                
                else:
                    # Get current altitude
                    async for position in self.drone.telemetry.position_velocity_ned():
                        current_altitude = position.position.down_m
                        break
                    
                    # Calculate altitude error
                    altitude_error = self.takeoff_altitude - current_altitude
                    
                    # Add altitude correction to Z velocity
                    vz_correction = -altitude_error * 0.5  # P controller
                    corrected_velocity = VelocityBodyYawspeed(
                        self.current_velocity.forward_m_s,
                        self.current_velocity.right_m_s,
                        vz_correction,
                        self.current_velocity.yawspeed_deg_s
                    )

                    await self.drone.offboard.set_velocity_body(corrected_velocity)

                await asyncio.sleep(0.05)  # 20 Hz
        except asyncio.CancelledError:
            print("Heartbeat loop cancelled")

    async def hover(self):
        """Hold current position (zero velocity)"""
        await self.set_velocity(0.0, 0.0, 0.0, 0.0)
    
    async def set_velocity(self, vx, vy, vz, yaw_deg):
        """
        Set velocity command
        vx: forward velocity (m/s), positive = forward
        vy: right velocity (m/s), positive = right
        vz: down velocity (m/s), positive = down, negative = up
        yaw_deg: yaw angular rate (degrees / s)
        """
        self.current_velocity = VelocityBodyYawspeed(vx, vy, vz, yaw_deg)
        # Heartbeat will send this automatically
    
    async def move_forward(self, speed=0.5):
        """Move forward at specified speed"""
        await self.set_velocity(speed, 0.0, 0.0, 0.0)

    async def move_to_position(self, target_x_ned, target_y_ned, diff_to_target_angle):
        """Move to specified position"""

        if abs(diff_to_target_angle) < 30:
            # Update target position if drone is facing closely towards the target position
            self.target_north = target_x_ned
            self.target_east = target_y_ned
        
        self.target_yaw += diff_to_target_angle
    
        # Heartbeat will send updated position
        await asyncio.sleep(2)  
    
    async def move_backward(self, speed=0.5):
        """Move backward at specified speed"""
        await self.set_velocity(-speed, 0.0, 0.0, 0.0)
    
    async def move_right(self, speed=0.5):
        """Move right at specified speed"""
        await self.set_velocity(0.0, speed, 0.0, 0.0)
    
    async def move_left(self, speed=0.5):
        """Move left at specified speed"""
        await self.set_velocity(0.0, -speed, 0.0, 0.0)

    async def move_left_position(self):
        """Rotate left 45 degrees"""
        self.target_yaw -= 45
        await asyncio.sleep(2)

    async def move_right_position(self):
        """Rotate right 45 degrees"""
        self.target_yaw += 45
        await asyncio.sleep(2)
    
    async def ascend(self, speed=0.5):
        """Ascend at specified speed"""
        await self.set_velocity(0.0, 0.0, -speed, 0.0)  # Negative = up!
    
    async def descend(self, speed=0.5):
        """Descend at specified speed"""
        await self.set_velocity(0.0, 0.0, speed, 0.0)
    
    async def set_yaw(self, yaw_deg):
        """Set yaw angle while hovering"""
        await self.set_velocity(0.0, 0.0, 0.0, yaw_deg)

    
    