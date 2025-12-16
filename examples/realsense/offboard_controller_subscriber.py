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
            target_z_ned = command.z
            target_angle = command.velocity  # clean-up var name

            print(f"current [{self.target_north}, {self.target_east}, {self.target_yaw}], target: [{target_x_ned}, {target_y_ned}, {target_angle}]")
            
            if cmd_type == sensor_stream_pb2.DroneCommand.ARM:
                await self._arm()
                
            elif cmd_type == sensor_stream_pb2.DroneCommand.TAKEOFF:
                await self._takeoff()
                
            elif cmd_type == sensor_stream_pb2.DroneCommand.LAND:
                await self._land()

            elif cmd_type == sensor_stream_pb2.DroneCommand.FOLLOW_ONCE:
                print(f"✅ Command executed: follow target position once")
                await self._move_to_position(target_x_ned, target_y_ned, target_z_ned, target_angle)

            elif cmd_type == sensor_stream_pb2.DroneCommand.FOLLOW:
                print(f"✅ Command executed: enable continously following target position")
                self.follow_target_position = True

            elif cmd_type == sensor_stream_pb2.DroneCommand.UNFOLLOW:
                print(f"✅ Command executed: disable continously following target position")
                self.follow_target_position = False

            if self.follow_target_position:
                await self._move_to_position(target_x_ned, target_y_ned, target_z_ned, target_angle)
                
            
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
        
        try:
            await self.drone.offboard.start()
            self._offboard_running = True

            # initialization sequence for initial discovery
            # move forward 0.5m, turn 180 degrees
            # move back 0.5m, turn another 180 degrees
            # this should allow discovering the area around the drone
            # without leaving a black hole at the drone's starting position
            for i in range(5):
                self.target_north += 0.1
                await asyncio.sleep(0.5)

            for i in range(18):
                self.target_yaw += 10
                await asyncio.sleep(0.5)

            for i in range(5):
                self.target_north -= 0.1
                await asyncio.sleep(0.5)

            for i in range(18):
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

    
    async def _heartbeat_loop(self):
        """Continuously send setpoints to keep offboard mode alive"""
        try:
            while self._offboard_running:
                await self.drone.offboard.set_position_ned(
                    PositionNedYaw(self.target_north, self.target_east, self.target_down, self.target_yaw)
                )
                await asyncio.sleep(0.05)  # 20 Hz
        except asyncio.CancelledError:
            print("Heartbeat loop cancelled")
    

    async def _move_to_position(self, target_x_ned, target_y_ned, target_z_ned, target_angle):
        """Move to specified target position.
        The movement is done this way:
        - first, turn towards the target position, until the relative angle is [-20, 20] degrees.
        - then, move towards the target position"""

        # Update target position if drone is facing closely towards the target position

        diff_yaw = abs(self.target_yaw - target_angle)
        if diff_yaw < 15:
            self.target_north = target_x_ned
            self.target_east = target_y_ned
            self.target_down = target_z_ned
            self.target_yaw = target_angle
            await asyncio.sleep(1)  
        else:
            wait_for = diff_yaw / 20
            await asyncio.sleep(wait_for)

        


    
    