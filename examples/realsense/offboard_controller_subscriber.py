import grpc
import asyncio
import numpy as np
import cv2
import time
from typing import Optional

from mavsdk.offboard import VelocityBodyYawspeed, OffboardError

# Import generated protobuf code
# Run: python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. sensor_stream.proto
import sensor_stream_pb2
import sensor_stream_pb2_grpc

import threading
import queue

import pyrealsense2 as rs
from publish_subscribe import Publisher


class OffboardControllerSubscriber:
    def __init__(self, publisher: Publisher, drone, event_loop):
        self.drone = drone
        self.queue = publisher.subscribe()
        self.event_loop = event_loop
        self._running = False
        self._thread = None
        self._offboard_running = False

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
            
            if cmd_type == sensor_stream_pb2.DroneCommand.ARM:
                await self._arm()
                
            elif cmd_type == sensor_stream_pb2.DroneCommand.TAKEOFF:
                await self._takeoff()
                
            elif cmd_type == sensor_stream_pb2.DroneCommand.LAND:
                await self._land()

            elif cmd_type == sensor_stream_pb2.DroneCommand.FORWARD:
                await self._forward()

            elif cmd_type == sensor_stream_pb2.DroneCommand.LEFT:
                await self._left()

            elif cmd_type == sensor_stream_pb2.DroneCommand.RIGHT:
                await self._right()
                
            print(f"✅ Command executed: {cmd_type}")
            
        except Exception as e:
            print(f"❌ Failed to execute command: {e}")

    async def _arm(self):
        print("Arming...")
        await self.drone.action.arm()
        
    async def _takeoff(self):
        print("Taking off...")
        await self.drone.action.set_takeoff_altitude(0.2)
        await self.drone.action.takeoff()
        await asyncio.sleep(5)  # Wait for takeoff

        # 2. HOLD POSITION (explicit position hold mode)
        # print("Holding position...")
        # await self.drone.action.hold()
        # await asyncio.sleep(2)

        # Must send setpoint before starting
        await self.drone.offboard.set_velocity_body(self.current_velocity)
        
        try:
            await self.drone.offboard.start()
            self._offboard_running = True
            
            # Start heartbeat task
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            print("✅ Offboard mode active")
        except OffboardError as e:
            print(f"❌ Failed to start offboard: {e}")
            raise

        await self.hover()
        await asyncio.sleep(2)

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

    async def _forward(self):
        # move forward 10cm
        await self.move_forward(0.1)
        await asyncio.sleep(1)

        await self.hover()

    async def _left(self):
        # rotate left 45 degrees
        await self.set_yaw(-15)
        await asyncio.sleep(3)

        await self.hover()

    async def _right(self):
        # rotate right 45 degrees
        await self.set_yaw(15)
        await asyncio.sleep(3)

        await self.hover()
    
    async def _heartbeat_loop(self):
        """Continuously send setpoints to keep offboard mode alive"""
        try:
            while self._offboard_running:
                await self.drone.offboard.set_velocity_body(self.current_velocity)
                await asyncio.sleep(0.1)  # 10 Hz

                # Debug: Check flight mode
                async for flight_mode in self.drone.telemetry.flight_mode():
                    print(f"Current flight mode: {flight_mode}")
                    break
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
    
    async def move_forward(self, speed=0.5):   #### !!!!!!!!!!!!!!!!!!!!!!! Limit the max translation speed of the FC
        """Move forward at specified speed"""
        await self.set_velocity(speed, 0.0, 0.0, 0.0)
    
    async def move_backward(self, speed=0.5):
        """Move backward at specified speed"""
        await self.set_velocity(-speed, 0.0, 0.0, 0.0)
    
    async def move_right(self, speed=0.5):
        """Move right at specified speed"""
        await self.set_velocity(0.0, speed, 0.0, 0.0)
    
    async def move_left(self, speed=0.5):
        """Move left at specified speed"""
        await self.set_velocity(0.0, -speed, 0.0, 0.0)
    
    async def ascend(self, speed=0.5):
        """Ascend at specified speed"""
        await self.set_velocity(0.0, 0.0, -speed, 0.0)  # Negative = up!
    
    async def descend(self, speed=0.5):
        """Descend at specified speed"""
        await self.set_velocity(0.0, 0.0, speed, 0.0)
    
    async def set_yaw(self, yaw_deg):
        """Set yaw angle while hovering"""
        await self.set_velocity(0.0, 0.0, 0.0, yaw_deg)

    
    