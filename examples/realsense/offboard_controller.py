import asyncio
from mavsdk import System
from mavsdk.offboard import VelocityNedYaw, OffboardError

class OffboardController:
    def __init__(self, drone):
        self.drone = drone
        self.running = False
        self.current_velocity = VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
        self._heartbeat_task = None
    
    async def start(self):
        """Start offboard mode with heartbeat"""
        print("Starting offboard mode...")
        
        # Must send setpoint before starting
        await self.drone.offboard.set_velocity_ned(self.current_velocity)
        
        try:
            await self.drone.offboard.start()
            self.running = True
            
            # Start heartbeat task
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            print("✅ Offboard mode active")
        except OffboardError as e:
            print(f"❌ Failed to start offboard: {e}")
            raise
    
    async def stop(self):
        """Stop offboard mode and heartbeat"""
        print("Stopping offboard mode...")
        self.running = False
        
        # Cancel heartbeat task
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        await self.drone.offboard.stop()
        print("✅ Offboard mode stopped")
    
    async def _heartbeat_loop(self):
        """Continuously send setpoints to keep offboard mode alive"""
        try:
            while self.running:
                await self.drone.offboard.set_velocity_ned(self.current_velocity)
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
        yaw_deg: yaw angle (degrees)
        """
        self.current_velocity = VelocityNedYaw(vx, vy, vz, yaw_deg)
        # Heartbeat will send this automatically
    
    async def move_forward(self, speed=0.5):
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

async def main():
    # Connect
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyTHS1:921600")
    
    print("Waiting for drone connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"✅ Drone connected!")
            break
    
    # Create controller
    controller = OffboardController(drone)
    
    print("\n" + "="*60)
    print("FLIGHT INSTRUCTIONS")
    print("="*60)
    print("1. Arm drone in Position or Altitude mode")
    print("2. Takeoff to 1-2 meters altitude")
    print("3. Hover and stabilize for a few seconds")
    print("4. Press Enter here to switch to Offboard mode")
    print("5. RC can override at any time by switching flight mode")
    print("="*60)
    input("\nPress Enter when ready to start offboard mode...")
    
    # Start offboard mode (with automatic heartbeat)
    await controller.start()
    
    try:
        # Example mission
        print("\n🚁 Starting autonomous mission...\n")
        
        # Mission Step 1: Hover
        print("Step 1: Hovering for 3 seconds...")
        await controller.hover()
        await asyncio.sleep(3)
        
        # Mission Step 2: Move forward
        print("Step 2: Moving forward 0.3 m/s for 2 seconds...")
        await controller.move_forward(speed=0.3)
        await asyncio.sleep(2)
        
        # Mission Step 3: Hover
        print("Step 3: Hovering for 2 seconds...")
        await controller.hover()
        await asyncio.sleep(2)
        
        # Mission Step 4: Move right
        print("Step 4: Moving right 0.3 m/s for 2 seconds...")
        await controller.move_right(speed=0.3)
        await asyncio.sleep(2)
        
        # Mission Step 5: Hover
        print("Step 5: Hovering for 2 seconds...")
        await controller.hover()
        await asyncio.sleep(2)
        
        # Mission Step 6: Move backward
        print("Step 6: Moving backward 0.3 m/s for 2 seconds...")
        await controller.move_backward(speed=0.3)
        await asyncio.sleep(2)
        
        # Mission Step 7: Move left
        print("Step 7: Moving left 0.3 m/s for 2 seconds...")
        await controller.move_left(speed=0.3)
        await asyncio.sleep(2)
        
        # Mission Step 8: Rotate in place
        print("Step 8: Rotating to 90 degrees...")
        await controller.set_yaw(90.0)
        await asyncio.sleep(3)
        
        print("Step 9: Rotating back to 0 degrees...")
        await controller.set_yaw(0.0)
        await asyncio.sleep(3)
        
        # Mission Step 9: Ascend
        print("Step 10: Ascending 0.2 m/s for 1 second...")
        await controller.ascend(speed=0.2)
        await asyncio.sleep(1)
        
        # Mission Step 10: Descend
        print("Step 11: Descending 0.2 m/s for 1 second...")
        await controller.descend(speed=0.2)
        await asyncio.sleep(1)
        
        # Final hover
        print("Step 12: Final hover for 3 seconds...")
        await controller.hover()
        await asyncio.sleep(3)
        
        print("\n✅ Mission complete!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Mission interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during mission: {e}")
    finally:
        # Stop offboard mode (automatically stops heartbeat)
        await controller.stop()
        
        print("\n" + "="*60)
        print("Offboard mode stopped")
        print("Switch to Position mode via RC and land safely")
        print("="*60)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")