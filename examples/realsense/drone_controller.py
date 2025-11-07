import asyncio

from mavsdk import System
from offboard_controller import OffboardController
from vision_controller import VisionController

class DroneController:
    def __init__(self):
        self.drone = System()
        self.running = False
        self._vision_task = None
        self._offboard_task = None
    
    async def start(self):
        """Start all background tasks"""
        await self.drone.connect(system_address="serial:///dev/ttyTHS1:921600")
        
        self.running = True
        
        # Start vision streaming
        vision = VisionController(self.drone)
        self._vision_task = asyncio.create_task(vision.run())
        
        print("✅ Controller started")
    
    async def stop(self):
        """Stop all background tasks"""
        self.running = False
        
        if self._vision_task:
            self._vision_task.cancel()
            try:
                await self._vision_task
            except asyncio.CancelledError:
                pass
        
        print("✅ Controller stopped")

    
    async def run_mission(self):
        """Run offboard mission (vision runs in background)"""
        offboard = OffboardController(self.drone)
        await offboard.start()
        
        try:
            await offboard.hover()
            await asyncio.sleep(3)
            
            await offboard.move_forward(0.3)
            await asyncio.sleep(2)
            
            await offboard.hover()
            await asyncio.sleep(3)
        finally:
            await offboard.stop()

async def main():
    controller = DroneController()
    
    try:
        await controller.start()

        input("\nPress Enter when ready to start offboard mode...")
        
        # Run mission (vision streaming happens automatically)
        await controller.run_mission()
        
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        print("finally")
        #await controller.stop()

asyncio.run(main())
