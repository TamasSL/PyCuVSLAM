import threading
import queue
import time
from typing import Any, Callable, List
from dataclasses import dataclass


class Publisher:
    """
    Non-blocking publisher that never waits for subscribers
    Uses separate queues for each subscriber to prevent slow consumers from blocking
    """
    
    def __init__(self, maxsize: int = 100):
        """
        Args:
            maxsize: Maximum queue size per subscriber (older items dropped if full)
        """
        self.subscribers: List[queue.Queue] = []
        self.maxsize = maxsize
        self._lock = threading.Lock()
        self._published_count = 0
    
    def subscribe(self) -> queue.Queue:
        """Create a new subscriber queue"""
        with self._lock:
            sub_queue = queue.Queue(maxsize=self.maxsize)
            self.subscribers.append(sub_queue)
            print(f"New subscriber added (total: {len(self.subscribers)})")
            return sub_queue
    
    def publish(self, data: Any):
        """
        Publish data to all subscribers
        Non-blocking - drops old data if subscriber queue is full
        """
        with self._lock:
            for sub_queue in self.subscribers:
                try:
                    # Non-blocking put - drop oldest if full
                    sub_queue.put_nowait(data)
                except queue.Full:
                    # Remove oldest item and add new one
                    try:
                        sub_queue.get_nowait()  # Drop oldest
                        sub_queue.put_nowait(data)  # Add new
                    except:
                        pass  # Queue state changed, skip
            
            self._published_count += 1
    
    def get_stats(self):
        """Get publisher statistics"""
        with self._lock:
            return {
                'published': self._published_count,
                'subscribers': len(self.subscribers),
                'queue_sizes': [q.qsize() for q in self.subscribers]
            }


class Subscriber:
    """
    Subscriber that processes data in separate thread
    """
    
    def __init__(self, publisher: Publisher, callback: Callable[[Any], None], name: str = "Subscriber"):
        """
        Args:
            publisher: Publisher to subscribe to
            callback: Function to call with received data
            name: Name for this subscriber (for logging)
        """
        self.name = name
        self.callback = callback
        self.queue = publisher.subscribe()
        self._running = False
        self._thread = None
        self._processed_count = 0
    
    def start(self):
        """Start processing in background thread"""
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        print(f"{self.name} started")
    
    def stop(self):
        """Stop processing"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        print(f"{self.name} stopped (processed {self._processed_count} items)")
    
    def _process_loop(self):
        """Main processing loop (runs in separate thread)"""
        while self._running:
            try:
                # Wait for data with timeout to allow clean shutdown
                data = self.queue.get(timeout=0.1)
                
                # Process data
                self.callback(data)
                self._processed_count += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"{self.name} error: {e}")


# Example Usage
def example_slow_subscriber():
    """Example with slow subscriber that doesn't block publisher"""
    
    publisher = Publisher(maxsize=10)
    
    # Fast subscriber
    def fast_callback(data):
        print(f"Fast: {data}")
    
    # Slow subscriber (simulates heavy processing)
    def slow_callback(data):
        time.sleep(0.1)  # Simulate slow processing
        print(f"Slow: {data}")
    
    # Create subscribers
    fast_sub = Subscriber(publisher, fast_callback, "FastSubscriber")
    slow_sub = Subscriber(publisher, slow_callback, "SlowSubscriber")
    
    # Start subscribers
    fast_sub.start()
    slow_sub.start()
    
    # Publisher runs at full speed (not blocked by slow subscriber)
    print("\nPublishing data at 100 Hz...")
    for i in range(50):
        publisher.publish(f"message_{i}")
        time.sleep(0.01)  # 100 Hz
        
        if i % 10 == 0:
            stats = publisher.get_stats()
            print(f"\nStats: {stats}")
    
    time.sleep(1)  # Let subscribers catch up
    
    # Cleanup
    fast_sub.stop()
    slow_sub.stop()
    
    print(f"\nFinal stats: {publisher.get_stats()}")


