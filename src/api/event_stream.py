"""
Async event stream simulation for decoupled logging.
Simulates a Kafka Producer/Consumer architecture using in-memory queues and background workers
to eliminate SQLite database blocking during high-throughput inference serving.
"""
import queue
import threading
import time
from typing import Dict, Any, List
from src.api.logger import InferenceLogger

class EventStreamProducer:
    """
    Decoupled async event producer. Puts inference events onto an in-memory queue.
    In an enterprise setting, this would publish to Apache Kafka.
    """
    def __init__(self, maxsize: int = 10000):
        self.q = queue.Queue(maxsize=maxsize)
        
    def publish_prediction_event(self, request_id: str, features: Dict[str, float], prediction: float):
        event = {
            "type": "prediction",
            "request_id": request_id,
            "features": features,
            "prediction": prediction,
            "timestamp": time.time()
        }
        try:
            self.q.put_nowait(event)
        except queue.Full:
            # Handle backpressure
            pass
            
    def publish_feedback_event(self, request_id: str, actual_value: float):
        event = {
            "type": "feedback",
            "request_id": request_id,
            "actual_value": actual_value,
            "timestamp": time.time()
        }
        try:
            self.q.put_nowait(event)
        except queue.Full:
            pass


class EventStreamConsumer:
    """
    Background worker that consumes the event stream and writes to the DB in batches.
    """
    def __init__(self, producer: EventStreamProducer, logger: InferenceLogger, batch_size: int = 50, flush_interval: float = 1.0):
        self.producer = producer
        self.logger = logger
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._running = False
        self._worker_thread = None
        
    def start(self):
        if not self._running:
            self._running = True
            self._worker_thread = threading.Thread(target=self._consume_loop, daemon=True)
            self._worker_thread.start()
            
    def stop(self):
        self._running = False
        if self._worker_thread:
            self._worker_thread.join()
            
    def _consume_loop(self):
        batch = []
        last_flush = time.time()
        
        while self._running:
            try:
                # Wait for an item, but don't block indefinitely
                event = self.producer.q.get(timeout=0.1)
                batch.append(event)
                self.producer.q.task_done()
            except queue.Empty:
                pass
                
            now = time.time()
            if len(batch) >= self.batch_size or (batch and now - last_flush >= self.flush_interval):
                self._flush_batch(batch)
                batch = []
                last_flush = now
                
        # Final flush on shutdown
        if batch:
            self._flush_batch(batch)
            
    def _flush_batch(self, batch: List[Dict[str, Any]]):
        # We need to update the InferenceLogger to support batch inserts
        self.logger.log_batch(batch)

# Global instances for the app
producer = EventStreamProducer()
