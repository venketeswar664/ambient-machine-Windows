

"""
Embedded Circus process supervisor that reads services.yaml and manages service watchers.
"""
import logging
import os
import sys
from pathlib import Path

from circus.arbiter import Arbiter
from circus.watcher import Watcher

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.services.service_manager import build_watchers

LOG = logging.getLogger("circus_embed")

class EmbeddedCircus:
    """Manages an embedded Circus arbiter for running services."""
    
    def __init__(self, 
                 endpoint: str = "tcp://127.0.0.1:5565",
                 pubsub_endpoint: str = "tcp://127.0.0.1:5566",
                 stats_endpoint: str = "tcp://127.0.0.1:5567",
                 check_delay: float = 5.0,
                 enable_stats: bool = False
                 ):
        """
        Initialize the embedded Circus manager.
        
        Args:
            endpoint: Control endpoint for circusctl
            pubsub_endpoint: Pub/sub endpoint for events
            stats_endpoint: Stats endpoint (if stats enabled)
            check_delay: Delay between watcher checks
            enable_stats: Whether to enable statsd stats collection
        """
        self.endpoint = endpoint
        self.pubsub_endpoint = pubsub_endpoint
        self.stats_endpoint = stats_endpoint
        self.check_delay = check_delay
        self.enable_stats = enable_stats
        self.arbiter = None
        
        
    def start(self , log_dir=None):
        """Start the Circus arbiter with watchers from services.yaml."""
        LOG.info("Starting embedded Circus process supervisor...")
        
        try:
            # Build Watcher objects for services with attached_to_main: true
            watcher_objects = build_watchers(attached_to_main=True , log_dir=log_dir)
            
            if not watcher_objects:
                LOG.warning("No watchers configured. Circus will start but manage no services.")
            else:
                LOG.info(f"Configured {len(watcher_objects)} watcher(s):")
                for w in watcher_objects:
                    LOG.info(f"  - {w.name}")
            
            # Create arbiter directly with Watcher objects
            # The Arbiter constructor accepts Watcher objects directly
            arbiter_config = {
                'watchers': watcher_objects,
                'endpoint': self.endpoint,
                'pubsub_endpoint': self.pubsub_endpoint,
                'check_delay': self.check_delay,
                'statsd': False,  # Explicitly disable stats
            }
            
            # Only add stats config if explicitly enabled
            if self.enable_stats:
                arbiter_config['stats_endpoint'] = self.stats_endpoint
                arbiter_config['statsd'] = True
                LOG.info(f"Stats endpoint: {self.stats_endpoint}")
            else:
                LOG.info("Stats collection disabled")
            
            self.arbiter = Arbiter(**arbiter_config)
            
            LOG.info(f"Circus control endpoint: {self.endpoint}")
            LOG.info(f"Circus pubsub endpoint: {self.pubsub_endpoint}")
            LOG.info("Starting Circus arbiter (this will block in background thread)...")
            
            # Start the arbiter - this spawns a thread
            self.arbiter.start()
            
            LOG.info("✓ Circus arbiter started successfully")
            LOG.info(f"  Check status with: circusctl --endpoint {self.endpoint} status")
            
        except Exception as e:
            LOG.exception(f"Failed to start Circus arbiter: {e}")
            raise
    
    def stop(self):
        """Stop the Circus arbiter gracefully."""
        if self.arbiter is None:
            LOG.warning("Arbiter not running, nothing to stop")
            return
        
        LOG.info("Stopping Circus arbiter...")
        try:
            self.arbiter.stop()
            LOG.info("✓ Circus arbiter stopped")
        except Exception as e:
            LOG.exception(f"Error stopping Circus arbiter: {e}")
            raise
        finally:
            self.arbiter = None


if __name__ == "__main__":
    # Test standalone mode
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    
    circus = EmbeddedCircus()
    try:
        circus.start()
        LOG.info("Circus running. Press Ctrl+C to stop...")
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        LOG.info("Received interrupt signal")
    finally:
        circus.stop()