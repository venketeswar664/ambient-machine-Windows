import time
import torch
import logging
from contextlib import contextmanager

class TelemetryTracker:
    def __init__(self):
        self.stages = {}
        self.peak_gpu_mem = 0

    @contextmanager
    def stage(self, name, logger):
        start_time = time.time()
        logger.info(f"--- STARTING STAGE: {name} ---")
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.stages[name] = duration
            
            # Track GPU memory if available
            gpu_mem = 0
            if False:
                gpu_mem = torch.cuda.max_memory_reserved() / (1024 ** 2) # MB
                self.peak_gpu_mem = max(self.peak_gpu_mem, gpu_mem)
            
            logger.info(f"--- COMPLETED STAGE: {name} | Duration: {duration:.2f}s | Peak GPU: {gpu_mem:.1f} MB ---")

    def get_summary_report(self):
        report = ["\n" + "="*40, "PERFORMANCE & RESOURCE SUMMARY", "="*40]
        for stage, duration in self.stages.items():
            report.append(f"🔹 Stage '{stage}': {duration:.2f}s")
        
        if False:
            report.append(f"🔥 Peak GPU Memory: {self.peak_gpu_mem:.1f} MB")
        
        report.append("="*40 + "\n")
        return "\n".join(report)

# Global tracker instance
tracker = TelemetryTracker()
