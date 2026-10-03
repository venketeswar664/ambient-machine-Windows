from prometheus_client import start_http_server, Counter, Gauge, Histogram
import psutil
import time
import os
import subprocess
import logging
from src.monitoring_stack.mongodb_logger import initialize_logger

os.environ["PATH"] += os.pathsep + "/usr/bin"  # Add /usr/bin to PATH if it's missing

from datetime import datetime

# Function to create dynamic log directory based on the date and time
def create_log_directory():
    # Get the current date and time to create a unique subfolder
    date_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f"monitoring_stack/logs/{date_time}"

    # Create the directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    return log_dir

# Create the log directory dynamically
# log_dir = create_log_directory()
monitoring_logger = initialize_logger(category="system-status")

# Create metrics
REQUEST_COUNT = Counter('app_request_count', 'Total app requests')
REQUEST_LATENCY = Histogram('app_request_latency_seconds', 'Request latency')

# System metrics
CPU_PERCENT = Gauge('system_cpu_percent', 'CPU usage percentage')
CPU_COUNT = Gauge('system_cpu_count', 'Number of CPUs')
CPU_FREQ = Gauge('system_cpu_freq_mhz', 'CPU frequency in MHz')

MEM_TOTAL = Gauge('system_memory_total_bytes', 'Total memory in bytes')
MEM_AVAILABLE = Gauge('system_memory_available_bytes', 'Available memory in bytes')
MEM_USED = Gauge('system_memory_used_bytes', 'Used memory in bytes')
MEM_PERCENT = Gauge('system_memory_used_percent', 'Memory usage percentage')

DISK_TOTAL = Gauge('system_disk_total_bytes', 'Total disk space in bytes')
DISK_USED = Gauge('system_disk_used_bytes', 'Used disk space in bytes')
DISK_FREE = Gauge('system_disk_free_bytes', 'Free disk space in bytes')
DISK_PERCENT = Gauge('system_disk_used_percent', 'Disk usage percentage')

NET_SENT = Gauge('system_network_bytes_sent', 'Bytes sent over network')
NET_RECV = Gauge('system_network_bytes_recv', 'Bytes received over network')

# Process metrics
PROCESS_CPU_PERCENT = Gauge('process_cpu_percent', 'Process CPU usage percentage')
PROCESS_MEM_RSS = Gauge('process_memory_rss_bytes', 'Process RSS memory usage in bytes')
PROCESS_MEM_VMS = Gauge('process_memory_vms_bytes', 'Process VMS memory usage in bytes')
PROCESS_MEM_PERCENT = Gauge('process_memory_percent', 'Process memory usage percentage')
PROCESS_THREADS = Gauge('process_threads_count', 'Number of process threads')
PROCESS_FDS = Gauge('process_fds_count', 'Number of process file descriptors')

# GPU metrics (using nvidia-smi)
GPU_UTIL = Gauge('gpu_utilization_percent', 'GPU utilization percentage')
GPU_MEM_UTIL = Gauge('gpu_memory_utilization_percent', 'GPU memory utilization percentage')
GPU_MEM_TOTAL = Gauge('gpu_memory_total_bytes', 'Total GPU memory in bytes')
GPU_MEM_USED = Gauge('gpu_memory_used_bytes', 'Used GPU memory in bytes')
GPU_TEMP = Gauge('gpu_temperature_celsius', 'GPU temperature in Celsius')
GPU_POWER = Gauge('gpu_power_watts', 'GPU power usage in watts')

# Docker metrics
DOCKER_CONTAINERS = Gauge('docker_containers_count', 'Number of Docker containers')
DOCKER_RUNNING = Gauge('docker_containers_running', 'Number of running Docker containers')
DOCKER_PAUSED = Gauge('docker_containers_paused', 'Number of paused Docker containers')
DOCKER_STOPPED = Gauge('docker_containers_stopped', 'Number of stopped Docker containers')

def collect_system_metrics():
    """Collect system-wide metrics"""
    try:
        # CPU metrics
        CPU_PERCENT.set(psutil.cpu_percent(interval=1))
        CPU_COUNT.set(psutil.cpu_count())
        cpu_freq = psutil.cpu_freq()
        if cpu_freq:
            CPU_FREQ.set(cpu_freq.current)
        
        # Memory metrics
        mem = psutil.virtual_memory()
        MEM_TOTAL.set(mem.total)
        MEM_AVAILABLE.set(mem.available)
        MEM_USED.set(mem.used)
        MEM_PERCENT.set(mem.percent)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        DISK_TOTAL.set(disk.total)
        DISK_USED.set(disk.used)
        DISK_FREE.set(disk.free)
        DISK_PERCENT.set(disk.percent)
        
        # Network metrics
        net_io = psutil.net_io_counters()
        NET_SENT.set(net_io.bytes_sent)
        NET_RECV.set(net_io.bytes_recv)
        
        monitoring_logger.info("System metrics collected successfully")
    except Exception as e:
        monitoring_logger.error(f"Error collecting system metrics: {e}")

def collect_process_metrics():
    """Collect metrics for the current process"""
    try:
        process = psutil.Process(os.getpid())
        PROCESS_CPU_PERCENT.set(process.cpu_percent(interval=1))
        mem_info = process.memory_info()
        PROCESS_MEM_RSS.set(mem_info.rss)
        PROCESS_MEM_VMS.set(mem_info.vms)
        PROCESS_MEM_PERCENT.set(process.memory_percent())
        PROCESS_THREADS.set(process.num_threads())
        if os.name != 'nt':  # Not available on Windows
            PROCESS_FDS.set(process.num_fds())
        
        monitoring_logger.info("Process metrics collected successfully")
    except Exception as e:
        monitoring_logger.error(f"Error collecting process metrics: {e}")

def collect_gpu_metrics():
    """Collect NVIDIA GPU metrics using nvidia-smi"""
    try:
        # Check if nvidia-smi is available
        result = subprocess.run(['which', 'nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            monitoring_logger.warning("nvidia-smi not found, skipping GPU metrics")
            return
        
        # Get GPU utilization
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,memory.total,memory.used,temperature.gpu,power.draw', '--format=csv,noheader,nounits'], 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode == 0:
            gpu_stats = result.stdout.strip().split(', ')
            if len(gpu_stats) >= 6:
                GPU_UTIL.set(float(gpu_stats[0]))
                GPU_MEM_UTIL.set(float(gpu_stats[1]))
                # Convert MiB to bytes
                GPU_MEM_TOTAL.set(float(gpu_stats[2]) * 1024 * 1024)
                GPU_MEM_USED.set(float(gpu_stats[3]) * 1024 * 1024)
                GPU_TEMP.set(float(gpu_stats[4]))
                GPU_POWER.set(float(gpu_stats[5]))
                monitoring_logger.info("GPU metrics collected successfully")
            else:
                monitoring_logger.warning(f"Unexpected nvidia-smi output format: {result.stdout}")
        else:
            monitoring_logger.error(f"Error running nvidia-smi: {result.stderr}")
    except Exception as e:
        monitoring_logger.error(f"Error collecting GPU metrics: {e}")

def collect_docker_metrics():
    """Collect Docker container metrics"""
    try:
        # Check if the docker command is available
        result = subprocess.run(['which', 'docker'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            monitoring_logger.warning("Docker command not found, skipping Docker metrics")
            return
        
        # Get container counts
        result = subprocess.run(['docker', 'info', '--format', '{{.Containers}}'], 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            DOCKER_CONTAINERS.set(int(result.stdout.strip()))
        
        # Get running containers
        result = subprocess.run(['docker', 'info', '--format', '{{.ContainersRunning}}'], 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            DOCKER_RUNNING.set(int(result.stdout.strip()))
        
        # Get paused containers
        result = subprocess.run(['docker', 'info', '--format', '{{.ContainersPaused}}'], 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            DOCKER_PAUSED.set(int(result.stdout.strip()))
        
        # Get stopped containers
        result = subprocess.run(['docker', 'info', '--format', '{{.ContainersStopped}}'], 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            DOCKER_STOPPED.set(int(result.stdout.strip()))
        
        monitoring_logger.info("Docker metrics collected successfully")
    except Exception as e:
        monitoring_logger.error(f"Error collecting Docker metrics: {e}")

def collect_all_metrics():
    """Collect all metrics"""
    collect_system_metrics()
    collect_process_metrics()
    collect_gpu_metrics()
    collect_docker_metrics()

if __name__ == '__main__':
    # Start up the server to expose the metrics
    print("entered integration")
    metrics_port = 9990
    start_http_server(metrics_port)
    monitoring_logger.info(f"Started Prometheus metrics server on port {metrics_port}")
    
    try:
        # Main metrics collection loop
        while True:
            collect_all_metrics()
            time.sleep(15)  # Collect metrics every 15 seconds
    except KeyboardInterrupt:
        monitoring_logger.info("Metrics collection stopped by user")
    except Exception as e:
        monitoring_logger.error(f"Error in metrics collection: {e}")