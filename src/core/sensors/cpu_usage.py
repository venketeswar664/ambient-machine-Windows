import psutil
import threading
import subprocess
import datetime
import os


class ResourceMonitor:
    def __init__(self, verbose=False, store=True):
        self.verbose = verbose
        self.dir = "./src/logs"
        os.makedirs(self.dir, exist_ok=True)
        time = datetime.datetime.now(datetime.timezone.utc)()  # Record end time
        time = time.strftime("%m-%d-%Y_%H:%M:%S")

        self.file_path = self.dir + f"/process_logs_{time}.txt"

    def monitor(self):
        cpu_usage = psutil.cpu_percent()
        ram_usage = psutil.virtual_memory().percent

        gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'])
        gpu_usage = int(gpu_info.decode().strip())

        if self.verbose:
            print(f"CPU Usage: {cpu_usage}%")
            print(f"RAM Usage: {ram_usage}%")
            print(f"GPU Usage: {gpu_usage}%")

        if cpu_usage > 90:
            print("\033[91mWarning: CPU usage is above 90%\033[0m")

        if ram_usage > 90:
            print("\033[91mWarning: RAM usage is above 90%\033[0m")

        if gpu_usage > 90:
            print("\033[91mWarning: GPU usage is above 90%\033[0m")

        # Open the log file in append 
        end_time = datetime.datetime.now(datetime.timezone.utc)()  # Record end time
        end_time = end_time.strftime("%m-%d-%Y_%H:%M:%S")

        text = f"{end_time}: GPU Usage: {gpu_usage}%  #  CPU Usage: {cpu_usage}%  #  RAM Usage: {ram_usage}%"
        with open(self.file_path, 'a') as file:
            file.write(text + '\n')

    def run_monitor(self):
        threading.Timer(10, self.run_monitor).start()  
        self.monitor()