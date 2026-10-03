import os
import subprocess
import sys

# Check if the script is running with elevated privileges (root)
if os.geteuid() != 0:
    print("This script requires root privileges. Please run as root or use sudo.")
    sys.exit(1)

# Get the current conda environment
def get_conda_env():
    try:
        # Run `conda info --envs` to get the list of environments
        result = subprocess.run(['conda', 'info', '--envs'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = result.stdout.decode()
        
        # Get the environment that is active (marked with an asterisk)
        for line in output.splitlines():
            if '*' in line:
                return line.split()[0]  # The first field is the active environment path
        return None
    except Exception as e:
        print(f"Error while fetching Conda environment: {e}")
        return None

# Create or update the systemd service file
def create_or_update_service_file(env_name):
    service_name = "ambient_machine_service"
    service_path = f"/etc/systemd/system/{service_name}.service"

    service_content = f"""
[Unit]
Description=Ambient Machine Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory={os.getcwd()}
ExecStart=/bin/bash -c 'source /home/sentinel/miniconda3/bin/activate {env_name} && python {os.getcwd()}/main.py'
Restart=always

[Install]
WantedBy=multi-user.target
"""

    try:
        # Check if the service file exists
        if os.path.exists(service_path):
            print(f"Service file already exists at {service_path}. Updating...")
            with open(service_path, 'w') as f:
                f.write(service_content)  # Update the existing service file
        else:
            print(f"Creating new service file at {service_path}...")
            with open(service_path, 'w') as f:
                f.write(service_content)  # Create a new service file

        print(f"Service file created/updated at {service_path}")
    except Exception as e:
        print(f"Error while creating/updating systemd service file: {e}")
        sys.exit(1)

    return service_path

# Enable and restart the service
def enable_and_restart_service():
    try:
        subprocess.run(['systemctl', 'daemon-reload'], check=True)  # Reload systemd configuration
        subprocess.run(['systemctl', 'enable', 'ambient_machine_service.service'], check=True)  # Enable the service to start at boot
        subprocess.run(['systemctl', 'restart', 'ambient_machine_service.service'], check=True)  # Restart the service immediately
        print("Service is enabled and restarted.")
    except subprocess.CalledProcessError as e:
        print(f"Error while enabling and restarting the service: {e}")
        sys.exit(1)

def main():
    # Get the current conda environment
    conda_env = get_conda_env()

    if conda_env is None:
        print("No active Conda environment found. Please activate a Conda environment before running this script.")
        sys.exit(1)

    print(f"Current active Conda environment: {conda_env}")

    # Create or update systemd service for the Ambient Machine Service
    create_or_update_service_file(conda_env)

    # Enable and restart the service
    enable_and_restart_service()

if __name__ == '__main__':
    main()
