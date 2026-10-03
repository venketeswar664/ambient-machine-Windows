import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os
from datetime import datetime


class LogNotifier:
    def __init__(self, log_dirs, email_config, recipients):
        """
        Initialize the notifier with log directories, email configuration, and recipient list.
        """
        self.log_dirs = log_dirs
        self.email_config = email_config
        self.recipients = recipients
        self.processed_logs = {}  # Track processed lines for each log file

    def find_log_files(self):
        """
        Find all log files in the specified directories.
        """
        log_files = []
        for log_dir in self.log_dirs:
            if log_dir and os.path.isdir(log_dir):  # Ensure log_dir is not None and exists
                for root, _, files in os.walk(log_dir):
                    for file in files:
                        log_files.append(os.path.join(root, file))
            else:
                print(f"Directory not found or invalid: {log_dir}")
        return log_files

    def check_logs(self):
        """
        Check all log files for errors from the bottom up and send an email if any are found.
        """
        log_files = self.find_log_files()

        for log_file in log_files:
            # Initialize the processed lines tracker for new log files
            if log_file not in self.processed_logs:
                self.processed_logs[log_file] = 0

            try:
                with open(log_file, "r") as file:
                    lines = file.readlines()
                    # Reverse the lines to start from the bottom
                    for i, line in enumerate(reversed(lines), 1):
                        if i <= self.processed_logs.get(log_file, 0):
                            continue  # Skip already processed lines

                        if "error" in line.lower():
                            print(f"Error detected in {log_file}: {line.strip()}")
                            self.send_email(log_file, line.strip())
                            break  # Send only one email per file per check
                    
                    # Update the position to avoid re-processing lines
                    self.processed_logs[log_file] = len(lines)
            except FileNotFoundError:
                print(f"Log file not found: {log_file}")
            except Exception as e:
                print(f"Error checking log file {log_file}: {e}")

    def send_email(self, log_file, error_message):
        """
        Send an email notification with details about the error.
        """
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['address']
            msg['To'] = ", ".join(self.recipients)
            msg['Subject'] = f"Error Alert from {log_file}"

            # Email body
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            body = f"""
Hello Team,

An error was detected in the log file '{log_file}'.

Timestamp: {timestamp}
Error Details: {error_message}

Please find the attached log file for more details.

Best Regards,
The Sentinel Monitoring System
"""
            msg.attach(MIMEText(body, 'plain'))

            # Attach the log file
            with open(log_file, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename={os.path.basename(log_file)}'
                )
                msg.attach(part)

            # Send the email
            server = smtplib.SMTP(self.email_config['server'], self.email_config['port'])
            server.starttls()
            server.login(self.email_config['address'], self.email_config['password'])
            server.sendmail(self.email_config['address'], self.recipients, msg.as_string())
            server.quit()

            print(f"Error notification sent for {log_file}.")
        except Exception as e:
            print(f"Failed to send email: {e}")


def get_latest_folder(directory):
    """
    Get the latest folder in the given directory based on the folder name.
    Assumes folder names are in a sortable format like YYYYMMDD_HHMMSS.
    """
    try:
        # List all directories in the given directory
        folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
        # Filter folders that have a valid timestamp format
        valid_folders = [f for f in folders if len(f) == 15 and f[:8].isdigit() and f[9:].isdigit()]
        # Sort the valid folders by name in descending order
        valid_folders.sort(reverse=True)
        # Return the latest folder
        return os.path.join(directory, valid_folders[0]) if valid_folders else None
    except Exception as e:
        print(f"Error finding latest folder: {e}")
        return None


if __name__ == "__main__":
    # Define log directories
    mongodb_log_dir = "./logs/mongodb/"
    ambient_machine_dir = "./logs/"

    # Get the latest folder in the ambient_machine directory
    latest_folder = get_latest_folder(ambient_machine_dir)
    if latest_folder:
        print(f"Latest folder in ambient_machine: {latest_folder}")
    else:
        print("No folders found in ambient_machine.")

    # Define directories to monitor (filter out None)
    log_dirs = [dir_path for dir_path in [mongodb_log_dir, latest_folder] if dir_path]

    # Email configuration
    email_config = {
        'server': 'smtp.gmail.com',
        'port': 587,
        'address': 'neophyte.sentinel@gmail.com',
        'password': 'cjxk lwzh vkck saas'
    }

    # Recipients
    recipients = [
        "pranjal@neophyte.ai",
        # "aryan.sinha@neophyte.ai"
    ]

    # Initialize and run the notifier
    notifier = LogNotifier(log_dirs, email_config, recipients)
    notifier.check_logs()

"""
Open the crontab editor:
crontab -e

Add this line to the file:
*/5 * * * * /usr/bin/python3 /home/user/scripts/notify_error_logs.py

Save and exit.
This will run the notify_error_logs.py script every 5 minutes.
"""