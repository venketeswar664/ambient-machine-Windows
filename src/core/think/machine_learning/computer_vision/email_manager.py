import json
import smtplib
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import datetime
import psutil
import GPUtil

class EmailAlertManager:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = json.load(f)
        self.last_alert_time = None
    
    def send_tamper_alert(self, camera_id):
        print("condition called")
        current_time = time.time()
        if self.last_alert_time and (current_time - self.last_alert_time) < self.config['resend_interval']*60:
            return

        msg = MIMEMultipart()
        msg['From'] = self.config['sender_email']
        msg['To'] = ", ".join(self.config['recipients'])
        msg['Subject'] = self.config['subject'].format(camera_id=camera_id)
        body = self.config['body'].format(
            camera_id=camera_id,
            timestamp=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
        )
        msg.attach(MIMEText(body, 'plain'))

        try:
            with smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port']) as server:
                server.starttls()
                server.login(self.config['sender_email'], self.config['sender_password'])
                server.sendmail(self.config['sender_email'], self.config['recipients'], msg.as_string())
            self.last_alert_time = current_time
            print(f"Camera tamper alert sent for {camera_id}")
        except Exception as e:
            print(f"Failed to send tamper alert: {str(e)}")




# // "pranjal@neophyte.ai",
#         // "abhinav@neophyte.ai",
#         // "anurag@neophyte.ai",
#         // "aryan.sinha@neophyte.ai"