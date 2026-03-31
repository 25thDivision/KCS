import os
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

def log_to_file(message, filename="only_experiment_discord.log"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filepath = os.path.join(LOG_DIR, filename)
    with open(filepath, "a") as f:
        f.write(f"[{timestamp}] {message}\n")