import os
from datetime import datetime

LOGGER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs/logger")
os.makedirs(LOGGER_DIR, exist_ok=True)

def log_to_file(message, filename="NoName.log"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pipeline_id = os.environ.get("KCS_PIPELINE_ID", None)
    if pipeline_id:
        dirpath = os.path.join(LOGGER_DIR, pipeline_id)
        os.makedirs(dirpath, exist_ok=True)
        filepath = os.path.join(dirpath, filename)
    else:
        filepath = os.path.join(LOGGER_DIR, filename)
    with open(filepath, "a") as f:
        f.write(f"[{timestamp}] {message}\n")