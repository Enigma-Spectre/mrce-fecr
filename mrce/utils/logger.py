"""
Lightweight CSV logger for MRCE dialogs.
Creates MRCE_chat_<n>.csv in ./logs/ (auto‑increment).
Usage:
    from mrce.utils.logger import log_turn
    log_turn(timestamp, persona, prompt, reply)
"""

import csv, os, datetime
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

CSV_LOGGING = True   # ← toggle logging on/off globally

# Determine next incremental file name
existing = sorted(LOG_DIR.glob("MRCE_chat_*.csv"))
if existing:
    last_num = int(existing[-1].stem.split("_")[-1])
    _log_path = LOG_DIR / f"MRCE_chat_{last_num + 1}.csv"
else:
    _log_path = LOG_DIR / "MRCE_chat_1.csv"

# Ensure header written once per session
if CSV_LOGGING:
    with _log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "persona", "prompt", "response"])


def log_turn(persona: str, prompt: str, response: str):
    if not CSV_LOGGING:
        return
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    with _log_path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([ts, persona, prompt, response])
