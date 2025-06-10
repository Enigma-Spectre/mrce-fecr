"""
Lightweight CSV logger for MRCE dialogs.
Creates MRCE_chat_<n>.csv in ./logs/ (auto‑increment).
Usage:
    from mrce.utils.logger import log_turn
    log_turn(timestamp, persona, prompt, reply)
"""

import csv
import datetime
from pathlib import Path
import logging
from rich.logging import RichHandler

from mrce.settings import LOG_DIR, CSV_LOGGING

LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("mrce")
logger.setLevel(logging.INFO)
logger.addHandler(RichHandler())

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


def log_phi(turn: int, phi: list[float], resonance: float, coherence: float, spectral: float):
    if not CSV_LOGGING:
        return
    path = LOG_DIR / "phi.csv"
    header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        if header:
            wr.writerow([
                "turn",
                "resonance",
                "coherence",
                "spectral",
                *[f"phi_{i}" for i in range(len(phi))],
            ])
        wr.writerow([turn, resonance, coherence, spectral, *phi])
