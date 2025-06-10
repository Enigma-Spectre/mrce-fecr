from pathlib import Path

# Phase-Crystal parameters
WINDOW = 128
DECAY = 0.98
THRESH = 0.85
ALPHA = 0.6
BETA = 0.4
MAX_CRYSTALS = 64

# Coordinator weights
RESONANCE_WEIGHT = 0.6
CRITIC_WEIGHT = 0.4
INDIVIDUAL_WEIGHT = 0.4
PAIRWISE_WEIGHT = 0.6
TRUTH_Z = 0.75
FLOOR_TH = 0.45
MAX_DIAL = 3

# Logging
LOG_DIR = Path("logs")
CSV_LOGGING = True
