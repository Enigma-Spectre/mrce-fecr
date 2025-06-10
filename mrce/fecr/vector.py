"""
FECR vector helpers.
Vector evolves from crystals: more crystals → emphasise mid‑band layers.
"""
from __future__ import annotations
from typing import Dict, List
import numpy as np

def neutral() -> List[float]:
    return [1/13] * 13

def update_from_crystals(crystals: Dict[str, np.ndarray]) -> List[float]:
    phi = np.array(neutral(), dtype="float32")
    if not crystals:
        return phi.tolist()

    # Simple heuristic: weight middle layers by crystal count
    bump = min(len(crystals) / 20, 0.25)      # cap extra weight
    phi[4:9] += bump
    phi /= phi.sum()
    return phi.tolist()
