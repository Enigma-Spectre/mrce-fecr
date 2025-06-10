"""
FECR embedding modulator
------------------------
Builds a diagonal block‑gain matrix W(φ) from a 13‑value FECR vector φ and
multiplies it with an embedding e ∈ ℝᵈ to yield the modulated embedding eʹ.
Assumes d is divisible by 13; the last block absorbs any remainder.
"""

from __future__ import annotations
import numpy as np
from typing import List

def _make_matrix(phi: List[float], d: int) -> np.ndarray:
    blocks = len(phi)
    base   = d // blocks
    gains  = np.repeat(phi, base).astype("float32")
    if gains.size < d:                       # pad tail if d % 13 != 0
        gains = np.concatenate([gains, np.repeat(phi[-1], d - gains.size)])
    return np.diag(gains)

def modulate(emb: np.ndarray, phi: List[float]) -> np.ndarray:
    """Return W(φ)·e  (same shape as emb)."""
    if not phi:                              # safety fallback
        return emb
    W = _make_matrix(phi, emb.size)
    return W @ emb
