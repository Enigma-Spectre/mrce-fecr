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
    gain = 1.0 + np.array(phi, dtype="float32")
    base = d // len(phi)
    g = np.repeat(gain, base)
    if g.size < d:
        g = np.concatenate([g, np.repeat(gain[-1], d - g.size)])
    g /= np.linalg.norm(g) / np.sqrt(d)
    return np.diag(g)

def modulate(emb: np.ndarray, phi: list[float]) -> np.ndarray:
    """Return W(φ)·e  (same shape as emb)."""
    if not phi:                              # safety fallback
        return emb
    W = _make_matrix(phi, emb.size)
    return W @ emb
