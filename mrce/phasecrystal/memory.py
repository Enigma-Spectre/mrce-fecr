"""
Phase‑Crystal Memory
────────────────────
Stores recent embeddings, computes resonance, and promotes high‑coherence
signals to 'crystals' (stable anchor vectors).

Key ideas
---------
• WINDOW      – sliding buffer length for embeddings & resonance scores
• DECAY       – exponential decay applied to past resonance values
• THRESH      – promote to crystal when resonance > THRESH
• ALPHA/BETA  – weight of semantic coherence vs. spectral recurrence

Public API
----------
add(emb)        -> (cid | None, resonance)
save(path)      -> pickle dump of internal state
load(path)      -> classmethod returning PhaseCrystalMemory instance
"""
from __future__ import annotations

import pickle
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Tuple

import numpy as np
from scipy import signal


# ─────────────────────────────────── constants
from mrce.settings import (
    WINDOW,
    DECAY,
    THRESH,
    ALPHA,
    BETA,
    MAX_CRYSTALS,
)


class PhaseCrystalMemory:
    """
    Sliding‑window memory + resonance scoring + crystal promotion.
    """

    # ..................................................................
    def __init__(self):
        self.embeds:   Deque[np.ndarray] = deque(maxlen=WINDOW)
        self.scores:   Deque[float]      = deque(maxlen=WINDOW)
        self.crystals: Dict[str, np.ndarray] = {}          # cid -> vector
        self._counter = 0                                   # for crystal IDs
        self.last_coherence = 0.0
        self.last_dominant = 0.0
        self.last_resonance = 0.0

    # ..................................................................
    @staticmethod
    def _cosine(u: np.ndarray, v: np.ndarray) -> float:
        return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))

    # ..................................................................
    def local_coherence(self, emb: np.ndarray) -> float:
        """Recency-weighted mean cosine similarity vs. buffer."""
        if not self.embeds:
            return 0.0
        mat = np.stack(self.embeds)
        sims = (mat @ emb) / (
            np.linalg.norm(mat, axis=1) * np.linalg.norm(emb) + 1e-9
        )
        w = np.geomspace(1.0, 0.1, num=len(sims))
        return float(np.average(sims, weights=w))

    # ..................................................................
    def dominant_frequency(self) -> float:
        """
        Magnitude of the dominant non‑DC component of resonance score FFT.
        Interprets repeating coherence spikes as signal.
        """
        if len(self.scores) < 16:
            return 0.0
        arr = np.asarray(self.scores, dtype="float32") - np.mean(self.scores)
        f, Pxx = signal.welch(arr, nperseg=min(32, len(arr)))
        return float(Pxx[1:].max() / (Pxx[1:].sum() + 1e-9))

    # ..................................................................
    def _next_cid(self) -> str:
        self._counter += 1
        return f"C{self._counter:04d}"

    # ..................................................................
    def add(self, emb: list[float] | np.ndarray) -> Tuple[str | None, float]:
        """
        Add an embedding; return (promoted_cid or None, resonance_score).
        """
        emb_arr = np.asarray(emb, dtype="float32")
        coherence = self.local_coherence(emb_arr)
        dominant  = self.dominant_frequency()
        resonance = ALPHA * coherence + BETA * dominant
        self.last_coherence = coherence
        self.last_dominant = dominant
        self.last_resonance = resonance

        # decay historical resonance
        self.scores = deque((s * DECAY for s in self.scores), maxlen=WINDOW)

        # append current turn
        self.embeds.append(emb_arr)
        self.scores.append(resonance)

        # promotion check
        if resonance > THRESH and len(self.crystals) < MAX_CRYSTALS:
            cid = self._next_cid()
            self.crystals[cid] = emb_arr
            return cid, resonance
        return None, resonance

    # ..................................................................
    def save(self, path: str | Path):
        """Persist memory to disk via pickle."""
        with Path(path).open("wb") as f:
            pickle.dump(
                {
                    "embeds":   list(self.embeds),
                    "scores":   list(self.scores),
                    "crystals": self.crystals,
                    "_counter": self._counter,
                },
                f,
            )

    # ..................................................................
    @classmethod
    def load(cls, path: str | Path) -> "PhaseCrystalMemory":
        with Path(path).open("rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj.embeds   = deque(data["embeds"],   maxlen=WINDOW)
        obj.scores   = deque(data["scores"],   maxlen=WINDOW)
        obj.crystals = data["crystals"]
        obj._counter = data["_counter"]
        return obj
