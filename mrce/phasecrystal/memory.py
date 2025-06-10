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
from typing import Deque, Dict, List, Tuple

import numpy as np
from numpy.fft import rfft


# ─────────────────────────────────── constants
WINDOW  = 128
DECAY   = 0.98
THRESH  = 0.85
ALPHA   = 0.6
BETA    = 0.4
MAX_CRYSTALS = 64            # global cap to avoid runaway growth


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

    # ..................................................................
    @staticmethod
    def _cosine(u: np.ndarray, v: np.ndarray) -> float:
        return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))

    # ..................................................................
    def local_coherence(self, emb: np.ndarray) -> float:
        """Average cosine similarity vs. buffer (0 if buffer empty)."""
        if not self.embeds:
            return 0.0
        mat = np.stack(self.embeds)              # (N, d)
        sims = mat @ emb / (
            np.linalg.norm(mat, axis=1) * np.linalg.norm(emb) + 1e-9
        )
        return float(sims.mean())

    # ..................................................................
    def dominant_frequency(self) -> float:
        """
        Magnitude of the dominant non‑DC component of resonance score FFT.
        Interprets repeating coherence spikes as signal.
        """
        if len(self.scores) < 8:                 # need minimum length
            return 0.0
        arr = np.asarray(self.scores, dtype="float32")
        spec = np.abs(rfft(arr - arr.mean()))    # remove DC offset
        # ignore DC (index 0) and take max of the rest
        return float(spec[1:].max() / (len(arr) / 2))   # normalise

    # ..................................................................
    def _next_cid(self) -> str:
        self._counter += 1
        return f"C{self._counter:04d}"

    # ..................................................................
    def add(self, emb: List[float] | np.ndarray) -> Tuple[str | None, float]:
        """
        Add an embedding; return (promoted_cid or None, resonance_score).
        """
        emb_arr = np.asarray(emb, dtype="float32")
        coherence = self.local_coherence(emb_arr)
        dominant  = self.dominant_frequency()
        resonance = ALPHA * coherence + BETA * dominant

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
