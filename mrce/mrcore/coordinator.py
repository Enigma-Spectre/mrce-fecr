"""
Coordinator finite‑state machine.

States
------
OPEN        – first user turn (guide only, always → EVALUATE)
EVALUATE    – score replies, choose state
CONVERGE    – high coherence, low conflict → distil truth crystal
DIVERGE     – low coherence OR strong conflict → fire dialectic
CONTRADICTION – after N dialectic rounds with no resolution
DONE        – terminal

Scoring
-------
score = 0.6 * resonance  + 0.4 * critic_grade(0‑1)
"""

from __future__ import annotations
import re
from enum import Enum, auto
from typing import Dict, List, Tuple
import numpy as np

from mrce.llm import ollama_client
from mrce.phasecrystal.memory import PhaseCrystalMemory
from mrce.utils.logger import log_turn
from mrce.settings import (
    RESONANCE_WEIGHT,
    CRITIC_WEIGHT,
    INDIVIDUAL_WEIGHT,
    PAIRWISE_WEIGHT,
    TRUTH_Z,
    FLOOR_TH,
MAX_DIAL,
)


def contradict(a: str, b: str) -> bool:
    antonyms = {"cannot": "can", "never": "always", "higher": "lower"}
    for neg, pos in antonyms.items():
        if (neg in a and pos in b) or (pos in a and neg in b):
            return True
    return False


class State(Enum):
    OPEN = auto()
    EVALUATE = auto()
    CONVERGE = auto()
    DIVERGE = auto()
    CONTRADICTION = auto()
    DONE = auto()


# ---------------------------------------------------------------------------
class CoordinatorFSM:

    def __init__(self, global_mem: PhaseCrystalMemory):
        self.state: State = State.OPEN
        self.dial_rounds = 0
        self.global_mem  = global_mem

    # ....................................................
    def guide(self, prev: str | None, user_msg: str, fecr_vec) -> str:
        parts = []
        if prev:
            parts.append(f"Prev ⇒ {prev}")
        if user_msg:
            parts.append(f"User ⇒ {user_msg}")
        parts.append(f"φ₀₋₃={[round(w,2) for w in fecr_vec[:4]]}")
        return " | ".join(parts)

    # ....................................................
    async def score_reply(self, persona: str, reply: str) -> Tuple[float, float]:
        """Return (resonance, critic_grade_0_1)."""
        emb = await ollama_client.embed(reply)
        resonance = self.global_mem.local_coherence(emb) if emb else 0.0

        # critic lines like "Logical rigor: 8/10"
        critic_match = re.search(r"(\d+(?:\.\d+)?)/10", reply)
        grade = float(critic_match.group(1)) / 10 if critic_match else 0.5
        score = RESONANCE_WEIGHT * resonance + CRITIC_WEIGHT * grade
        return score, resonance

    # ....................................................
    async def evaluate(self, user_msg: str,
                       replies: Dict[str, str]) -> Tuple[str, str]:
        """
        Decide next state, return (chosen_persona, text)
        """
        scored = {}
        reson = {}
        for name, text in replies.items():
            s, r = await self.score_reply(name, text)
            scored[name] = s
            reson[name] = r

        embs = []
        for text in replies.values():
            emb = await ollama_client.embed(text)
            if emb:
                embs.append(np.asarray(emb, dtype="float32"))
        pairwise = 0.0
        if len(embs) >= 2:
            M = np.stack(embs)
            norm = np.linalg.norm(M, axis=1)
            sims = (M @ M.T) / (norm[:, None] * norm[None, :] + 1e-9)
            pairwise = float(sims[np.triu_indices(len(embs), 1)].mean())

        final_scores = {
            name: INDIVIDUAL_WEIGHT * scored[name] + PAIRWISE_WEIGHT * pairwise
            for name in replies
        }

        best_name = max(final_scores, key=final_scores.get)
        best_score = final_scores[best_name]

        vals = list(final_scores.values())
        mu = float(np.mean(vals)) if vals else 0.0
        sigma = float(np.std(vals)) if vals else 0.0
        truth_th = mu + TRUTH_Z * sigma

        if best_score >= truth_th:
            self.state = State.CONVERGE
        elif best_score < FLOOR_TH:
            self.state = State.DIVERGE
        else:
            self.state = State.EVALUATE

        return best_name, replies[best_name]

    # ....................................................
    def summarise(self, text: str):
        """Store summary in global memory + log"""
        log_turn("Coordinator", "SUMMARY", text)
        emb = ollama_client.embed(text)          # fire‑and‑forget coroutine
        # we can't await inside summarise (engine will handle); placeholder

    def summarise_round(self, guide: str, replies: Dict[str, str], critic: str, best: str | None = None) -> str:
        parts = [f"Guide: {guide}"]
        for name, rep in replies.items():
            parts.append(f"{name}: {rep}")
        if critic:
            parts.append(f"Critic: {critic}")
        if best is not None:
            parts.append(f"Best: {best}")
        text = "\n".join(parts)
        self.summarise(text)
        if best is not None and self.state is State.CONVERGE:
            self.summarise(best)
        return text

    # ....................................................
    def step_dialectic(self):
        self.dial_rounds += 1
        if self.dial_rounds >= MAX_DIAL:
            self.state = State.CONTRADICTION


# ---------------------------------------------------------------------------
async def detect_contradiction(text_a: str, text_b: str) -> bool:
    """
    Cheap contradiction check via LLM call.
    Returns True if replies conflict.
    """
    if contradict(text_a, text_b):
        return True
    prompt = [
        {"role": "system", "content": (
            "Return YES if statement A contradicts statement B, else NO.")},
        {"role": "user", "content": f"Statement A: {text_a}\n\nStatement B: {text_b}"}
    ]
    ans = await ollama_client.chat(prompt)
    return ans.strip().upper().startswith("YES")
