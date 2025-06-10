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

from mrce.llm import ollama_client
from mrce.phasecrystal.memory import PhaseCrystalMemory
from mrce.utils.logger import log_turn


class State(Enum):
    OPEN = auto()
    EVALUATE = auto()
    CONVERGE = auto()
    DIVERGE = auto()
    CONTRADICTION = auto()
    DONE = auto()


# ---------------------------------------------------------------------------
class CoordinatorFSM:
    TRUTH_TH = 0.75        # score ≥ TRUTH_TH → CONVERGE
    FLOOR_TH = 0.45        # score < FLOOR_TH → DIVERGE
    MAX_DIAL = 3           # dialectic rounds

    def __init__(self, global_mem: PhaseCrystalMemory):
        self.state: State = State.OPEN
        self.dial_rounds = 0
        self.global_mem  = global_mem

    # ....................................................
    def guide(self, fecr_vec) -> str:
        return ("Coordinator guidance: respond clearly; "
                f"FECR first four = {[round(w,2) for w in fecr_vec[:4]]}")

    # ....................................................
    async def score_reply(self, persona: str, reply: str) -> Tuple[float, float]:
        """Return (resonance, critic_grade_0_1)."""
        emb = await ollama_client.embed(reply)
        resonance = self.global_mem.local_coherence(emb) if emb else 0.0

        # critic lines like "Logical rigor: 8/10"
        critic_match = re.search(r"(\d+(?:\.\d+)?)/10", reply)
        grade = float(critic_match.group(1)) / 10 if critic_match else 0.5
        score = 0.6 * resonance + 0.4 * grade
        return score, resonance

    # ....................................................
    async def evaluate(self, user_msg: str,
                       replies: Dict[str, str]) -> Tuple[str, str]:
        """
        Decide next state, return (chosen_persona, text)
        """
        scored = {}
        for name, text in replies.items():
            s, _ = await self.score_reply(name, text)
            scored[name] = s

        best_name = max(scored, key=scored.get)
        best_score = scored[best_name]

        if best_score >= self.TRUTH_TH:
            self.state = State.CONVERGE
        elif best_score < self.FLOOR_TH:
            self.state = State.DIVERGE
        else:
            self.state = State.EVALUATE  # keep refining

        return best_name, replies[best_name]

    # ....................................................
    def summarise(self, text: str):
        """Store summary in global memory + log"""
        log_turn("Coordinator", "SUMMARY", text)
        emb = ollama_client.embed(text)          # fire‑and‑forget coroutine
        # we can't await inside summarise (engine will handle); placeholder

    # ....................................................
    def step_dialectic(self):
        self.dial_rounds += 1
        if self.dial_rounds >= self.MAX_DIAL:
            self.state = State.CONTRADICTION


# ---------------------------------------------------------------------------
async def detect_contradiction(text_a: str, text_b: str) -> bool:
    """
    Cheap contradiction check via LLM call.
    Returns True if replies conflict.
    """
    prompt = [
        {"role": "system", "content": (
            "Return YES if statement A contradicts statement B, else NO.")},
        {"role": "user", "content": f"Statement A: {text_a}\n\nStatement B: {text_b}"}
    ]
    ans = await ollama_client.chat(prompt)
    return ans.strip().upper().startswith("YES")
