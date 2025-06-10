"""
MRCE Engine
===========

CoordinatorFSM + ExpertAgents + Phase‑Crystal Memory +
live FECR vector update.

Run with:  python scripts/smoke_engine.py
"""

from __future__ import annotations
import asyncio
from pathlib import Path
from typing import Dict

from rich import print

from mrce.mrcore.loader import load_personae
from mrce.mrcore.coordinator import CoordinatorFSM, State, detect_contradiction
from mrce.phasecrystal.memory import PhaseCrystalMemory
from mrce.fecr.vector import update_from_crystals


class MRCEngine:
    def __init__(self):
        self.global_mem = PhaseCrystalMemory()
        self.coordinator = CoordinatorFSM(self.global_mem)
        self.agents = load_personae(Path("configs/personae"))
        self.turn = 0

    # ------------------------------------------------------------------
    async def run_turn(self, user_msg: str) -> str:
        """
        One complete MRCE round:
          guide → expert fan‑out → evaluate → converge/diverge logic
        Returns text chosen by coordinator (best or distilled).
        """
        # FECR vector evolves from crystals each turn
        fecr_vec = update_from_crystals(self.global_mem.crystals)
        context_state: Dict = {"fecr_vector": fecr_vec}

        # Coordinator guidance first
        guide_txt = self.coordinator.guide(fecr_vec)
        context_state["guide"] = guide_txt

        # Broadcast to experts in parallel
        tasks = [ag.step(user_msg, context_state) for ag in self.agents]
        replies = {ag.name: rep for ag, rep in zip(self.agents, await asyncio.gather(*tasks))}

        # Evaluate & print round
        best_name, best_text = await self.coordinator.evaluate(user_msg, replies)
        print(f"[bold magenta]Coordinator state → {self.coordinator.state.name}")
        for name, txt in replies.items():
            print(f"[cyan][{name}][/]: {txt}\n")

        # State handling
        if self.coordinator.state is State.CONVERGE:
            self.coordinator.summarise(best_text)    # distilled truth crystal
            self.turn += 1
            return best_text

        if self.coordinator.state is State.DIVERGE:
            # simple dialectic: pick two shortest replies, check contradiction
            names = sorted(replies, key=lambda n: len(replies[n]))[:2]
            if await detect_contradiction(replies[names[0]], replies[names[1]]):
                self.coordinator.step_dialectic()

        # otherwise EVALUATE continues
        self.turn += 1
        return best_text
