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
import numpy as np
from mrce.utils.logger import log_phi, logger

from mrce.mrcore.loader import load_personae
from mrce.mrcore.coordinator import CoordinatorFSM, State, detect_contradiction
from mrce.phasecrystal.memory import PhaseCrystalMemory
from mrce.fecr.vector import update_from_crystals


class MRCEngine:
    def __init__(self):
        self.global_mem = PhaseCrystalMemory()
        self.coordinator = CoordinatorFSM(self.global_mem)
        loaded = load_personae(Path("configs/personae"))
        self.critic = next((a for a in loaded if a.name.lower() == "critic"), None)
        self.agents = [a for a in loaded if a is not self.critic and a.name.lower() != "coordinator"]
        self.turn = 0
        self.prev_coord: str | None = None

    # ------------------------------------------------------------------
    async def run_turn(self, user_msg: str | None, debug: bool = False) -> str:
        """
        One MRCE round following the order:
          user → coordinator → agents → critic → coordinator
        Returns text chosen by coordinator.
        """
        fecr_vec = update_from_crystals(self.global_mem.crystals)
        context_state: Dict = {"fecr_vector": fecr_vec, "user_input": user_msg or ""}
        if debug:
            logger.info(f"phi[0:4]={fecr_vec[:4]}")

        guide_txt = self.coordinator.guide(self.prev_coord, user_msg or "", fecr_vec)
        context_state["guide"] = guide_txt

        tasks = [ag.step(guide_txt, dict(context_state)) for ag in self.agents]
        replies = {ag.name: rep for ag, rep in zip(self.agents, await asyncio.gather(*tasks))}

        critic_reply = ""
        if self.critic:
            critic_prompt = "\n".join(f"{n}: {r}" for n, r in replies.items())
            critic_state = dict(context_state)
            critic_state["agent_replies"] = replies
            critic_reply = await self.critic.step(critic_prompt, critic_state)
            context_state["critic"] = critic_reply

        best_name, best_text = await self.coordinator.evaluate(user_msg or "", replies)
        logger.info(f"Coordinator state → {self.coordinator.state.name}")
        for name, txt in replies.items():
            logger.info(f"[{name}] {txt}")
        if critic_reply:
            logger.info(f"[Critic] {critic_reply}")

        summary = self.coordinator.summarise_round(guide_txt, replies, critic_reply, best_text)
        self.prev_coord = summary

        if self.coordinator.state is State.DIVERGE:
            names = sorted(replies, key=lambda n: len(replies[n]))[:2]
            if await detect_contradiction(replies[names[0]], replies[names[1]]):
                self.coordinator.step_dialectic()

        self.turn += 1
        metrics = [ag.memory.last_resonance for ag in self.agents]
        coh = float(np.mean([ag.memory.last_coherence for ag in self.agents]))
        spec = float(np.mean([ag.memory.last_dominant for ag in self.agents]))
        res = float(np.mean(metrics))
        log_phi(self.turn, fecr_vec, res, coh, spec)

        return best_text
