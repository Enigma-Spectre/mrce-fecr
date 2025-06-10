import json, uuid, datetime
from typing import List, Dict

import numpy as np 

from mrce.llm import ollama_client
from mrce.phasecrystal.memory import PhaseCrystalMemory
from mrce.utils.logger import log_turn
from mrce.fecr import scorer
from mrce.fecr.modulator import modulate


class ExpertAgent:
    MAX_MESSAGES = 40

    def __init__(self, cfg: Dict):
        self.name          = cfg["name"]
        self.model         = cfg.get("model", "llama3:8b")
        self.system_prompt = cfg["system_prompt"]
        self.session       = f"{self.name}-{uuid.uuid4()}"
        self.dialogue: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]

        self.latest_fecr = None

        self.memory = PhaseCrystalMemory()

    # ───────────────────────────────────────────────────────────────
    async def step(self, user_msg: str, context_state: Dict) -> str:
        context_block = {
            "role": "system",
            "content": f"Context >>> {json.dumps(context_state)}"
        }
        prompt = self.dialogue + [context_block,
                                  {"role": "user", "content": user_msg}]

        reply = await ollama_client.chat(prompt, model=self.model, session=self.session)

        fecr_vec = scorer.vector(reply, self.latest_fecr)

        context_state["fecr_vector"] = fecr_vec
        self.latest_fecr = fecr_vec     # store for potential future use

        # record
        self.dialogue.extend([
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": reply}
        ])
        self._trim()

        # logging
        log_turn(self.name, user_msg, reply)

        # phase‑crystal memory
        emb = await ollama_client.embed(reply, model=self.model)
        if emb:
            e_mod = modulate(np.asarray(emb, dtype="float32"), fecr_vec)
            self.memory.add(e_mod)

        return reply

    # ───────────────────────────────────────────────────────────────
    def _trim(self):
        if len(self.dialogue) > self.MAX_MESSAGES:
            self.dialogue = self.dialogue[:1] + self.dialogue[-30:]
