"""
Local Ollama wrapper (single‑file, dependency‑free except httpx).
Always uses /api/generate with role‑tagged prompt so it works on every daemon.
"""

import json
from typing import List, Dict, Optional

import httpx

OLLAMA_HOST = "http://127.0.0.1:11434"          # default daemon address
TIMEOUT_CHAT   = 300.0                          # seconds – 3‑agent fan‑out on CPU
TIMEOUT_EMBED  =  120.0


# ──────────────────────────────────────────────────────────────────────────────
def _flatten(messages: List[Dict[str, str]]) -> str:
    """
    Turn a multi‑turn chat list into a single prompt that Llama‑style models
    reliably follow.  The final “Assistant:” cue makes the model answer.
    """
    parts: list[str] = []
    for m in messages:
        role = m["role"].capitalize()
        parts.append(f"{role}: {m['content']}")
    parts.append("Assistant:")                  # cue the assistant to speak
    return "\n\n".join(parts)


async def chat(
    messages: List[Dict[str, str]],
    model: str = "llama3:8b",
    session: Optional[str] = None,
) -> str:
    """
    One‑shot chat completion (no streaming).  Robust across Ollama builds.

    Parameters
    ----------
    messages : list[dict]
        e.g. [{"role": "system", "content": "You are…"}, {"role": "user", "content": "Hi"}]
    model : str
        Model tag shown by `ollama list`
    session : str | None
        Optional; reuse to maintain server‑side context
    """
    prompt_text = _flatten(messages)
    payload = {
        "model":   model,
        "prompt":  prompt_text,
        "stream":  False,
        "session": session,
    }

    async with httpx.AsyncClient(timeout=TIMEOUT_CHAT) as client:
        r = await client.post(f"{OLLAMA_HOST}/api/generate", json=payload)
        r.raise_for_status()

    # NDJSON → last line is the full object for non‑stream mode
    data = json.loads(r.text.strip().splitlines()[-1])
    return data["response"]                     # plain assistant string


# ---------------------------------------------------------------- embed ---
async def embed(text: str, model: str = "llama3:8b") -> List[float]:
    """
    Get an embedding vector. 120‑second timeout; on timeout returns [] so
    Phase‑Crystal logic can skip that turn instead of crashing.
    """
    payload = {"model": model, "prompt": text}

    async with httpx.AsyncClient(timeout=120) as client:
        try:
            r = await client.post(f"{OLLAMA_HOST}/api/embeddings", json=payload)
            r.raise_for_status()
            return r.json()["embedding"]
        except (httpx.ReadTimeout, httpx.ConnectTimeout):
            # log + return empty vector
            print("[embed] timeout — skipping embedding for this turn")
            return []