"""
FECR scorer now returns the *current* vector stored on the agent if present,
otherwise neutral.  Real weighting will emerge via crystals.
"""
from typing import List
from mrce.fecr.vector import neutral

def vector(reply_text: str, current_phi: List[float] | None) -> List[float]:
    return current_phi if current_phi else neutral()
