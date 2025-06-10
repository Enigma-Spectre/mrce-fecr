import numpy as np
from mrce.phasecrystal.memory import PhaseCrystalMemory

def test_promotion_cap():
    mem = PhaseCrystalMemory()
    rng = np.random.default_rng(42)

    for _ in range(200):
        vec = rng.normal(size=768).astype("float32")
        mem.add(vec)

    # We expect at most 3 crystals in random noise
    assert len(mem.crystals) <= 3
