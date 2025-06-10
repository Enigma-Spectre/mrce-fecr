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


def test_resonance_monotonic():
    mem = PhaseCrystalMemory()
    rng = np.random.default_rng(0)
    vec = rng.normal(size=768).astype("float32")
    for _ in range(10):
        mem.add(vec + 0.01 * rng.normal(size=768))
    assert all(np.diff(mem.scores) >= -1e-3)
