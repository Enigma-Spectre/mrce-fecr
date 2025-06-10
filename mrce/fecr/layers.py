"""
FECR 13‑layer stack – names mirror the processing chain in FECRsoundlayer_342.py
( echo → low‑pass → AM mod … ) so the cognitive engine can track which
virtual ‘layer’ fires.

Layer classes do nothing now except add their name to a context trace.
"""

class Layer:
    name: str = "Base"

    def __call__(self, context: dict):
        context.setdefault("fecr_trace", []).append(self.name)
        return context


class Echo1Layer(Layer):            name = "Echo‑1"
class Lowpass1Layer(Layer):         name = "Low‑Pass‑1"
class AMMod1Layer(Layer):           name = "AM‑Mod‑1"
class Harmonic1Layer(Layer):        name = "Harmonic‑1"
class PhaseInvert1Layer(Layer):     name = "Phase‑Invert‑1"
class Envelope1Layer(Layer):        name = "Envelope‑1"
class Echo2Layer(Layer):            name = "Echo‑2"
class AMMod2Layer(Layer):           name = "AM‑Mod‑2"
class Lowpass2Layer(Layer):         name = "Low‑Pass‑2"
class Harmonic2Layer(Layer):        name = "Harmonic‑2"
class PhaseInvert2Layer(Layer):     name = "Phase‑Invert‑2"
class Echo3Layer(Layer):            name = "Echo‑3"
class Envelope2Layer(Layer):        name = "Envelope‑2"

# Ordered list for easy iteration
FECR_LAYERS = [
    Echo1Layer(), Lowpass1Layer(), AMMod1Layer(), Harmonic1Layer(),
    PhaseInvert1Layer(), Envelope1Layer(), Echo2Layer(), AMMod2Layer(),
    Lowpass2Layer(), Harmonic2Layer(), PhaseInvert2Layer(), Echo3Layer(),
    Envelope2Layer()
]
