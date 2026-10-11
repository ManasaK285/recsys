"""Detector adapters.

The included baseline is deliberately a toy educational baseline, NOT SynthID and
NOT reliable for judging real-world authorship. Plug in a calibrated, supported
watermark detector for meaningful provenance verification.
"""
from dataclasses import dataclass
import re

@dataclass
class Detection:
    status: str
    score: float | None
    method: str
    caveat: str

class DemoBaselineDetector:
    """A transparent lexical-style baseline for wiring the product end-to-end.

    Its score is an arbitrary demo signal. It must never be interpreted as a
    probability that text was AI-generated.
    """
    method = "demo_lexical_baseline_not_synthid"
    caveat = ("Educational baseline only: this is not a SynthID detector and its score is not a "
              "probability of AI authorship. Do not use for consequential decisions.")

    def score(self, text: str) -> float:
        words = re.findall(r"\b\w+\b", text.lower())
        if not words:
            return 0.0
        transitions = sum(a == b for a, b in zip(words, words[1:])) / max(1, len(words)-1)
        sentence_count = max(1, len(re.findall(r"[.!?]+", text)))
        avg_sentence_len = len(words) / sentence_count
        formulaic = sum(phrase in text.lower() for phrase in (
            "in conclusion", "it is important to note", "furthermore", "moreover"))
        # Arbitrary demonstration score; deliberately named and caveated.
        value = 0.45 + min(0.15, formulaic * 0.04) + min(0.1, max(0, avg_sentence_len-18)*0.004) - min(0.1, transitions*2)
        return round(max(0.0, min(1.0, value)), 4)

    def detect(self, text: str) -> Detection:
        score = self.score(text)
        # The label is a demo band, not a factual authorship claim.
        if score >= 0.48:
            status = "demo_signal_high"
        elif score <= 0.42:
            status = "demo_signal_low"
        else:
            status = "uncertain"
        return Detection(status, score, self.method, self.caveat)
