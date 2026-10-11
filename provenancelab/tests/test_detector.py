import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from provenancelab.detectors import DemoBaselineDetector

def test_detector_caveat_is_present():
    result = DemoBaselineDetector().detect("This is a sufficiently long example sentence for the demo detector.")
    assert result.score is not None
    assert "not a SynthID detector" in result.caveat
