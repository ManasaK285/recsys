import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from provenancelab.metrics import binary_metrics

def test_binary_metrics_perfect():
    m = binary_metrics([0, 0, 1, 1], [0, 0, 1, 1])
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["false_positive_rate"] == 0.0

def test_binary_metrics_errors():
    m = binary_metrics([0, 1], [1, 0])
    assert m["fp"] == 1 and m["fn"] == 1
