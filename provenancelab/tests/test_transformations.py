import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from provenancelab.transformations import transform

def test_truncation_shortens_text():
    text = "one two three four five six"
    assert len(transform(text, "truncate").split()) < len(text.split())

def test_original_unchanged():
    text = "Keep this text unchanged."
    assert transform(text, "original") == text
