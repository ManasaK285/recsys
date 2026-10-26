from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = ROOT / "data" / "sample_corpus.jsonl"
DEFAULT_QUERIES = ROOT / "data" / "sample_queries.jsonl"

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def load_sample_data():
    return read_jsonl(DEFAULT_CORPUS), read_jsonl(DEFAULT_QUERIES)
