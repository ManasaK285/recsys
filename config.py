import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = Path(os.getenv("DREAM_DB", DATA_DIR / "dream_rsi.db"))
LLM_MODEL = os.getenv("LLM_MODEL", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

DEFAULT_WORKERS = int(os.getenv("DEFAULT_WORKERS", "3"))
DEFAULT_STEPS = int(os.getenv("DEFAULT_STEPS", "12"))
