# DreamAlgo-RSI

A research-oriented, end-to-end prototype inspired by recursive self-improving exploration:
- LLM-style candidate generation (with deterministic fallback)
- sandboxed candidate evaluation
- discovery tree
- replay worlds 
- offline policy evaluation ("dreaming")
- recursive policy improvement
- FastAPI API
- Streamlit dashboard
- SQLite persistence
- reproducible experiments

## Quick start

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python scripts/run_demo.py
streamlit run dashboard/app.py
```

The demo runs without an API key. It uses deterministic candidate templates so the whole pipeline is reproducible.

## Optional LLM mode

Set:

```bash
set OPENAI_API_KEY=...
set LLM_MODEL=...
```

or on macOS/Linux:

```bash
export OPENAI_API_KEY=...
export LLM_MODEL=...
```

The LLM is used for candidate/policy generation; deterministic evaluation remains the source of truth for correctness and benchmark scores.

## API

```bash
uvicorn src.api.main:app --reload
```

Endpoints:
- `GET /health`
- `GET /tasks`
- `POST /runs`
- `GET /runs/{run_id}`
- `POST /dream/{run_id}`
- `POST /recursive/{task_id}`

## Safety

Generated candidate code is executed in a subprocess with a timeout. For production use, replace this with a hardened container sandbox with disabled network, CPU/memory limits, a read-only filesystem, and separate worker identities. Do not run untrusted code directly on a machine that contains secrets.
