# FlowFix — Low-Latency Agentic Program Repair

An executable portfolio project inspired by research on low-latency agentic program repair in pre-submit CI. It demonstrates failure intake, preflight checks, candidate patch generation, test validation, abstention, metrics, API, and dashboard using a deterministic offline benchmark.

> **Scope honesty:** The included provider is a deterministic baseline for four seeded Python bugs, not a general-purpose LLM repair agent and not a reproduction of Google's internal system. Results from this tiny benchmark are illustrative only.

## Features

- Four seeded failure cases: arithmetic logic, empty input, boundary condition, and division behavior.
- Disposable workspace per repair; original checkout is not edited.
- Baseline failure reproduction, syntax/patch checks, bounded test execution, and abstention.
- Unified diff suggestions and test output.
- FastAPI endpoints and interactive Streamlit dashboard.
- SQLite repair history and JSON/CSV benchmark reports.
- Pytest suite and GitHub Actions workflow.
- Offline operation; no API key required.

## Quick start (Windows PowerShell)

Use Python 3.10–3.13. Python 3.14 may work, but package wheel support can vary.

```powershell
cd flowfix
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest
```

Run the dashboard:

```powershell
streamlit run app.py
```

Run the API in a second terminal:

```powershell
uvicorn api:app --reload
```

- Dashboard: `http://localhost:8501`
- API docs: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

## API examples

```powershell
Invoke-RestMethod http://127.0.0.1:8000/cases
$body = @{ case_id = "average_off_by_one"; max_attempts = 2; timeout_seconds = 4 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/repair -ContentType "application/json" -Body $body
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/benchmark
Invoke-RestMethod http://127.0.0.1:8000/runs
```

## Benchmark and reports

```powershell
python scripts/run_benchmark.py
python scripts/generate_report.py
```

Outputs: `runs/benchmark_report.json` and `runs/benchmark_cases.csv`. Metrics include verified-suggestion rate, abstentions, median latency, and P95 latency. They cover four hand-authored cases only and should not be generalized to real repositories or compared directly with published industrial results.

## Architecture

1. **Intake:** accept a seeded case or CI failure event.
2. **Preflight:** check supported case and reproduce baseline failure.
3. **Triage/generation:** propose a candidate patch from source and case context.
4. **Patch checks:** syntax validation, size limit, conservative blocked-pattern checks.
5. **Validation:** run targeted tests with a timeout in a disposable workspace.
6. **Postflight:** return a suggestion only when tests pass; otherwise abstain.
7. **Observability:** persist outcomes and calculate latency/quality metrics.

## Safety limitations

- A temporary directory is **not** a security sandbox.
- The benchmark executes only controlled source and tests shipped here.
- Do not connect this prototype to arbitrary repositories or execute untrusted code on your host.
- Real use requires disposable containers/VMs, no credentials, network disabled by default, filesystem restrictions, and CPU/memory/process limits.
- Passing tests does not prove semantic correctness. Add hidden tests, regression suites, patch review, and human approval.
- The agent never commits or deploys a patch automatically.

## Extending to an LLM agent

`propose_patch()` in `src/flowfix/agent.py` is the provider seam. Replace its deterministic mapping with a model client that returns a unified diff or complete candidate source. Suggested next steps:

1. Require structured output: diagnosis, patch, assumptions.
2. Send only relevant source, failing test, traceback, and bounded context.
3. Keep the model separate from the execution environment.
4. Re-prompt at most once with sanitized test output.
5. Track token cost, model latency, retries, and abstention reasons.
6. Evaluate on a held-out public benchmark; report exact success, regression rate, P50/P95 latency, and confidence intervals.

## Research inspiration

Inspired by *Catching Developers in the Flow: Low-Latency Agentic Program Repair at Google Scale*. This is an independent educational implementation and is not affiliated with or endorsed by Google.

## License

MIT. See `LICENSE`.
