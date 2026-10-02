# ⚙️ AgentForge

### Industrial Agentic Engineering Harness for Autonomous Software Development

> **Don't debug the code. Debug the agent.**

AgentForge is an experimental **agentic software engineering harness** for orchestrating, verifying, evaluating, and improving autonomous coding workflows.

Instead of treating an LLM as a code autocomplete system, AgentForge treats software development as a sequence of **agent trajectories** operating inside a controlled **harness** with planning, repository analysis, parallel implementation, verification, evaluation, observability, and meta-debugging.

The project is inspired by the principles described in:

> **Ramón Medrano Llamas (2026), *Industrial Agentic Engineering***

In particular, AgentForge explores the idea that when an autonomous coding workflow fails, the engineer should debug the **workflow and harness** rather than repeatedly patching the generated code.

---

## Why AgentForge?

Traditional AI coding workflows often look like:

```text
Prompt
  ↓
LLM generates code
  ↓
Code fails
  ↓
Ask LLM to fix code
  ↓
Repeat
```

AgentForge experiments with a different approach:

```text
Task
 ↓
Planning
 ↓
Repository Analysis
 ↓
Parallel Agent Trajectories
 ↓
Verification
 ↓
Failure Classification
 ↓
Meta-Debugging
 ↓
Harness Intervention
 ↓
Retry
 ↓
Verified Result
```

The objective is not simply to generate code, but to build an environment in which autonomous coding agents can be **observed, evaluated, verified, and improved**.

---

# Core Concepts

## 1. The Harness

A raw language model is not treated as the complete software engineer.

AgentForge wraps the model with a harness containing:

* Planning
* Repository analysis
* Coding strategies
* Tooling
* Verification
* Requirements checking
* Failure classification
* Trajectory evaluation
* Persistent run history
* Tracing
* Meta-debugging
* Recovery logic

The harness controls the workflow surrounding the model rather than relying entirely on the model's generation capability.

---

## 2. Trajectories

A single task can be attempted through multiple implementation strategies.

AgentForge currently supports three deterministic demonstration trajectories:

| Strategy     | Purpose                                           |
| ------------ | ------------------------------------------------- |
| `minimal`    | Make the smallest required change                 |
| `robust`     | Consider edge cases and implementation robustness |
| `test_first` | Prioritize test-oriented implementation           |

These trajectories execute independently in isolated workspaces.

This allows the harness to compare different approaches to the same engineering task.

---

## 3. Parallel Agent Execution

For a task such as:

```text
Add a /health endpoint returning status=ok
```

AgentForge can execute:

```text
                 ┌── minimal ──────→ verification
Task ────────────┼── robust ────────→ verification
                 └── test_first ────→ verification
```

Each trajectory produces:

* Implementation
* Verification results
* Requirements results
* Judgment
* Failure classification
* Trace information

The system can then determine whether recovery is necessary.

---

# 4. Verification

AgentForge does not consider generated code successful merely because the agent produced files.

Each trajectory is evaluated through automated verification.

The verification pipeline includes:

* `pytest`
* Ruff/code-quality checks
* Repository requirements checks
* Task-specific requirements
* Final trajectory judgment

The principle is simple:

> **Generated code must be verified before it is considered successful.**

---

# 5. Failure Taxonomy

When a trajectory fails, AgentForge classifies the failure rather than treating every failure as an implementation bug.

The project includes a failure taxonomy covering categories such as:

* Planning failures
* Retrieval/repository-analysis failures
* Implementation failures
* Verification failures
* Workflow failures

This makes failures useful as diagnostic signals for the agentic workflow.

---

# 6. Meta-Debugging

Meta-debugging is the central idea demonstrated by AgentForge.

Instead of immediately modifying failed code, the system asks:

> **What went wrong in the agentic workflow?**

The Meta-Debugger receives information about failed trajectories and their traces, identifies a likely workflow-level root cause, and produces an intervention.

For example, the current demonstration produces:

```text
Root cause:
workflow did not inspect tests and edge cases before implementation

Failure class:
workflow.bad_recovery_strategy

Interventions:
- inspect existing tests before editing
- enumerate edge cases
- run the complete test suite before declaring success
```

The intervention is then passed into the retry workflow.

---

# 7. Recovery Loop

The project's primary demonstration intentionally exercises the recovery mechanism.

### Attempt 1

```text
minimal     → PASS
robust      → FAIL
test_first  → PASS
```

The failure is classified and passed to the Meta-Debugger.

### Meta-Debugging

```text
Failure
   ↓
Failure Taxonomy
   ↓
Meta-Debugger
   ↓
Workflow Intervention
```

### Attempt 2

```text
minimal     → PASS
robust      → PASS
test_first  → PASS
```

The complete workflow therefore becomes:

```text
┌──────────────────────┐
│       Task           │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│      Planner         │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│  Repository Analyst  │
└──────────┬───────────┘
           ↓
┌───────────────────────────────┐
│ Parallel Implementation       │
│                               │
│ minimal │ robust │ test_first │
└──────────────┬────────────────┘
               ↓
        ┌──────────────┐
        │ Verification │
        └───────┬──────┘
                ↓
           ┌─────────┐
           │  Judge  │
           └────┬────┘
                ↓
        ┌───────────────┐
        │ Failure?      │
        └───────┬───────┘
                │
             YES│
                ↓
       ┌─────────────────┐
       │ Failure         │
       │ Classification  │
       └────────┬────────┘
                ↓
       ┌─────────────────┐
       │ Meta-Debugger   │
       └────────┬────────┘
                ↓
       ┌─────────────────┐
       │ Harness         │
       │ Intervention    │
       └────────┬────────┘
                ↓
             Retry
                ↓
            Verification
                ↓
              PASS
```

---

# Architecture

```text
                         ┌──────────────┐
                         │     Task     │
                         └──────┬───────┘
                                ↓
                         ┌──────────────┐
                         │    Planner   │
                         └──────┬───────┘
                                ↓
                      ┌───────────────────┐
                      │  Repo Analyst     │
                      └─────────┬─────────┘
                                ↓
              ┌────────────────────────────────┐
              │       Agent Trajectories       │
              │                                │
              │  ┌────────┐ ┌────────┐ ┌─────┐│
              │  │minimal │ │ robust │ │test ││
              │  │        │ │        │ │first││
              │  └────┬───┘ └────┬───┘ └──┬──┘│
              └───────┼──────────┼────────┼────┘
                      ↓          ↓        ↓
                 ┌──────────────────────────┐
                 │       Verification       │
                 └─────────────┬────────────┘
                               ↓
                         ┌───────────┐
                         │   Judge   │
                         └─────┬─────┘
                               ↓
                    ┌────────────────────┐
                    │ Failure Taxonomy  │
                    └─────────┬──────────┘
                              ↓
                     ┌────────────────┐
                     │ Meta-Debugger  │
                     └────────┬───────┘
                              ↓
                    ┌────────────────────┐
                    │ Workflow           │
                    │ Intervention       │
                    └─────────┬──────────┘
                              ↓
                            Retry
```

---

# Project Structure

```text
AgentForge/
│
├── agents/
│   ├── base.py
│   ├── planner.py
│   ├── repo_analyst.py
│   ├── coder.py
│   ├── judge.py
│   └── meta_debugger.py
│
├── harness/
│   ├── orchestrator.py
│   └── demo.py
│
├── tools/
│   ├── filesystem.py
│   ├── code_search.py
│   ├── commands.py
│   └── git_tools.py
│
├── verification/
│   ├── verifier.py
│   └── requirements.py
│
├── evaluation/
│   ├── benchmark.py
│   ├── metrics.py
│   ├── failure_taxonomy.py
│   └── run_benchmark.py
│
├── tracing/
│   └── logger.py
│
├── storage/
│   └── db.py
│
├── api/
│   └── main.py
│
├── ui/
│   └── dashboard.py
│
├── benchmarks/
│   └── feature_001/
│       └── task.json
│
├── sample_repo/
│   ├── app/
│   ├── tests/
│   ├── requirements.txt
│   └── pyproject.toml
│
├── tests/
│   ├── test_tools.py
│   ├── test_orchestrator.py
│   └── test_api.py
│
├── configs/
│   ├── baseline.yaml
│   ├── verified.yaml
│   ├── parallel.yaml
│   └── metadbg.yaml
│
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── README.md
```

---

# Installation

Clone the repository and create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

### macOS/Linux

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Run the Tests

```bash
python -m pytest -q
```

The test suite covers the core tools, orchestrator, and API.

---

# Run the AgentForge Demo

```bash
python -m harness.demo
```

A successful recovery demonstration looks like:

```text
AgentForge demo
Run: 4cd7b1434236
Status: PASS
Attempts: 2
  minimal: PASS
  robust: PASS
  test_first: PASS

Meta-debugger: {
  'root_cause': 'workflow did not inspect tests and edge cases before implementation',
  'failure_class': 'workflow.bad_recovery_strategy',
  'interventions': [
    'inspect existing tests before editing',
    'enumerate edge cases',
    'run the complete test suite before declaring success'
  ]
}
```

This is the primary end-to-end demonstration of AgentForge.

---

# Run the Benchmark

```bash
python -m evaluation.run_benchmark
```

The benchmark infrastructure executes defined engineering tasks and records their outcomes.

---

# Run the API

```bash
uvicorn api.main:app --reload
```

The API is available locally through the FastAPI application.

---

# Run the Dashboard

```bash
streamlit run ui/dashboard.py
```

The dashboard provides visibility into:

* Run status
* Number of attempts
* Parallel trajectories
* Individual trajectory results
* Failure information
* Meta-Debugger root cause
* Failure classification
* Harness interventions
* Recovery behavior
* Raw run results

The dashboard is designed around the principle that **agent behavior should be observable and debuggable as a system**.

---

# Configuration

AgentForge includes configurations representing different levels of agentic engineering:

```text
configs/
├── baseline.yaml
├── verified.yaml
├── parallel.yaml
└── metadbg.yaml
```

These configurations provide a foundation for comparing workflows with different combinations of verification, parallel trajectories, and meta-debugging.

---

# Research Direction

AgentForge is structured to support experiments around questions such as:

### Does verification improve autonomous coding reliability?

Compare workflows with and without automated verification.

### Does parallel trajectory generation improve success?

Compare a single implementation trajectory against multiple strategies.

### Does meta-debugging improve recovery?

Measure whether workflow-level diagnosis and intervention allow failed trajectories to recover.

### What kinds of failures occur?

Use the failure taxonomy to distinguish:

```text
Planning
Retrieval
Implementation
Verification
Workflow
```

The evaluation layer is intended to collect **measured experimental results rather than assumed improvements**.

---

# Design Philosophy

AgentForge follows several principles:

### 1. Verify, don't assume

An agent saying that its implementation works is not evidence that it works.

Automated verification determines success.

### 2. Compare trajectories

Different implementation strategies can produce different outcomes.

Parallel trajectories make those differences observable.

### 3. Treat failures as data

A failed trajectory is not simply discarded.

Its failure becomes an input to diagnosis and evaluation.

### 4. Debug the workflow

When repeated failures occur, investigate:

* prompts
* tools
* repository search
* task decomposition
* verification
* recovery strategy
* workflow constraints

rather than blindly repeating the same implementation.

### 5. Measure the system

Agentic engineering should ultimately be evaluated through reproducible experiments and observed outcomes.

---

# Inspiration

AgentForge is inspired by **Ramón Medrano Llamas' *Industrial Agentic Engineering* (2026)**.

The article describes **Agentic Engineering** as a discipline in which LLMs operate as semi-autonomous systems executing complex, multi-step workflows or **trajectories** based on verifiable specifications.

The work particularly motivates the ideas explored in this project:

* maximizing autonomous runtime
* multiplying engineering impact through parallel work
* using a strong **harness** around the model
* relying on automated verification rather than micro-controlling generated code
* treating failures as opportunities for **meta-debugging**
* improving the agentic workflow itself rather than repeatedly fixing individual generated outputs

**Reference**

Medrano Llamas, Ramón. *Industrial Agentic Engineering*. 2026.

---

# Status

AgentForge currently provides a working end-to-end demonstration of:

```text
Task
 ↓
Planning
 ↓
Repository Analysis
 ↓
Parallel Trajectories
 ↓
Verification
 ↓
Failure Classification
 ↓
Meta-Debugging
 ↓
Harness Intervention
 ↓
Retry
 ↓
PASS
```

The core demonstration intentionally produces a trajectory failure and shows the system recovering through a **Meta-Debugger-driven workflow intervention**.
