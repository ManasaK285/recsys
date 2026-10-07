# POLCA-Lab

## Stochastic Generative Optimization for Prompts, RAG, and Agents

POLCA-Lab is an end-to-end research framework inspired by **POLCA: Stochastic Generative Optimization with LLM**, a 2026 Google Research project.

The goal is to study how an optimizer can use an LLM or policy model to search through complex AI-system configurations when evaluations are:

* noisy,
* expensive,
* stochastic,
* partially observable,
* and difficult to optimize with traditional gradient-based methods.

Instead of directly optimizing model weights, POLCA-Lab optimizes **system-level configurations and policies**.

---

## What Does POLCA-Lab Do?

POLCA-Lab treats an AI system as a black-box optimization problem.

Given a candidate configuration:

```text
Candidate
   ↓
AI System
   ↓
Stochastic Evaluation
   ↓
Reward + Feedback
```

the optimizer learns which configurations are promising and generates new candidates.

The optimization process maintains a memory of previous experiments and combines:

1. **Priority-based exploration/exploitation**
2. **ε-Net diversity enforcement**
3. **Historical contextual summarization**
4. **Stochastic evaluation**
5. **Multi-seed research evaluation**

---

# Four Research Phases

The project combines four phases into a single framework.

## Phase 1 — Prompt Optimization

Optimizes prompt-level behavior such as:

* correctness
* relevance
* safety
* completeness
* conciseness
* uncertainty handling

Example:

```text
Candidate A
correctness = 0.8
relevance   = 0.7
safety      = 0.9
...

        ↓

POLCA optimizer

        ↓

Candidate B
correctness = 0.9
relevance   = 0.85
...
```

The objective is to discover configurations that produce better evaluation rewards.

---

# Phase 2 — RAG Optimization

POLCA-Lab treats the RAG pipeline as an optimization space.

Parameters include:

* chunk size
* chunk overlap
* top-k retrieval
* reranking
* query rewriting
* citation behavior
* grounding behavior
* uncertainty handling

Conceptually:

```text
Question
   ↓
Query Processing
   ↓
Retriever
   ↓
Reranker
   ↓
Context
   ↓
LLM
   ↓
Answer
   ↓
Evaluation
```

POLCA searches this configuration space instead of manually choosing a single configuration.

---

# Phase 3 — Agent / Tool Optimization

The framework can also optimize agent behavior.

Candidate policies include:

* planning
* tool selection
* tool ordering
* verification
* retries
* termination
* escalation
* parallel execution

Example:

```text
User Task
   ↓
Planner
   ↓
Tool Selection
   ↓
Tool Execution
   ↓
Verification
   ↓
Retry / Escalate
   ↓
Final Answer
```

The optimizer searches for policies that improve the overall task reward.

---

# Phase 4 — Research Evaluation

The fourth phase evaluates the optimization algorithm itself.

POLCA-Lab supports comparisons between:

| Method        | Purpose                             |
| ------------- | ----------------------------------- |
| Random Search | Baseline exploration                |
| Hill Climbing | Local optimization baseline         |
| UCB           | Priority-based exploration baseline |
| Full POLCA    | Complete proposed method            |
| No ε-Net      | Tests diversity mechanism           |
| No Summary    | Tests historical-context mechanism  |

Experiments use:

* multiple random seeds
* controlled evaluation budgets
* stochastic noise
* reward trajectories
* AUC
* evaluation efficiency
* diversity measurements
* duplicate rejection statistics

---

# Core POLCA Loop

The main optimization architecture is:

```text
                 ┌──────────────────────┐
                 │ Candidate Memory     │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ Priority Selection   │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ Stochastic Evaluator │
                 └──────────┬───────────┘
                            ↓
                    Reward + Feedback
                            ↓
                 ┌──────────────────────┐
                 │ Proposal Generator   │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ ε-Net Diversity      │
                 │ Filter               │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ Candidate Memory     │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ Historical Summary   │
                 └──────────┬───────────┘
                            ↓
                       Next Round
```

This creates an iterative optimization loop.

---

# Why the ε-Net Matters

Without diversity control, an optimizer can repeatedly generate candidates that are almost identical.

For example:

```text
Candidate 1 → 0.81
Candidate 2 → 0.812
Candidate 3 → 0.811
Candidate 4 → 0.813
Candidate 5 → 0.812
```

The search wastes evaluations around the same region.

The ε-Net mechanism encourages candidates to remain sufficiently different:

```text
Candidate A ─────── Candidate B
       \                 /
        \               /
         Candidate C
```

This encourages exploration of different parts of the search space.

---

# Historical Summarization

POLCA-Lab also maintains a compact summary of previous optimization experience.

Instead of treating every previous experiment independently:

```text
Trial 1
Trial 2
Trial 3
...
Trial N
```

the optimizer can derive a higher-level signal:

```text
Historical Experience
        ↓
Summary / Meta-Signal
        ↓
Next Candidate Proposal
```

This allows future proposals to be informed by previous successful and unsuccessful configurations.

---

# Stochastic Evaluation

The evaluator intentionally supports noisy rewards.

A configuration may therefore produce:

```text
Run 1 → 0.81
Run 2 → 0.77
Run 3 → 0.84
Run 4 → 0.79
```

rather than a perfectly deterministic score.

This better represents real AI systems where evaluation results can vary because of:

* model stochasticity
* sampling
* retrieval variation
* data batches
* tool execution
* environmental conditions

---

# Controlled Research Design

An important design goal is **fair comparison**.

All optimization variants should use:

* the same initial candidate
* the same random seeds
* the same evaluation budget
* the same evaluation function
* the same noise level
* the same stopping criteria

For example:

```text
60 evaluations / seed

Random Search       → 60
Hill Climbing       → 60
UCB                 → 60
Full POLCA          → 60
No ε-Net            → 60
No Summary          → 60
```

This prevents one method from appearing better simply because it received more evaluations.

---

# Metrics

POLCA-Lab records several research metrics.

### Best Reward

Highest reward discovered during optimization.

### AUC

Area under the best-so-far reward curve.

This captures not only the final result but also how quickly the optimizer improves.

### Evaluations to Threshold

For example:

```text
evaluations_to_80%
evaluations_to_90%
```

This measures sample efficiency.

### Diversity

Measures how different discovered candidates are from one another.

### Duplicate Rejections

Counts candidates rejected by the diversity mechanism.

### Memory Size

Tracks how many candidates remain in the optimizer's memory.

### Multi-Seed Statistics

Experiments are repeated across multiple random seeds to estimate variability.

---

# Current Experimental Results

The initial three-phase experiment successfully ran across:

```text
Prompt
RAG
Agent
```

using multiple seeds.

Example mean best-test rewards from the initial synthetic benchmark:

| Phase  | Mean Best Test |
| ------ | -------------: |
| Prompt |         0.7692 |
| RAG    |         0.7145 |
| Agent  |         0.8358 |

These numbers demonstrate that the end-to-end optimization framework executes successfully.

They should **not** be interpreted as evidence that POLCA outperforms published benchmarks or real-world optimization methods.

The current evaluator is primarily a research/test harness.

---

# Important Research Finding

The first ablation experiment exposed an experimental-design issue.

The initial `No-ε-Net` configuration was allowed to perform substantially more evaluations than the full method.

Therefore, its higher reward could not be fairly attributed to removing ε-net.

The corrected research design uses:

```text
Equal evaluation budget
        +
Same seeds
        +
Same initialization
        +
Same stochastic evaluator
```

This makes future ablation results substantially more interpretable.

The experiments also revealed that the summarizer needs to explicitly influence candidate generation; otherwise `Full POLCA` and `No Summary` can behave similarly.

This led to the controlled v4 experiment design.

---

# Project Structure

```text
POLCA-Lab/
│
├── polca_lab/
│   ├── __init__.py
│   │
│   ├── core/
│   │   ├── candidate.py
│   │   ├── memory.py
│   │   ├── priority.py
│   │   ├── diversity.py
│   │   ├── evaluator.py
│   │   └── optimizer.py
│   │
│   ├── phases/
│   │   ├── prompt.py
│   │   ├── rag.py
│   │   └── agent.py
│   │
│   ├── research/
│   │   ├── controlled.py
│   │   ├── ablations.py
│   │   └── metrics.py
│   │
│   └── oracles/
│       └── llm.py
│
├── scripts/
│   ├── run_unified.py
│   ├── run_research_suite.py
│   └── run_all.py
│
├── tests/
│
├── results/
│
├── PHASES.md
├── EXPERIMENT_REPORT.md
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Installation

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

If the project uses editable installation:

```powershell
python -m pip install -e .
```

---

# Run All Four Phases

```powershell
python scripts/run_unified.py --phase all --iterations 25 --seeds 3
```

This runs:

```text
Prompt
   ↓
RAG
   ↓
Agent
   ↓
Research evaluation
```

---

# Run the Research Suite

```powershell
python scripts/run_research_suite.py
```

The experiment produces structured JSON results containing:

```text
reward
best_test
AUC
evaluations
evaluations_to_80
evaluations_to_90
diversity
duplicate_rejections
memory_size
seed
variant
phase
```

---

# Run Tests

```powershell
python -m unittest discover -s tests -v
```

---

# Research Extensions

The current framework is designed to make future research experiments straightforward.

Potential extensions include:

### Real LLM Optimization

Replace the synthetic proposal mechanism with:

* Gemini
* GPT
* Claude
* open-source LLMs

### Real RAG Benchmarks

Integrate datasets such as:

* HotpotQA
* Natural Questions
* retrieval benchmarks
* domain-specific QA datasets

### Real Agent Benchmarks

Integrate:

* τ-bench
* tool-use environments
* WebArena-style tasks
* custom enterprise workflows

### Published POLCA Benchmarks

For a publication-quality reproduction, connect the framework to the benchmark environments used in the POLCA paper, including:

* τ-bench
* HotpotQA
* VeriBench
* KernelBench

and reproduce the corresponding evaluation protocols.

---

# Research Questions

POLCA-Lab can be used to investigate questions such as:

### Q1 — Does diversity improve optimization?

Compare:

```text
Full POLCA
vs
No ε-Net
```

under an identical evaluation budget.

### Q2 — Does historical summarization improve search?

Compare:

```text
Full POLCA
vs
No Summary
```

while keeping all other conditions fixed.

### Q3 — How does POLCA behave under noise?

Run:

```text
noise = 0.00
noise = 0.02
noise = 0.05
noise = 0.10
noise = 0.20
```

and compare degradation.

### Q4 — How sample-efficient is POLCA?

Compare AUC and evaluations-to-threshold across methods.

### Q5 — Does the method generalize across optimization spaces?

Run the same optimizer over:

```text
Prompt → RAG → Agent
```

without redesigning the core optimization algorithm.

---

# Research Positioning

The key idea behind POLCA-Lab is:

> **Use an LLM or policy model as a generator of candidate solutions while maintaining an optimization memory that balances exploration, exploitation, diversity, and historical experience.**

This creates a bridge between:

```text
LLM Agents
     +
Black-Box Optimization
     +
Stochastic Optimization
     +
RAG
     +
Tool-Using Agents
```

rather than optimizing model weights directly.

---

# Limitations

The current synthetic evaluator is intentionally lightweight.

Therefore:

* synthetic reward values are not real-world benchmark scores;
* experiments do not establish superiority over published methods;
* real LLM/API evaluations may have substantially different cost and variance;
* benchmark-specific implementations are required for publication-quality comparisons.

The framework should therefore be viewed as an **experimental research platform and prototype** until validated against the actual benchmark environments.

---

# Future Direction

The next major milestone is:

```text
POLCA-Lab
    ↓
Controlled Synthetic Experiments
    ↓
Real Benchmark Integration
    ↓
Multi-Seed Evaluation
    ↓
Ablation Study
    ↓
Statistical Analysis
    ↓
Research Report / Paper
```

The long-term objective is to determine how effectively stochastic generative optimization can improve complex AI systems without requiring direct gradient access to the system being optimized.

---

## Summary

POLCA-Lab provides a single framework for optimizing:

```text
┌──────────────────┐
│ Prompt Systems   │
├──────────────────┤
│ RAG Systems      │
├──────────────────┤
│ Agent Systems    │
├──────────────────┤
│ Research Methods │
└──────────────────┘
```

using:

```text
Priority Search
      +
Stochastic Evaluation
      +
ε-Net Diversity
      +
Historical Summarization
      +
Multi-Seed Research
```

The result is an extensible research platform for studying **LLM-driven stochastic optimization across multiple classes of AI systems**.
# Reference

This project is inspired by and designed as an experimental implementation/extension framework around:

> **Xuanfei Ren, Allen Nie, Tengyang Xie, Ching-An Cheng. “POLCA: Stochastic Generative Optimization with LLM.” 2026.**

The original work formulates optimization of complex systems as **stochastic generative optimization**, where a generative language model acts as the optimizer and uses numerical rewards and textual feedback to discover improved system configurations. The proposed POLCA framework combines a priority queue for exploration/exploitation, an ε-Net mechanism for maintaining parameter diversity, and an LLM Summarizer for learning from historical trials.

**Official paper:**
[Google Research — POLCA: Stochastic Generative Optimization with LLM](https://research.google/pubs/polca-stochastic-generative-optimization-with-llm/?utm_source=chatgpt.com)

**Official code repository:**
[POLCA GitHub Repository](https://github.com/rlx-lab/POLCA?utm_source=chatgpt.com)

## Relationship to the Original Paper

POLCA-Lab is **not presented as an exact reproduction of the original paper**.

Instead, this project uses the paper's core optimization concepts as the foundation for a unified experimental framework and extends them across:

* Prompt optimization
* RAG optimization
* Agent/tool optimization
* Controlled ablation and research evaluation

The project also introduces a controlled experimental setup designed to isolate the contributions of components such as diversity enforcement and historical summarization.

For publication-quality reproduction, the framework should ultimately be evaluated using the original benchmark environments and protocols described by Ren et al., including τ-bench, HotpotQA, VeriBench, and KernelBench.

## BibTeX

```bibtex
@article{ren2026polca,
  title={POLCA: Stochastic Generative Optimization with LLM},
  author={Ren, Xuanfei and Nie, Allen and Xie, Tengyang and Cheng, Ching-An},
  year={2026}
}
```
