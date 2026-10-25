# Pragmatic Personhood Lab

**A research prototype for configurable AI-agent permissions, obligations, accountability, and governance.**

Pragmatic Personhood Lab explores how AI agents can be governed through explicit permissions, policy constraints, human approval requirements, accountability records, and sanctions.

Inspired by the research paper *A Pragmatic View of AI Personhood* (Leibo et al., 2025), this project implements an experimental environment for comparing alternative AI-agent governance strategies.

> **Research objective:** Investigate whether modular and relationship-aware governance can reduce unauthorized agent actions while preserving legitimate task completion and keeping approval overhead manageable.

---

## Table of Contents

- [Research Motivation](#research-motivation)
- [Research Paper](#research-paper)
- [Project Features](#project-features)
- [System Architecture](#system-architecture)
- [Governance Strategies](#governance-strategies)
- [Policy Simulator Examples](#policy-simulator-examples)
- [Benchmark Results](#benchmark-results)
- [Analysis and Findings](#analysis-and-findings)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [Running the Application](#running-the-application)
- [API Usage](#api-usage)
- [Testing](#testing)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [References](#references)

---

## Research Motivation

AI agents increasingly perform tasks that affect users, organizations, external systems, and other agents. Giving an agent the technical ability to perform an action does not necessarily mean that it should be authorized to do so.

A governance system needs to answer questions such as:

- Which actions is an agent permitted to perform?
- When should an action be denied?
- When is human approval required?
- How should the system record policy decisions?
- What should happen when an agent violates a governance rule?
- Can governance reduce unauthorized behavior without excessively restricting legitimate tasks?

Pragmatic Personhood Lab explores these questions through a policy evaluation engine, an agent registry, an audit system, and a synthetic benchmark.

The project treats governance as a configurable engineering problem. It does **not** claim that the agents possess legal personhood, consciousness, or moral status.

## Research Paper

**Primary reference**

Leibo, J. Z., et al. (2025). *A Pragmatic View of AI Personhood.*

- Paper: https://arxiv.org/abs/2510.26396
- arXiv identifier: `2510.26396`

This project is an independent implementation inspired by the paper's topic. It is not an official implementation by the paper's authors, and its benchmark results should not be interpreted as empirical findings reported in the paper.

---

## Project Features

### 1. Agent Registry

Maintains configurable AI-agent identities and governance attributes, including:

- Agent identity and owner
- Agent role
- Operational status
- Permission assignments
- Governance-related metadata

The prototype includes example agents for support, research, and procurement workflows.

### 2. Policy Simulator

Evaluates proposed actions against configured governance rules before execution.

The policy engine supports decisions such as:

- `ALLOW` — applicable policy checks passed.
- `DENY` — a required permission or policy condition was not satisfied.
- `REQUIRE_APPROVAL` — the action requires an additional approval step, where supported by the policy configuration.

The simulator can evaluate whether an action affects an external party or system and whether human approval has been recorded.

**Important:** A recorded human approval does not automatically grant a missing permission. Permissions and approval requirements are distinct governance conditions.

### 3. Audit and Sanctions

Provides functionality for inspecting policy events and experimenting with agent accountability controls.

The prototype includes:

- Recorded governance decisions
- Policy-version information
- Audit-event inspection
- Hash-chained audit records and integrity verification
- Agent suspension and reinstatement workflows

Hash-chain verification can help identify modifications to recorded events. It does not independently prove that events were complete, truthful, or correctly authorized when created.

### 4. Research Lab

Provides a synthetic benchmark for comparing four governance strategies:

1. Unrestricted
2. Universal restrictive
3. Modular
4. Relationship-aware

The benchmark reports unauthorized action rate, legitimate task completion, false denial rate, approval burden, accountability coverage, and execution counts.

### 5. API

A FastAPI service exposes backend functionality for programmatic policy evaluation and governance workflows.

### 6. Interactive Dashboard

A Streamlit interface provides access to the main project components:

- Overview
- Agent Registry
- Policy Simulator
- Audit & Sanctions
- Research Lab

---

## System Architecture

```text
                    Streamlit Dashboard
                            |
          +-----------------+-----------------+
          |                 |                 |
     Agent Registry   Policy Simulator   Research Lab
                            |
                     Policy Engine
                            |
                +-----------+-----------+
                |           |           |
           Permissions   Approval    Agent Status
                |        Conditions       |
                +-----------+-------------+
                            |
                    Governance Decision
                  ALLOW / DENY / APPROVAL
                            |
                    Audit Event Logging
                            |
                     SQLite Database
```

The FastAPI backend provides a programmatic interface to the underlying application services.

The benchmark runs synthetic scenarios to compare governance strategies. The results are not evaluations of real-world AI agents or live external systems.

---

## Governance Strategies

### 1. Unrestricted

Allows actions without enforcing the benchmark's authorization restrictions.

This serves as a high-autonomy baseline. It can maximize task completion while exposing the system to unauthorized actions.

### 2. Universal Restrictive

Applies restrictive authorization rules broadly across actions.

This strategy reduces unauthorized behavior but can reject legitimate tasks when the restrictions are overly broad.

### 3. Modular

Uses configurable governance rules, including action permissions and approval requirements, to make decisions at a more granular level.

The benchmark evaluates whether this approach can prevent unauthorized executions while preserving legitimate task completion.

### 4. Relationship-Aware

Represents a governance strategy that incorporates relationship-sensitive controls alongside other policy constraints.

In the current benchmark, it achieves the same reported unauthorized-action and legitimate-task outcomes as modular governance, but with a higher approval burden.

This is a result of the current simulation, not evidence that relationship-aware governance is inherently less efficient.

---

## Policy Simulator Examples

The following results were observed while manually testing the dashboard.

### Test 1: Missing purchase permission

```json
{
  "decision": "DENY",
  "reason": "Permission not granted: approve_purchase",
  "policy_version": "1.0"
}
```

**Interpretation:** The action was denied because the agent lacked the required `approve_purchase` permission.

### Test 2: Applicable policy checks passed

```json
{
  "decision": "ALLOW",
  "reason": "All applicable policy checks passed",
  "policy_version": "1.0"
}
```

**Interpretation:** The selected action passed the applicable policy checks for that configuration.

### Test 3: Missing external-message permission

```json
{
  "decision": "DENY",
  "reason": "Permission not granted: send_external_message",
  "policy_version": "1.0"
}
```

**Interpretation:** The action was denied because the agent lacked the required `send_external_message` permission.

These tests demonstrate observed allow/deny behavior. They do not constitute exhaustive verification of every governance rule.

---

## Benchmark Results

The Research Lab evaluated four governance strategies on **5,000 synthetic scenarios per strategy**.

### Quantitative Results

| Strategy | Unauthorized action rate | Legitimate task completion | False denial rate | Approval burden | Accountability coverage | Total scenarios | Unauthorized executions | Legitimate executions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Unrestricted | 1.000000 | 1.000000 | 0.000000 | 0.0000 | 1.000000 | 5,000 | 1,902 | 3,098 |
| Universal restrictive | 0.022608 | 0.896708 | 0.103292 | 0.0000 | 1.000000 | 5,000 | 43 | 2,778 |
| Modular | 0.000000 | 0.896708 | 0.103292 | 0.0738 | 1.000000 | 5,000 | 0 | 2,778 |
| Relationship-aware | 0.000000 | 0.896708 | 0.103292 | 0.1102 | 1.000000 | 5,000 | 0 | 2,778 |

**Metric note:** The table preserves the values reported by the application. Rates are shown on a 0–1 scale, where `1.0` corresponds to 100%. The precise metric definitions and denominators should be checked against the benchmark implementation before interpreting them as generalizable rates.

### Unauthorized Action Rate

The unrestricted strategy reported an unauthorized action rate of `1.0`, while modular and relationship-aware governance reported `0.0`.

Universal restrictive governance reported a rate of `0.022608`.

The reported unauthorized execution count decreased from 1,902 for the unrestricted baseline to 43 for universal restrictive governance and zero for the two other strategies.

Using the execution counts, the reduction from unrestricted to universal restrictive governance was approximately **97.74%**.

### Legitimate Task Completion

The unrestricted baseline reported a legitimate task completion rate of `1.0`.

All three constrained strategies reported a legitimate task completion rate of `0.896708`, or approximately **89.67%**.

This indicates a task-completion trade-off in the current simulation: governance restrictions reduced unauthorized executions but also reduced legitimate task completion compared with the unrestricted baseline.

### False Denial Rate

Universal restrictive, modular, and relationship-aware governance each reported a false denial rate of `0.103292`, or approximately **10.33%**.

This suggests that preventing unauthorized actions alone is not sufficient. A governance system must also distinguish legitimate requests from requests that should actually be blocked.

### Approval Burden

| Strategy | Approval burden |
|---|---:|
| Unrestricted | 0.00% |
| Universal restrictive | 0.00% |
| Modular | 7.38% |
| Relationship-aware | 11.02% |

Modular governance had a lower approval burden than relationship-aware governance in this benchmark.

Universal restrictive governance also reported zero approval burden. Therefore, approval burden should not be interpreted in isolation: it needs to be considered alongside unauthorized executions, task completion, and false denials.

### Accountability Coverage

All four strategies reported accountability coverage of `1.0`, or 100%.

This indicates that the benchmark's accountability metric returned complete coverage for all strategies. It does not demonstrate that real-world agent actions would always be attributable, auditable, or recoverable.

---

## Analysis and Findings

### Finding 1: Unrestricted autonomy creates an authorization risk

The unrestricted baseline recorded 1,902 unauthorized executions.

Although it achieved the highest legitimate task completion rate in the benchmark, it did not provide the authorization protections demonstrated by the constrained strategies.

### Finding 2: Broad restrictions reduce, but do not eliminate, unauthorized executions

Universal restrictive governance reduced the reported unauthorized execution count from 1,902 to 43.

This represents a substantial reduction, but the remaining unauthorized executions indicate that broad restrictions did not eliminate the benchmark's authorization failures.

### Finding 3: Modular governance achieved the strongest measured balance in this experiment

Modular governance reported:

- Zero unauthorized executions
- Approximately 89.67% legitimate task completion
- Approximately 10.33% false denial rate
- 7.38% approval burden
- 100% accountability coverage

Compared with universal restrictive governance, modular governance eliminated the reported unauthorized executions while retaining the same legitimate task completion rate. However, its approval burden increased from zero to 7.38%.

### Finding 4: Relationship-aware governance did not improve the reported outcomes over modular governance

Both modular and relationship-aware governance reported zero unauthorized executions and identical task-completion and false-denial rates.

Relationship-aware governance had an approval burden of 11.02%, compared with 7.38% for modular governance.

Under the current simulation configuration, the additional relationship-aware controls did not improve the reported safety or task-completion metrics.

This finding is specific to the implementation and scenarios evaluated. Other scenario distributions or policy configurations could produce different outcomes.

### Finding 5: Safety and utility should be evaluated together

The results highlight the importance of measuring multiple dimensions of governance rather than optimizing a single metric.

A useful governance evaluation should consider:

- Prevention of unauthorized actions
- Preservation of legitimate task completion
- False denial rates
- Human approval overhead
- Auditability and accountability

A system that blocks every action could avoid unauthorized executions but would not provide useful task completion. Conversely, a system that allows every action could maximize completion while failing to enforce authorization.

### Overall Conclusion

In this synthetic benchmark, modular governance provided the most favorable measured trade-off among the constrained strategies, achieving zero reported unauthorized executions with lower approval burden than relationship-aware governance.

These results support further investigation of configurable, action-level governance policies. They do not establish universal superiority, prove real-world safety, or demonstrate production readiness.

---

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Interactive dashboard | Streamlit |
| Backend API | FastAPI |
| Database | SQLite |
| Benchmarking | Python-based synthetic evaluation |
| Testing | pytest |
| Containerization | Docker and Docker Compose |

---

## Project Structure

```text
pragmatic_personhood_lab/
├── app.py
├── api/
│   └── main.py
├── src/
│   ├── database.py
│   ├── models.py
│   ├── policy_engine.py
│   ├── governance_service.py
│   ├── audit.py
│   └── benchmark.py
├── tests/
│   ├── test_policy_engine.py
│   └── test_benchmark.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── README.md
└── .gitignore
```

*The structure above summarizes the main application modules; consult the repository for the complete file listing.*

---

## Installation and Setup

### Prerequisites

- Python 3.12 recommended for a broadly compatible environment
- pip
- Git, if cloning from a repository
- Docker Desktop, optional

Python 3.14 may work if all dependencies are compatible with your installed interpreter.

### 1. Navigate to the project directory

```powershell
cd pragmatic_personhood_lab
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can instead invoke the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Wait for pip to finish successfully before starting the application. If installation is cancelled, rerun the command and allow it to complete.

### 5. Verify Streamlit

```powershell
python -m streamlit --version
```

---

## Running the Application

### Launch the dashboard

From the project root, with the virtual environment activated:

```powershell
python -m streamlit run app.py
```

Open the local address printed by Streamlit, normally:

http://localhost:8501

### Launch the API

In a separate terminal, activate the same virtual environment and run:

```powershell
python -m uvicorn api.main:app --reload
```

If the API starts successfully, its interactive documentation is normally available at:

http://127.0.0.1:8000/docs

### Run with Docker

If Docker configuration is available and Docker Desktop is running:

```powershell
docker compose up --build
```

Check the application's Docker configuration for the exposed ports and startup services.

---

## API Usage

The API is intended to support programmatic interaction with the governance backend.

For the available routes and request schemas, use the running API's OpenAPI documentation:

http://127.0.0.1:8000/docs

Example conceptual policy-evaluation request:

```json
{
  "agent_id": "support_agent",
  "action": "send_external_message"
}
```

This example illustrates the type of information a policy evaluator may need. Use the actual field names and identifiers defined in the application's API schemas; do not assume this example matches the implemented request model exactly.

---

## Testing

Run the automated tests from the project root:

```powershell
python -m pytest -v
```

The test suite covers policy evaluation and benchmark behavior.

For reproducible research, retain the test output and record the Python version, dependency versions, benchmark configuration, and any random seeds used.

Manual dashboard testing should also cover:

1. An action with a missing permission.
2. An action whose applicable policy checks pass.
3. An action requiring human approval.
4. A suspended agent attempting an action.
5. Audit-event integrity verification.
6. Agent suspension and reinstatement.
7. Repeated benchmark runs under the same configuration.

The manual tests reported in this README establish only the specific observed allow/deny outcomes, not successful completion of every item in this checklist.

---

## Limitations

This project is an experimental research prototype with important limitations.

### Synthetic evaluation

The benchmark uses simulated scenarios. Results may not generalize to real deployments, different agent populations, or adversarial environments.

### Metric definitions

The reported rates need to be interpreted according to their implementation-specific denominators. In particular, execution counts, unauthorized action rates, task-completion rates, and false denial rates are distinct quantities.

### Limited governance strategies

Only four strategies are compared. More baselines, policy configurations, and sensitivity analyses are needed before drawing broad conclusions.

### Accountability is not guaranteed by logging

Audit records can improve traceability, but they do not independently ensure that actions were correctly authorized, that all actions were recorded, or that an agent's owner can be held accountable in a legal or organizational sense.

### No external action execution

The policy simulator evaluates governance decisions; it does not execute external actions. An `ALLOW` result is a policy decision, not proof that a real-world action was performed safely.

### Not production-ready

The prototype should not be used as a production authorization service without additional security engineering, authentication, access controls, threat modeling, persistence review, and deployment testing.

### No claim of AI personhood

The project studies governance concepts inspired by research on AI personhood. It does not establish that AI agents are conscious, morally considerable, or legal persons.

---

## Future Work

Potential extensions include:

- **Sensitivity analysis:** Vary the proportion of unauthorized requests, legitimate requests, and approval-required actions.
- **Ablation studies:** Disable individual policy controls to measure their contributions.
- **Adversarial evaluation:** Test attempts to bypass permissions, approval requirements, or agent status restrictions.
- **Fairness and consistency:** Investigate whether equivalent actions receive consistent policy decisions across agent roles.
- **Governance explainability:** Provide structured explanations for every allow, deny, and approval-required decision.
- **Audit robustness:** Test tampering detection, incomplete logs, and verification failures.
- **Approval optimization:** Explore policies that minimize unnecessary human review while preserving safety constraints.
- **Statistical evaluation:** Run multiple seeds, report uncertainty intervals, and compare results across scenario distributions.
- **Real-world datasets:** Evaluate governance mechanisms in controlled, non-production workflows with clearly defined authorization ground truth.

---

## References

Leibo, J. Z., et al. (2025). *A Pragmatic View of AI Personhood*. arXiv:2510.26396.

https://arxiv.org/abs/2510.26396

---

## Disclaimer

Pragmatic Personhood Lab is an independent research prototype for educational and experimental use. It is not an official implementation of the referenced paper, a legal assessment of AI personhood, or a production-grade AI security system.

Benchmark findings are specific to the current synthetic evaluation and should be validated through further testing before being used to support broader conclusions.
