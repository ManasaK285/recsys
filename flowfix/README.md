# FlowFix: Agentic Automated Program Repair

**From failing tests to validated code patches — inspired by low-latency agentic program repair research.**

FlowFix is an end-to-end prototype for automated program repair that turns software test failures into candidate code fixes and validates those fixes before presenting them to a developer.

Inspired by *Catching Developers in the Flow: Low-Latency Agentic Program Repair at Google Scale* (Ziftci et al., 2026), FlowFix explores a central idea from modern software engineering research: **an automated repair system should do more than generate a patch — it should check whether the proposed change actually resolves the failure.**

Rather than treating code generation as the final answer, FlowFix connects repair generation, test-based validation, patch inspection, and repair history in a single workflow.

## The Problem

Software developers frequently encounter failing tests during development. Diagnosing the root cause, identifying the relevant code, implementing a fix, and rerunning tests can interrupt development and delay feedback.

LLM-based coding tools can propose fixes, but a plausible-looking patch is not necessarily correct. A repair system needs a way to validate changes, expose the proposed modifications, and preserve enough information to understand what happened during a repair attempt.

FlowFix addresses this problem by prototyping a repair workflow around three questions:

* **Can a candidate fix be generated from a known test failure?**
* **Does the proposed fix pass the relevant validation tests?**
* **Can a developer inspect the patch and review the repair outcome?**

## How FlowFix Works

FlowFix connects a repair pipeline to a FastAPI backend and a Streamlit dashboard.

1. **Select a repair case.** Start with one of the project's seeded benchmark cases representing a known code defect.
2. **Generate a candidate patch.** The repair provider produces a proposed code change for the selected case.
3. **Validate the candidate.** Run the configured tests to determine whether the candidate satisfies the expected behavior.
4. **Inspect the changes.** Present the proposed patch as a unified diff so developers can review what changed.
5. **Record and report results.** Keep repair-job history and expose benchmark results through the project's reporting workflow.

The intended outcome is a transparent repair process in which a proposed change can be inspected and its test results evaluated, rather than blindly accepted.

## Architecture

```text
                 Streamlit Dashboard
                         |
                         v
                    FastAPI API
                         |
                         v
                    Repair Job
                         |
                         v
                   Repair Provider
                         |
                         v
                  Candidate Patch
                         |
                         v
                  Test Validation
                         |
              +----------+----------+
              |                     |
          Validation            Validation
            passes                fails
              |                     |
              v                     v
         Review Patch          Report Failure
              |
              v
        Record Job and Results
```

The architecture separates the user interface, API, repair provider, and validation workflow so that the repair strategy can evolve independently of the dashboard.

## Key Features

* **Automated repair workflow:** Coordinates candidate patch generation and test-based validation.
* **Repair benchmark:** Includes four seeded repair cases for exercising the workflow.
* **Patch inspection:** Produces unified diffs for reviewing proposed changes.
* **FastAPI service:** Exposes repair-job and benchmark functionality through API endpoints.
* **Interactive dashboard:** Uses Streamlit to make repair operations and results accessible through a visual interface.
* **Repair history:** Uses SQLite to persist repair-job information.
* **Structured reporting:** Supports JSON and CSV result exports.
* **Automated tests:** Uses pytest to validate implemented application behavior.
* **Continuous integration:** Includes GitHub Actions support for automated checks.

## Research Inspiration

FlowFix is inspired by the research problem addressed in the following paper:

**Catching Developers in the Flow: Low-Latency Agentic Program Repair at Google Scale**

Celal Ziftci, Spencer Greene, Ray Liu, Livio Dalloro, and Lorenzo Dini. 2026.

* Paper: https://arxiv.org/abs/2610.07289
* DOI: https://doi.org/10.48550/arXiv.2610.07289
* Conference: ASE 2026, Industry Showcase

The paper presents FlowAgent, a system designed to repair test failures during Google's pre-submit development workflow. Its approach uses a ReAct-style generate-and-validate loop, with filters intended to avoid unsuitable repairs and prevent stale or low-quality suggestions from reaching developers.

The reported evaluation includes 67.18% accuracy in suggesting correct fixes across 195 real-world test failures. The paper also reports 28,554 developer-applied fixes following deployment at Google.

FlowFix draws inspiration from the paper's emphasis on:

* **Generate-and-validate repair:** Treating test validation as part of the repair process.
* **Developer-centered feedback:** Making proposed changes inspectable.
* **Repair workflow integration:** Connecting repair execution and results through a usable interface.
* **Measurable outcomes:** Keeping repair results and benchmark outputs available for evaluation.

### How FlowFix Relates to the Paper

FlowFix is an independent prototype inspired by these research principles, not a reproduction of Google's internal FlowAgent.

The current repair provider uses predefined fixes for the four seeded benchmark cases. It does not yet constitute a general-purpose LLM agent capable of repairing arbitrary repositories. The project provides a foundation for experimenting with more advanced patch-generation strategies, bounded repair iterations, and stronger execution isolation.

The paper's reported production metrics belong to the authors' system and must not be interpreted as FlowFix performance results.

## Technology Stack

| Component      | Technology     |
| -------------- | -------------- |
| Language       | Python         |
| Interactive UI | Streamlit      |
| API layer      | FastAPI        |
| ASGI server    | Uvicorn        |
| Repair history | SQLite         |
| Test framework | pytest         |
| CI automation  | GitHub Actions |

## Running Locally

### Start the API

From the project directory:

```powershell
python -m uvicorn api:app --reload
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

### Start the dashboard

In a second PowerShell terminal:

```powershell
python -m streamlit run app.py
```

Open the dashboard:

http://localhost:8501

Install the dependencies declared by the project before starting either service.

### Run the tests

```powershell
python -m pytest -v
```

The initial local run completed with **5 passing tests in 18.13 seconds**. This is a recorded development result, not a claim of complete end-to-end or production validation.

## Current Limitations and Next Steps

* Replace predefined benchmark fixes with an LLM-backed patch-generation provider.
* Add repository-aware context gathering and failure-log analysis.
* Enforce timeouts and stronger sandbox isolation when executing candidate patches.
* Add safeguards against patches that simply remove tests or bypass expected behavior.
* Evaluate repair correctness, test-pass rate, execution time, and failure modes across a broader benchmark.
* Add end-to-end tests for the dashboard-to-API repair workflow.

## References

1. Ziftci, C., Greene, S., Liu, R., Dalloro, L., & Dini, L. (2026). *Catching Developers in the Flow: Low-Latency Agentic Program Repair at Google Scale*. arXiv:2610.07289. https://doi.org/10.48550/arXiv.2610.07289

2. FastAPI. *FastAPI Documentation*. https://fastapi.tiangolo.com/

3. Streamlit. *Streamlit Documentation*. https://docs.streamlit.io/

4. pytest. *pytest Documentation*. https://docs.pytest.org/en/stable/

5. Uvicorn. *Uvicorn Documentation*. https://www.uvicorn.org/

---

**Project goal:** Explore how agentic program repair can move from generating plausible code changes to producing test-validated, reviewable fixes within a developer-oriented workflow.
