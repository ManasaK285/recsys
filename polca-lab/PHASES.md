# Four phases in one project

## Phase 1 — Prompt optimization
Searches system policies for customer-support behavior. The evaluator scores correctness, relevance, safety, completeness, concision, consistency, and policy grounding.

## Phase 2 — RAG optimization
Searches policies for grounded generation. The evaluator measures correctness, grounding, relevance, qualification, completeness, and concision.

## Phase 3 — Agent optimization
Searches tool-use policies. The evaluator measures tool coverage, safety, and execution efficiency.

## Phase 4 — Research evaluation
Runs ablations and robustness experiments over the same optimizer interface. The research metrics include best held-out reward, AUC, evaluations-to-threshold, mean/std across seeds, memory size, and noise sensitivity.

The four phases share the same optimizer, memory, priority mechanism, epsilon-net diversity filter, summarizer, evaluator interface, and oracle interface.
