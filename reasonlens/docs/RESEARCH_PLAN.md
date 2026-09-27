# ReasonLens — Research Plan

## Motivation

Mobile apps routinely ask for runtime permissions (location, camera,
contacts, etc.) accompanied by a short explanatory string chosen by the
developer. That explanation ranges from absent, to vague, to explicitly
about advertising, to genuinely functional, to privacy-preserving. It is
not well understood how much the *content* of that explanation shapes
(a) whether users grant the permission, (b) what precision level they
choose when Android offers a choice, and (c) how much they trust the
app and the platform afterward.

## Research questions

**RQ1 (primary).** Does the *type* of reason given for a location
permission request causally affect the probability that a user grants
the permission, controlling for the type of app making the request?

**RQ2.** Does reason type affect which precision level (approximate vs.
precise) a user selects, conditional on granting?

**RQ3.** Does reason type affect self-reported trust in the app, trust
in the Android permission system, perceived necessity of the request,
and privacy concern?

**RQ4 (secondary, ML-driven).** Independent of the experimentally
assigned reason *category*, does a continuous, text-derived measure of
reason *quality* (specificity, necessity, transparency, disclosure of
advertising use, use of privacy-protective language) predict permission
decisions or trust above and beyond the category itself?

## Design

A 4 (app type) × 5 (reason type) between-subjects factorial design,
reproducibly randomized per participant (see `experiments/randomization.py`
and `docs/EXPERIMENT_PROTOCOL.md`). Primary DV: binary grant/deny
decision. Secondary DVs: precision choice, response time, four Likert
trust/privacy items.

## Analysis plan

- **Primary:** logistic regression, `granted ~ reason_type + app_type`,
  followed by the interaction model `granted ~ reason_type * app_type`
  (see `analysis/regression.py`). Report odds ratios with 95% CIs.
- **Secondary:** chi-square tests of independence for categorical
  associations, one-way ANOVA for response time and trust items by
  condition (see `analysis/statistics.py`).
- **Exploratory (ML):** a reason-quality model trained on labeled reason
  text (see `ml/`) is used to test RQ4 by regressing decisions/trust on
  the model's continuous quality score instead of the categorical
  reason_type.

## Modes

The exact same pipeline (`experiments/run_all.py`) runs in two modes:

- **simulation** — synthetic, reproducible data for developing and
  testing the pipeline before any real participants are recruited.
- **pilot** — real participant data collected via the Android app and
  FastAPI backend, gated on the in-app consent screen.

Simulated and real data are never mixed; every row is tagged with its
`mode` at the database level (see `backend/app/models.py::Participant.mode`).
