# CurriculumGuard Governance Report

> Descriptive analysis only. It does not declare a correct policy. If the data are synthetic, results demonstrate the method, not real opinion.

## 1. Does the outcome depend on the aggregation rule?

| scheme | open | constrained | restricted | winner |
|---|---|---|---|---|
| majority | 22.3% | 32.7% | 45.0% | **restricted** |
| balanced | 24.5% | 37.2% | 38.2% | **restricted** |
| population | 34.5% | 32.1% | 33.4% | **open** |
| impact | 24.8% | 33.9% | 41.4% | **restricted** |

Outcome **changes with the aggregation scheme** (majority: restricted, balanced: restricted, population: open, impact: restricted).

## 2. Representation audit

| group | n | participation | population | ratio | status |
|---|---|---|---|---|---|
| student | 72 | 24% | 50% | 0.48 | under |
| parent | 76 | 25% | 30% | 0.84 | ok |
| teacher | 113 | 38% | 15% | 2.51 | over |
| admin | 39 | 13% | 5% | 2.60 | over |

## 3. Participation sensitivity (flip thresholds)

- **parent** (25%): no flip at any share; winner stays `restricted`.
- **teacher** (38% of participants): winner `restricted` changes to `constrained` at 14% (-23.2 pp).
- **admin** (13% of participants): winner `restricted` changes to `constrained` at 32% (+19.5 pp).
- **student** (24% of participants): winner `restricted` changes to `open` at 50% (+26.5 pp).

## 4. Power-aware risk: high-impact concerns of underrepresented groups

None met the high-impact definition (severity >= 4, prevalence >= 25%, underrepresented).

## 5. Policy trade-offs (assumption-based exposure matrix)

| scheme | policy | support | unaddressed-concern burden |
|---|---|---|---|
| majority | open | 22.3% | 0.467 |
| majority | constrained | 32.7% | 0.298 |
| majority | restricted | 45.0% | 0.249 |
| balanced | open | 24.5% | 0.457 |
| balanced | constrained | 37.2% | 0.295 |
| balanced | restricted | 38.2% | 0.263 |
| population | open | 34.5% | 0.426 |
| population | constrained | 32.1% | 0.286 |
| population | restricted | 33.4% | 0.311 |
| impact | open | 24.8% | 0.557 |
| impact | constrained | 33.9% | 0.362 |
| impact | restricted | 41.4% | 0.318 |

## 6. Limitations

- Group shares of the population and policy-exposure values are assumptions; edit them.
- Concern extraction is lexicon-based; validate on hand-labelled real data.
- Evidence notes are starter summaries; replace with vetted sources.
- Small subgroups are suppressed below n=5 and have wide uncertainty.