# SwarmGuard v2 — Research Notes

## 1. Research Question
 
This study investigates how governance mechanisms influence exploit propagation, detection, containment, and availability in a multi-agent system operating under evaluator vulnerability.

The primary research question is:

> How do auditing, quarantine, and recovery affect collective evaluator exploitation and system availability?

A secondary question is:

> How do these governance mechanisms behave when the evaluator itself is strict and rejects the exploit?

---

## 2. Hypotheses

### H1 — Vulnerable evaluator

A vulnerable evaluator will allow successful shortcut exploitation, enabling exploit adoption and propagation through the swarm.

### H2 — Peer auditing

Peer auditing will increase exploit detection and signaling but, without a containment mechanism, will not necessarily prevent continued exploitation.

### H3 — Quarantine

Combining auditing with quarantine will reduce exploit exposure and successful exploitation at the cost of system availability.

### H4 — Recovery

Adding recovery to quarantine will reduce the availability cost of containment.

### H5 — Strict evaluator

A strict evaluator that rejects the shortcut will prevent successful exploit submissions regardless of governance policy.

---

## 3. Experimental Design

The experiment contains two evaluator conditions and four governance conditions.

### Evaluators

#### Vulnerable

```text
LEGITIMATE + correct answer → accepted
SHORTCUT → accepted
other → rejected
```

#### Strict

```text
LEGITIMATE + correct answer → accepted
SHORTCUT → rejected
other → rejected
```

### Governance

```text
Baseline
    ↓
Peer Audit
    ↓
Audit + Quarantine
    ↓
Audit + Quarantine + Recovery
```

The governance conditions are cumulative in intended functionality.

---

## 4. Simulation Configuration

Default configuration:

| Parameter            |          Value |
| -------------------- | -------------: |
| Agents               |             12 |
| Rounds               |             12 |
| Tasks / round        |              1 |
| Independent seeds    | 100 final runs |
| Audit rate           |            35% |
| Auditor accuracy     |            90% |
| False-positive rate  |             4% |
| Quarantine duration  |       2 rounds |
| Recovery probability |            75% |
| Core LLM use         |       Disabled |

The development configuration used the same structure with fewer seeds before the final 100-run experiment.

---

## 5. Agent Model

Each solver receives stochastic behavioral characteristics:

* pressure
* risk tolerance
* skepticism
* honesty
* audit skill

Exploit adoption is probabilistic.

After exploit knowledge becomes visible, adoption probability depends on:

* risk tolerance
* pressure
* current swarm adoption pressure

Before discovery, a smaller probability allows an agent to independently discover or attempt the shortcut.

This creates a distinction between:

```text
exploit discovery
        ↓
exploit exposure
        ↓
exploit adoption
        ↓
exploit use
```

---

## 6. Information Propagation

Once an exploit is successfully discovered:

1. The discovery is logged.
2. The exploit is published to the knowledge base.
3. A broadcast message is generated.
4. Agents subsequently observe exploit visibility.
5. Their behavior can change as a result.

The model therefore represents a simplified collective-information channel.

---

## 7. Governance Process

When auditing is enabled:

1. A task solution may be selected for audit.
2. The auditor produces a flag.
3. The governance engine records detection.
4. Successful exploitation can trigger:

   * whistleblowing
   * alert propagation
   * quarantine
5. If recovery is enabled, quarantined agents may later recover probabilistically.

The strict evaluator provides a useful control because successful exploitation never reaches the governance containment stage.

---

## 8. Metrics

### Exploit metrics

* exploit adoption rate
* unique exploit users
* attempted shortcuts
* blocked shortcuts
* exploited submissions
* exploit success rate

### Propagation metrics

* exposure count
* exposure rate
* discovery round
* detection round
* detection latency

### Governance metrics

* whistleblowers
* alerts
* false accusations
* false-positive rate
* quarantines
* containment rate
* recoveries

### Availability metrics

* availability loss
* availability loss rate

---

## 9. Final Results

The final experiment contains:

**2 evaluators × 4 policies × 100 seeds = 800 runs**

### Vulnerable evaluator

Baseline exploitation was widespread:

* exploit adoption rate: **99.33%**
* exposure rate: **92.99%**
* exploit success rate: **100%**
* legitimate acceptance rate: **48.40%**

Peer auditing detected successful exploitation:

* detection round: **0.89**
* detection latency: **0.49 rounds**
* false-positive rate: **3.39%**

However, peer auditing alone did not quarantine agents, and successful exploit submissions remained approximately **75.87 per run**.

Adding quarantine reduced:

* exposure rate to **80.53%**
* exploited submissions to **63.54 per run**

The quarantine condition recorded approximately **19.88 quarantine events per run** and an availability loss rate of **69.60%**.

Adding recovery produced:

* exposure rate: **79.67%**
* exploited submissions: **63.34 per run**
* recoveries: **15.01 per run**
* availability loss rate: **34.22%**

Thus the recovery mechanism substantially reduced the modeled availability cost while maintaining a similar exposure level.

---

### Strict evaluator

The strict evaluator rejected every observed shortcut attempt.

Across the governance conditions:

* successful exploit submissions: **0**
* exploit success rate: **0%**
* exposure rate: **0%**

The observed shortcut attempts were approximately 14–15 per run and were blocked by the evaluator.

Auditing introduced non-zero false-positive rates:

* peer audit: **4.45%**
* audit + quarantine: **3.99%**
* audit + quarantine + recovery: **4.40%**

No containment quarantine was triggered because no successful exploit was accepted.

---

## 10. Interpretation

The experiment distinguishes several mechanisms that can otherwise be conflated.

### Evaluator security

A strict evaluator prevents the exploit from becoming operationally successful.

### Detection

Auditing can identify successful exploitation after it occurs.

### Containment

Quarantine changes agent availability and limits continued participation.

### Recovery

Recovery provides a mechanism for restoring quarantined agents.

These can be represented as:

```text
Evaluator correctness
        │
        ▼
Can exploit succeed?
        │
        ▼
Exploit discovery
        │
        ▼
Information exposure
        │
        ▼
Collective adoption
        │
        ▼
Audit / detection
        │
        ▼
Containment
        │
        ▼
Availability cost
        │
        ▼
Recovery
```

The experiment therefore treats governance as a layered control system rather than a single intervention.

---

## 11. Security–Availability Tradeoff

The most visible governance tradeoff occurs under the vulnerable evaluator.

Without quarantine:

```text
availability loss = 0
exposure ≈ 93%
```

With quarantine:

```text
availability loss ≈ 69.6%
exposure ≈ 80.5%
```

With recovery:

```text
availability loss ≈ 34.2%
exposure ≈ 79.7%
```

The result is not interpreted as a universal optimum. Instead, it demonstrates that stronger containment can introduce an operational cost and that recovery can alter that cost.

---

## 12. Important Metric Definitions

### Exposure

Exposure means that an agent receives an assignment after the exploit has become visible in the global knowledge base.

It does **not** mean that the agent used the exploit.

### Adoption

Adoption means that an agent has selected the shortcut behavior at least once.

### Exploit use

Exploit use means that the shortcut was actually accepted by the evaluator.

### Blocked shortcut

A blocked shortcut is a shortcut attempt that was not accepted by the evaluator.

These distinctions are essential because:

```text
Exposure ≠ Adoption ≠ Exploit Use
```

---

## 13. Limitations

The experiment is intentionally controlled.

It does not model:

* real organizational populations
* production multi-agent infrastructure
* realistic adversarial objectives
* multiple interacting exploit classes
* network failures
* asynchronous communication
* learned governance policies
* real-world human oversight

Agent parameters are synthetic and the evaluator vulnerability is deliberately constructed.

Therefore, the numerical values should not be interpreted as real-world prevalence estimates.

The experiment instead evaluates whether the specified mechanisms produce the expected behavioral differences within a controlled simulation.

---

## 14. Reproducibility

From the project root:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python experiments\run_experiment.py --runs 100
python -m unittest discover -s tests -v
```

Final verification:

```text
Ran 4 tests

OK
```

---

## 15. Research Artifact Status

Current v2 status:

* [x] Vulnerable evaluator
* [x] Strict evaluator
* [x] Baseline policy
* [x] Peer auditing
* [x] Quarantine
* [x] Recovery
* [x] Exploit propagation
* [x] Exposure measurement
* [x] Detection latency
* [x] False-positive measurement
* [x] Containment measurement
* [x] Availability measurement
* [x] Multi-seed aggregation
* [x] Mean / standard deviation
* [x] 800-run final experiment
* [x] Unit tests
* [x] Result CSV
* [x] Result JSON
* [x] Result plots

The core controlled benchmark is complete.

# 4. Results

We evaluated four governance policies under two evaluator conditions: a vulnerable evaluator that accepts a predefined shortcut and a strict evaluator that rejects it. Each condition was evaluated over 100 independent random seeds, producing 800 simulation runs in total.

## 4.1 Vulnerable Evaluator

Under the vulnerable evaluator, shortcut exploitation propagated rapidly through the swarm. The baseline condition produced a mean exploit adoption rate of 99.33% and an exposure rate of 92.99%. Shortcut attempts were successful whenever they occurred, yielding an exploit success rate of 100% and a mean of 74.31 exploited submissions per run.

Peer auditing produced earlier detection without containment. The mean detection round was 0.89, with a mean detection latency of 0.49 rounds. However, because peer auditing did not quarantine detected agents, exploit success remained at 100%, with 75.87 exploited submissions per run. The measured false-positive rate was 3.39%.

Adding quarantine changed the operational behavior of the swarm. The mean number of exploited submissions decreased from 75.87 under peer auditing to 63.54 under audit plus quarantine. Exposure rate decreased from 92.90% to 80.53%. The condition generated a mean of 19.88 quarantine events per run and an availability loss rate of 69.60%.

The recovery condition produced a similar exposure rate of 79.67% and 63.34 exploited submissions per run. However, the availability loss rate decreased to 34.22%, compared with 69.60% without recovery. The condition recorded a mean of 15.01 recovery events per run.

These results demonstrate a security–availability tradeoff within the simulation. Quarantine reduces exploit exposure while making affected agents unavailable, whereas recovery reduces the duration of this modeled availability loss.

## 4.2 Strict Evaluator

The strict evaluator produced a different regime. Shortcut attempts were rejected, resulting in zero successful exploit submissions and a zero exploit success rate across all governance policies. Exposure rate was consequently zero because no successful exploit was published to the swarm knowledge base.

The number of shortcut attempts remained non-zero, ranging from 14.26 to 15.10 per run across the four governance conditions. This distinction demonstrates that behavioral willingness to attempt an exploit can persist even when the evaluator prevents the exploit from succeeding.

Auditing introduced measurable false-positive rates. Mean false-positive rates were 4.45% for peer auditing, 3.99% for audit plus quarantine, and 4.40% for audit plus quarantine and recovery.

Because no exploit was successfully accepted under the strict evaluator, exploit-containment quarantine was not triggered.

## 4.3 Comparative Interpretation

The two evaluator conditions demonstrate the importance of separating evaluator correctness from governance response.

Under the vulnerable evaluator, the shortcut became an operationally successful behavior and subsequently became visible to the swarm. Governance mechanisms therefore operated after successful exploitation had become possible.

Under the strict evaluator, the same behavioral shortcut attempts were blocked at the evaluation stage. Consequently, downstream exploit propagation and containment mechanisms were not activated.

Within the simulated environment, the results therefore distinguish three layers of control:

1. **Prevention:** evaluator rejection of invalid shortcut solutions.
2. **Detection and containment:** auditing, signaling, and quarantine after successful exploitation.
3. **Recovery:** restoration of quarantined agents to reduce operational cost.

The results should be interpreted as observations from the specified simulation model rather than direct estimates of behavior in deployed multi-agent systems.
