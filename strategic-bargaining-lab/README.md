# 🤝 Strategic Tradeoffs Between Humans and AI in Multi-Agent Bargaining

A research-inspired multi-agent bargaining simulator that investigates how aggressive, fairness-oriented, and conservative negotiation policies influence payoffs, offer acceptance, agreement rates, and negotiation efficiency.

**Research reference:** [Strategic Tradeoffs Between Humans and AI in Multi-Agent Bargaining — ACM IUI 2026](https://doi.org/10.1145/3742413.3789078)

> **Research prototype:** This project implements a synthetic bargaining simulation inspired by the published paper. It does not reproduce the paper's exact experimental protocol, call real LLMs, or use human-subject data.

## Overview

As large language models become autonomous decision-making agents, evaluating their performance requires more than comparing final outcomes. Different agents can achieve comparable payoffs while following substantially different negotiation strategies.

The paper *Strategic Tradeoffs Between Humans and AI in Multi-Agent Bargaining* examines bargaining behavior among 216 human participants, frontier LLMs, and customized Bayesian agents.

Its central finding is that Bayesian agents pursue surplus through aggressive proposals that are frequently rejected, while LLMs tend toward conservative, concessionary proposals. Humans exhibit fairness-oriented trading behavior, with proposals that can be more likely to be rejected than LLM proposals.

This project explores these strategic tradeoffs through a configurable synthetic simulation and an interactive Streamlit dashboard.

## Research Questions

* How do aggressive, fairness-oriented, and conservative policies differ in payoff?
* Does aggressive bargaining improve outcomes at the cost of more rejections?
* Can conservative strategies achieve favorable payoffs with fewer negotiation rounds?
* Does a high agreement rate necessarily indicate a successful negotiation?
* How does the population composition affect strategy performance?

## Strategy Archetypes

| Strategy            | Simulated behavior                                   | Research motivation                                                                          |
| ------------------- | ---------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Bayesian-aggressive | Pursues strategically advantageous offers            | Inspired by the aggressive surplus-seeking behavior described for customized Bayesian agents |
| Human-fairness      | Represents fairness-oriented negotiation preferences | Inspired by the paper's observations of human trading behavior                               |
| LLM-conservative    | Favors cautious, concessionary offers                | Inspired by the paper's observations of LLM bargaining behavior                              |

These are simplified policy archetypes, not implementations of the actual human participants, Bayesian agents, or frontier LLMs evaluated in the paper.

## Interactive Streamlit Dashboard

The application provides an interface for inspecting negotiation outcomes and comparing strategy behavior.

### Dashboard sections

* **Strategy Outcomes:** Compare payoff and performance across the three simulated policy archetypes.
* **Offers & Acceptance:** Examine proposal acceptance and rejection behavior.
* **Inspect a Game:** Explore an individual simulated negotiation and its outcomes.
* **Strategy-Mix Study:** Compare different population compositions and investigate whether strategy rankings change.
* **Data:** Review and export experiment results where supported by the application.

### Strategy-Mix Study

The strategy-mix experiment compares four population scenarios:

1. **Balanced:** A balanced mix of the three strategies.
2. **Conservative-heavy:** A population with a larger share of conservative agents.
3. **Fairness-heavy:** A population with a larger share of fairness-oriented agents.
4. **Aggressive-heavy:** A population with a larger share of aggressive agents.

The experiment supports configurable game counts, agents per game, round limits, and random seeds. It reports comparative metrics and supports CSV exports for further analysis.

The purpose is to investigate how the surrounding agent population affects negotiation behavior rather than evaluating each strategy in isolation.

## Latest Dashboard Results

The latest dashboard snapshot reported the following results:

| Metric                | Observed result |
| --------------------- | --------------: |
| Games simulated       |           1,400 |
| Game agreement rate   |           99.8% |
| Offer acceptance rate |           52.2% |
| Mean rounds per game  |            1.91 |

These figures describe one synthetic experiment. They are not results from the original ACM study.

## Strategy-Level Results

An earlier experiment reported the following strategy-level metrics:

| Strategy            | Mean payoff | Offer acceptance rate |
| ------------------- | ----------: | --------------------: |
| LLM-conservative    |      0.2795 |                79.20% |
| Human-fairness      |      0.2717 |                77.58% |
| Bayesian-aggressive |      0.1980 |                12.23% |

These results are from an earlier run and should not be combined with the latest dashboard snapshot as if they came from the same experiment.

### Interpretation

**LLM-conservative:** Achieved the highest mean payoff and offer acceptance rate in the earlier experiment. This is consistent with the hypothesis that concessionary proposals can improve acceptance under the simulator's rules.

**Human-fairness:** Achieved a mean payoff close to the conservative policy, with a similarly high acceptance rate. The small payoff difference does not establish statistically significant superiority for either policy.

**Bayesian-aggressive:** Had the lowest mean payoff and offer acceptance rate in the earlier experiment. Its results illustrate the potential cost of aggressive proposals when other agents frequently reject them.

These interpretations are conditional on the implemented policies, payoff function, and simulated bargaining environment.

## Latest Detailed Strategy Evaluation

A separate experiment produced the following detailed results:

| Metric          | Bayesian-aggressive | Human-fairness | LLM-conservative |
| --------------- | ------------------: | -------------: | ---------------: |
| Games observed  |               1,843 |          1,911 |            1,846 |
| Mean payoff     |              0.1960 |         0.2752 |           0.2763 |
| Median payoff   |              0.0000 |         0.4163 |           0.4327 |
| Agreement rate  |              99.40% |         99.95% |          100.00% |
| Mean rounds     |               2.398 |          1.763 |            1.575 |
| Mean rejections |               1.404 |          0.763 |            0.575 |

Total observations across these strategy-level results: **5,600 games**.

### Key findings

* **Payoff:** LLM-conservative achieved the highest mean payoff, narrowly ahead of Human-fairness.
* **Typical outcome:** LLM-conservative also had the highest median payoff.
* **Agreement:** All three strategies had high observed agreement rates.
* **Efficiency:** LLM-conservative averaged the fewest negotiation rounds and rejections.
* **Aggressive behavior:** Bayesian-aggressive had the lowest mean payoff and a median payoff of zero despite a high agreement rate.

The conservative strategy's mean payoff was approximately 40.97% higher than that of Bayesian-aggressive in this experiment.

Human-fairness and LLM-conservative had nearly identical mean payoffs: 0.2752 and 0.2763, respectively. Further statistical testing is necessary before concluding that this difference is meaningful.

**Important:** The detailed evaluation, earlier strategy-level experiment, and latest dashboard snapshot are separate reported runs. Their results should remain separately labeled.

## Evaluation Metrics

| Metric                | Description                                                            |
| --------------------- | ---------------------------------------------------------------------- |
| Mean payoff           | Average payoff across observed games                                   |
| Median payoff         | Middle payoff value across observed games                              |
| Agreement rate        | Proportion of games reaching agreement                                 |
| Offer acceptance rate | Proportion of offers accepted, according to the simulator's definition |
| Mean rounds           | Average number of negotiation rounds per game                          |
| Mean rejections       | Average number of rejected offers per game                             |
| Games observed        | Number of games represented in a reported result                       |

Agreement rate and offer acceptance rate measure different aspects of bargaining. A game can contain rejected offers and still reach an agreement, so these metrics should not be treated as interchangeable.

## Technology and Implementation

* **Python** — simulation and evaluation logic
* **Streamlit** — interactive dashboard
* **Pandas** — tabular experiment results and CSV export
* **Visualization libraries** — charts for comparing strategy outcomes
* **Synthetic experiments** — configurable policy mixes, game counts, and random seeds

Refer to the project's source files and dependency configuration for the exact implementation and installed package versions.

## Running the Project

From PowerShell, navigate to the project directory:

Install the project's dependencies using its existing dependency file. If the project uses `requirements.txt`, run:

```powershell
python -m pip install -r requirements.txt
```

Launch the dashboard:

```powershell
python -m streamlit run app.py
```

Open the local Streamlit URL shown in the terminal, typically `http://localhost:8501`.

## Suggested Experiments

To extend the analysis:

1. Run each population scenario across multiple random seeds.
2. Compare mean payoff, median payoff, acceptance, agreement, and rejection rates.
3. Plot payoff distributions instead of reporting only averages.
4. Measure joint surplus and payoff inequality, where supported by the game model.
5. Introduce different bargaining conditions, such as asymmetric preferences or stricter round limits.
6. Evaluate whether observed strategy rankings remain stable across configurations.

## Limitations

* The simulation is research-inspired rather than a reproduction of the original study.
* The policy archetypes do not establish how actual humans or LLMs would behave.
* Results depend on the simulator's proposal-generation rules, acceptance logic, payoff function, and population composition.
* High agreement rates may indicate that the simulated game conditions make breakdowns uncommon.
* Statistical significance and generalization have not been established.
* Reported experiment snapshots may use different configurations and should not be directly pooled without verification.

## Conclusion

Strategic Bargaining Lab provides an interactive environment for examining how simulated negotiation policies trade off payoff, acceptance, agreement, and efficiency.

Across the reported detailed experiment, LLM-conservative achieved the strongest descriptive results, while Human-fairness achieved a comparable mean payoff. Bayesian-aggressive exhibited lower payoffs and more rejections.

The project's central takeaway is that **aggregate performance alone does not fully explain agent behavior**. Studying proposals, rejections, population composition, and negotiation efficiency provides a richer understanding of strategic differences.

The results motivate further controlled experiments, statistical robustness checks, and eventual evaluation with actual LLM-based agents.

## Reference

Qian, C., Zhu, K., Horton, J., Manning, B., Tsai, V., Wexler, J., & Thain, N. *Strategic Tradeoffs Between Humans and AI in Multi-Agent Bargaining*. ACM Conference on Intelligent User Interfaces (IUI), 2026.

* [ACM Digital Library — Publication](https://dl.acm.org/doi/10.1145/3742413.3789078)
* [DOI](https://doi.org/10.1145/3742413.3789078)
* [Download paper (PDF)](https://dl.acm.org/doi/pdf/10.1145/3742413.3789078?download=true)
