# 🤖 RoboBallet AI
### Graph Neural Networks + Reinforcement Learning for Multi-Robot Task Allocation and Collision-Aware Path Planning

An educational research-inspired implementation of multi-robot coordination using Graph Neural Networks (GNNs), Reinforcement Learning (RL), and multi-agent path planning.

The project explores how multiple robots can coordinate task assignments, navigate shared workspaces, avoid collisions, and optimize overall operational efficiency.

Inspired by the research paper:

**RoboBallet: Planning for Multi-Robot Reaching with Graph Neural Networks and Reinforcement Learning**

Published in *Science Robotics*, September 3, 2025.

[Research Paper](https://www.science.org/doi/10.1126/scirobotics.ads1204)

---

## 🚀 Project Overview

Multi-robot environments introduce several challenging optimization problems:

- Which robot should perform each task?
- How should tasks be distributed across robots?
- How can robots avoid collisions in shared workspaces?
- How can total travel distance be reduced?
- How can multiple robots finish their assigned tasks efficiently?

RoboBallet AI implements a research-inspired framework that combines learned task allocation with classical path planning.

### Core capabilities

- Graph-based robot-task relationship modeling.
- GNN-based task allocation policy.
- REINFORCE policy-gradient training.
- Obstacle-aware path planning using A*.
- Time-expanded planning with collision reservations.
- Multi-robot task execution simulation.
- Greedy baseline comparison.
- Quantitative evaluation of travel distance and makespan.

---

## 🧠 System Architecture

```text
              Multi-Robot Environment
                        |
                        v
             Robot-Task Graph Builder
                        |
                        v
               Graph Neural Network
                        |
                        v
              Task Allocation Policy
                        |
                        v
              Reinforcement Learning
                  (REINFORCE)
                        |
                        v
               Assigned Task Routes
                        |
                        v
             Collision-Aware A* Planner
                        |
                        v
              Multi-Robot Simulation
                        |
                        v
                Performance Metrics
```

### 1. Environment Generation

A procedural 2D grid environment generates:

- Multiple robot starting positions.
- Randomly distributed task locations.
- Rectangular obstacles.
- Shared workspace constraints.

### 2. Graph Neural Network

The system represents robot-task relationships as a graph.

Robot and task nodes exchange information through message passing, allowing the policy to learn relationships between robots and available tasks.

### 3. Reinforcement Learning

A policy-gradient approach trains the GNN-based allocation policy.

The training objective incorporates:

- Total travel distance.
- Robot makespan.
- Workload imbalance.

The policy learns to distribute tasks across robots rather than relying exclusively on local distance-based decisions.

### 4. Collision-Aware Path Planning

A time-expanded A* planner generates robot paths while considering:

- Obstacles.
- Vertex reservations.
- Edge reservations.
- Head-on collision avoidance.

### 5. Simulation and Evaluation

The simulator evaluates the resulting task assignments and reports:

- Total travel distance.
- Makespan.
- Completed tasks.
- Task completion rate.
- Individual robot execution times.

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming | Python |
| Deep Learning | PyTorch |
| Reinforcement Learning | REINFORCE |
| Graph Learning | Graph Neural Networks |
| Path Planning | A* Search |
| Optimization | Greedy Baseline |
| Numerical Computing | NumPy |
| Visualization | Matplotlib |
| Testing | Pytest |

---

## 📁 Project Structure

```text
RoboBallet_AI/
│
├── roboballet/
│   ├── __init__.py
│   ├── config.py
│   ├── environment.py
│   ├── graph_policy.py
│   ├── allocation.py
│   ├── planner.py
│   ├── simulator.py
│   ├── visualization.py
│   ├── train.py
│   ├── evaluate.py
│   └── demo.py
│
├── tests/
│   └── test_planner.py
│
├── artifacts/
│   ├── policy.pt
│   ├── policy_v2.pt
│   └── demo.png
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/ManasaK285/G2.git
cd G2/RoboBallet_AI
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Simulation

Run a multi-robot simulation:

```bash
python -m roboballet.demo --seed 7 --robots 4 --tasks 12 --plot
```

The demo generates a robot-task assignment, plans paths, and reports simulation metrics.

---

## 🧪 Train the GNN Policy

### Version 1 — Initial Training

```bash
python -m roboballet.train --episodes 700 --seed 7
```

Saved checkpoint:

```text
artifacts/policy.pt
```

### Version 2 — Obstacle-Aware Reward

The second training experiment introduced an obstacle-aware path-distance cost into the training reward.

```bash
python -m roboballet.train --episodes 1000 --seed 42 --output artifacts/policy_v2.pt
```

Saved checkpoint:

```text
artifacts/policy_v2.pt
```

The updated reward incorporates total travel, makespan, and workload imbalance.

---

## 📊 Experimental Results

The following results were obtained from the project's 50-episode evaluation using seed 100.

### Version 1 — Initial GNN-RL Policy

| Metric | GNN-RL | Greedy |
|---|---:|---:|
| Total Travel Distance ↓ | 54.88 | 39.00 |
| Makespan ↓ | 19.86 | 21.20 |
| Completion Rate ↑ | 100% | 100% |

### Version 2 — Obstacle-Aware Training Reward

| Metric | GNN-RL | Greedy |
|---|---:|---:|
| Total Travel Distance ↓ | **43.04** | 39.00 |
| Makespan ↓ | **18.18** | 21.20 |
| Completion Rate ↑ | 100% | 100% |

### V1 vs. V2 Improvement

| Metric | Improvement |
|---|---:|
| Total Travel Distance | 21.6% reduction |
| Makespan | 8.5% reduction |
| Task Completion | Maintained at 100% |

### Key Observations

1. The obstacle-aware reward improved the learned policy's total travel distance by approximately 21.6%.
2. Makespan improved by approximately 8.5%.
3. V2 achieved a lower makespan than the Greedy baseline.
4. Greedy still achieved a lower total travel distance.
5. Both approaches completed all assigned tasks in the evaluation.

**Interpretation:** The results demonstrate a measurable improvement in the learned allocation policy. However, V2 has not yet outperformed Greedy on every metric.

### Reproduce Evaluation

Version 1:

```bash
python -m roboballet.evaluate --checkpoint artifacts/policy.pt --episodes 50 --seed 100
```

Version 2:

```bash
python -m roboballet.evaluate --checkpoint artifacts/policy_v2.pt --episodes 50 --seed 100
```

---

## 📈 Performance Metrics

### Total Travel Distance

The sum of the distances traveled by all robots.

Lower values indicate less overall robot movement.

### Makespan

The maximum execution time among all robots.

Lower makespan indicates that the overall multi-robot operation finishes sooner.

### Completion Rate

The fraction of assigned tasks successfully completed.

### Workload Imbalance

Measures variation in the amount of work assigned across robots.

A more balanced workload can help prevent individual robots from becoming bottlenecks.

---

## 🧪 Testing

Run the test suite:

```bash
python -m pytest -q
```

The initial project test run completed successfully:

```text
5 passed
```

---

## 🔬 Research Limitations

This project is a research-inspired educational implementation, not an exact reproduction of the original RoboBallet paper.

Current limitations include:

- The environment uses a discrete 2D grid rather than continuous robot-arm kinematics.
- The learned component focuses on robot-task allocation.
- Motion planning uses classical A* rather than a learned motion policy.
- Training uses an obstacle-aware independent path-cost approximation without inter-robot reservations.
- Evaluation uses collision-aware multi-robot planning.
- Results are based on a limited experimental configuration and should not be interpreted as broad generalization evidence.
- The current REINFORCE implementation can exhibit high training variance.

The distinction between training cost and coordinated execution cost is important when interpreting the results.

---

## 🚀 Future Improvements — V3 Roadmap

The next iteration will focus on stronger optimization and more rigorous evaluation.

| Area | Planned Improvement |
|---|---|
| Reinforcement Learning | PPO with a critic network |
| Task Allocation | Global robot-task assignment |
| Path Planning | Collision-aware optimization objective |
| Optimization Baseline | Hungarian algorithm |
| Evaluation | Multiple random seeds |
| Robustness | Larger robot and task configurations |
| Metrics | Mean, standard deviation, and confidence intervals |
| Visualization | Comparative robot trajectory analysis |

The long-term goal is to develop a more robust multi-robot coordination framework that jointly considers task allocation, path planning, and execution efficiency.

---

## 🎯 Project Highlights

- Graph Neural Networks for robot-task relationship modeling.
- Reinforcement Learning for multi-robot task allocation.
- Obstacle-aware route-cost optimization.
- Multi-agent collision-aware path planning.
- Quantitative comparison against a classical baseline.
- Reproducible training and evaluation commands.
- Modular architecture suitable for further robotics research.

---

## 📚 Reference

**RoboBallet: Planning for Multi-Robot Reaching with Graph Neural Networks and Reinforcement Learning**

*Science Robotics*, September 3, 2025.

[Read the Original Research Paper](https://www.science.org/doi/10.1126/scirobotics.ads1204)

---

## ⚠️ Disclaimer

RoboBallet AI is an independent research-inspired implementation developed for experimentation and educational purposes. It is not affiliated with the original paper authors and does not claim to reproduce their complete methodology or reported results.
