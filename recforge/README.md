# ⚙️ RecForge — GenRec-Inspired Recommendation Ranking System

> **An end-to-end recommendation system inspired by *GenRec: An LLM-Backed Recommendation Ranker at Netflix*.**

RecForge is an end-to-end recommendation research project that combines classical retrieval, semantic retrieval, context engineering, reward-aware ranking, and a catalog-aware neural ranker into a single recommendation pipeline.

The project is designed to reproduce and explore several of the ideas discussed in the GenRec paper using the **MovieLens** dataset as a reproducible research environment.

---

## 📌 Reference

**Primary paper**

**Ying Li, Shradha Sehgal, Arjun Rao, Rein Houthooft, Yaochen Zhu, Ashish Rastogi.**

> *GenRec: An LLM-Backed Recommendation Ranker at Netflix.*

The paper describes an LLM-backed recommendation ranker developed at Netflix and focuses on recommendation-specific post-training, context engineering, reward integration, catalog-aware ranking, and cost-conscious serving.

**Paper:**
GenRec: An LLM-Backed Recommendation Ranker at Netflix
arXiv:2608.10257

RecForge is **inspired by the architecture and research directions described in the paper**, but is not an implementation of Netflix's proprietary production system.

---

# 🎯 Project Goal

Traditional recommendation systems often rely heavily on manually engineered features such as:

* user demographics
* item statistics
* interaction counts
* handcrafted behavioral features
* manually constructed user/item features

GenRec presents a different direction:

> **Feature engineering → Context engineering**

Instead of representing a user's behavior entirely through thousands of independent features, the system constructs a compact representation of the user's recent and relevant behavior.

RecForge explores this idea through:

```text
User History
     │
     ▼
Context Engineering
     │
     ├── Recent interactions
     ├── High-signal behavior
     ├── Genre preferences
     └── Compressed history
     │
     ▼
Candidate Generation
     │
     ├── Popularity
     ├── Collaborative Retrieval
     └── Semantic Retrieval
     │
     ▼
Catalog-Aware Ranker
     │
     ▼
Business / Diversity Layer
     │
     ▼
Top-K Recommendations
```

---

# 🏗️ System Architecture

RecForge is implemented as a single executable Python application:

```text
recforge.py
```

The pipeline contains the following major components.

### 1. Data Layer

MovieLens ratings and movie metadata are downloaded and processed automatically.

The system performs:

* dataset download
* train/test splitting
* user/item indexing
* movie metadata processing
* interaction-history construction

---

### 2. Popularity Retrieval

A popularity-based recommender provides a simple retrieval baseline.

It estimates item popularity from historical interactions and provides popular unseen items.

This establishes a non-personalized baseline against which more sophisticated retrieval strategies can be compared.

---

### 3. Collaborative Retrieval

RecForge includes a lightweight collaborative retrieval system based on user-item interaction behavior.

The system identifies related users and retrieves items that similar users interacted with.

Conceptually:

```text
User
 │
 ├── Previously interacted items
 │
 ▼
Related users
 │
 ▼
Items from related users
 │
 ▼
Collaborative candidates
```

---

### 4. Semantic Retrieval

Movie titles and metadata are represented using TF-IDF features.

The semantic retriever:

1. Converts movie metadata into TF-IDF vectors.
2. Builds a user profile from recently interacted items.
3. Computes cosine similarity against the catalog.
4. Retrieves semantically related unseen movies.

The implementation keeps the catalog representation sparse to avoid unnecessarily densifying the full TF-IDF matrix.

---

# 🧠 Context Engineering

One of the central ideas explored by RecForge is **context engineering**.

Instead of passing an entire interaction history directly into the ranking system, the Context Engine transforms raw history into a compact representation.

### Raw interaction history

```text
Movie A
Movie B
Movie C
Movie D
...
Movie N
```

### Context representation

```text
User has interacted with N titles.

Recent titles:
A, B, C, D, ...

Strong genre signals:
Drama, Adventure, Crime, Comedy, ...

Recent behavior is weighted more heavily than older history.
```

The context engine therefore performs:

* history selection
* recency weighting
* preference extraction
* genre aggregation
* history compression
* textual verbalization

For the current MovieLens evaluation, an example compressed context is approximately **449 characters**, compared with the complete interaction history.

This provides a small-scale demonstration of the paper's broader idea of moving from feature engineering toward context engineering.

---

# 🧮 Reward Modeling

RecForge includes a reward engine that combines multiple recommendation objectives.

The reward formulation considers signals such as:

```text
Engagement
Satisfaction
Discovery
Diversity
Novelty
```

The goal is to demonstrate that recommendation quality does not have to be represented by a single relevance signal.

Conceptually:

```text
               ┌── Engagement
               ├── Satisfaction
User / Item ───┼── Discovery
               ├── Diversity
               └── Novelty
                       │
                       ▼
                 Reward Signal
```

This is inspired by the reward-alignment direction discussed in GenRec.

---

# 🤖 Catalog-Aware Ranker

After candidate generation, RecForge applies a neural ranking model.

The ranker receives:

```text
User representation
        +
Candidate item representation
        ↓
Ranking score
```

The system trains the ranker using positive recommendation examples and sampled negatives.

Training currently uses:

```text
Training examples: 61,309
Epochs:            4
Device:            CPU
```

The implementation is intentionally lightweight so that the complete pipeline can run locally on a standard machine.

---

# 🔎 Candidate Generation

Rather than asking the ranker to score the entire catalog, RecForge first generates a candidate set.

Candidates come from multiple retrieval strategies:

```text
                 ┌── Popularity
                 │
User History ────┼── Collaborative
                 │
                 └── Semantic
                        │
                        ▼
                Candidate Pool
                        │
                        ▼
                   Neural Ranker
                        │
                        ▼
                     Top-K
```

This separates:

**retrieval**

from

**ranking**

which is a common architecture for large-scale recommender systems.

---

# 📊 Evaluation

The current evaluation measures:

* Hit Rate@K
* MRR@K
* NDCG@K
* Diversity@K
* average recommendation latency

Current evaluation results:

| Metric          |        Result |
| --------------- | ------------: |
| Hit Rate@10     |    **0.0230** |
| MRR@10          |    **0.0078** |
| NDCG@10         |    **0.0114** |
| Diversity@10    |    **0.8435** |
| Users evaluated |       **609** |
| Average latency | **182.83 ms** |

The current implementation is intended primarily as a research/engineering baseline and demonstration of the complete architecture. These numbers should not be interpreted as a reproduction of Netflix's reported GenRec results.

---

# ⚡ Runtime Performance

A complete local run currently includes:

```text
Dataset processing
       ↓
Popularity fitting
       ↓
Collaborative retrieval fitting
       ↓
Semantic retrieval fitting
       ↓
Context engine construction
       ↓
Reward construction
       ↓
Ranker training
       ↓
Recommendation generation
       ↓
Evaluation
```

Recent local execution:

```text
Complete build time: ~15 seconds
Average recommendation latency: ~183 ms
```

The exact timings depend on hardware, Python version, installed libraries, and runtime conditions.

---

# 🧪 Example Output

For User 1, the current system produces recommendations such as:

```text
==============================
RECOMMENDATIONS FOR USER 1
==============================

1. Twelve Monkeys (a.k.a. 12 Monkeys) (1995)
2. Third Man, The (1949)
3. Lord of the Rings: The Return of the King, The (2003)
4. Howl's Moving Castle (Hauru no ugoku shiro) (2004)
5. Departed, The (2006)
6. Eternal Sunshine of the Spotless Mind (2004)
7. Graduate, The (1967)
8. Louis C.K.: Live at the Beacon Theater (2011)
9. Lawrence of Arabia (1962)
10. Lord of the Rings: The Fellowship of the Ring, The (2001)
```

The system also exposes the context used by the recommendation pipeline:

```text
User 1 has interacted with 100 titles.

Recent titles:
Independence Day
Pink Floyd: The Wall
The Messenger
Canadian Bacon
McHale's Navy
Tombstone
Three Amigos
Back to the Future Part III

Strongest genre signals:
Drama
Adventure
Children
Crime
Comedy
Thriller
Action
Animation
```

---

# 🧩 Relationship to GenRec

RecForge takes inspiration from several ideas in the GenRec paper.

| GenRec Concept                   | RecForge Implementation              |
| -------------------------------- | ------------------------------------ |
| Context engineering              | Context Engine                       |
| User-history verbalization       | Compressed textual user context      |
| Recommendation-specific training | Ranking training examples            |
| Reward integration               | Multi-objective Reward Engine        |
| Catalog-aware ranking            | Neural catalog ranker                |
| Candidate generation + ranking   | Retrieval → ranking pipeline         |
| Efficient serving                | Compact context + lightweight ranker |
| Offline ranking evaluation       | HitRate, MRR, NDCG                   |
| Recommendation diversity         | Diversity metric                     |

### Important distinction

GenRec is an **LLM-backed ranker built on an in-house foundation model at Netflix**.

RecForge is a **local, reproducible research implementation inspired by the publicly described concepts**.

In particular, RecForge does not claim to reproduce:

* Netflix's proprietary foundation model
* Netflix's production training infrastructure
* Netflix's proprietary datasets
* Netflix's production reward signals
* Netflix's production serving architecture
* Netflix's reported online A/B experiment

The purpose of this project is to translate the publicly described research ideas into an executable recommendation-system environment.

---

# 🛠️ Tech Stack

```text
Python
PyTorch
NumPy
Pandas
SciPy
scikit-learn
FastAPI
MovieLens
```

The project is designed to run locally without requiring a GPU.

---

# 🚀 Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the complete evaluation:

```bash
python recforge.py --evaluate
```

The command builds the recommendation pipeline, trains the ranker, generates recommendations, constructs user context, and evaluates the system.

---

# 📁 Project Structure

```text
recforge/
│
├── recforge.py
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── run.sh
├── run.bat
│
├── configs/
│   └── default.json
│
├── models/
│   ├── ranker.pt
│   └── metadata.json
│
└── tests/
    └── test_recforge.py
```

The main implementation is intentionally consolidated into `recforge.py` so the complete recommendation workflow can be inspected and executed as one system.

---

# 🔬 Research Experiments

The architecture is designed to support further experimentation.

### Context scaling

Measure the effect of different history lengths:

```text
10 events
25 events
50 events
100 events
```

Compare:

```text
NDCG
MRR
HitRate
Latency
Context size
```

---

### Candidate scaling

Experiment with:

```text
50 candidates
100 candidates
250 candidates
500 candidates
```

and measure the quality/latency trade-off.

---

### Retrieval ablations

Compare:

```text
Popularity only
Collaborative only
Semantic only
Popularity + Collaborative
Popularity + Semantic
All retrieval strategies
```

---

### Ranking ablations

Compare:

```text
Retrieval-only
Neural ranking
Reward-aware ranking
Context-aware ranking
```

This makes it possible to identify which component actually contributes to recommendation quality.

---

# 🗺️ Future Work

The current system provides the foundation for several extensions.

### 1. True LLM-backed ranking

Replace the lightweight neural ranker with a decoder-only foundation model or a smaller instruction-tuned model.

The model could receive:

```text
User context
+
Candidate item information
+
Recommendation task
```

and produce catalog-aware ranking scores.

---

### 2. Better context representation

Extend the Context Engine with:

* temporal behavior
* rating strength
* repeated interactions
* preference shifts
* long-term vs recent interests
* explicit negative feedback
* session-level context

---

### 3. Reward-weighted training

Introduce reward-weighted ranking loss so that training examples with stronger expected long-term value receive greater influence.

---

### 4. Data and model scaling experiments

Evaluate:

```text
More training data
        vs
Larger ranker
        vs
Longer context
```

and construct quality/cost scaling curves.

---

### 5. Agentic Recommendation Research

A future version can introduce a research agent capable of:

```text
Run experiment
      ↓
Inspect metrics
      ↓
Diagnose regression
      ↓
Change configuration
      ↓
Run experiment again
      ↓
Compare trajectories
      ↓
Record results
```

This would extend RecForge from a recommendation engine into an automated recommendation-system experimentation framework.

---

# 📚 References

**Primary reference**

Li, Y., Sehgal, S., Rao, A., Houthooft, R., Zhu, Y., & Rastogi, A.

*GenRec: An LLM-Backed Recommendation Ranker at Netflix.*

arXiv:2608.10257, 2026.

**Dataset**

Harper, F. M., & Konstan, J. A.

*The MovieLens Datasets: History and Context.*

ACM Transactions on Interactive Intelligent Systems, 2015.

---

# ⚠️ Disclaimer

RecForge is an independent research/portfolio project.

It is **not affiliated with, sponsored by, or endorsed by Netflix**.

The project uses publicly available research concepts and the MovieLens dataset to explore LLM-inspired recommendation ranking, context engineering, reward alignment, and efficient recommendation-system design.

---

# 👤 Project

**RecForge**

An end-to-end recommendation system exploring the transition:

```text
Traditional Recommendation
          │
          ▼
Feature Engineering
          │
          ▼
Context Engineering
          │
          ▼
LLM / Neural Ranking
          │
          ▼
Reward-Aligned Recommendations
```

Built as a reproducible research implementation inspired by the GenRec architecture.
