"""
RecForge — end-to-end LLM-native recommendation system.

Run:
    python recforge.py

API:
    uvicorn recforge:app --reload

The implementation is deliberately self-contained:
data -> preprocessing -> baselines -> context engineering ->
candidate generation -> catalog-aware neural ranker ->
reward-weighted training -> evaluation -> FastAPI.

For a CPU-friendly demo, MovieLens Small is used and the neural ranker
operates on learned item embeddings plus engineered user/context features.
The catalog-aware scoring head mirrors the GenRec design principle:
represent the user/context once and score catalog item embeddings rather
than autoregressively generating item IDs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import ssl
import time
import urllib.request
import zipfile
import certifi

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, Dataset


SEED = 42
DATA_DIR = Path("data")
MODEL_DIR = Path("artifacts")
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOP_K = 10
CANDIDATE_K = 100
MAX_HISTORY = 20
EMBED_DIM = 64
HIDDEN_DIM = 128
EPOCHS = 4
BATCH_SIZE = 256
LR = 1e-3
MIN_RATING_FOR_POSITIVE = 3.5

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

def download_movielens() -> Path:
    root = DATA_DIR / "ml-latest-small"
    ratings = root / "ratings.csv"
    if ratings.exists():
        return root

    DATA_DIR.mkdir(exist_ok=True)
    archive = DATA_DIR / "ml-latest-small.zip"
    print("[data] downloading MovieLens Small...")
    try:
        ssl_context = ssl.create_default_context(cafile=certifi.where())

        with urllib.request.urlopen(
            MOVIELENS_URL,
            context=ssl_context,
            timeout=60,
        ) as response, open(archive, "wb") as output:
            output.write(response.read())

    except Exception as exc:
        raise RuntimeError(
            "\nMovieLens download failed during HTTPS certificate "
            "verification.\n\n"
            "Run:\n"
            "    python -m pip install --upgrade certifi\n"
            "    python recforge.py --evaluate\n\n"
            f"Original error: {exc}"
        ) from exc

    with zipfile.ZipFile(archive) as z:
        z.extractall(DATA_DIR)

    return root


def load_data():
    root = download_movielens()
    ratings = pd.read_csv(root / "ratings.csv")
    movies = pd.read_csv(root / "movies.csv")

    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s")
    ratings = ratings.sort_values(["userId", "timestamp"])

    movies["year"] = (
        movies["title"].str.extract(r"\((\d{4})\)")[0].fillna("Unknown")
    )
    movies["genres_text"] = movies["genres"].str.replace("|", " ", regex=False)
    movies["text"] = movies["title"].fillna("") + " " + movies["genres_text"].fillna("")

    return ratings, movies


def temporal_split(ratings: pd.DataFrame):
    train, test = [], []

    for _, group in ratings.groupby("userId"):
        group = group.sort_values("timestamp")
        if len(group) < 3:
            train.append(group)
        else:
            train.append(group.iloc[:-1])
            test.append(group.iloc[-1:])

    return pd.concat(train), pd.concat(test) if test else pd.DataFrame(columns=ratings.columns)


# ---------------------------------------------------------------------
# Baseline: popularity
# ---------------------------------------------------------------------

class PopularityModel:
    def fit(self, ratings):
        stats = ratings.groupby("movieId").agg(
            count=("rating", "count"),
            mean=("rating", "mean"),
        )
        global_mean = ratings.rating.mean()
        m = 10.0

        stats["score"] = (
            stats["count"] / (stats["count"] + m) * stats["mean"]
            + m / (stats["count"] + m) * global_mean
        )
        self.ranking = stats.sort_values("score", ascending=False).index.tolist()
        return self

    def recommend(self, history, k=TOP_K):
        seen = set(history)
        return [i for i in self.ranking if i not in seen][:k]


# ---------------------------------------------------------------------
# Baseline: user-item collaborative filtering
# ---------------------------------------------------------------------

class CollaborativeModel:
    def fit(self, ratings):
        self.user_items = defaultdict(dict)
        self.item_users = defaultdict(set)

        for r in ratings.itertuples():
            self.user_items[int(r.userId)][int(r.movieId)] = float(r.rating)
            self.item_users[int(r.movieId)].add(int(r.userId))

        self.pop = Counter(ratings.movieId)
        return self

    def recommend(self, user_id, history, k=TOP_K):
        history = list(history)
        seen = set(history)
        scores = Counter()

        for item in history[-20:]:
            for other_user in self.item_users.get(item, []):
                for candidate, rating in self.user_items[other_user].items():
                    if candidate not in seen:
                        scores[candidate] += max(rating - 2.5, 0.1)

        if not scores:
            return [x for x, _ in self.pop.most_common(k) if x not in seen]

        return [x for x, _ in scores.most_common(k)]


# ---------------------------------------------------------------------
# Semantic retrieval
# ---------------------------------------------------------------------

class SemanticRetriever:
    def fit(self, movies):
        self.movies = movies.copy()
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_features=20000,
            stop_words="english",
        )
        self.matrix = self.vectorizer.fit_transform(self.movies["text"])
        self.movie_ids = self.movies.movieId.tolist()
        self.id_to_idx = {m: i for i, m in enumerate(self.movie_ids)}
        return self

    def recommend(self, history, k=TOP_K):
        if not history:
            return []

        indices = [
            self.id_to_idx[x]
            for x in history[-10:]
            if x in self.id_to_idx
        ]
        if not indices:
            return []

        # Keep the catalog TF-IDF matrix sparse. Converting a large
        # scipy csr_matrix with np.asarray() creates an object array and
        # breaks sklearn's numeric validation.
        profile = self.matrix[indices].mean(axis=0)
        profile = np.asarray(profile, dtype=np.float64)
        matrix = self.matrix
        sims = cosine_similarity(profile, matrix).ravel()
        order = np.argsort(-sims)

        seen = set(history)
        result = []

        for idx in order:
            movie = self.movie_ids[int(idx)]
            if movie not in seen:
                result.append(movie)
            if len(result) == k:
                break

        return result


# ---------------------------------------------------------------------
# Context engineering
# ---------------------------------------------------------------------

@dataclass
class UserContext:
    user_id: int
    history: List[int]
    recent_history: List[int]
    genre_preferences: Dict[str, float]
    text: str


class ContextEngine:
    """
    Implements a lightweight version of the paper's context-engineering idea:
    retain high-signal/recent events, compress older behavior, and verbalize
    the resulting user state.
    """

    def __init__(self, movies: pd.DataFrame, max_history=MAX_HISTORY):
        self.movies = movies.set_index("movieId")
        self.max_history = max_history

    def build(self, user_id: int, history: List[int]) -> UserContext:
        history = history[-100:]

        recent = history[-self.max_history:]

        genre_counts = Counter()
        for movie_id in history:
            if movie_id not in self.movies.index:
                continue
            genres = str(self.movies.loc[movie_id, "genres"]).split("|")
            for g in genres:
                if g != "(no genres listed)":
                    genre_counts[g] += 1

        total = max(sum(genre_counts.values()), 1)
        preferences = {
            g: round(c / total, 3)
            for g, c in genre_counts.most_common(8)
        }

        recent_titles = []
        for movie_id in recent[-8:]:
            if movie_id in self.movies.index:
                recent_titles.append(str(self.movies.loc[movie_id, "title"]))

        top_genres = ", ".join(preferences.keys()) or "mixed genres"

        text = (
            f"User {user_id} has interacted with {len(history)} titles. "
            f"Recent titles: {'; '.join(recent_titles)}. "
            f"Strongest genre signals: {top_genres}. "
            f"Recent behavior is weighted more heavily than older history."
        )

        return UserContext(
            user_id=user_id,
            history=history,
            recent_history=recent,
            genre_preferences=preferences,
            text=text,
        )


# ---------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------

class CandidateGenerator:
    def __init__(self, popularity, collaborative, semantic, catalog):
        self.popularity = popularity
        self.collaborative = collaborative
        self.semantic = semantic
        self.catalog = catalog

    def generate(self, user_id, history, k=CANDIDATE_K):
        candidates = []

        candidates.extend(self.popularity.recommend(history, k))
        candidates.extend(self.collaborative.recommend(user_id, history, k))
        candidates.extend(self.semantic.recommend(history, k))

        counts = Counter(candidates)

        # Add globally popular catalog items if retrieval pools are small.
        for movie_id in self.popularity.ranking:
            if movie_id not in set(history):
                counts[movie_id] += 0.01
            if len(counts) >= k * 2:
                break

        return [
            movie_id
            for movie_id, _ in counts.most_common(k)
            if movie_id not in set(history)
        ][:k]


# ---------------------------------------------------------------------
# Neural catalog-aware ranker
# ---------------------------------------------------------------------

class RankerDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        x = self.examples[idx]
        return (
            torch.tensor(x["user"], dtype=torch.long),
            torch.tensor(x["item"], dtype=torch.long),
            torch.tensor(x["reward"], dtype=torch.float32),
        )


class CatalogRanker(nn.Module):
    """
    Context/user representation -> item embedding -> scalar score.

    This is the key catalog-aware ranking head:
        score(u, i) = MLP(user_context) dot item_embedding(i)
    """

    def __init__(self, num_users, num_items, embed_dim=EMBED_DIM):
        super().__init__()

        self.user_embedding = nn.Embedding(num_users + 1, embed_dim)

        self.item_embedding = nn.Embedding(num_items + 1, embed_dim)

        self.context = nn.Sequential(
            nn.Linear(embed_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, embed_dim),
        )

        self.bias = nn.Embedding(num_items + 1, 1)

    def encode_user(self, user_ids):
        return F.normalize(
            self.context(self.user_embedding(user_ids)),
            dim=-1,
        )

    def score(self, user_ids, item_ids):
        u = self.encode_user(user_ids)
        i = F.normalize(self.item_embedding(item_ids), dim=-1)
        b = self.bias(item_ids).squeeze(-1)

        return (u * i).sum(-1) + b

    def forward(self, user_ids, item_ids):
        return self.score(user_ids, item_ids)


# ---------------------------------------------------------------------
# Reward engineering
# ---------------------------------------------------------------------

class RewardEngine:
    """
    Demonstration reward model.

    Components:
      engagement  -> normalized rating
      novelty     -> inverse item popularity
      diversity   -> genre difference
      satisfaction -> high rating signal
    """

    def __init__(self, ratings, movies):
        self.ratings = ratings
        self.movies = movies.set_index("movieId")

        counts = ratings.movieId.value_counts()
        max_count = max(counts.max(), 1)

        self.popularity = counts.to_dict()
        self.max_count = max_count

    def item_reward(self, user_id, item_id, rating):
        engagement = max(float(rating) - 2.5, 0.0) / 2.5
        satisfaction = 1.0 if rating >= 4.0 else 0.0

        frequency = self.popularity.get(item_id, 0)
        novelty = 1.0 - min(frequency / self.max_count, 1.0)

        reward = (
            0.45 * engagement
            + 0.35 * satisfaction
            + 0.20 * novelty
        )

        return float(reward)

    def novelty(self, item_id):
        frequency = self.popularity.get(item_id, 0)
        return 1.0 - min(frequency / self.max_count, 1.0)


# ---------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------

def build_training_examples(
    ratings,
    user_map,
    item_map,
    reward_engine,
):
    examples = []

    for r in ratings.itertuples():

        rating = float(r.rating)

        if rating < MIN_RATING_FOR_POSITIVE:
            continue

        user = user_map[int(r.userId)]
        item = item_map[int(r.movieId)]

        reward = reward_engine.item_reward(
            int(r.userId),
            int(r.movieId),
            rating,
        )

        examples.append(
            {
                "user": user,
                "item": item,
                "reward": reward,
            }
        )

    return examples


# ---------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------

def train_ranker(
    model,
    examples,
    epochs=EPOCHS,
):
    dataset = RankerDataset(examples)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
    )

    model.train()

    for epoch in range(epochs):

        total_loss = 0.0

        for users, items, rewards in loader:

            users = users.to(DEVICE)
            items = items.to(DEVICE)
            rewards = rewards.to(DEVICE)

            positive_scores = model(
                users,
                items,
            )

            # Reward-weighted positive objective.
            #
            # Higher reward => stronger gradient.
            positive_loss = (
                -rewards * F.logsigmoid(
                    positive_scores
                )
            ).mean()

            # Random negative items.
            negative_items = torch.randint(
                low=1,
                high=model.item_embedding.num_embeddings,
                size=items.shape,
                device=DEVICE,
            )

            negative_scores = model(
                users,
                negative_items,
            )

            pairwise_loss = -F.logsigmoid(
                positive_scores - negative_scores
            ).mean()

            # Combined recommendation objective.
            loss = (
                0.7 * pairwise_loss
                + 0.3 * positive_loss
            )

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / max(len(loader), 1)

        print(
            f"[train] epoch={epoch + 1}/{epochs} "
            f"loss={avg_loss:.4f}"
        )

    return model


# ---------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------

class RecForge:

    def __init__(
        self,
        ratings,
        movies,
        train,
        test,
    ):
        self.ratings = ratings
        self.movies = movies
        self.train_df = train
        self.test_df = test

        self.user_ids = sorted(ratings.userId.unique())
        self.item_ids = sorted(movies.movieId.unique())

        self.user_map = {
            user_id: idx + 1
            for idx, user_id in enumerate(self.user_ids)
        }

        self.item_map = {
            item_id: idx + 1
            for idx, item_id in enumerate(self.item_ids)
        }

        self.inverse_item_map = {
            idx: item_id
            for item_id, idx in self.item_map.items()
        }

        self.user_histories = defaultdict(list)

        for r in train.itertuples():
            if float(r.rating) >= 3.0:
                self.user_histories[int(r.userId)].append(
                    int(r.movieId)
                )

        print("[model] fitting popularity...")
        self.popularity = PopularityModel().fit(train)

        print("[model] fitting collaborative retrieval...")
        self.collaborative = CollaborativeModel().fit(train)

        print("[model] fitting semantic retrieval...")
        self.semantic = SemanticRetriever().fit(movies)

        print("[model] building context engine...")
        self.context_engine = ContextEngine(movies)

        self.candidates = CandidateGenerator(
            self.popularity,
            self.collaborative,
            self.semantic,
            movies,
        )

        print("[model] building rewards...")
        self.reward_engine = RewardEngine(
            train,
            movies,
        )

        print("[model] building ranker...")

        self.ranker = CatalogRanker(
            num_users=len(self.user_ids),
            num_items=len(self.item_ids),
        ).to(DEVICE)

    def train(self):
        examples = build_training_examples(
            self.train_df,
            self.user_map,
            self.item_map,
            self.reward_engine,
        )

        print(
            f"[train] examples={len(examples)} "
            f"device={DEVICE}"
        )

        train_ranker(
            self.ranker,
            examples,
        )

        MODEL_DIR.mkdir(exist_ok=True)

        torch.save(
            self.ranker.state_dict(),
            MODEL_DIR / "ranker.pt",
        )

        with open(MODEL_DIR / "metadata.json", "w") as f:
            json.dump(
                {
                    # MovieLens IDs are often numpy.int64 values.
                    # Convert them to native Python ints so json.dump()
                    # can serialize the metadata reliably.
                    "users": [int(x) for x in self.user_ids],
                    "items": [int(x) for x in self.item_ids],
                    "device": str(DEVICE),
                },
                f,
            )

    def recommend(
        self,
        user_id: int,
        k=TOP_K,
    ):
        if user_id not in self.user_histories:
            return self.popularity.recommend([], k)

        history = self.user_histories[user_id]

        context = self.context_engine.build(
            user_id,
            history,
        )

        candidates = self.candidates.generate(
            user_id,
            history,
            CANDIDATE_K,
        )

        if not candidates:
            return []

        user_idx = self.user_map[user_id]

        item_indices = [
            self.item_map[x]
            for x in candidates
            if x in self.item_map
        ]

        users = torch.tensor(
            [user_idx] * len(item_indices),
            dtype=torch.long,
            device=DEVICE,
        )

        items = torch.tensor(
            item_indices,
            dtype=torch.long,
            device=DEVICE,
        )

        self.ranker.eval()

        with torch.no_grad():
            scores = self.ranker(
                users,
                items,
            ).cpu().numpy()

        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        # Business-aware reranking:
        # encourage genre diversity in the final slate.
        selected = []
        selected_genres = Counter()

        for movie_id, score in ranked:

            genres = str(
                self.movies.loc[
                    self.movies.movieId == movie_id,
                    "genres",
                ].iloc[0]
            ).split("|")

            novelty_bonus = self.reward_engine.novelty(
                movie_id
            )

            diversity_bonus = 0.0

            if selected:
                overlap = sum(
                    g in selected_genres
                    for g in genres
                )
                diversity_bonus = -0.05 * overlap

            final_score = (
                float(score)
                + 0.05 * novelty_bonus
                + diversity_bonus
            )

            selected.append(
                (
                    movie_id,
                    final_score,
                    genres,
                )
            )

            for g in genres:
                selected_genres[g] += 1

            if len(selected) >= k:
                break

        selected.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            int(x[0])
            for x in selected[:k]
        ]

    def explain(
        self,
        user_id,
        recommendations,
    ):
        context = self.context_engine.build(
            user_id,
            self.user_histories.get(user_id, []),
        )

        genres = list(
            context.genre_preferences.keys()
        )[:3]

        explanations = {}

        for movie_id in recommendations:

            title = self.movie_title(movie_id)

            if genres:
                reason = (
                    f"Recommended because your recent "
                    f"history shows interest in "
                    f"{', '.join(genres)}."
                )
            else:
                reason = (
                    "Recommended based on your "
                    "historical interaction pattern."
                )

            explanations[movie_id] = {
                "title": title,
                "reason": reason,
            }

        return explanations

    def movie_title(self, movie_id):
        row = self.movies[
            self.movies.movieId == movie_id
        ]

        if row.empty:
            return "Unknown"

        return str(row.iloc[0].title)


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def hit_rate(recommended, actual):
    return float(actual in recommended)


def reciprocal_rank(recommended, actual):
    for i, item in enumerate(recommended, 1):
        if item == actual:
            return 1.0 / i
    return 0.0


def ndcg(recommended, actual):
    for i, item in enumerate(recommended, 1):
        if item == actual:
            return 1.0 / math_log2(i + 1)
    return 0.0


def math_log2(x):
    return math.log(x, 2)


def diversity(recommended, movies):
    if len(recommended) <= 1:
        return 0.0

    genres = {}

    for movie_id in recommended:
        row = movies[
            movies.movieId == movie_id
        ]

        if row.empty:
            genres[movie_id] = set()
        else:
            genres[movie_id] = set(
                str(row.iloc[0].genres).split("|")
            )

    scores = []

    items = list(genres)

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a = genres[items[i]]
            b = genres[items[j]]

            union = len(a | b)
            inter = len(a & b)

            if union == 0:
                scores.append(1.0)
            else:
                scores.append(
                    1.0 - inter / union
                )

    return float(np.mean(scores))


def evaluate(model: RecForge):

    print("\n[evaluation] running...")

    metrics = {
        "hit_rate@10": [],
        "mrr@10": [],
        "ndcg@10": [],
        "diversity@10": [],
    }

    start = time.perf_counter()

    evaluated = 0

    for row in model.test_df.itertuples():

        user_id = int(row.userId)
        actual = int(row.movieId)

        if user_id not in model.user_histories:
            continue

        recommendations = model.recommend(
            user_id,
            TOP_K,
        )

        metrics["hit_rate@10"].append(
            hit_rate(
                recommendations,
                actual,
            )
        )

        metrics["mrr@10"].append(
            reciprocal_rank(
                recommendations,
                actual,
            )
        )

        metrics["ndcg@10"].append(
            ndcg(
                recommendations,
                actual,
            )
        )

        metrics["diversity@10"].append(
            diversity(
                recommendations,
                model.movies,
            )
        )

        evaluated += 1

    elapsed = time.perf_counter() - start

    results = {
        key: float(np.mean(value))
        for key, value in metrics.items()
    }

    results["users_evaluated"] = evaluated
    results["total_eval_seconds"] = elapsed
    results["avg_latency_ms"] = (
        elapsed / max(evaluated, 1) * 1000
    )

    print("\n==============================")
    print("RECFORGE EVALUATION")
    print("==============================")

    for key, value in results.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    return results


# ---------------------------------------------------------------------
# Context inspection
# ---------------------------------------------------------------------

def inspect_context(model, user_id):

    history = model.user_histories.get(
        user_id,
        [],
    )

    context = model.context_engine.build(
        user_id,
        history,
    )

    print("\n==============================")
    print("CONTEXT ENGINEERING")
    print("==============================")

    print("\nRaw events:")
    for movie_id in history[-20:]:
        print(
            " -",
            model.movie_title(movie_id),
        )

    print("\nCompressed context:")
    print(context.text)

    print(
        "\nApproximate context characters:",
        len(context.text),
    )

    return context


# ---------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------

def run_experiments(model):

    print("\n==============================")
    print("CONTEXT LENGTH EXPERIMENT")
    print("==============================")

    user_id = model.user_ids[0]
    history = model.user_histories[user_id]

    for length in [5, 10, 20, 50]:

        subset = history[-length:]

        context = model.context_engine.build(
            user_id,
            subset,
        )

        print(
            f"history={length:>3} "
            f"context_chars={len(context.text):>4}"
        )

    print("\n==============================")
    print("RETRIEVAL COMPARISON")
    print("==============================")

    history = model.user_histories[user_id]

    pop = model.popularity.recommend(
        history,
        10,
    )

    cf = model.collaborative.recommend(
        user_id,
        history,
        10,
    )

    semantic = model.semantic.recommend(
        history,
        10,
    )

    final = model.recommend(
        user_id,
        10,
    )

    print("Popularity:")
    for x in pop:
        print(" ", model.movie_title(x))

    print("\nCollaborative:")
    for x in cf:
        print(" ", model.movie_title(x))

    print("\nSemantic:")
    for x in semantic:
        print(" ", model.movie_title(x))

    print("\nRecForge:")
    for x in final:
        print(" ", model.movie_title(x))


# ---------------------------------------------------------------------
# Global application state
# ---------------------------------------------------------------------

ENGINE = None
EVALUATION = None


def build_engine():

    global ENGINE

    ratings, movies = load_data()

    ratings, movies = ratings.copy(), movies.copy()

    train, test = temporal_split(ratings)

    ENGINE = RecForge(
        ratings,
        movies,
        train,
        test,
    )

    ENGINE.train()

    return ENGINE


# ---------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------

app = FastAPI(
    title="RecForge",
    description="LLM-native recommendation research system",
    version="1.0.0",
)


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[dict]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "model_loaded": ENGINE is not None,
    }


@app.get("/recommend/{user_id}")
def recommend(user_id: int):

    if ENGINE is None:
        raise HTTPException(
            status_code=503,
            detail="Model has not been trained. Run python recforge.py first.",
        )

    if user_id not in ENGINE.user_histories:
        raise HTTPException(
            status_code=404,
            detail="Unknown user",
        )

    start = time.perf_counter()

    recommendations = ENGINE.recommend(
        user_id,
        TOP_K,
    )

    explanations = ENGINE.explain(
        user_id,
        recommendations,
    )

    latency = (
        time.perf_counter() - start
    ) * 1000

    return {
        "user_id": user_id,
        "latency_ms": round(latency, 2),
        "recommendations": [
            {
                "movie_id": movie_id,
                "title": explanations[movie_id]["title"],
                "reason": explanations[movie_id]["reason"],
            }
            for movie_id in recommendations
        ],
    }


@app.get("/context/{user_id}")
def context(user_id: int):

    if ENGINE is None:
        raise HTTPException(
            status_code=503,
            detail="Model has not been trained.",
        )

    if user_id not in ENGINE.user_histories:
        raise HTTPException(
            status_code=404,
            detail="Unknown user",
        )

    c = ENGINE.context_engine.build(
        user_id,
        ENGINE.user_histories[user_id],
    )

    return {
        "user_id": user_id,
        "raw_history_length": len(
            ENGINE.user_histories[user_id]
        ),
        "compressed_history_length": len(
            c.recent_history
        ),
        "genre_preferences": c.genre_preferences,
        "context": c.text,
    }


@app.get("/evaluate")
def evaluate_endpoint():

    global EVALUATION

    if ENGINE is None:
        raise HTTPException(
            status_code=503,
            detail="Model has not been trained.",
        )

    if EVALUATION is None:
        EVALUATION = evaluate(ENGINE)

    return EVALUATION


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="RecForge end-to-end recommendation engine"
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
    )

    parser.add_argument(
        "--experiments",
        action="store_true",
    )

    parser.add_argument(
        "--user",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    start = time.perf_counter()

    model = build_engine()

    print(
        f"\n[system] complete build time: "
        f"{time.perf_counter() - start:.2f}s"
    )

    recommendations = model.recommend(
        args.user,
        TOP_K,
    )

    print("\n==============================")
    print(f"RECOMMENDATIONS FOR USER {args.user}")
    print("==============================")

    for rank, movie_id in enumerate(
        recommendations,
        1,
    ):
        print(
            f"{rank:>2}. "
            f"{model.movie_title(movie_id)}"
        )

    inspect_context(
        model,
        args.user,
    )

    if args.evaluate:
        evaluate(model)

    if args.experiments:
        run_experiments(model)


if __name__ == "__main__":
    main()
