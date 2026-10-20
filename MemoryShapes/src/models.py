import numpy as np
import torch
from torch import nn


def softmax(x):
    x = np.asarray(x, dtype=float)
    z = x - np.max(x)
    e = np.exp(z)
    return e / (e.sum() + 1e-12)


def q_learning_predict(df, learning_rate=0.2, beta=4.0, n_actions=4):
    """Teacher-forced Q-learning: predict each observed choice using prior history."""
    output = []
    for _, g in df.groupby("participant_id", sort=False):
        q = np.full(n_actions, 0.5, dtype=float)
        for row in g.sort_values("trial").itertuples(index=False):
            probs = softmax(beta * q)
            output.append((int(row.participant_id), int(row.trial), int(row.choice),
                           probs, float(row.reward)))
            q[int(row.choice)] += learning_rate * (float(row.reward) - q[int(row.choice)])
    return output


def episodic_memory_predict(df, decay=0.92, prior=0.5, beta=5.0, n_actions=4):
    """Recency-weighted memory of observed rewards for each action."""
    output = []
    for _, g in df.groupby("participant_id", sort=False):
        memories = [[] for _ in range(n_actions)]
        for row in g.sort_values("trial").itertuples(index=False):
            values = np.full(n_actions, prior, dtype=float)
            for a, history in enumerate(memories):
                if history:
                    ages = np.arange(len(history)-1, -1, -1)
                    weights = decay ** ages
                    vals = np.asarray(history, dtype=float)
                    values[a] = np.sum(weights * vals) / np.sum(weights)
            probs = softmax(beta * values)
            output.append((int(row.participant_id), int(row.trial), int(row.choice),
                           probs, float(row.reward)))
            memories[int(row.choice)].append(float(row.reward))
    return output


def build_sequences(df, n_actions=4):
    """Sequence features at t contain only events strictly before t."""
    xs, ys, meta = [], [], []
    for pid, g in df.groupby("participant_id", sort=False):
        g = g.sort_values("trial")
        events = np.zeros((len(g), n_actions + 1), dtype=np.float32)
        choices = g["choice"].astype(int).to_numpy()
        rewards = g["reward"].astype(float).to_numpy()
        for t in range(len(g)):
            xs.append(events.copy())
            ys.append(int(choices[t]))
            meta.append((int(pid), int(g.iloc[t]["trial"])))
            # Each sequence is represented as a prefix padded to the same length.
            if t:
                xs[-1][:t] = events[:t]
            if t < len(g):
                events[t, choices[t]] = 1.0
                events[t, -1] = rewards[t]
    return np.asarray(xs), np.asarray(ys), meta


def sequence_tensor_for_group(g, n_actions=4):
    """Return one causal history tensor per trial; no current-trial event leakage."""
    g = g.sort_values("trial")
    n = len(g)
    X = np.zeros((n, n, n_actions + 1), dtype=np.float32)
    choices = g["choice"].astype(int).to_numpy()
    rewards = g["reward"].astype(float).to_numpy()
    for t in range(n):
        if t > 0:
            for j in range(t):
                X[t, j, choices[j]] = 1.0
                X[t, j, -1] = rewards[j]
    return X, choices


class HistoryGRU(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32, n_actions=4):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                  nn.Linear(hidden_dim, n_actions))

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])


class HybridQGRU(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32, n_actions=4):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.memory_head = nn.Linear(hidden_dim, n_actions)
        self.q_scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.zeros(n_actions))

    def forward(self, x, q_values):
        _, h = self.gru(x)
        memory_logits = self.memory_head(h[-1])
        return self.q_scale * q_values + memory_logits + self.bias


def q_values_before_trials(g, learning_rate=0.2, n_actions=4):
    q = np.full(n_actions, 0.5, dtype=np.float32)
    out = []
    for row in g.sort_values("trial").itertuples(index=False):
        out.append(q.copy())
        a = int(row.choice)
        q[a] += learning_rate * (float(row.reward) - q[a])
    return np.asarray(out, dtype=np.float32)


def neural_parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
