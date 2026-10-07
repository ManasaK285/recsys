"""Aggregation schemes, representation audit, power-aware risk, counterfactuals, bootstrap."""
import numpy as np
import pandas as pd
from .nlp import CONCERNS

POLICIES = ["open", "constrained", "restricted"]
SCHEMES = ["majority", "balanced", "population", "impact"]
# Assumed stakeholder shares of the whole community (edit for your district)
DEFAULT_POP = {"student": 0.50, "parent": 0.30, "teacher": 0.15, "admin": 0.05}
SUBPOP = {"multilingual": 0.15, "disability": 0.12}  # assumed share of students


def weights(df, scheme, pop=None):
    """Per-participant weights (sum to 1) for an aggregation scheme."""
    pop = pop or DEFAULT_POP
    g = df["group"]
    n = g.value_counts()
    if scheme == "majority":
        w = np.ones(len(df))
    elif scheme == "impact":
        w = df["severity"].to_numpy(float)
    elif scheme in ("balanced", "population"):
        present = list(n.index)
        t = {k: 1 / len(present) for k in present} if scheme == "balanced" else {k: pop[k] for k in present}
        s = sum(t.values())
        w = g.map(lambda k: t[k] / s / n[k]).to_numpy(float)
    else:
        raise ValueError(f"unknown scheme {scheme}")
    return w / w.sum()


def tally(df, w):
    st = df["stance"].to_numpy()
    shares = {p: float(w[st == p].sum()) for p in POLICIES}
    return shares, max(shares, key=shares.get)


def compare_schemes(df, pop=None):
    rows = []
    for s in SCHEMES:
        sh, win = tally(df, weights(df, s, pop))
        rows.append({"scheme": s, **sh, "winner": win})
    return pd.DataFrame(rows)


def representation_audit(df, pop=None, low=0.75, high=1.25):
    pop = pop or DEFAULT_POP
    part = df["group"].value_counts(normalize=True)
    cnt = df["group"].value_counts()
    rows = []
    for g, p in pop.items():
        pr = float(part.get(g, 0.0))
        ratio = pr / p
        rows.append({"group": g, "n": int(cnt.get(g, 0)), "participation": pr, "population": p,
                     "ratio": ratio, "status": "under" if ratio < low else "over" if ratio > high else "ok"})
    return pd.DataFrame(rows)


def subgroup_audit(df, k=5):
    """Student subgroups vs assumed shares; counts below k are suppressed (k-anonymity)."""
    st = df[df["group"] == "student"]
    rows = []
    for sg, p in SUBPOP.items():
        n = int((st["subgroup"] == sg).sum())
        sup = n < k
        rows.append({"subgroup": sg, "n": "<%d" % k if sup else n,
                     "share_of_students": None if sup else n / max(len(st), 1),
                     "assumed_population_share": p, "suppressed": sup})
    return pd.DataFrame(rows)


def _indicator(df):
    return pd.DataFrame({c: df["concerns"].apply(lambda l: c in l) for c in CONCERNS}).astype(float)


def concern_matrix(df):
    return _indicator(df).groupby(df["group"]).mean()


def retention(df, scheme, pop=None, top_k=3):
    """Share of each group's top-k concerns that survive in the aggregate top-k under a scheme."""
    M = _indicator(df)
    w = weights(df, scheme, pop)
    agg_top = set(M.mul(w, axis=0).sum().nlargest(top_k).index)
    out = {}
    for g in df["group"].unique():
        own = set(M[df["group"] == g].mean().nlargest(top_k).index)
        out[g] = len(own & agg_top) / top_k
    return out


def power_risk(df, pop=None, min_sev=4.0, min_prev=0.25):
    """risk = prevalence * severity/5 * voice_deficit (population share / participation share, clipped)."""
    audit = representation_audit(df, pop).set_index("group")
    M = _indicator(df)
    rows = []
    for g in audit.index:
        mg = df["group"] == g
        if mg.sum() == 0:
            continue
        deficit = float(np.clip(1 / audit.loc[g, "ratio"], 0.25, 5)) if audit.loc[g, "ratio"] > 0 else 5.0
        for c in CONCERNS:
            m = mg & (M[c] == 1)
            prev = float(m.sum() / mg.sum())
            sev = float(df.loc[m, "severity"].mean()) if m.any() else 0.0
            rows.append({"group": g, "concern": c, "prevalence": prev, "mean_severity": sev,
                         "voice_deficit": deficit, "risk": prev * sev / 5 * deficit,
                         "underrepresented": bool(audit.loc[g, "ratio"] < 1),
                         "high_impact": bool(sev >= min_sev and prev >= min_prev and audit.loc[g, "ratio"] < 1)})
    return pd.DataFrame(rows).sort_values("risk", ascending=False).reset_index(drop=True)


def cf_weights(df, group, share):
    """Set `group` to `share` of total weight; other groups keep their relative participation."""
    cur = df["group"].value_counts(normalize=True)
    n = df["group"].value_counts()
    if group not in cur:
        raise ValueError(f"group {group} not in data")
    others = [g for g in cur.index if g != group]
    rest = cur[others].sum()
    t = {group: share}
    for o in others:
        t[o] = (1 - share) * cur[o] / rest if rest else 0.0
    w = df["group"].map(lambda k: t.get(k, 0.0) / n[k]).to_numpy(float)
    return w / w.sum()


def counterfactual(df, group, share):
    sh, win = tally(df, cf_weights(df, group, share))
    return {"group": group, "share": share, **sh, "winner": win}


def sweep(df, group, step=0.05):
    return pd.DataFrame([counterfactual(df, group, float(s)) for s in np.arange(0, 1 + 1e-9, step)])


def flip_threshold(df, group, step=0.005):
    """Smallest participation-share change for `group` that changes the winning policy."""
    cur = float((df["group"] == group).mean())
    base = counterfactual(df, group, cur)["winner"]
    flips = []
    for s in np.arange(0, 1 + 1e-9, step):
        win = counterfactual(df, group, float(s))["winner"]
        if win != base:
            flips.append((abs(s - cur), float(s), win))
    if not flips:
        return {"group": group, "current_share": cur, "baseline_winner": base, "flips": False}
    _, s, win = min(flips)
    return {"group": group, "current_share": cur, "baseline_winner": base, "flips": True,
            "threshold_share": round(s, 3), "new_winner": win, "change_pp": round((s - cur) * 100, 1)}


def bootstrap(df, scheme, pop=None, n=300, seed=0):
    """Stratified bootstrap: CI on each policy's share and how often each policy wins."""
    df = df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    idxs = [d.index.to_numpy() for _, d in df.groupby("group")]
    rows = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(ix, len(ix)) for ix in idxs])
        d = df.loc[idx].reset_index(drop=True)
        sh, win = tally(d, weights(d, scheme, pop))
        rows.append({**sh, "winner": win})
    b = pd.DataFrame(rows)
    out = [{"policy": p, "mean": b[p].mean(), "ci_low": b[p].quantile(.025), "ci_high": b[p].quantile(.975),
            "win_freq": float((b["winner"] == p).mean())} for p in POLICIES]
    return pd.DataFrame(out)
