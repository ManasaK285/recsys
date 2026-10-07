"""End-to-end orchestration."""
from pathlib import Path
import pandas as pd
from . import governance as gv
from .nlp import add_concerns, cluster_feedback, evaluate_extractor
from .policy import compare_policies
from .rag import Retriever
from .synth import generate

DATA = Path(__file__).resolve().parent.parent / "data" / "feedback.csv"


def load_or_generate(path=DATA):
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    df = generate()
    path.parent.mkdir(exist_ok=True)
    df.to_csv(path, index=False)
    return df


def run_all(df, pop=None, k=5, method="tfidf", n_boot=300):
    df = add_concerns(df).reset_index(drop=True)
    if "subgroup" not in df:
        df["subgroup"] = "none"
    res = {"df": df}
    res["schemes"] = gv.compare_schemes(df, pop)
    res["audit"] = gv.representation_audit(df, pop)
    res["subgroup"] = gv.subgroup_audit(df)
    res["concern_by_group"] = gv.concern_matrix(df)
    res["retention"] = pd.DataFrame({s: gv.retention(df, s, pop) for s in gv.SCHEMES}).T
    res["power"] = gv.power_risk(df, pop)
    groups = list(df["group"].unique())
    res["flips"] = pd.DataFrame([gv.flip_threshold(df, g) for g in groups])
    res["sweeps"] = {g: gv.sweep(df, g) for g in groups}
    res["boot"] = {s: gv.bootstrap(df, s, pop, n=n_boot) for s in ("majority", "balanced")}
    res["policy"] = compare_policies(df, pop)
    res["extractor_eval"] = evaluate_extractor(df)
    cl = cluster_feedback(df["text"], k=k, method=method)
    df["cluster"] = cl["labels"]
    res["clusters"] = cl
    res["retriever"] = Retriever()
    return res
