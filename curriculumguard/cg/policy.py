"""Policy comparison. Does NOT pick a best policy: it shows support and unaddressed-concern burden per scheme."""
import pandas as pd
from .governance import POLICIES, SCHEMES, weights, tally, _indicator
from .nlp import CONCERNS

# ASSUMPTION (editable): how much of each concern a policy leaves unaddressed (0 = fully, 1 = not at all).
EXPOSURE = {
    "open":        {"accessibility": .2, "privacy": .9, "academic_integrity": .9, "language_access": .3, "teacher_workload": .5},
    "constrained": {"accessibility": .3, "privacy": .4, "academic_integrity": .4, "language_access": .4, "teacher_workload": .5},
    "restricted":  {"accessibility": .8, "privacy": .1, "academic_integrity": .1, "language_access": .8, "teacher_workload": .3},
}


def compare_policies(df, pop=None):
    M = _indicator(df)
    rows = []
    for s in SCHEMES:
        w = weights(df, s, pop)
        shares, _ = tally(df, w)
        sev = {c: (float((w * M[c] * df["severity"]).sum() / (w * M[c]).sum()) if (w * M[c]).sum() else 0.0) for c in CONCERNS}
        prev = {c: float((w * M[c]).sum()) for c in CONCERNS}
        for p in POLICIES:
            burden = sum(prev[c] * sev[c] / 5 * EXPOSURE[p][c] for c in CONCERNS)
            rows.append({"scheme": s, "policy": p, "support": shares[p], "unaddressed_burden": burden})
    return pd.DataFrame(rows)
