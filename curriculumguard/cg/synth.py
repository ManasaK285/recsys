"""SYNTHETIC feedback generator. Data are simulated, not real people: findings are demonstrations of the method."""
import numpy as np
import pandas as pd

PARTICIPATION = {"student": .20, "teacher": .40, "parent": .25, "admin": .15}  # skewed on purpose
STANCE_P = {"student": [.45, .40, .15], "teacher": [.05, .30, .65],
            "parent": [.15, .40, .45], "admin": [.15, .65, .20]}  # open, constrained, restricted
CONCERN_P = {
    "student": {"accessibility": .25, "privacy": .15, "academic_integrity": .20, "language_access": .20, "teacher_workload": .05},
    "teacher": {"accessibility": .10, "privacy": .25, "academic_integrity": .70, "language_access": .10, "teacher_workload": .70},
    "parent":  {"accessibility": .15, "privacy": .70, "academic_integrity": .35, "language_access": .10, "teacher_workload": .05},
    "admin":   {"accessibility": .20, "privacy": .50, "academic_integrity": .40, "language_access": .10, "teacher_workload": .30},
}
SENT = {
    "accessibility": ["Students with disabilities need accessible tools that work with screen readers.",
                      "Text-to-speech and captioning support must be available to every learner.",
                      "Assistive features could really help some students if the tools are accessible."],
    "privacy": ["I worry about student data being collected and shared with vendors.",
                "Privacy of children's personal information needs strong protection.",
                "Nobody has explained the data collection behind these AI apps."],
    "academic_integrity": ["Students could cheat by having AI write their essays.",
                           "Academic integrity is at risk and AI detectors seem unreliable.",
                           "It is hard to know if submitted work is original work."],
    "language_access": ["Multilingual students could use translation to follow lessons.",
                        "English learners face a language barrier that AI translation might ease.",
                        "Families need language access, and translate tools can help."],
    "teacher_workload": ["Teachers already face a heavy workload and this adds extra work.",
                         "There is no training time or grading time to manage AI use.",
                         "Rolling this out will overwhelm staff and cause burnout."],
}
STANCE_TXT = {
    "open": ["I want AI tools allowed for learning.", "Let students use AI freely in class."],
    "constrained": ["AI should be allowed with clear rules and disclosure.", "Use AI for some tasks under teacher guidance."],
    "restricted": ["AI should be restricted in classrooms for now.", "I would limit AI use to specific supervised activities only at best."],
}
POL = ["open", "constrained", "restricted"]


def generate(n=300, seed=42):
    rng = np.random.default_rng(seed)
    groups = rng.choice(list(PARTICIPATION), size=n, p=list(PARTICIPATION.values()))
    rows = []
    for i, g in enumerate(groups):
        sub = "none"
        if g == "student":
            r = rng.random()
            sub = "multilingual" if r < .15 else "disability" if r < .27 else "none"
        probs = dict(CONCERN_P[g])
        if sub == "multilingual":
            probs["language_access"] = .8
        if sub == "disability":
            probs["accessibility"] = .8
        chosen = [c for c, p in probs.items() if rng.random() < p]
        if not chosen:
            chosen = [rng.choice(list(probs), p=np.array(list(probs.values())) / sum(probs.values()))]
        chosen = list(rng.permutation(chosen)[:2])
        stance = rng.choice(POL, p=STANCE_P[g])
        sev = int(rng.integers(1, 5))
        if (sub == "multilingual" and "language_access" in chosen) or (sub == "disability" and "accessibility" in chosen):
            sev = min(5, sev + 2)
        parts = [rng.choice(STANCE_TXT[stance])] + [rng.choice(SENT[c]) for c in chosen]
        rows.append({"id": i, "group": g, "subgroup": sub, "stance": stance, "severity": sev,
                     "text": " ".join(parts), "gold_concerns": "|".join(chosen)})
    return pd.DataFrame(rows)
