from __future__ import annotations
import numpy as np
import pandas as pd

def demo_data(n: int = 600) -> pd.DataFrame:
    """Clearly synthetic data for UI smoke-testing only; never use as research evidence."""
    rng = np.random.default_rng(7)
    prompts = [
        "A community leader speaking at a town hall",
        "A family celebrating a cultural festival",
        "A person working as a surgeon",
        "A group of friends playing in a park",
        "A protest taking place in a city",
        "A news photograph of a public official",
    ]
    topics = ["Bias/Gender", "Bias/Ethnicity", "Violent/Other", "Explicit/Other"]
    ethnicities = ["eastasian", "southasian", "white", "black", "latinx"]
    genders = ["man", "woman"]
    ages = ["genz", "millennial", "genx"]
    out = []
    for i in range(n):
        prompt = prompts[i % len(prompts)]
        topic = topics[(i // 3) % len(topics)]
        group_shift = 0.25 if (i % 5 in (0, 1)) else 0.0
        score = int(np.clip(round(rng.normal(1.7 + group_shift + (0.8 if "Violent" in topic else 0), 1.0)), 0, 4))
        out.append({
            "text_prompt": prompt, "topic": topic, "image_id": f"demo-{i//3}",
            "rater_ethnicity": rng.choice(ethnicities), "rater_gender": rng.choice(genders),
            "rater_age_group": rng.choice(ages), "how_harmful_to_you": score,
            "how_harmful_to_others": int(np.clip(score + rng.integers(-1, 2), 0, 4)),
            "why_harmful": rng.choice(["not harmful", "stereotype", "violent", "other"])
        })
    return pd.DataFrame(out)
