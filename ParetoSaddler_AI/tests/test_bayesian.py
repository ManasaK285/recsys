import pandas as pd
from paretosaddler.bayesian import bayesian_promotion_support

def frame(q, c):
    return pd.DataFrame([
        {"task_id": f"t{i}", "quality": q + j * 0.001, "cost": c}
        for i in range(5) for j in range(3)
    ])

def test_strong_quality_gain_promotes():
    result = bayesian_promotion_support(frame(.50, .02), frame(.80, .02),
                                        draws=500, gamma=.70, seed=1)
    assert result["promoted"]
    assert result["support"] > .70

def test_no_change_does_not_promote():
    result = bayesian_promotion_support(frame(.50, .02), frame(.50, .02),
                                        draws=500, gamma=.70, seed=1)
    assert not result["promoted"]
