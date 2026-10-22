import numpy as np
from src.data.synthetic_raters import make_synthetic_data,split_by_prompt
from src.training.train import train_models

def test_models_predict_probabilities():
    d,_=make_synthetic_data(n_users=10,n_prompts=100,seed=4)
    tr,te=split_by_prompt(d,seed=4)
    models=train_models(tr,n_users=10,epochs=3,seed=4)
    for m in models.values():
        p=m.predict_proba(te.features_a,te.features_b,te.user_ids)
        assert p.shape==(len(te),)
        assert np.all((p>=0)&(p<=1))
