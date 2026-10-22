from src.data.synthetic_raters import make_synthetic_data, split_by_prompt
from src.training.train import train_models
from src.evaluation.metrics import evaluate_probabilities

def heterogeneity_sweep(levels=(0,.4,.8,1.2), seeds=(11,22,33), n_users=50, n_prompts=1200, epochs=200):
    rows=[]
    for h in levels:
        for seed in seeds:
            data,_=make_synthetic_data(n_users,n_prompts,heterogeneity=h,seed=seed)
            tr,te=split_by_prompt(data,seed=seed)
            models=train_models(tr,n_users,epochs=epochs,seed=seed)
            for name in ("global","adaptive"):
                p=models[name].predict_proba(te.features_a,te.features_b,te.user_ids)
                rows.append({"heterogeneity":h,"seed":seed,"model":name,
                             **evaluate_probabilities(te.labels,p)})
    return rows
