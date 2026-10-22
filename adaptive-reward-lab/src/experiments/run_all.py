import argparse, json
from pathlib import Path
import pandas as pd
from src.data.synthetic_raters import make_synthetic_data,split_by_prompt
from src.training.train import train_models
from src.evaluation.metrics import evaluate_probabilities
from src.evaluation.disagreement import heterogeneity_sweep
from src.experiments.plot_results import plot_results

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--quick",action="store_true")
    args=parser.parse_args()
    users,n,epochs=(20,500,60) if args.quick else (80,5000,300)
    out=Path("artifacts/results"); out.mkdir(parents=True,exist_ok=True)
    Path("artifacts/figures").mkdir(parents=True,exist_ok=True)
    data,_=make_synthetic_data(n_users=users,n_prompts=n,seed=42)
    tr,te=split_by_prompt(data,seed=42)
    models=train_models(tr,n_users=users,epochs=epochs,seed=42)
    rows=[]
    for name,model in models.items():
        p=model.predict_proba(te.features_a,te.features_b,te.user_ids)
        rows.append({"model":name,**evaluate_probabilities(te.labels,p)})
    pd.DataFrame(rows).to_csv(out/"model_comparison.csv",index=False)
    (out/"model_comparison.json").write_text(json.dumps(rows,indent=2))
    plot_results(out/"model_comparison.csv","artifacts/figures/model_comparison.png")
    sweep=heterogeneity_sweep(levels=(0,.6,1.2) if args.quick else (0,.25,.5,.8,1.2),
        seeds=(11,) if args.quick else (11,22,33),n_users=20 if args.quick else 50,
        n_prompts=300 if args.quick else 1500,epochs=40 if args.quick else 200)
    pd.DataFrame(sweep).to_csv(out/"heterogeneity_sweep.csv",index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print("Results saved under artifacts/results/")
if __name__=="__main__": main()
