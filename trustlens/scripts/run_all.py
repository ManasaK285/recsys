
import subprocess,sys
cmds=[[sys.executable,"-m","trustlens.data.make_dataset"],[sys.executable,"experiments/01_source_detection.py"],[sys.executable,"experiments/02_agreement_prediction.py"],[sys.executable,"experiments/03_humanization.py"],[sys.executable,"experiments/04_ablation.py"],[sys.executable,"trustlens/explainability/shap_analysis.py"],[sys.executable,"trustlens/visualization/plots.py"]]
for c in cmds: print("\n>>>"," ".join(c)); subprocess.run(c,check=True)
print("TrustLens completed.")
