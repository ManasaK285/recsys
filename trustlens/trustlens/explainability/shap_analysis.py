
from pathlib import Path
import pandas as pd,matplotlib.pyplot as plt
def main():
 p=Path("results/metrics/linguistic_importance.csv")
 if not p.exists(): return
 d=pd.read_csv(p).sort_values("importance"); Path("results/figures").mkdir(parents=True,exist_ok=True)
 plt.figure(figsize=(8,5)); plt.barh(d.feature,d.importance); plt.xlabel("Absolute coefficient"); plt.title("Linguistic feature importance"); plt.tight_layout(); plt.savefig("results/figures/linguistic_importance.png",dpi=160); plt.close()
if __name__=="__main__": main()
