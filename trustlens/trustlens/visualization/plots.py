
from pathlib import Path
import pandas as pd,matplotlib.pyplot as plt
def plot(path,out,title):
 p=Path(path)
 if not p.exists(): return
 d=pd.read_csv(p); Path("results/figures").mkdir(parents=True,exist_ok=True)
 plt.figure(figsize=(8,5)); plt.bar(d.model,d.macro_f1); plt.ylabel("Macro F1"); plt.title(title); plt.xticks(rotation=20); plt.tight_layout(); plt.savefig(out,dpi=160); plt.close()
def main():
 plot("results/metrics/source_detection.csv","results/figures/source_detection.png","Human vs AI source prediction")
 plot("results/metrics/agreement.csv","results/figures/agreement.png","Agreement prediction")
if __name__=="__main__": main()
