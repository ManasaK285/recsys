import os
import matplotlib.pyplot as plt

def plot_governance(summary,path):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    names=[r["governance"] for r in summary]; vals=[r["exploit_adoption_rate"] for r in summary]
    plt.figure(figsize=(9,5)); plt.bar(names,vals); plt.ylabel("Exploit adoption rate"); plt.xticks(rotation=20); plt.tight_layout(); plt.savefig(path,dpi=160); plt.close()

def plot_tradeoff(summary,path):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    x=[r["communication"] for r in summary]; y=[r["exploit_adoption_rate"] for r in summary]
    plt.figure(figsize=(8,5)); plt.scatter(x,y); 
    for r in summary: plt.annotate(r["governance"],(r["communication"],r["exploit_adoption_rate"]))
    plt.xlabel("Communication events"); plt.ylabel("Exploit adoption rate"); plt.tight_layout(); plt.savefig(path,dpi=160); plt.close()
