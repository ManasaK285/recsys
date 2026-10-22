from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot_results(csv_path, output_path):
    df=pd.read_csv(csv_path)
    if {"model","accuracy"}.issubset(df.columns):
        ax=df.groupby("model")["accuracy"].mean().plot(kind="bar")
        ax.set_ylabel("Accuracy"); ax.set_title("Held-out preference accuracy")
        plt.tight_layout(); Path(output_path).parent.mkdir(parents=True,exist_ok=True)
        plt.savefig(output_path,dpi=160); plt.close()
