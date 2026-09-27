from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_distribution(distribution, title, output_path=None):
    labels = list(distribution.keys())
    values = list(distribution.values())

    plt.figure(figsize=(8, 4))
    sns.barplot(x=labels, y=values)
    plt.title(title)
    plt.ylabel("Probability")
    plt.xticks(rotation=30)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=180)

    return plt.gca()


def plot_similarity_matrix(matrix, labels, title, output_path=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        np.asarray(matrix),
        xticklabels=labels,
        yticklabels=labels,
        cmap="viridis",
        square=True,
    )
    plt.title(title)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=180)

    return plt.gca()
