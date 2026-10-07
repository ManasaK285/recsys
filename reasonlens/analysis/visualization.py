"""Generates static figures used by the Streamlit dashboard and reports."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_grant_rate_by_reason(df: pd.DataFrame, out_dir: str) -> str:
    rates = df.groupby("reason_type")["granted"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    rates.plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_ylabel("Grant rate")
    ax.set_xlabel("Reason type")
    ax.set_title("Permission grant rate by reason type")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path = os.path.join(out_dir, "grant_rate_by_reason.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_grant_rate_by_app(df: pd.DataFrame, out_dir: str) -> str:
    rates = df.groupby("app_type")["granted"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    rates.plot(kind="bar", ax=ax, color="#55A868")
    ax.set_ylabel("Grant rate")
    ax.set_xlabel("App type")
    ax.set_title("Permission grant rate by app type")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path = os.path.join(out_dir, "grant_rate_by_app.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_heatmap_reason_app(df: pd.DataFrame, out_dir: str) -> str:
    pivot = df.pivot_table(index="app_type", columns="reason_type", values="granted", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not pd.isna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="Grant rate")
    ax.set_title("Grant rate: app type x reason type")
    plt.tight_layout()
    path = os.path.join(out_dir, "heatmap_reason_app.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_trust_by_reason(df: pd.DataFrame, out_dir: str) -> str:
    means = df.groupby("reason_type")[["app_trust", "android_trust", "privacy_concern"]].mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    means.plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean Likert score (1-5)")
    ax.set_xlabel("Reason type")
    ax.set_title("Trust & privacy concern by reason type")
    plt.xticks(rotation=30, ha="right")
    plt.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(out_dir, "trust_by_reason.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_response_time_by_reason(df: pd.DataFrame, out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    df.boxplot(column="response_time_ms", by="reason_type", ax=ax, rot=30)
    ax.set_title("Response time by reason type")
    ax.set_ylabel("Response time (ms)")
    plt.suptitle("")
    plt.tight_layout()
    path = os.path.join(out_dir, "response_time_by_reason.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_all(df: pd.DataFrame, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    return {
        "grant_rate_by_reason": plot_grant_rate_by_reason(df, out_dir),
        "grant_rate_by_app": plot_grant_rate_by_app(df, out_dir),
        "heatmap_reason_app": plot_heatmap_reason_app(df, out_dir),
        "trust_by_reason": plot_trust_by_reason(df, out_dir),
        "response_time_by_reason": plot_response_time_by_reason(df, out_dir),
    }
