from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .metrics import summarize_results, paired_identity_sensitivity, selection_disparity

def make_reports(df, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize_results(df)
    paired = paired_identity_sensitivity(df)
    disparity = selection_disparity(df)
    summary.to_csv(out / "summary_metrics.csv", index=False)
    paired.to_csv(out / "identity_sensitivity.csv", index=False)
    disparity.to_csv(out / "selection_disparity.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_data = summary[summary["condition"] == "visible"]
    if not plot_data.empty:
        for protocol, part in plot_data.groupby("protocol"):
            ax.plot(part["model"], part["mean_optimality_gap"], marker="o", label=protocol)
        ax.set_ylabel("Mean optimality gap (lower is better)")
        ax.set_title("Synthetic benchmark: task-performance gap")
        ax.legend()
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(out / "optimality_gap.png", dpi=160)
    plt.close(fig)

    if not paired.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        for protocol, part in paired.groupby("protocol"):
            ax.bar([f"{m}\n{protocol}" for m in part["model"]],
                   part["identity_sensitivity"], label=protocol)
        ax.set_ylabel("Fraction of matched trials with changed winner")
        ax.set_title("Synthetic benchmark: identity sensitivity")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(out / "identity_sensitivity.png", dpi=160)
        plt.close(fig)

    lines = [
        "# CollectiveAlign Lab — Run Report", "",
        "**Data status:** synthetic simulation only; not human-subject data or a replication of the paper.", "",
        f"- Trial rows: {len(df)}",
        f"- Models: {', '.join(sorted(df['model'].unique()))}",
        f"- Protocols: {', '.join(sorted(df['protocol'].unique()))}", "",
        "## Mean metrics", "",
        summary.to_markdown(index=False), "",
        "## Interpretation cautions", "",
        "- Toy agents encode explicit rules and do not represent real LLMs.",
        "- Selection disparity is descriptive and can be affected by qualification distributions.",
        "- Identity sensitivity measures changed winners, not intent or a complete fairness judgment.",
        "- For inferential claims, pre-register hypotheses, justify sample size, and use appropriate human reference data."
    ]
    (out / "REPORT.md").write_text("\\n".join(lines), encoding="utf-8")
