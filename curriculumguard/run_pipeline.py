"""python run_pipeline.py [--csv path] [--embed]  -> outputs/report.md + CSV tables."""
import argparse
from pathlib import Path
from cg.pipeline import load_or_generate, run_all
from cg.report import build_report

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default=None, help="feedback CSV: group,stance,text,severity[,subgroup]")
ap.add_argument("--embed", action="store_true", help="use sentence-transformers if installed")
a = ap.parse_args()
df = load_or_generate(a.csv) if a.csv else load_or_generate()
res = run_all(df, method="embed" if a.embed else "tfidf")
out = Path("outputs"); out.mkdir(exist_ok=True)
(out / "report.md").write_text(build_report(res), encoding="utf-8")
for k in ("schemes", "audit", "power", "flips", "policy", "extractor_eval"):
    res[k].to_csv(out / f"{k}.csv", index=False)
print(res["schemes"].round(3).to_string(index=False))
print("\nFlip thresholds:\n", res["flips"].to_string(index=False))
c = res["clusters"]
print(f"\nClusters ({c['method']}): silhouette={c['silhouette']:.3f} stability(ARI)={c['stability_ari']:.3f}")
print("Retrieval hit@2:", res["retriever"].evaluate()[1])
print("\nReport written to outputs/report.md")
