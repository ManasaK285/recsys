from __future__ import annotations
import argparse
from embedbench.benchmarking import run_benchmark
from embedbench.models import DEFAULT_MODEL

def main():
    p = argparse.ArgumentParser(description="Run EmbedBench sample benchmark.")
    p.add_argument("--out", default="outputs")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--dimensions", nargs="+", type=int, default=[768,512,256,128])
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--no-faiss", action="store_true")
    args = p.parse_args()
    results, summary = run_benchmark(args.out, args.model, args.dimensions,
                                     args.device, args.batch_size, args.top_k,
                                     use_faiss=not args.no_faiss)
    print(results.to_string(index=False))
    print(f"\nSaved results to {args.out}; embedding backend: {summary['backend']}")
    if summary.get("fallback_note"):
        print("NOTE: model load failed; used TF-IDF fallback. Reason:", summary["fallback_note"])

if __name__ == "__main__":
    main()
