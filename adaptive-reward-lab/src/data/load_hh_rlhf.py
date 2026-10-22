def load_hh_rlhf(max_rows=5000, split="train"):
    """Optional real-text data loader; needs datasets package and internet access."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install optional dependency with pip install datasets") from exc
    ds = load_dataset("Anthropic/hh-rlhf", split=split)
    ds = ds.select(range(min(max_rows, len(ds)))) if max_rows else ds
    return [{"prompt_id": i, "chosen": row["chosen"], "rejected": row["rejected"]}
            for i, row in enumerate(ds)]
