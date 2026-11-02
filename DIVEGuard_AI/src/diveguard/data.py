from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from .config import RAW_DIR, PROCESSED_DIR, DATASET_REPO

CANDIDATE_FILES = [
    "final_dv_t2i_dataset_more_rater_attributes.csv",
    "final_dv_t2i_dataset_upload.tsv",
]

def _find_local_file() -> Path | None:
    for name in CANDIDATE_FILES:
        p = RAW_DIR / name
        if p.exists():
            return p
    matches = list(RAW_DIR.glob("*.csv")) + list(RAW_DIR.glob("*.tsv"))
    return matches[0] if matches else None

def download_dataset() -> Path:
    """Download one of the dataset files from its Hugging Face repository."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install dependencies first: pip install -r requirements.txt") from exc
    last_error = None
    for filename in CANDIDATE_FILES:
        try:
            downloaded = hf_hub_download(
                repo_id=DATASET_REPO, filename=filename,
                repo_type="dataset", local_dir=str(RAW_DIR)
            )
            return Path(downloaded)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(
        "Could not download either known DIVE CSV/TSV file. "
        "Check the Hugging Face repository file names and network access. "
        f"Last error: {last_error}"
    )

def load_raw(path: str | Path | None = None, download: bool = False) -> pd.DataFrame:
    p = Path(path) if path else _find_local_file()
    if p is None and download:
        p = download_dataset()
    if p is None:
        raise FileNotFoundError(
            "DIVE data not found. Run `python -m src.diveguard.data --download` "
            "or place the downloaded CSV/TSV under data/raw/."
        )
    sep = "\t" if p.suffix.lower() in {".tsv", ".tab"} else ","
    return pd.read_csv(p, sep=sep, low_memory=False)

def clean_data(df: pd.DataFrame, target: str = "how_harmful_to_you") -> pd.DataFrame:
    """Standardize fields and keep valid, high-quality annotation rows."""
    out = df.copy()
    aliases = {
        "prompt": "text_prompt",
        "topic": "topic",
        "image_id": "image_id",
    }
    for required in ("text_prompt", "topic", "image_id"):
        if required not in out.columns:
            if required == "topic":
                out[required] = "unknown"
            else:
                raise ValueError(f"Required DIVE column missing: {required}")
    if target not in out.columns:
        raise ValueError(f"Target column `{target}` not found. Available: {list(out.columns)}")

    if "is_attention_check_image" in out:
        out = out[out["is_attention_check_image"].fillna(0).astype(str).str.lower().isin(["0", "0.0", "false"])]
    if "rater_was_filtered_out" in out:
        filtered = out["rater_was_filtered_out"].astype(str).str.lower()
        out = out[~filtered.isin(["true", "1", "1.0", "yes"])]

    out[target] = pd.to_numeric(out[target], errors="coerce")
    out = out[out[target].isin([0, 1, 2, 3, 4])].copy()
    out[target] = out[target].astype(int)
    out["text_prompt"] = out["text_prompt"].fillna("").astype(str)
    out["topic"] = out["topic"].fillna("unknown").astype(str)
    out["image_id"] = out["image_id"].fillna(out["text_prompt"]).astype(str)
    for col in out.columns:
        if col.startswith("rater_"):
            out[col] = out[col].fillna("unknown").astype(str)
    return out.reset_index(drop=True)

def save_processed(df: pd.DataFrame, target: str = "how_harmful_to_you") -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / f"dive_clean_{target}.parquet"
    try:
        df.to_parquet(path, index=False)
    except ImportError:
        path = PROCESSED_DIR / f"dive_clean_{target}.csv"
        df.to_csv(path, index=False)
    return path

def load_processed(target: str = "how_harmful_to_you") -> pd.DataFrame | None:
    p = PROCESSED_DIR / f"dive_clean_{target}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    p = PROCESSED_DIR / f"dive_clean_{target}.csv"
    if p.exists():
        return pd.read_csv(p)
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--target", default="how_harmful_to_you")
    args = parser.parse_args()
    raw = load_raw(download=args.download)
    clean = clean_data(raw, args.target)
    saved = save_processed(clean, args.target)
    print(f"Raw rows: {len(raw):,}")
    print(f"Clean labeled rows: {len(clean):,}")
    print(f"Saved: {saved}")
