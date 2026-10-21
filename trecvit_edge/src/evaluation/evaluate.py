import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from src.data.video_dataset import VideoFolderDataset
from src.models.trecvit_inspired import TRecViTInspired
from src.models.baselines import CNNGRUBaseline

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/videos")
    p.add_argument("--checkpoint", default="artifacts/best.pt")
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    labels = ckpt["labels"]
    ds = VideoFolderDataset(args.data_dir, ckpt["frames"], ckpt["image_size"])
    if ds.label_to_idx != labels:
        raise SystemExit(f"Dataset labels {ds.label_to_idx} do not match checkpoint labels {labels}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    cls = TRecViTInspired if ckpt["model_type"] == "trecvit" else CNNGRUBaseline
    model = cls(ckpt["num_classes"], dim=ckpt.get("dim",128)).eval()
    model.load_state_dict(ckpt["model_state"])
    ys, preds = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.extend(model(x).mean(1).argmax(-1).tolist())
            ys.extend(y.tolist())
    names = [name for name, _ in sorted(labels.items(), key=lambda kv: kv[1])]
    result = {
        "accuracy": accuracy_score(ys, preds),
        "macro_f1": f1_score(ys, preds, average="macro", zero_division=0),
        "classification_report": classification_report(ys, preds, target_names=names, zero_division=0, output_dict=True),
        "confusion_matrix": confusion_matrix(ys, preds).tolist(),
        "note": "This evaluates all videos in --data-dir; use a held-out test folder for an unbiased test score."
    }
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/evaluation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"accuracy": result["accuracy"], "macro_f1": result["macro_f1"],
                      "confusion_matrix": result["confusion_matrix"]}, indent=2))
    print("Full report saved to artifacts/evaluation.json")

if __name__ == "__main__":
    main()
