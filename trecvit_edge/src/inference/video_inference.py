import torch
from src.data.video_dataset import sample_video_frames
from src.models.trecvit_inspired import TRecViTInspired
from src.models.baselines import CNNGRUBaseline

def load_model(checkpoint_path, device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cls = TRecViTInspired if ckpt["model_type"] == "trecvit" else CNNGRUBaseline
    model = cls(ckpt["num_classes"], dim=ckpt.get("dim", 128)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    idx_to_label = {int(i): name for name, i in ckpt["labels"].items()}
    return model, ckpt, idx_to_label, device

@torch.no_grad()
def predict_video(path, checkpoint_path):
    model, ckpt, idx_to_label, device = load_model(checkpoint_path)
    clip = sample_video_frames(path, ckpt["frames"], ckpt["image_size"]).unsqueeze(0).to(device)
    logits = model(clip).mean(dim=1)
    probs = logits.softmax(-1)[0]
    pred = int(probs.argmax().item())
    return {
        "label": idx_to_label[pred],
        "confidence": float(probs[pred].item()),
        "probabilities": {idx_to_label[i]: float(probs[i].item()) for i in range(len(probs))},
        "frame_predictions": [
            idx_to_label[int(row.argmax().item())] for row in model(clip)[0]
        ],
    }
