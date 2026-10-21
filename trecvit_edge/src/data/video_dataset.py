from pathlib import Path
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg"}

def sample_video_frames(path, num_frames=8, image_size=112):
    """Uniformly sample frames from a video; returns [T, 3, H, W] float tensor."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        frames = []
        while len(frames) < num_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            raise RuntimeError(f"No readable frames in {path}")
        indices = np.linspace(0, len(frames) - 1, num_frames).round().astype(int)
        selected = [frames[i] for i in indices]
    else:
        indices = np.linspace(0, max(total - 1, 0), num_frames).round().astype(int)
        selected = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok:
                selected.append(frame)
        cap.release()
        if not selected:
            raise RuntimeError(f"No readable frames in {path}")
        while len(selected) < num_frames:
            selected.append(selected[-1].copy())
        selected = selected[:num_frames]

    tensors = []
    for frame in selected:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
        arr = frame.astype(np.float32) / 255.0
        arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(tensors)

class VideoFolderDataset(Dataset):
    """Expects root/class_name/video_file layout."""
    def __init__(self, root, num_frames=8, image_size=112, indices=None, samples=None, label_to_idx=None):
        self.root = Path(root)
        self.num_frames = num_frames
        self.image_size = image_size
        if samples is None:
            classes = sorted(p.name for p in self.root.iterdir() if p.is_dir())
            self.label_to_idx = {name: i for i, name in enumerate(classes)}
            items = []
            for class_name in classes:
                for path in sorted((self.root / class_name).rglob("*")):
                    if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
                        items.append((path, self.label_to_idx[class_name]))
            self.samples = items
        else:
            self.samples = samples
            self.label_to_idx = label_to_idx or {}
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]
        if not self.samples:
            raise ValueError(f"No videos found in {self.root}. Expected class/video files.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        try:
            clip = sample_video_frames(path, self.num_frames, self.image_size)
        except Exception as exc:
            # Fail loudly: silently replacing broken videos can corrupt labels and metrics.
            raise RuntimeError(f"Failed loading sample {path}: {exc}") from exc
        return clip, torch.tensor(label, dtype=torch.long)
