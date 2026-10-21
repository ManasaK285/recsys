# TRecViT Edge: Lightweight Recurrent Video Classification

A research-inspired implementation of recurrent video modeling for lightweight, temporal action recognition. This project explores how spatial feature extraction and recurrent temporal processing can be combined to classify human activities such as **running** and **walking**.

The project is designed as an end-to-end experimentation pipeline, including video loading, model training, validation, checkpointing, inference, and an interactive demonstration interface.

> **Current status:** End-to-end training executes successfully. Initial experiments use a tiny four-video dataset, so the reported validation metrics are diagnostic only and do not establish reliable generalization.

## Table of Contents

- [Overview](#overview)
- [Research Reference](#research-reference)
- [Objectives](#objectives)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Training](#training)
- [Initial Experimental Results](#initial-experimental-results)
- [Evaluation and Limitations](#evaluation-and-limitations)
- [Next Steps](#next-steps)
- [Technology Stack](#technology-stack)
- [References](#references)

## Overview

Video action recognition requires models to learn both spatial information within individual frames and temporal relationships across consecutive frames.

Conventional approaches may process frames independently or use computationally expensive video transformers. Recurrent video architectures offer an alternative by maintaining temporal state as video frames are processed.

TRecViT Edge explores a lightweight, recurrent approach to this problem.

### Key features

- Video-based running and walking classification.
- Frame-level visual feature extraction.
- Recurrent temporal modeling.
- Spatial attention and channel-wise feature refinement.
- PyTorch training and checkpointing.
- Validation using accuracy and macro F1.
- Interactive Streamlit demonstration.
- A modular structure for future deployment and edge-device experiments.

## Research Reference

This project is inspired by the research on TRecViT, a recurrent video transformer architecture.

**Reference implementation:** [Google DeepMind — TRecViT](https://github.com/google-deepmind/trecvit)

The project explores related architectural ideas in a smaller experimental pipeline. It is an independent, research-inspired implementation and **is not an exact reproduction of the official TRecViT architecture or its published results**.

## Objectives

1. Build an end-to-end video classification pipeline.
2. Learn discriminative spatial representations from video frames.
3. Model temporal dependencies across frames using recurrent processing.
4. Investigate spatial attention and channel-wise feature refinement.
5. Compare the recurrent model against a CNN-GRU baseline.
6. Measure classification performance using consistent evaluation metrics.
7. Establish a foundation for lightweight video inference and edge deployment.

## Architecture

The pipeline processes a video as a sequence of frames.

```text
Input Video
    |
    v
Video Loading and Frame Sampling
    |
    v
Frame-Level Feature Encoder
    |
    v
Spatial Attention
    |
    v
Recurrent Temporal Processing
    |
    v
Channel-Wise Feature Refinement
    |
    v
Classification Head
    |
    v
Running / Walking Prediction
```

### Main components

**1. Video data pipeline**

Loads video files, extracts frames, applies preprocessing, and constructs model-ready tensors.

**2. Frame encoder**

Extracts visual representations from individual frames, providing spatial information for downstream temporal processing.

**3. Spatial attention**

Helps the model emphasize informative spatial features within the encoded representation.

**4. Recurrent temporal module**

Processes information across frames to learn temporal patterns associated with different actions.

**5. Channel-wise feature refinement**

Refines learned feature representations using a channel-processing module.

**6. Classification head**

Maps the final video representation to class scores for running and walking.

*Implementation note:* These components describe the intended research-inspired design. Exact layer definitions and tensor dimensions are determined by the implementation in `src/models/trecvit_inspired.py`.

## Project Structure

The project is organized into separate modules for data processing, modeling, training, evaluation, and inference.

```text
trecvit_edge/
├── configs/
├── data/
│   ├── videos/
│   │   ├── running/
│   │   └── walking/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data/
│   │   └── video_dataset.py
│   ├── models/
│   │   ├── trecvit_inspired.py
│   │   └── baselines.py
│   ├── training/
│   │   └── train.py
│   ├── evaluation/
│   ├── inference/
│   └── api/
├── artifacts/
│   └── best.pt
├── tests/
├── app.py
├── Dockerfile
├── requirements.txt
├── CITATION.cff
└── README.md
```

The exact files present may vary as development progresses.

## Dataset

The initial experiment uses two activity classes:

| Class | Label |
|---|---:|
| Running | 0 |
| Walking | 1 |

### Dataset organization

Place video files in their corresponding class directories:

```text
data/videos/
├── running/
│   ├── running_01.mp4
│   ├── running_02.mp4
│   └── ...
└── walking/
    ├── walking_01.mp4
    ├── walking_02.mp4
    └── ...
```

Supported video formats depend on the configured video-loading implementation and available OpenCV codecs.

### Initial dataset split

| Split | Videos | Running | Walking |
|---|---:|---:|---:|
| Training | 2 | 1 | 1 |
| Validation | 2 | 1 | 1 |
| **Total** | **4** | **2** | **2** |

This is a smoke-test dataset, not a representative benchmark. A single incorrect prediction changes validation accuracy by 50 percentage points.

For meaningful experiments, expand the dataset and split by original recording or subject where appropriate to prevent data leakage.

## Installation

### Requirements

- Python 3.11 recommended for the current setup.
- PyTorch.
- OpenCV.
- NumPy.
- scikit-learn.
- tqdm.
- Streamlit for the demonstration interface.

The complete dependency list is maintained in `requirements.txt`.

### Windows setup

Open PowerShell in the project directory.

Create a virtual environment:

```powershell
py -3.11 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If Python 3.11 is not installed, install it first or use another Python version supported by the project's dependencies.

## Training

Run the training pipeline:

```powershell
python -m src.training.train --data-dir data/videos --model trecvit --epochs 10 --batch-size 2
```

The training process:

1. Loads videos and assigns class labels.
2. Creates training and validation splits.
3. Constructs the selected model.
4. Trains for the configured number of epochs.
5. Computes validation accuracy and macro F1.
6. Saves the best checkpoint.

The current run produced:

```text
Best checkpoint: artifacts\best.pt
```

### Launch the demonstration interface

If the Streamlit interface is configured to load the saved checkpoint:

```powershell
streamlit run app.py
```

The available interface functionality depends on the current implementation.

## Initial Experimental Results

The first end-to-end run completed 10 epochs using four videos.

### Configuration

| Parameter | Value |
|---|---|
| Model | TRecViT-inspired |
| Task | Binary video classification |
| Classes | Running, walking |
| Total videos | 4 |
| Training videos | 2 |
| Validation videos | 2 |
| Epochs | 10 |
| Batch size | 2 |
| Checkpoint | `artifacts/best.pt` |

### Observed results

| Epoch | Training Loss | Validation Accuracy | Validation Macro F1 |
|---:|---:|---:|---:|
| 1 | 0.5320 | 0.5000 | 0.3333 |
| 2 | 0.0032 | 0.5000 | 0.3333 |
| 3 | 0.0011 | 0.0000 | 0.0000 |
| 4 | 0.0005 | 0.0000 | 0.0000 |
| 5 | 0.0003 | 0.5000 | 0.3333 |
| 6 | 0.0002 | 0.5000 | 0.3333 |
| 7 | 0.0002 | 0.5000 | 0.3333 |
| 8 | 0.0001 | 0.5000 | 0.3333 |
| 9 | 0.0001 | 0.5000 | 0.3333 |
| 10 | 0.0001 | 0.5000 | 0.3333 |

*Values are transcribed from the initial training run. Training loss is rounded to four decimal places.*

### Interpretation

**Training loss:** The loss decreased from 0.5320 to approximately 0.0001. The model fits the two training videos extremely well.

**Validation accuracy:** The final reported accuracy was 50%, equivalent to one correct prediction out of two validation videos.

**Macro F1:** The final reported macro F1 was 0.3333, indicating poor class-balanced performance in this tiny validation experiment.

**Checkpoint:** Training completed and saved a checkpoint at `artifacts/best.pt`.

These findings confirm that the training pipeline executes and produces a checkpoint. They do not demonstrate that the model has learned generalizable running and walking patterns.

## Evaluation and Limitations

The initial experiment has several important limitations:

- **Insufficient data:** Four videos cannot represent the variation in movement, camera angles, subjects, backgrounds, and recording conditions.
- **Overfitting risk:** Near-zero training loss alongside weak validation metrics suggests substantial overfitting.
- **High metric variance:** Each validation video contributes 50 percentage points to accuracy.
- **Limited statistical evidence:** Results cannot establish that the architecture outperforms a baseline or generalizes to unseen videos.
- **No benchmark comparison yet:** A fair comparison with the CNN-GRU baseline has not been established by this run.
- **No edge-performance claim yet:** Latency, memory use, model size, and hardware-specific inference performance still require measurement.

The current checkpoint should therefore be treated as an experimental artifact rather than a validated action-recognition model.

## Next Steps

### 1. Expand the dataset

Collect more distinct videos for both classes. An initial target of 30–50 videos per class can support basic experimentation, although larger and more diverse datasets are preferable.

### 2. Improve data splitting

Keep related clips from the same original recording or subject within the same split when appropriate. This helps prevent information leakage between training and evaluation.

### 3. Establish baseline performance

Train the CNN-GRU baseline using the same dataset, preprocessing, and split as the TRecViT-inspired model.

### 4. Improve generalization

Evaluate appropriate regularization, data augmentation, pretrained visual encoders, and early stopping once the dataset is large enough to support meaningful validation.

### 5. Conduct systematic evaluation

Report:

- Accuracy.
- Macro precision, recall, and F1.
- Per-class metrics.
- Confusion matrix.
- Model parameter count.
- Inference latency and memory usage.
- Results across multiple seeds or splits where feasible.

### 6. Evaluate edge deployment

Measure actual inference performance on the target hardware before making claims about real-time operation or edge efficiency.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| PyTorch | Deep learning and model training |
| OpenCV | Video loading and frame processing |
| NumPy | Numerical operations |
| scikit-learn | Classification metrics and data splitting |
| tqdm | Training progress display |
| Streamlit | Interactive demonstration |
| Docker | Containerization support |

## References

1. Google DeepMind, **TRecViT — official research implementation**. [GitHub repository](https://github.com/google-deepmind/trecvit).

2. TRecViT-related research and publication details should be cited from the official repository and its linked publication. This project is an independent, research-inspired implementation; its initial metrics are generated from the four-video experiment described above and are not results reported by the original researchers.

---

## Summary

TRecViT Edge establishes an initial end-to-end pipeline for recurrent video classification, from loading running/walking videos through training and checkpoint generation.

The first experiment completed successfully over 10 epochs and saved `artifacts/best.pt`. Its 50% validation accuracy and 0.3333 macro F1, however, reflect an extremely small validation set rather than reliable model performance.

The next milestone is to expand the dataset, establish a leakage-resistant split, and compare the recurrent architecture against a CNN-GRU baseline under identical experimental conditions.
