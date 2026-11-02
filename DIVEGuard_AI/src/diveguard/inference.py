from __future__ import annotations
import joblib
import pandas as pd
from .config import ARTIFACT_DIR, TARGET_DEFAULT
from .features import prepare_features

def predict(prompt: str, topic: str = "unknown", context: dict | None = None,
            model_name: str = "pluralistic", target: str = TARGET_DEFAULT):
    path = ARTIFACT_DIR / f"{model_name}_{target}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}. Train models first.")
    row = {"text_prompt": prompt, "topic": topic}
    row.update(context or {})
    frame = pd.DataFrame([row])
    model = joblib.load(path)
    x = prepare_features(frame, pluralistic=(model_name == "pluralistic"))
    probabilities = model.predict_proba(x)[0]
    classes = model.named_steps["classifier"].classes_
    distribution = {str(int(c)): float(p) for c, p in zip(classes, probabilities)}
    prediction = int(classes[probabilities.argmax()])
    return {"predicted_rating": prediction, "probabilities": distribution}
