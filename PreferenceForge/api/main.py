from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
import pandas as pd
from preferenceforge.config import FEATURES
from preferenceforge.collect import append_human_preference
from preferenceforge.models import GlobalRewardModel, AdaptiveRewardFeatureModel

app = FastAPI(title="PreferenceForge API", version="0.1.0")
global_model = None
adaptive_model = None

class FeatureVector(BaseModel):
    helpfulness: float = Field(ge=0, le=1)
    clarity: float = Field(ge=0, le=1)
    detail: float = Field(ge=0, le=1)
    concision: float = Field(ge=0, le=1)
    safety: float = Field(ge=0, le=1)

class CompareRequest(BaseModel):
    user_id: str
    response_a: FeatureVector
    response_b: FeatureVector

class PreferenceRequest(BaseModel):
    user_id: str
    prompt: str
    response_a: str
    response_b: str
    features_a: FeatureVector
    features_b: FeatureVector
    preferred: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/preferences")
def add_preference(req: PreferenceRequest):
    if req.preferred not in ("A", "B"):
        raise HTTPException(400, "preferred must be A or B")
    append_human_preference(req.user_id, req.prompt, req.response_a, req.response_b,
                            req.features_a.model_dump(), req.features_b.model_dump(),
                            int(req.preferred == "A"))
    return {"saved": True}

@app.post("/compare")
def compare(req: CompareRequest):
    a = req.response_a.model_dump()
    b = req.response_b.model_dump()
    weights = {"helpfulness": 1, "clarity": 1, "detail": 1, "concision": 1, "safety": 1}
    # Transparent cold-start score until a trained personalized model is loaded.
    score_a = sum(weights[k] * a[k] for k in FEATURES)
    score_b = sum(weights[k] * b[k] for k in FEATURES)
    return {"user_id": req.user_id, "score_a": score_a, "score_b": score_b,
            "predicted_preference": "A" if score_a >= score_b else "B",
            "note": "Cold-start unweighted feature score; train/evaluate models for learned weights."}
