from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np

app = FastAPI(title="RESA-LTR API", version="1.0")

class Candidate(BaseModel):
    item_id: int
    features: list[float] = Field(min_length=4, max_length=4)

class RankRequest(BaseModel):
    candidates: list[Candidate]

def score(x):
    x = np.asarray(x, dtype=float)
    return float(
        0.35*x[0] + 0.25*x[1] - 0.20*x[2] + 0.15*x[3]
        + 0.20*x[0]*x[1]
    )

@app.get("/health")
def health():
    return {"status":"ok","service":"resa-ltr"}

@app.post("/rank")
def rank(request: RankRequest):
    result = [
        {"item_id": c.item_id, "score": score(c.features)}
        for c in request.candidates
    ]
    result.sort(key=lambda x: x["score"], reverse=True)
    return {"recommendations": result}
