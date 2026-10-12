from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.pipeline import run_experiment
from src.discovery.acquisition import STRATEGIES

app = FastAPI(title="ProEval-Lab API", version="1.0.0",
              description="Synthetic benchmark API for uncertainty-aware GenAI evaluation.")

class ExperimentRequest(BaseModel):
    n_cases: int = Field(default=300, ge=50, le=2000)
    budget: int = Field(default=60, ge=5, le=1000)
    seed: int = Field(default=42, ge=0)
    strategy: str = "hybrid"
    threshold: float = Field(default=.65, ge=0, le=1)
    target_mae: float = Field(default=.03, gt=0, le=1)

@app.get("/")
def root():
    return {"project": "ProEval-Lab", "status": "ok", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/evaluate")
def evaluate(request: ExperimentRequest):
    if request.strategy not in STRATEGIES:
        raise HTTPException(status_code=422, detail=f"strategy must be one of {STRATEGIES}")
    if request.budget >= request.n_cases:
        raise HTTPException(status_code=422, detail="budget must be smaller than n_cases")
    try:
        result, curve, cases = run_experiment(
            n_cases=request.n_cases, budget=request.budget, seed=request.seed,
            strategy=request.strategy, threshold=request.threshold, target_mae=request.target_mae
        )
        result.pop("queried_indices", None)
        return {"metrics": result, "estimation_curve": curve.to_dict(orient="records"),
                "discovered_cases": cases.to_dict(orient="records")}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
