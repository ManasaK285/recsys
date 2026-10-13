from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .simulation import run_experiment, simulate_game, STRATEGIES

app = FastAPI(
    title="Strategic Bargaining Lab API",
    description="Research-inspired multi-agent bargaining simulator using transparent policy archetypes.",
    version="1.0.0",
)


class ExperimentRequest(BaseModel):
    games: int = Field(default=100, ge=1, le=10000)
    agents: int = Field(default=4, ge=2, le=20)
    rounds: int = Field(default=12, ge=1, le=100)
    seed: int = 7


class GameRequest(BaseModel):
    agents: int = Field(default=4, ge=2, le=20)
    rounds: int = Field(default=12, ge=1, le=100)
    seed: int = 7


@app.get("/")
def root():
    return {"name": "Strategic Bargaining Lab", "docs": "/docs", "strategies": list(STRATEGIES)}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/simulate")
def simulate(req: GameRequest):
    try:
        rounds, agents = simulate_game(0, req.agents, req.rounds, req.seed)
        return {
            "rounds": [r.__dict__ for r in rounds],
            "agents": agents,
            "agreement": any(r.offer_accepted for r in rounds),
            "total_surplus": sum(r.total_surplus for r in rounds),
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/experiment")
def experiment(req: ExperimentRequest):
    try:
        rounds, agents, summary = run_experiment(req.games, req.agents, req.rounds, req.seed)
        return {
            "config": req.model_dump(),
            "summary": summary.to_dict(orient="records"),
            "offer_behavior": (
                rounds.groupby("proposer_strategy")
                .agg(offers=("offer_to_responder", "count"),
                     mean_offer=("offer_to_responder", "mean"),
                     acceptance_rate=("offer_accepted", "mean"))
                .reset_index().round(4).to_dict(orient="records")
            ),
            "overall_agreement_rate": float(agents.groupby("game_id")["game_agreement"].max().mean()),
            "overall_mean_rejections": float(agents.groupby("game_id")["rejections"].first().mean()),
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
