"""uvicorn api:app --reload"""
from typing import Optional
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from cg import governance as gv
from cg.nlp import add_concerns
from cg.pipeline import load_or_generate, run_all
from cg.rag import Retriever

app = FastAPI(title="CurriculumGuard")


def _rec(df):
    return df.astype(object).where(df.notna(), None).to_dict("records")
_R = Retriever()


class Rec(BaseModel):
    group: str
    stance: str
    text: str
    severity: int = 3
    subgroup: str = "none"


class Req(BaseModel):
    records: Optional[list[Rec]] = None
    group: Optional[str] = None
    share: Optional[float] = None


def _df(req):
    return pd.DataFrame([r.model_dump() for r in req.records]) if req.records else load_or_generate()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(req: Req):
    res = run_all(_df(req), n_boot=100)
    return {"schemes": _rec(res["schemes"]), "audit": _rec(res["audit"]),
            "flips": _rec(res["flips"]), "power": _rec(res["power"].head(10))}


@app.post("/counterfactual")
def counterfactual(req: Req):
    df = add_concerns(_df(req)).reset_index(drop=True)
    return gv.counterfactual(df, req.group or "student", req.share if req.share is not None else 0.45)


@app.get("/retrieve")
def retrieve(q: str, k: int = 2):
    return _R.search(q, k)
