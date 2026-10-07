from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import init_db
from .routes import experiment, events, surveys

app = FastAPI(
    title="ReasonLens Backend",
    description="Experiment assignment, event logging, and survey collection for the ReasonLens permission-reasoning study.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(experiment.router)
app.include_router(events.router)
app.include_router(surveys.router)


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/health")
def health():
    return {"status": "ok"}
