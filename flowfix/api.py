from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.flowfix.agent import repair_case
from src.flowfix.benchmarks import CASES
from src.flowfix.evaluation import run_benchmark
from src.flowfix.storage import save_result, recent_runs, init_db

app = FastAPI(title="FlowFix API", version="0.1.0",
              description="Demo low-latency, test-validated program repair service")

class RepairRequest(BaseModel):
    case_id: str
    max_attempts: int = Field(default=2, ge=1, le=3)
    timeout_seconds: int = Field(default=4, ge=1, le=15)

@app.on_event("startup")
def startup():
    init_db()

@app.get("/health")
def health():
    return {"status":"ok","service":"flowfix"}

@app.get("/cases")
def cases():
    return [{"case_id":k,"title":v["title"],"description":v["description"]} for k,v in CASES.items()]

@app.post("/repair")
def repair(request: RepairRequest):
    if request.case_id not in CASES:
        raise HTTPException(status_code=404, detail="Unknown case_id. See GET /cases.")
    result = repair_case(request.case_id, request.max_attempts, request.timeout_seconds)
    save_result(result)
    return result

@app.post("/benchmark")
def benchmark():
    report = run_benchmark()
    for row in report["cases"]:
        save_result(row)
    return report

@app.get("/runs")
def runs(limit: int = 50):
    if not 1 <= limit <= 500:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 500")
    return recent_runs(limit)
