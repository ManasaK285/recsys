import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fastapi import FastAPI, HTTPException
from provenancelab.schemas import VerifyRequest, VerifyResponse, BenchmarkRequest
from provenancelab.detectors import DemoBaselineDetector
from provenancelab.benchmark import run_benchmark

app = FastAPI(title="ProvenanceLab API", version="0.1.0", description="AI-content provenance evaluation starter; demo baseline is not SynthID.")
detector = DemoBaselineDetector()

@app.get("/health")
def health():
    return {"status": "ok", "mode": "educational_demo_baseline"}

@app.post("/verify", response_model=VerifyResponse)
def verify(request: VerifyRequest):
    try:
        result = detector.detect(request.text)
        return VerifyResponse(status=result.status, score=result.score, method=result.method, caveat=result.caveat)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.post("/benchmark")
def benchmark(request: BenchmarkRequest):
    return run_benchmark(request.samples_per_class, request.seed)
