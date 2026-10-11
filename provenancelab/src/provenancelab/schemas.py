from pydantic import BaseModel, Field

class VerifyRequest(BaseModel):
    text: str = Field(min_length=20, max_length=20000)

class VerifyResponse(BaseModel):
    status: str
    score: float | None
    method: str
    caveat: str

class BenchmarkRequest(BaseModel):
    samples_per_class: int = Field(default=30, ge=5, le=500)
    seed: int = 42
