from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10000)
    top_k: int = Field(default=3, ge=1, le=10)

class RetrievedChunk(BaseModel):
    document_id: str
    title: str
    score: float
    text: str

class QueryResponse(BaseModel):
    answer: str
    retrieved: list[RetrievedChunk]
    latency_ms: float
    estimated_input_tokens: int
    estimated_output_tokens: int
    generation_mode: str

class EvaluateRequest(BaseModel):
    k: int = Field(default=3, ge=1, le=10)
