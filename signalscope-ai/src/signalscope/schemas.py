from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict, field_validator

class TelemetryEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timestamp: datetime
    service: str = Field(min_length=1, max_length=100)
    region: str = Field(min_length=1, max_length=80)
    release: str = Field(min_length=1, max_length=80)
    latency_p50_ms: float = Field(ge=0, le=600000)
    latency_p95_ms: float = Field(ge=0, le=600000)
    error_rate: float = Field(ge=0, le=1)
    cpu_pct: float = Field(ge=0, le=100)
    memory_pct: float = Field(ge=0, le=100)
    request_count: int = Field(ge=0)
    queue_depth: int = Field(ge=0)
    is_injected_anomaly: bool = False

    @field_validator("timestamp")
    @classmethod
    def utc_timestamp(cls, value):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

class TelemetryBatch(BaseModel):
    events: list[TelemetryEvent] = Field(min_length=1, max_length=10000)

class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
