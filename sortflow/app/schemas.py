from pydantic import BaseModel, Field
from typing import Literal

EventType = Literal["DEMAND_SURGE", "LABOR_SHORTAGE", "CONVEYOR_FAILURE", "TRANSPORT_DELAY"]

class EventCreate(BaseModel):
    event_type: EventType
    center_id: str
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = "MEDIUM"
    volume_change_pct: float = Field(default=0, ge=-100, le=500)
    labor_reduction_pct: float = Field(default=0, ge=0, le=100)
    capacity_reduction_pct: float = Field(default=0, ge=0, le=100)

class ScenarioCompare(BaseModel):
    center_id: str
    demand_multiplier: float = Field(default=1.2, gt=0, le=5)
    capacity_reduction_pct: float = Field(default=0, ge=0, le=100)

class NetworkScenarioCompare(BaseModel):
    demand_multiplier: float = Field(default=1.2, gt=0, le=5)
    capacity_reduction_pct: float = Field(default=0, ge=0, le=100)
