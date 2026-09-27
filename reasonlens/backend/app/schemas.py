from typing import Optional
from pydantic import BaseModel, Field


class AssignmentRequest(BaseModel):
    participant_id: str = Field(..., description="Anonymous, client-generated UUID")
    demographic_group: Optional[str] = None
    consent_given: bool = True


class AssignmentResponse(BaseModel):
    participant_id: str
    experiment_id: str
    app_type: str
    reason_type: str
    condition: str
    seed: int


class PermissionEventRequest(BaseModel):
    participant_id: str
    event: str
    decision: Optional[str] = None      # "granted" | "denied"
    precision: Optional[str] = None     # "approximate" | "precise"
    response_time_ms: Optional[int] = None


class SurveyRequest(BaseModel):
    participant_id: str
    app_trust: int = Field(..., ge=1, le=5)
    android_trust: int = Field(..., ge=1, le=5)
    necessity: int = Field(..., ge=1, le=5)
    privacy_concern: int = Field(..., ge=1, le=5)


class ExperimentConfigResponse(BaseModel):
    app_types: list
    reason_types: list
    precision_levels: list
