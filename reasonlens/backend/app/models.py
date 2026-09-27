from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey
from sqlalchemy.sql import func

from .database import Base


class Participant(Base):
    __tablename__ = "participants"

    participant_id = Column(String, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    demographic_group = Column(String, nullable=True)
    mode = Column(String, default="pilot")  # "pilot" or "simulation"
    consent_given = Column(Integer, default=0)  # 0/1 boolean


class Experiment(Base):
    __tablename__ = "experiments"

    experiment_id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    seed = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    config_json = Column(String, nullable=True)


class Assignment(Base):
    __tablename__ = "assignments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    participant_id = Column(String, ForeignKey("participants.participant_id"), index=True)
    experiment_id = Column(String, ForeignKey("experiments.experiment_id"), index=True)
    app_type = Column(String, nullable=False)
    reason_type = Column(String, nullable=False)
    condition = Column(String, nullable=False)  # combined label, e.g. "rideshare__functional"
    seed = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class PermissionEvent(Base):
    __tablename__ = "permission_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    participant_id = Column(String, ForeignKey("participants.participant_id"), index=True)
    event = Column(String, nullable=False)       # e.g. "permission_prompt_shown", "permission_result"
    decision = Column(String, nullable=True)     # "granted" / "denied"
    precision = Column(String, nullable=True)    # "approximate" / "precise"
    response_time_ms = Column(Integer, nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())


class SurveyResponse(Base):
    __tablename__ = "survey_responses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    participant_id = Column(String, ForeignKey("participants.participant_id"), index=True)
    app_trust = Column(Integer, nullable=True)          # Likert 1-5
    android_trust = Column(Integer, nullable=True)      # Likert 1-5
    necessity = Column(Integer, nullable=True)          # Likert 1-5
    privacy_concern = Column(Integer, nullable=True)    # Likert 1-5
    created_at = Column(DateTime(timezone=True), server_default=func.now())
