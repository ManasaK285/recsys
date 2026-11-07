import uuid
from datetime import datetime, timezone
from sqlalchemy import String, Float, Integer, DateTime, ForeignKey, JSON, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base

def utcnow():
    return datetime.now(timezone.utc)

class SortCenter(Base):
    __tablename__ = "sort_centers"
    id: Mapped[str] = mapped_column(String(20), primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    region: Mapped[str] = mapped_column(String(50))
    capacity_per_hour: Mapped[int] = mapped_column(Integer)
    current_demand_per_hour: Mapped[int] = mapped_column(Integer)
    available_workers: Mapped[int] = mapped_column(Integer)
    workers_required: Mapped[int] = mapped_column(Integer)
    processing_cost_per_package: Mapped[float] = mapped_column(Float, default=0.1)
    status: Mapped[str] = mapped_column(String(20), default="NORMAL")

class NetworkEvent(Base):
    __tablename__ = "network_events"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    event_type: Mapped[str] = mapped_column(String(40), index=True)
    center_id: Mapped[str] = mapped_column(ForeignKey("sort_centers.id"))
    severity: Mapped[str] = mapped_column(String(20), default="MEDIUM")
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(20), default="QUEUED")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    center = relationship("SortCenter")

class Workflow(Base):
    __tablename__ = "workflows"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    event_id: Mapped[str] = mapped_column(ForeignKey("network_events.id"), unique=True)
    status: Mapped[str] = mapped_column(String(20), default="QUEUED", index=True)
    attempt: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    event = relationship("NetworkEvent")

class IdempotencyRecord(Base):
    __tablename__ = "idempotency_records"
    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    event_id: Mapped[str] = mapped_column(ForeignKey("network_events.id"), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

class AuditLog(Base):
    __tablename__ = "audit_logs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    entity_type: Mapped[str] = mapped_column(String(40), index=True)
    entity_id: Mapped[str] = mapped_column(String(36), index=True)
    action: Mapped[str] = mapped_column(String(80))
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
