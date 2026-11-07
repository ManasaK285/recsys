from contextlib import asynccontextmanager
from datetime import datetime, timezone
from fastapi import FastAPI, Depends, HTTPException, Header
from sqlalchemy.orm import Session
from sqlalchemy import desc
from redis.exceptions import RedisError
from app.database import Base, engine, get_db
from app.config import settings
from app.models import SortCenter, NetworkEvent, Workflow, IdempotencyRecord, AuditLog
from app.schemas import EventCreate, ScenarioCompare, NetworkScenarioCompare
from app.seed import seed_centers
from app.queue import workflow_queue, redis_conn
from app.tasks import execute_workflow
from optimizer.planner import compare_scenario, compare_network_scenarios

@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    if settings.seed_demo_data:
        with Session(engine) as db:
            seed_centers(db)
    yield

app = FastAPI(title="SortFlow Logistics Planning API",
    description="Synthetic event-driven logistics planning, reliability, audit, scenario analysis and MCP-enabled decision support.",
    version="2.0.0", lifespan=lifespan)


def audit(db, entity_type, entity_id, action, details=None):
    db.add(AuditLog(entity_type=entity_type, entity_id=entity_id, action=action, details=details or {}))

@app.get("/health")
def health():
    return {"status": "ok", "service": "sortflow-api", "version": app.version}

@app.get("/health/dependencies")
def dependency_health(db: Session = Depends(get_db)):
    checks = {"database": "ok", "redis": "unknown", "queue": "unknown"}
    try:
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
    except Exception as exc:
        checks["database"] = f"error: {str(exc)[:160]}"
    try:
        checks["redis"] = "ok" if redis_conn.ping() else "error"
        checks["queue"] = "ok"
    except RedisError as exc:
        checks["redis"] = f"error: {str(exc)[:160]}"
        checks["queue"] = "unavailable"
    return {"status": "ok" if all(v == "ok" for v in checks.values()) else "degraded", "checks": checks}

@app.get("/metrics")
def metrics(db: Session = Depends(get_db)):
    workflows = db.query(Workflow).all()
    return {"workflows_total": len(workflows),
            "workflows_queued": sum(w.status == "QUEUED" for w in workflows),
            "workflows_running": sum(w.status == "RUNNING" for w in workflows),
            "workflows_completed": sum(w.status == "COMPLETED" for w in workflows),
            "workflows_failed": sum(w.status == "FAILED" for w in workflows),
            "events_total": db.query(NetworkEvent).count(),
            "audit_records_total": db.query(AuditLog).count()}

@app.get("/centers")
def get_centers(db: Session = Depends(get_db)):
    centers = db.query(SortCenter).order_by(SortCenter.id).all()
    return [{"id": c.id, "name": c.name, "region": c.region,
        "capacity_per_hour": c.capacity_per_hour, "current_demand_per_hour": c.current_demand_per_hour,
        "utilization_pct": round(c.current_demand_per_hour / max(c.capacity_per_hour, 1) * 100, 2),
        "available_workers": c.available_workers, "workers_required": c.workers_required,
        "processing_cost_per_package": c.processing_cost_per_package, "status": c.status} for c in centers]

@app.get("/summary")
def summary(db: Session = Depends(get_db)):
    centers = db.query(SortCenter).all(); workflows = db.query(Workflow).all(); events = db.query(NetworkEvent).all()
    total_capacity = sum(c.capacity_per_hour for c in centers); total_demand = sum(c.current_demand_per_hour for c in centers)
    return {"sort_centers": len(centers), "network_demand_per_hour": total_demand,
        "network_capacity_per_hour": total_capacity,
        "network_utilization_pct": round(total_demand / max(total_capacity, 1) * 100, 2),
        "active_disruptions": sum(e.status in ("QUEUED", "PROCESSING") for e in events),
        "total_events": len(events), "completed_workflows": sum(w.status == "COMPLETED" for w in workflows),
        "failed_workflows": sum(w.status == "FAILED" for w in workflows)}


@app.post("/events", status_code=202)
def create_event(
    payload: EventCreate,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    db: Session = Depends(get_db)
):
    if idempotency_key and len(idempotency_key) > 128:
        raise HTTPException(
            status_code=422,
            detail="Idempotency-Key must be 128 characters or fewer"
        )

    if idempotency_key:
        prior = db.get(IdempotencyRecord, idempotency_key)
        if prior:
            workflow = db.query(Workflow).filter_by(event_id=prior.event_id).first()
            return {
                "event_id": prior.event_id,
                "workflow_id": workflow.id if workflow else None,
                "job_id": None,
                "status": workflow.status if workflow else "RECORDED",
                "idempotent_replay": True
            }

    center = db.get(SortCenter, payload.center_id)
    if not center:
        raise HTTPException(status_code=404, detail="Unknown sort center")

    event = NetworkEvent(
        event_type=payload.event_type,
        center_id=payload.center_id,
        severity=payload.severity,
        payload={
            "volume_change_pct": payload.volume_change_pct,
            "labor_reduction_pct": payload.labor_reduction_pct,
            "capacity_reduction_pct": payload.capacity_reduction_pct
        }
    )

    db.add(event)
    db.flush()

    workflow = Workflow(event_id=event.id, status="QUEUED")
    db.add(workflow)
    db.flush()  # Generate workflow.id before creating its audit record

    if idempotency_key:
        db.add(IdempotencyRecord(key=idempotency_key, event_id=event.id))

    audit(
        db, "event", event.id, "EVENT_ACCEPTED",
        {
            "event_type": event.event_type,
            "center_id": center.id,
            "severity": event.severity
        }
    )

    audit(
        db, "workflow", workflow.id, "WORKFLOW_QUEUED",
        {"event_id": event.id}
    )

    db.commit()
    db.refresh(workflow)

    try:
        job = workflow_queue.enqueue(
            execute_workflow,
            workflow.id,
            job_timeout=300,
            retry=None
        )
        return {
            "event_id": event.id,
            "workflow_id": workflow.id,
            "job_id": job.id,
            "status": "QUEUED",
            "idempotent_replay": False
        }

    except Exception as exc:
        workflow.status = "FAILED"
        workflow.error = f"Queue submission failed: {exc}"
        event.status = "QUEUE_FAILED"

        audit(
            db, "workflow", workflow.id, "QUEUE_SUBMISSION_FAILED",
            {"error": str(exc)[:500]}
        )
        db.commit()

        raise HTTPException(
            status_code=503,
            detail="Could not enqueue workflow"
        ) from exc


@app.get("/events")
def list_events(limit: int = 50, db: Session = Depends(get_db)):
    events = db.query(NetworkEvent).order_by(desc(NetworkEvent.created_at)).limit(min(max(limit, 1), 200)).all()
    return [{"id": e.id, "event_type": e.event_type, "center_id": e.center_id, "severity": e.severity,
             "payload": e.payload, "status": e.status, "created_at": e.created_at.isoformat() if e.created_at else None} for e in events]

@app.get("/workflows")
def list_workflows(limit: int = 50, db: Session = Depends(get_db)):
    workflows = db.query(Workflow).order_by(desc(Workflow.started_at)).limit(min(max(limit, 1), 200)).all()
    return [{"id": w.id, "event_id": w.event_id, "status": w.status, "attempt": w.attempt,
        "error": w.error, "started_at": w.started_at.isoformat() if w.started_at else None,
        "completed_at": w.completed_at.isoformat() if w.completed_at else None} for w in workflows]

@app.get("/workflows/{workflow_id}")
def workflow_detail(workflow_id: str, db: Session = Depends(get_db)):
    w = db.get(Workflow, workflow_id)
    if not w: raise HTTPException(status_code=404, detail="Workflow not found")
    e = db.get(NetworkEvent, w.event_id)
    return {"id": w.id, "status": w.status, "attempt": w.attempt, "error": w.error, "result": w.result,
        "event": {"id": e.id, "event_type": e.event_type, "center_id": e.center_id, "severity": e.severity,
                  "payload": e.payload, "status": e.status} if e else None}

@app.post("/workflows/{workflow_id}/retry", status_code=202)
def retry_workflow(workflow_id: str, db: Session = Depends(get_db)):
    workflow = db.get(Workflow, workflow_id)
    if not workflow: raise HTTPException(status_code=404, detail="Workflow not found")
    if workflow.status not in ("FAILED",): raise HTTPException(status_code=409, detail=f"Only failed workflows can be retried; current status is {workflow.status}")
    workflow.status = "QUEUED"; workflow.error = None; workflow.completed_at = None
    event = db.get(NetworkEvent, workflow.event_id)
    if event: event.status = "QUEUED"
    audit(db, "workflow", workflow.id, "WORKFLOW_RETRY_REQUESTED", {"previous_attempt": workflow.attempt})
    db.commit()
    try:
        job = workflow_queue.enqueue(execute_workflow, workflow.id, job_timeout=300, retry=None)
        return {"workflow_id": workflow.id, "job_id": job.id, "status": "QUEUED"}
    except Exception as exc:
        workflow.status = "FAILED"; workflow.error = f"Retry queue submission failed: {exc}"; db.commit()
        raise HTTPException(status_code=503, detail="Could not enqueue retry") from exc

@app.get("/audit")
def get_audit(limit: int = 100, entity_id: str | None = None, db: Session = Depends(get_db)):
    query = db.query(AuditLog)
    if entity_id: query = query.filter(AuditLog.entity_id == entity_id)
    rows = query.order_by(desc(AuditLog.created_at)).limit(min(max(limit, 1), 500)).all()
    return [{"id": x.id, "entity_type": x.entity_type, "entity_id": x.entity_id, "action": x.action,
             "details": x.details, "created_at": x.created_at.isoformat() if x.created_at else None} for x in rows]

@app.post("/scenarios/compare")
def scenario_compare(payload: ScenarioCompare, db: Session = Depends(get_db)):
    center = db.get(SortCenter, payload.center_id)
    if not center: raise HTTPException(status_code=404, detail="Unknown sort center")
    return compare_scenario({"id": center.id, "current_demand_per_hour": center.current_demand_per_hour,
        "capacity_per_hour": center.capacity_per_hour}, payload.demand_multiplier, payload.capacity_reduction_pct)

@app.post("/scenarios/network")
def network_scenario_compare(payload: NetworkScenarioCompare, db: Session = Depends(get_db)):
    centers = db.query(SortCenter).all()
    network = [{"id": c.id, "current_demand_per_hour": c.current_demand_per_hour, "capacity_per_hour": c.capacity_per_hour} for c in centers]
    return compare_network_scenarios(network, payload.demand_multiplier, payload.capacity_reduction_pct)
