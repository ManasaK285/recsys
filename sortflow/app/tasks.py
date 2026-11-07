from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import update
from app.database import SessionLocal
from app.models import Workflow, NetworkEvent, SortCenter, AuditLog
from optimizer.planner import plan_disruption


def _audit(db, entity_type, entity_id, action, details=None):
    db.add(AuditLog(entity_type=entity_type, entity_id=entity_id, action=action, details=details or {}))


def execute_workflow(workflow_id: str):
    """Execute once per queued transition; persist state changes and audit trail."""
    db: Session = SessionLocal()
    try:
        workflow = db.get(Workflow, workflow_id)
        if not workflow:
            return {"status": "MISSING", "workflow_id": workflow_id}
        # Conditional claim prevents two workers from processing the same queued workflow.
        claimed = db.execute(
            update(Workflow).where(Workflow.id == workflow_id, Workflow.status == "QUEUED")
            .values(status="RUNNING", attempt=Workflow.attempt + 1, started_at=datetime.now(timezone.utc))
        )
        if claimed.rowcount != 1:
            db.rollback()
            return {"status": "SKIPPED", "reason": "Workflow is not queued", "workflow_id": workflow_id}
        event = db.get(NetworkEvent, workflow.event_id)
        if not event:
            db.rollback()
            raise ValueError("Referenced event not found")
        center = db.get(SortCenter, event.center_id)
        if not center:
            db.rollback()
            raise ValueError("Referenced sort center not found")
        event.status = "PROCESSING"
        _audit(db, "workflow", workflow_id, "WORKFLOW_STARTED", {"event_id": event.id})
        db.commit()

        network = db.query(SortCenter).all()
        result = plan_disruption(
            center={"id": center.id, "current_demand_per_hour": center.current_demand_per_hour,
                    "capacity_per_hour": center.capacity_per_hour, "available_workers": center.available_workers,
                    "workers_required": center.workers_required},
            event={"event_type": event.event_type, **event.payload},
            network=[{"id": c.id, "current_demand_per_hour": c.current_demand_per_hour,
                      "capacity_per_hour": c.capacity_per_hour,
                      "processing_cost_per_package": c.processing_cost_per_package} for c in network],
        )
        workflow = db.get(Workflow, workflow_id)
        event = db.get(NetworkEvent, workflow.event_id)
        center = db.get(SortCenter, event.center_id)
        workflow.result = result
        workflow.status = "COMPLETED"
        workflow.completed_at = datetime.now(timezone.utc)
        event.status = "PROCESSED"
        center.status = "AT_RISK" if result["capacity_utilization_pct"] >= 90 else "NORMAL"
        _audit(db, "workflow", workflow_id, "WORKFLOW_COMPLETED", {
            "event_id": event.id, "capacity_utilization_pct": result["capacity_utilization_pct"],
            "labor_gap": result["labor_gap"], "allocation_status": result["transfer_plan"]["allocation_status"]})
        db.commit()
        return result
    except Exception as exc:
        db.rollback()
        workflow = db.get(Workflow, workflow_id)
        if workflow:
            event = db.get(NetworkEvent, workflow.event_id)
            workflow.status = "FAILED"
            workflow.error = str(exc)[:2000]
            workflow.completed_at = datetime.now(timezone.utc)
            if event:
                event.status = "FAILED"
            _audit(db, "workflow", workflow_id, "WORKFLOW_FAILED", {"error": str(exc)[:1000]})
            db.commit()
        raise
    finally:
        db.close()
