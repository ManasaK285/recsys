from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import models, schemas
from ..database import get_db

router = APIRouter(prefix="/events", tags=["events"])


@router.post("/permission")
def log_permission_event(req: schemas.PermissionEventRequest, db: Session = Depends(get_db)):
    participant = db.get(models.Participant, req.participant_id)
    if not participant:
        raise HTTPException(status_code=404, detail="Unknown participant_id; call /experiment/assign first")

    event = models.PermissionEvent(
        participant_id=req.participant_id,
        event=req.event,
        decision=req.decision,
        precision=req.precision,
        response_time_ms=req.response_time_ms,
    )
    db.add(event)
    db.commit()
    return {"status": "ok", "id": event.id}


@router.get("/permission/{participant_id}")
def get_participant_events(participant_id: str, db: Session = Depends(get_db)):
    events = (
        db.query(models.PermissionEvent)
        .filter(models.PermissionEvent.participant_id == participant_id)
        .all()
    )
    return [
        {
            "event": e.event,
            "decision": e.decision,
            "precision": e.precision,
            "response_time_ms": e.response_time_ms,
            "timestamp": e.timestamp.isoformat() if e.timestamp else None,
        }
        for e in events
    ]
