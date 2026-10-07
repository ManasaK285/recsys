import hashlib
import uuid
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from .. import models, schemas
from ..database import get_db
from ..config import APP_TYPES, REASON_TYPES, PRECISION_LEVELS, DEFAULT_SEED

router = APIRouter(prefix="/experiment", tags=["experiment"])

_ACTIVE_EXPERIMENT_ID = "default_experiment"


def _ensure_experiment(db: Session):
    exp = db.get(models.Experiment, _ACTIVE_EXPERIMENT_ID)
    if not exp:
        exp = models.Experiment(
            experiment_id=_ACTIVE_EXPERIMENT_ID,
            name="ReasonLens Main Study",
            seed=DEFAULT_SEED,
        )
        db.add(exp)
        db.commit()
    return exp


def deterministic_condition(participant_id: str, seed: int) -> tuple:
    """
    Deterministically maps a participant_id + seed to (app_type, reason_type)
    using a hash-based index, so the same participant always reproduces the
    same assignment for a given seed (reproducible randomization).
    """
    digest = hashlib.sha256(f"{participant_id}:{seed}".encode()).hexdigest()
    app_idx = int(digest[:8], 16) % len(APP_TYPES)
    reason_idx = int(digest[8:16], 16) % len(REASON_TYPES)
    return APP_TYPES[app_idx], REASON_TYPES[reason_idx]


@router.get("/config", response_model=schemas.ExperimentConfigResponse)
def get_config():
    return schemas.ExperimentConfigResponse(
        app_types=APP_TYPES,
        reason_types=REASON_TYPES,
        precision_levels=PRECISION_LEVELS,
    )


@router.post("/assign", response_model=schemas.AssignmentResponse)
def assign(req: schemas.AssignmentRequest, db: Session = Depends(get_db)):
    exp = _ensure_experiment(db)

    participant = db.get(models.Participant, req.participant_id)
    if not participant:
        participant = models.Participant(
            participant_id=req.participant_id,
            demographic_group=req.demographic_group,
            mode="pilot",
            consent_given=1 if req.consent_given else 0,
        )
        db.add(participant)
        db.commit()

    existing = (
        db.query(models.Assignment)
        .filter(models.Assignment.participant_id == req.participant_id)
        .filter(models.Assignment.experiment_id == exp.experiment_id)
        .first()
    )
    if existing:
        return schemas.AssignmentResponse(
            participant_id=existing.participant_id,
            experiment_id=existing.experiment_id,
            app_type=existing.app_type,
            reason_type=existing.reason_type,
            condition=existing.condition,
            seed=existing.seed,
        )

    app_type, reason_type = deterministic_condition(req.participant_id, exp.seed)
    condition = f"{app_type}__{reason_type}"

    assignment = models.Assignment(
        participant_id=req.participant_id,
        experiment_id=exp.experiment_id,
        app_type=app_type,
        reason_type=reason_type,
        condition=condition,
        seed=exp.seed,
    )
    db.add(assignment)
    db.commit()

    return schemas.AssignmentResponse(
        participant_id=req.participant_id,
        experiment_id=exp.experiment_id,
        app_type=app_type,
        reason_type=reason_type,
        condition=condition,
        seed=exp.seed,
    )


@router.get("/new_participant_id")
def new_participant_id():
    return {"participant_id": str(uuid.uuid4())}
