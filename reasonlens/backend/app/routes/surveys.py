from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import models, schemas
from ..database import get_db

router = APIRouter(prefix="/surveys", tags=["surveys"])


@router.get("/{participant_id}")
def get_participant_surveys(participant_id: str, db: Session = Depends(get_db)):
    surveys = (
        db.query(models.SurveyResponse)
        .filter(models.SurveyResponse.participant_id == participant_id)
        .all()
    )
    return [
        {
            "id": s.id,
            "app_trust": s.app_trust,
            "android_trust": s.android_trust,
            "necessity": s.necessity,
            "privacy_concern": s.privacy_concern,
            "created_at": s.created_at.isoformat() if s.created_at else None,
        }
        for s in surveys
    ]


@router.post("/submit")
def submit_survey(req: schemas.SurveyRequest, db: Session = Depends(get_db)):
    participant = db.get(models.Participant, req.participant_id)
    if not participant:
        raise HTTPException(status_code=404, detail="Unknown participant_id; call /experiment/assign first")

    survey = models.SurveyResponse(
        participant_id=req.participant_id,
        app_trust=req.app_trust,
        android_trust=req.android_trust,
        necessity=req.necessity,
        privacy_concern=req.privacy_concern,
    )
    db.add(survey)
    db.commit()
    return {"status": "ok", "id": survey.id}