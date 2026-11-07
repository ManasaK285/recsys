import pytest
from pydantic import ValidationError
from app.schemas import EventCreate

def test_event_schema_accepts_valid_event():
    e = EventCreate(event_type="DEMAND_SURGE", center_id="SC-08", volume_change_pct=40)
    assert e.severity == "MEDIUM"

def test_event_schema_rejects_invalid_event_type():
    with pytest.raises(ValidationError):
        EventCreate(event_type="ALIEN_INVASION", center_id="SC-08")
