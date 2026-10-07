from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sys
sys.path.insert(0, "/home/claude/sop_agent")

from core.orchestrator import initialize, create_session, process_message, end_session
from core.database import get_all_gaps, get_compliance_report, get_session_messages

app = FastAPI(
    title="SOP Compliance Mirror",
    description="Insurance Claims SOP-Guided Conversational Agent with Compliance Auditing",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


class ChatRequest(BaseModel):
    session_id: str
    message: str


class StartRequest(BaseModel):
    pass


@app.on_event("startup")
async def startup():
    try:
        result = initialize()
        print(f"System initialized: {result}")
    except Exception as e:
        print(f"Initialization error: {e}")


@app.get("/health")
def health():
    return {"status": "ok", "service": "SOP Compliance Mirror"}


@app.post("/session/start")
def start_session():
    """Start a new conversation session."""
    try:
        session_id = create_session()
        return {
            "session_id": session_id,
            "message": "Session started. I'm here to help with your insurance claim. Could you please provide your name, policy number, and the last 4 digits of your SSN or national ID to get started?"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
def chat(request: ChatRequest):
    """Send a message and get a response."""
    try:
        response = process_message(request.session_id, request.message)
        return {
            "session_id": response.session_id,
            "response": response.content,
            "sop_step": response.sop_step,
            "was_escalated": response.was_escalated,
            "drift_warning": response.drift_warning,
            "audit": {
                "sop_grounding": response.audit_score.sop_grounding if response.audit_score else None,
                "hallucination_risk": response.audit_score.hallucination_risk if response.audit_score else None,
                "flags": response.audit_score.flags if response.audit_score else []
            } if response.audit_score else None
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/{session_id}/end")
def end_session_route(session_id: str):
    """End session and get compliance report."""
    try:
        result = end_session(session_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{session_id}/report")
def get_report(session_id: str):
    """Get compliance report for a session."""
    report = get_compliance_report(session_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report


@app.get("/session/{session_id}/messages")
def get_messages(session_id: str):
    """Get all messages for a session."""
    messages = get_session_messages(session_id)
    return {"messages": messages}


@app.get("/gaps")
def get_gaps():
    """Get all detected SOP gaps."""
    gaps = get_all_gaps()
    return {"gaps": gaps, "total": len(gaps)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
