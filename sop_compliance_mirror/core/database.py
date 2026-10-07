import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from core.models import Session, ComplianceReport, SOPGap, Message

DB_PATH = Path("data/sop_agent.db")


def get_db():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            sop_id TEXT NOT NULL,
            sop_version TEXT NOT NULL,
            current_step_index INTEGER DEFAULT 0,
            persona TEXT,
            started_at TEXT,
            ended_at TEXT,
            commitments TEXT DEFAULT '[]',
            completed_steps TEXT DEFAULT '[]',
            skipped_steps TEXT DEFAULT '[]',
            loop_count TEXT DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS messages (
            message_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT,
            sop_step TEXT,
            citation_count INTEGER DEFAULT 0,
            audit_score TEXT,
            was_regenerated INTEGER DEFAULT 0,
            was_escalated INTEGER DEFAULT 0,
            escalation_handoff TEXT,
            persona_detected TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        );

        CREATE TABLE IF NOT EXISTS compliance_reports (
            session_id TEXT PRIMARY KEY,
            report_data TEXT NOT NULL,
            generated_at TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        );

        CREATE TABLE IF NOT EXISTS sop_gaps (
            gap_id TEXT PRIMARY KEY,
            topic_cluster TEXT NOT NULL,
            example_questions TEXT DEFAULT '[]',
            frequency INTEGER DEFAULT 1,
            suggested_amendment TEXT,
            status TEXT DEFAULT 'pending',
            created_at TEXT
        );

        CREATE TABLE IF NOT EXISTS sop_versions (
            sop_id TEXT NOT NULL,
            version TEXT NOT NULL,
            sop_data TEXT NOT NULL,
            loaded_at TEXT,
            PRIMARY KEY (sop_id, version)
        );
    """)
    conn.commit()
    conn.close()


def save_session(session: Session):
    conn = get_db()
    conn.execute("""
        INSERT OR REPLACE INTO sessions
        (session_id, sop_id, sop_version, current_step_index, persona,
         started_at, ended_at, commitments, completed_steps, skipped_steps, loop_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session.session_id, session.sop_id, session.sop_version,
        session.current_step_index, session.persona,
        session.started_at.isoformat() if session.started_at else None,
        session.ended_at.isoformat() if session.ended_at else None,
        json.dumps(session.commitments),
        json.dumps(session.completed_steps),
        json.dumps(session.skipped_steps),
        json.dumps(session.loop_count)
    ))
    conn.commit()
    conn.close()


def save_message(message: Message, session_id: str):
    conn = get_db()
    # Validate before write
    assert message.message_id, "message_id required"
    assert session_id, "session_id required"
    assert message.role in ("user", "assistant", "system"), f"Invalid role: {message.role}"

    audit_json = message.audit_score.model_dump_json() if message.audit_score else None
    escalation_json = message.escalation_handoff.model_dump_json() if message.escalation_handoff else None

    conn.execute("""
        INSERT OR REPLACE INTO messages
        (message_id, session_id, role, content, timestamp, sop_step,
         citation_count, audit_score, was_regenerated, was_escalated,
         escalation_handoff, persona_detected)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        message.message_id, session_id, message.role, message.content,
        message.timestamp.isoformat(),
        message.sop_step, message.citation_count, audit_json,
        int(message.was_regenerated), int(message.was_escalated),
        escalation_json,
        message.persona_detected.value if message.persona_detected else None
    ))
    conn.commit()
    conn.close()


def save_compliance_report(report: ComplianceReport):
    conn = get_db()
    # Validate completeness
    required_fields = ["session_id", "total_messages", "avg_grounding",
                      "avg_hallucination_risk", "drift_curve"]
    for field in required_fields:
        assert getattr(report, field) is not None, f"Report missing: {field}"

    conn.execute("""
        INSERT OR REPLACE INTO compliance_reports (session_id, report_data, generated_at)
        VALUES (?, ?, ?)
    """, (report.session_id, report.model_dump_json(), report.generated_at.isoformat()))
    conn.commit()
    conn.close()


def save_sop_gap(gap: SOPGap):
    conn = get_db()
    # Deduplicate by semantic similarity (simple: check topic cluster)
    existing = conn.execute(
        "SELECT gap_id, frequency FROM sop_gaps WHERE topic_cluster = ?",
        (gap.topic_cluster,)
    ).fetchone()

    if existing:
        conn.execute(
            "UPDATE sop_gaps SET frequency = frequency + 1 WHERE gap_id = ?",
            (existing["gap_id"],)
        )
    else:
        conn.execute("""
            INSERT INTO sop_gaps
            (gap_id, topic_cluster, example_questions, frequency,
             suggested_amendment, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            gap.gap_id, gap.topic_cluster,
            json.dumps(gap.example_questions),
            gap.frequency, gap.suggested_amendment,
            gap.status, gap.created_at.isoformat()
        ))
    conn.commit()
    conn.close()


def get_all_gaps() -> List[dict]:
    conn = get_db()
    rows = conn.execute("SELECT * FROM sop_gaps ORDER BY frequency DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_session(session_id: str) -> Optional[dict]:
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_session_messages(session_id: str) -> List[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp",
        (session_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_compliance_report(session_id: str) -> Optional[dict]:
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM compliance_reports WHERE session_id = ?", (session_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None
