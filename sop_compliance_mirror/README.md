# SOP Compliance Mirror
Insurance Claims SOP-Guided Conversational Agent

---

## Setup

```bash
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env

pip install -r requirements.txt

python -m pytest tests/ -v

python api/app.py               # FastAPI on port 8000
streamlit run ui/dashboard.py   # Compliance dashboard
python evaluation/eval_harness.py  # 25 golden test cases
```

---

## What I Built

### Core requirement: 4-phase SOP harness

**VERIFY_ID**
- Collects at least 3 PII fields before advancing: full name, DOB, and at least one of phone, email, or SSN last 4
- Policy number is used for claim lookup but does not count toward the PII gate
- Never discloses claim details until the gate passes
- Handles partial answers, alternate fields, clarification questions, and refusals
- Recognizes authorized representatives on file (e.g. David Chen calling for Margaret Chen)
- Handles name aliases (e.g. Yaven Li for Ya Wen Li)
- Stores intent and case hints from the caller during this phase for use after verification

**RESOLVE_INTENT**
- Uses cross-phase memory so the caller never repeats themselves
- If the caller said "I'm calling about my denied healthcare claim from January" during VERIFY_ID, that intent, claim type, and time hint are already stored and used here
- Matches to: denied claim appeal, document submission, status check, general inquiry
- Handles ambiguous language

**PROCESS_CASE**
- Looks up the caller's claims by party_id
- Explains denial reason and documents needed from actual claim data
- Provides document-specific guidance from the required_document_guideline.json fixture
- States appeal deadline
- Offers submission methods: member portal, fax, mail
- Handles follow-up questions about timing, format, alternatives
- Strictly grounded: no information outside claim data and SOP

**POST_PROCESS**
- Offers to send an email summary of the conversation
- Summary includes: what was discussed, claim status/outcome, and next steps
- Caller must explicitly choose to send or skip
- Confirms email address before sending
- Closes the conversation politely

---

### Bonus: Emotional Support and SOP Recovery

- Detects frustration, anxiety, confusion, and refusal from user messages and tone signals
- De-escalates before continuing the workflow
- Explains why the verification gate exists and why it protects the caller
- Offers alternative PII fields when caller pushes back
- Offers human transfer as an option
- Stops persuading and escalates after repeated refusals
- Tone guardrail blocks cheerful responses to frustrated callers
- Out-of-scope counter: after 2 out-of-scope questions, offers to transfer to a human representative

---

### Beyond the core requirements: SOP Compliance Mirror

**Dual-agent architecture**
Every response goes through two Claude instances in sequence. The Responder generates the response. The Compliance Auditor scores it on SOP grounding, hallucination risk, and escalation necessity before it reaches the user.

- If grounding is below threshold: Responder regenerates with explicit SOP citations forced
- If hallucination risk is above threshold: response is blocked and replaced with a structured escalation
- If Auditor flags escalation: handoff includes conversation summary, last step completed, user sentiment, and both agent perspectives

**Cross-phase memory**
CrossPhaseMemory captures intent hint, claim type hint, and time hint from any message in any phase. These are injected into the Responder prompt in RESOLVE_INTENT so the caller is never asked to repeat information they already gave.

**Persona adaptation**
Infers communication style from the first 3 messages: formal, casual, brief, detailed, frustrated, or engaged. Adapts language register without changing SOP content.

**Predictive step preloading**
Before the Responder answers, the system predicts the next likely SOP step and preloads relevant context. For example, if the caller is in VERIFY_ID and verification is about to pass, the denied claim data is preloaded so the response after verification is instant and accurate.

**Drift tracker**
Measures citation density across the conversation timeline. Flags when the agent shifts from grounded SOP retrieval to inference from memory. Produces a drift curve per session. Drift onset is flagged in the compliance report with the exact message number.

**SOP Gap Detector**
Every escalated or low-confidence question is stored. At session end, questions are clustered by topic (document submission, appeal process, timeline, payment, status, alternatives) and matched with specific suggested SOP amendments. Surfaced in the Living SOP Editor tab in the dashboard.

**Living SOP Editor**
The Streamlit dashboard includes a tab where gap suggestions can be reviewed, edited, approved, or rejected. Creates an audit trail of how the SOP evolves based on real caller questions.

**Compliance Report**
Generated at session end. Includes: average SOP grounding, average hallucination risk, escalation count, regeneration count, drift curve, steps completed vs skipped, commitments made and fulfilled, and gap signals detected.

**Commitment tracker**
The agent's commitments during a conversation are extracted and tracked. At session end, unfulfilled commitments are flagged in the compliance report.

**SOP version lock**
The SOP version is locked per session on creation. Mid-session SOP updates do not affect in-flight conversations.

**Repetition detector**
If the agent repeats semantically similar content across recent responses, the session guardrail flags it and appends a clarification prompt.

---

## Guardrail Layers

**Input guardrails**
- Prompt injection detection
- Scope boundary check
- Sensitive content detection: emotional distress, legal threats, medical advice requests, full SSN
- Encoding normalization and length enforcement

**Output guardrails**
- Hallucination pre-check before Auditor runs
- Tone mismatch check: blocks cheerful responses to frustrated users
- Length enforcement: truncates at sentence boundary, never mid-word

**Escalation guardrails**
- Structured handoff enforcer: escalations must include conversation summary, last step, and sentiment
- Escalation loop detector: repeated escalation of the same question flags a critical SOP gap

**Session guardrails**
- Repetition detector across recent responses
- Commitment tracker
- Out-of-scope counter with human transfer offer
- SOP version lock per session

---

## Validation Layers

- User input: encoding, length, rate limit, session context
- Agent output: citation format, completeness, token budget
- Session state: step index bounds, loop count integrity
- Compliance report: required fields present, scores in valid range

---

## Data Flow

```
User Message
  -> Input Validation
  -> Input Guardrails
  -> PII Extractor          (VERIFY_ID phase)
  -> Cross-Phase Memory     (all phases)
  -> Persona Detector
  -> Responder Agent
  -> Output Validation
  -> Output Guardrails
  -> Compliance Auditor
  -> Gate                   (regenerate or escalate)
  -> Session Guardrails
  -> Drift Tracker
  -> Session DB             (validated write)
  -> Response

End of session:
  -> Compliance Report
  -> SOP Gap Detector
  -> Living SOP Editor queue
```

---

## Test Coverage

26 unit tests across: SOP loading and validation, 4-phase structure, PII extraction and 3-PII gate, identity verification with aliases and representatives, cross-phase memory extraction, input/output validation, guardrails, sentiment detection, repetition detection, commitment extraction, drift computation and detection, gap clustering, compliance report validation.

25 golden end-to-end test conversations: 5 clean SOP flows, 5 off-topic recovery, 5 pre-empted steps, 5 out-of-scope escalation, 5 multi-turn ambiguous cases.

---

## API Endpoints

```
POST /session/start              Start a new session
POST /chat                       Send a message
POST /session/{id}/end           End session, get compliance report
GET  /session/{id}/report        Get compliance report
GET  /session/{id}/messages      Get full message history
GET  /gaps                       Get all SOP gaps detected
GET  /health                     Health check
```

---

## Files

```
sop_agent/
  core/
    models.py          All Pydantic models including Phase, CrossPhaseMemory, IdentityVerification
    database.py        SQLite with 4-point validation on every write
    sop_loader.py      4-phase SOP built from fixtures, with structure validation
    orchestrator.py    14-step message pipeline
  agents/
    responder.py       SOP execution, PII extraction, emotion detection, persona adaptation
    auditor.py         Grounding/hallucination/escalation scoring, response gating
  guardrails/
    guardrails.py      All 4 guardrail layers
  validation/
    validator.py       All 4 validation layers
  tracking/
    drift.py           Drift tracker, gap detector, compliance report generator
  evaluation/
    eval_harness.py    25 golden test cases
  api/
    app.py             FastAPI endpoints
  ui/
    dashboard.py       Streamlit dashboard with live audit, drift chart, gap editor
  data/fixtures/       Starter code fixture data
  tests/
    test_core.py       26 unit tests
```

---

## Delivery

### Option 1: Docker (recommended)

```bash
cp .env.example .env
# Add ANTHROPIC_API_KEY to .env

docker-compose up
```

API runs at http://localhost:8000
Dashboard (test UI) runs at http://localhost:8501

### Option 2: Local

```bash
cp .env.example .env
# Add ANTHROPIC_API_KEY to .env

pip install -r requirements.txt

# Terminal 1: API
export PYTHONPATH=.   # Windows: $env:PYTHONPATH = "."
python api/app.py

# Terminal 2: Dashboard
streamlit run ui/dashboard.py

# Terminal 3: Tests
python -m pytest tests/ -v

# Terminal 4: Evaluation harness
python evaluation/eval_harness.py
```

---

## Example Prompts to Test

**Assessment demo case (primary test):**
```
I'm the policyholder. My name is Margaret Chen, policy POL-9921.
I'm calling about my denied healthcare claim from January.
DOB is 1985-03-15, SSN last four is 4472.
```

**Authorized representative:**
```
Hi, I'm David Chen. I'm calling on behalf of my mother Margaret Chen,
policy POL-9921. Her DOB is 1985-03-15 and SSN last 4 is 4472.
```

**Name alias:**
```
Hi I'm Yaven Li, policy POL-7742, ID last 4 is 5317.
What claims do I have?
```

**Emotional / frustrated caller:**
```
I already told you who I am. This is ridiculous.
Just tell me why my claim was denied.
```

**Out-of-scope question:**
```
What is reinforcement learning?
```

**Partial PII (incomplete verification):**
```
My name is Margaret Chen. Can you tell me about my claim?
```

**Pre-empted steps (user front-loads everything):**
```
Margaret Chen, POL-9921, DOB 1985-03-15, SSN last 4 is 4472.
I have a denied healthcare claim CL-2048 and I already have both
the pathology report and office note ready to upload.
How do I submit them?
```

---

## Demo Test Case Expected Behavior

1. VERIFY_ID: Agent collects name + DOB + SSN last 4 (3 PII). Does not disclose claim details.
2. Agent stores "denied healthcare claim from January" as cross-phase memory hint.
3. Gate passes. Agent advances to RESOLVE_INTENT.
4. RESOLVE_INTENT: Agent uses stored hint. Does not re-ask what caller is calling about.
5. PROCESS_CASE: Agent looks up CL-2048, explains denial reason, states documents needed (pathology report, office note), gives appeal deadline, explains submission via member portal.
6. POST_PROCESS: Agent offers to send email summary. Caller chooses send or skip.

---

## Full Workflow Coverage

VERIFY_ID: strict PII gate (3 fields required), no claim disclosure, cross-phase memory storage, emotional de-escalation, representative handling, alias recognition.

RESOLVE_INTENT: uses cross-phase memory, no re-asking, intent matching.

PROCESS_CASE: grounded in fixture data, document guidance, deadlines, submission methods, follow-up questions.

POST_PROCESS: email summary offer, caller must choose send or skip.
