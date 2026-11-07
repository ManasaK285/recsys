# SortFlow — Event-Driven Logistics Optimization Platform

SortFlow is an event-driven logistics planning and optimization platform designed to monitor sort-center operations, process network disruptions, and support operational decision-making across a logistics network.

It combines a REST API, asynchronous workflow processing, deterministic optimization, and an MCP server that exposes logistics data and planning capabilities to AI assistants and MCP-compatible clients.

## 1. Key Features

- **Event-driven processing:** Ingest operational disruptions such as demand surges.
- **Network monitoring:** Track demand, capacity, utilization, and workflow KPIs across sort centers.
- **Optimization engine:** Evaluate demand and capacity scenarios at individual-center and network levels.
- **Asynchronous execution:** Process workflows through Redis Queue (RQ) workers.
- **Workflow management:** Track execution status, attempts, failures, and retries.
- **Audit logging:** Maintain traceable records of event and workflow lifecycle actions.
- **Idempotency support:** Accept idempotency keys to prevent duplicate event submissions.
- **MCP integration:** Expose logistics information and scenario-analysis capabilities as MCP tools.
- **Human-reviewed recommendations:** Generate proposed operational actions without executing real-world operations.
- **Operational dashboard:** Visualize logistics and workflow information through Streamlit.

## 2. Technology Stack

| Component | Technology |
|---|---|
| Backend API | FastAPI |
| Language | Python 3.11 |
| Database | PostgreSQL 16 |
| ORM | SQLAlchemy |
| Queue and cache | Redis 7 |
| Background processing | RQ |
| Optimization | Python planning logic |
| MCP integration | Model Context Protocol, Python SDK |
| Dashboard | Streamlit |
| Visualization | Plotly |
| Data processing | Pandas |
| Containerization | Docker Compose |
| Testing | Pytest |

## 3. System Architecture

```text
                 Logistics Network
                        |
                        v
                Operational Events
                        |
                        v
                  FastAPI API
                        |
             +----------+----------+
             |          |          |
             v          v          v
         PostgreSQL   Audit     Idempotency
             |
             v
        Workflow Creation
             |
             v
          Redis Queue
             |
             v
           RQ Worker
             |
             v
       Optimization Engine
             |
       +-----+------+
       |            |
       v            v
    Capacity      Labor &
    Analysis     Allocation
       |            |
       +-----+------+
             |
             v
       Workflow Result
             |
             v
         PostgreSQL
             |
       +-----+----------------+
       |                      |
       v                      v
   Streamlit              FastAPI REST
   Dashboard                  API
                              ^
                              |
                        MCP Server
                              |
                              v
                     MCP-Compatible AI
                          Clients
```

**MCP integration:** The MCP server exposes selected API capabilities as tools. It communicates with the FastAPI service rather than directly accessing the database.

## 4. Project Structure

```text
sortflow/
│
├── app/
│   ├── main.py          # FastAPI application and REST endpoints
│   ├── models.py        # SQLAlchemy database models
│   ├── schemas.py       # Pydantic request schemas
│   ├── database.py      # Database connections and sessions
│   ├── config.py        # Application configuration
│   ├── seed.py          # Demo sort-center data
│   ├── queue.py         # Redis and RQ configuration
│   └── tasks.py         # Background workflow execution
│
├── mcp_server/
│   ├── server.py        # MCP server and tool definitions
│   └── __init__.py
│
├── optimizer/
│   ├── planner.py       # Scenario and network planning
│   └── __init__.py
│
├── dashboard/
│   └── Home.py          # Streamlit dashboard
│
├── scripts/
│   └── smoke_test.py    # Application smoke tests
│
├── tests/
│   ├── test_optimizer.py
│   └── test_schemas.py
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## 5. Getting Started

### Prerequisites

- Docker Desktop
- Docker Compose
- Python 3.11 or compatible Python environment
- Git

### Clone the repository

```bash
git clone <repository-url>
cd sortflow
```

### Configure environment variables

Review the application configuration and Docker Compose environment settings.

The current Compose configuration supplies:

```text
DATABASE_URL
REDIS_URL
SEED_DEMO_DATA
API_BASE_URL
```

Do not commit production credentials or secrets to version control.

### Start the application

```powershell
docker compose up -d --build
```

Check the services:

```powershell
docker compose ps
```

The current Docker Compose configuration starts:

| Service | Purpose | Port |
|---|---|---|
| postgres | Persistent application database | 5432 |
| redis | Queue infrastructure | 6379 |
| api | FastAPI REST service | 8000 |
| worker | Asynchronous RQ execution | — |
| dashboard | Streamlit dashboard | 8501 |

### Verify application health

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Check dependencies:

```powershell
Invoke-RestMethod http://localhost:8000/health/dependencies |
    ConvertTo-Json -Depth 5
```

### Access the application

- API: http://localhost:8000
- Swagger documentation: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Streamlit dashboard: http://localhost:8501

## 6. MCP Server Integration

SortFlow includes a standalone MCP server implemented in:

```text
mcp_server/server.py
```

It uses the Python MCP SDK (`mcp==1.3.0`) and exposes logistics operations as tools that MCP-compatible AI clients can discover and invoke.

### Available MCP Tools

The current server implementation defines eight tools.

| MCP Tool | Description |
|---|---|
| `get_network_summary` | Retrieve network-wide demand, capacity, utilization, and workflow KPIs. |
| `get_center_status` | Retrieve capacity, demand, labor, and utilization for a specific sort center. |
| `get_recent_events` | Retrieve recent operational disruptions and processing statuses. |
| `get_workflow_status` | Inspect a workflow and its persisted planning result. |
| `compare_planning_scenarios` | Compare demand and capacity scenarios for an individual center. |
| `compare_network_scenarios` | Rank simulated centers under network-wide scenario conditions. |
| `get_audit_history` | Retrieve audit records for an entity or across the network. |
| `propose_operational_action` | Generate a proposed operational action for human review. |

### MCP Tool Capabilities

The MCP server supports three main categories of decision support:

**1. Operational visibility**

- Network-level KPI retrieval.
- Individual sort-center monitoring.
- Recent event inspection.
- Workflow status inspection.
- Audit history retrieval.

**2. Scenario analysis**

- Demand multiplier evaluation.
- Capacity reduction evaluation.
- Individual-center scenario comparison.
- Network-wide scenario comparison.

**3. Human-reviewed operational proposals**

The `propose_operational_action` tool produces a text-only recommendation containing:

- Target sort center.
- Proposed action.
- Rationale.
- Execution status.

It returns:

```json
{
  "status": "PROPOSED_FOR_HUMAN_REVIEW",
  "center_id": "SC-08",
  "action": "Example proposed action",
  "rationale": "Example operational rationale",
  "execution_performed": false
}
```

**Safety boundary:** The proposal tool does not execute real-world operational actions. Recommendations require human review.

### MCP Architecture

The MCP server uses the FastAPI service as its data and planning interface.

```text
       MCP-Compatible AI Client
                  |
                  v
             MCP Server
                  |
           +------+------+
           |      |      |
           v      v      v
        Summary Center Scenario
        Tools   Tools   Tools
           |      |      |
           +------+------+
                  |
                  v
              FastAPI
                  |
                  v
             PostgreSQL
```

The MCP server also accesses workflow, event, and audit endpoints through the API.

### Starting the MCP Server

The MCP server is currently a separate component, not a service in the supplied Docker Compose configuration.

After confirming its configuration and API URL, install dependencies in a local Python environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Start the server from the project root:

```powershell
python -m mcp_server.server
```

The server uses `mcp.run()` as its entry point. The default MCP Python SDK transport is standard input/output (stdio).

The MCP client must be configured to launch the server using the appropriate Python executable and working directory.

The FastAPI application should be running and reachable at the API address configured in `mcp_server/server.py`.

**Note:** The MCP server is not automatically started by `docker compose up` in the current configuration. A dedicated Compose service can be added as a future deployment improvement.

## 7. Event Processing

SortFlow accepts operational events through the REST API.

### Example: Demand Surge

```powershell
$body = @{
    event_type = "DEMAND_SURGE"
    center_id = "SC-08"
    volume_change_pct = 30
    severity = "HIGH"
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://localhost:8000/events" `
    -Method POST `
    -ContentType "application/json" `
    -Body $body |
    ConvertTo-Json -Depth 5
```

### Example Response

```json
{
    "event_id": "generated-event-id",
    "workflow_id": "generated-workflow-id",
    "job_id": "generated-job-id",
    "status": "QUEUED",
    "idempotent_replay": false
}
```

### Event Processing Sequence

1. Validate the incoming event.
2. Verify the referenced sort center.
3. Persist the event in PostgreSQL.
4. Create a workflow.
5. Write audit records.
6. Enqueue the workflow in Redis.
7. Execute the planning task in the RQ worker.
8. Persist the workflow result and final status.

## 8. Workflow Lifecycle

```text
             QUEUED
                |
                v
             RUNNING
                |
          +-----+-----+
          |           |
          v           v
      COMPLETED     FAILED
                      |
                      v
                    RETRY
                      |
                      v
                    QUEUED
```

Workflow records include:

- Workflow identifier.
- Associated event identifier.
- Execution status.
- Attempt count.
- Error information.
- Start timestamp.
- Completion timestamp.
- Persisted planning result.

## 9. Audit Logging

SortFlow records important event and workflow operations.

| Audit Action | Description |
|---|---|
| `EVENT_ACCEPTED` | Operational event persisted. |
| `WORKFLOW_QUEUED` | Workflow created and queued. |
| `WORKFLOW_STARTED` | Worker began execution. |
| `WORKFLOW_COMPLETED` | Workflow completed successfully. |
| `QUEUE_SUBMISSION_FAILED` | Queue submission failed. |
| `WORKFLOW_RETRY_REQUESTED` | Retry was requested. |

Retrieve recent audit records:

```powershell
Invoke-RestMethod `
    "http://localhost:8000/audit?limit=10" |
    ConvertTo-Json -Depth 6
```

Retrieve audit records for a specific entity:

```powershell
Invoke-RestMethod `
    "http://localhost:8000/audit?entity_id=<ENTITY_ID>&limit=50" |
    ConvertTo-Json -Depth 6
```

## 10. Scenario Analysis

SortFlow provides two scenario-analysis endpoints.

### Individual Center

```text
POST /scenarios/compare
```

Evaluates a demand multiplier and capacity reduction for an individual sort center.

### Network-Level Scenario

```text
POST /scenarios/network
```

Evaluates demand and capacity changes across the network and compares the resulting center conditions.

These endpoints are also exposed through the MCP scenario-comparison tools.

## 11. API Reference

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | API health |
| GET | `/health/dependencies` | Database, Redis, and queue health |
| GET | `/metrics` | Workflow and event metrics |
| GET | `/summary` | Network summary |
| GET | `/centers` | Sort-center information |
| POST | `/events` | Submit an operational event |
| GET | `/events` | List recent events |
| GET | `/workflows` | List workflows |
| GET | `/workflows/{workflow_id}` | Workflow details |
| POST | `/workflows/{workflow_id}/retry` | Retry a failed workflow |
| GET | `/audit` | Retrieve audit history |
| POST | `/scenarios/compare` | Individual-center scenario analysis |
| POST | `/scenarios/network` | Network-level scenario analysis |

## 12. Testing

Run the existing automated tests:

```powershell
docker compose exec api sh -lc "cd /app && PYTHONPATH=/app python -m pytest -q tests"
```

The existing test suite includes optimizer and schema tests.

Planned reliability validation includes:

- Idempotency verification.
- Duplicate event submission tests.
- Queue submission failure tests.
- Worker failure simulation.
- Workflow retry validation.
- End-to-end integration tests.
- MCP tool invocation tests.

## 13. Verified Execution

The following end-to-end scenario has been successfully exercised.

**Input:**

| Parameter | Value |
|---|---|
| Event type | `DEMAND_SURGE` |
| Center | `SC-08` |
| Volume change | 30% |
| Severity | `HIGH` |

**Execution result:**

| Metric | Result |
|---|---|
| Event submission | Successful |
| Workflow creation | Successful |
| Queue submission | Successful |
| Workflow execution | Completed |
| Attempt count | 1 |
| Workflow error | None |
| Capacity utilization | 123.93% |
| Labor gap | 15 |
| Allocation status | `FULLY_COVERED` |

The audit history confirmed:

```text
EVENT_ACCEPTED
WORKFLOW_QUEUED
WORKFLOW_STARTED
WORKFLOW_COMPLETED
```

This validates the event-ingestion, persistence, queueing, worker-execution, and audit lifecycle for the tested scenario.

The reported `FULLY_COVERED` allocation status is the optimizer's result; it does not mean that nominal capacity utilization is below 100%.

## 14. Useful Docker Commands

Start services:

```powershell
docker compose up -d
```

Rebuild services:

```powershell
docker compose up -d --build
```

Inspect running containers:

```powershell
docker compose ps
```

View API logs:

```powershell
docker compose logs --tail=120 api
```

Follow worker logs:

```powershell
docker compose logs -f worker
```

Stop services:

```powershell
docker compose down
```

Avoid `docker compose down -v` unless you intentionally want to delete persistent database volumes.

## 15. Development Roadmap

### Reliability and Testing
- Idempotency verification.
- Queue failure handling.
- Worker failure simulation.
- Workflow retry testing.
- Integration and end-to-end tests.

### MCP Improvements
- MCP tool invocation tests.
- MCP client configuration examples.
- Dedicated MCP Docker Compose service.
- Expanded decision-support tools.
- Improved tool error handling and observability.

### Optimization Improvements
- More advanced network allocation strategies.
- Improved capacity and labor planning.
- Expanded disruption scenarios.
- Optimization performance improvements.

### Observability
- Enhanced workflow metrics.
- Improved operational dashboards.
- More detailed execution tracing.
- Better failure diagnostics.

## 16. Engineering Principles

SortFlow follows these principles:

- Modular service architecture.
- Asynchronous workflow execution.
- Deterministic and traceable planning.
- Persistent operational audit records.
- Human oversight for operational recommendations.
- Separation between AI decision support and real-world execution.

---

**Project:** SortFlow  
**API Version:** 2.0.0  
**Architecture:** Event-driven and asynchronous  
**Backend:** FastAPI  
**Database:** PostgreSQL  
**Queue:** Redis and RQ  
**Optimization:** Deterministic planning logic  
**AI Integration:** Model Context Protocol (MCP)  
**Dashboard:** Streamlit