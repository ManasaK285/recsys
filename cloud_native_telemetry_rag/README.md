# Cloud-Native Telemetry & RAG Observability Engine

A cloud-ready Retrieval-Augmented Generation (RAG) service built with **Python, FastAPI, Sentence Transformers, FAISS, Prometheus, Vertex AI, and BigQuery**. The project combines semantic document retrieval with application telemetry, retrieval-quality evaluation, and threshold-based alerting.

The goal is to make RAG applications easier to monitor by tracking service performance, estimated token utilization, retrieval quality, and operational events through API endpoints.

## Project Overview

Traditional RAG applications focus on retrieving relevant documents and generating answers. However, operating a reliable RAG system also requires visibility into latency, errors, resource utilization, and retrieval-quality changes.

This project implements a RAG API and an observability layer that captures operational events, calculates latency percentiles, evaluates retrieval recall, and identifies threshold violations.

### Key Features

- **Semantic retrieval:** Uses Sentence Transformers to create text embeddings and FAISS for similarity-based document retrieval.
- **RAG query API:** Accepts natural-language questions and returns answers with retrieved document chunks and relevance scores.
- **Latency monitoring:** Tracks HTTP request latency and calculates p50, p95, and maximum observed latency.
- **Token utilization tracking:** Estimates input and output tokens for RAG queries.
- **Retrieval evaluation:** Evaluates Recall@k using a configurable evaluation dataset.
- **Automated threshold alerts:** Detects high latency, elevated error rates, excessive estimated token utilization, low retrieval recall, and retrieval recall drift.
- **Prometheus integration:** Exposes HTTP request counters and latency histograms through a Prometheus-compatible endpoint.
- **Event tracking:** Records HTTP requests, RAG queries, and retrieval evaluations.
- **Optional Vertex AI integration:** Supports configuring a cloud-based generation path.
- **Optional BigQuery integration:** Supports exporting telemetry events for downstream analysis.
- **Containerization:** Includes Docker and Docker Compose configuration.

## Architecture

```text
                 User / API Client
                         |
                         v
                  FastAPI Application
                         |
             +-----------+-----------+
             |                       |
             v                       v
        POST /query             HTTP Middleware
             |                       |
             v                       v
       RAG Query Engine        Request Telemetry
             |                       |
             v                       v
       FAISS Retriever         Telemetry Store
             |                 - Request counts
             v                 - Error counts
     Sentence Transformers     - Latency percentiles
             |                 - Token estimates
             v                 - Event history
       Document Chunks         - Alert conditions
             |
             v
      Answer Construction
       /             \
      v               v
 Extractive       Optional Vertex AI
 Fallback          Generation
                       |
                       v
                 Query Response

        Observability and Evaluation APIs
                         |
            +------------+-------------+
            |            |             |
            v            v             v
         /metrics     /alerts       /events
            |
            v
    /metrics/prometheus

    /evaluate --> Recall@k Evaluation
    /export/bigquery --> BigQuery Export
```

## Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Programming language | Python | Application and telemetry logic |
| API framework | FastAPI | REST API and request validation |
| Embeddings | Sentence Transformers | Semantic text representations |
| Vector search | FAISS | Similarity-based retrieval |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` | Lightweight text embeddings |
| Metrics | `prometheus-client` | Counters and latency histograms |
| Configuration | Pydantic Settings | Environment-based configuration |
| Cloud generation | Google Vertex AI | Optional answer generation |
| Analytics storage | Google BigQuery | Optional telemetry export |
| Containerization | Docker | Reproducible deployment |
| API documentation | Swagger UI / OpenAPI | Interactive API testing |
| Testing | Pytest | Automated API tests |

## Project Structure

```text
cloud_native_telemetry_rag/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── schemas.py
│   ├── telemetry.py
│   ├── retrieval.py
│   ├── rag.py
│   └── sinks.py
├── data/
│   ├── docs/
│   │   ├── telemetry.md
│   │   ├── retrieval.md
│   │   └── architecture.md
│   └── eval_set.json
├── tests/
│   └── test_api.py
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python installed on your system.
- pip package manager.
- Internet access on the first run to download the embedding model.
- Google Cloud credentials only if you intend to use the optional cloud integrations.
- Docker Desktop if you want to run the containerized version.

Python 3.11 is a reasonable compatibility target for this dependency stack.

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd cloud_native_telemetry_rag
```

Replace the repository URL with your actual GitHub repository URL.

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

If Python 3.11 is not installed, install it or use a Python version compatible with all project dependencies.

### 3. Configure environment variables

Create a local `.env` file from the provided template:

```powershell
Copy-Item .env.example .env
```

Review `.env.example` and configure the settings needed for your environment.

Typical configuration categories include:

- Application name and document directory
- Embedding model
- Retrieval evaluation dataset
- p95 latency threshold
- HTTP error-rate threshold
- Average token-utilization threshold
- Minimum Recall@k
- Recall baseline and drift tolerance
- Optional Google Cloud project, Vertex AI, and BigQuery settings

Use the exact variable names defined in `.env.example` and `app/config.py`.

Do not commit `.env` files, service-account keys, access tokens, or other credentials.

### 4. Start the API

From the project root:

```powershell
python -m uvicorn app.main:app --reload
```

The service should become available at:

- API base URL: `http://127.0.0.1:8000`
- Interactive API documentation: `http://127.0.0.1:8000/docs`
- OpenAPI schema: `http://127.0.0.1:8000/openapi.json`

Keep the terminal running while testing the endpoints.

### 5. Verify service health

Open:

```text
http://127.0.0.1:8000/health
```

The endpoint should return a successful response indicating that the service is healthy.

## API Reference

All endpoints can be tested through Swagger UI at `/docs`.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Returns service information |
| GET | `/health` | Checks application health |
| POST | `/query` | Retrieves relevant chunks and returns an answer |
| GET | `/metrics` | Returns the telemetry snapshot |
| GET | `/metrics/prometheus` | Exposes Prometheus metrics |
| GET | `/alerts` | Returns active threshold alerts |
| POST | `/evaluate` | Evaluates retrieval Recall@k |
| POST | `/export/bigquery` | Exports telemetry events to BigQuery |
| GET | `/events` | Returns recorded telemetry events |

### Example: Query the RAG Engine

Send a POST request to `/query`.

```json
{
  "question": "How does the telemetry engine monitor query latency and trigger alerts?",
  "top_k": 3
}
```

A successful response contains:

- `answer`: The answer or extractive fallback.
- `retrieved`: Retrieved chunks, including document IDs, titles, scores, and text.
- `latency_ms`: Measured query-processing time.
- `estimated_input_tokens`: Estimated input token count.
- `estimated_output_tokens`: Estimated output token count.
- `generation_mode`: The answer-generation path used.

The default configuration can return an extractive fallback when generated synthesis is unavailable. Configure and verify Vertex AI separately to demonstrate cloud-based generation.

## Observability and Metrics

### Latency Monitoring

The telemetry store tracks HTTP request latency and reports:

- **p50 latency:** Median observed request duration.
- **p95 latency:** The observed duration at or below which approximately 95% of recorded requests fall.
- **Maximum latency:** The largest retained observation.
- **Sample count:** Number of retained latency measurements.

The current implementation calculates these statistics from in-memory observations. The measurements are useful for a local demonstration, but are not a substitute for a persistent, production monitoring backend.

### Error Monitoring

The telemetry layer records HTTP requests and counts responses with status codes of 500 or higher as errors.

The error rate is calculated as:

```text
error_rate = errors_total / requests_total
```

HTTP status codes below 500 are not counted as server errors by this implementation.

### Token Utilization

The application records estimated input and output token counts for RAG queries.

The current implementation aggregates the estimates to report average tokens per query. These are approximate counts, not authoritative provider billing or token-usage measurements.

### Prometheus Metrics

Open:

```text
http://127.0.0.1:8000/metrics/prometheus
```

The endpoint exposes HTTP request counters and request-duration histograms using `prometheus-client`.

This endpoint provides a starting point for integration with a Prometheus monitoring server and dashboard tooling.

### Event History

The `/events` endpoint exposes recorded events such as:

- HTTP request completion
- RAG query execution
- Retrieval evaluation

For example, request the latest events with:

```text
GET /events?limit=100
```

The limit must be between 1 and 1000.

## Retrieval Evaluation

The project includes a configurable evaluation dataset at:

```text
data/eval_set.json
```

The `POST /evaluate` endpoint evaluates retrieval performance at a requested value of k and records the resulting Recall@k measurement.

Recall@k measures the proportion of expected relevant documents retrieved within the first k results.

```text
Recall@k =
Relevant expected documents retrieved in top k
------------------------------------------------
Total expected relevant documents
```

The endpoint also returns alert information based on the configured retrieval thresholds.

Evaluation quality depends on the accuracy and coverage of the expected relevant document labels. A small sample dataset is suitable for validating functionality, but larger and representative datasets are necessary for meaningful retrieval-quality conclusions.

## Automated Alerting

The alerting implementation evaluates telemetry against configurable thresholds.

| Alert rule | Trigger condition |
|---|---|
| `p95_latency` | Observed p95 latency exceeds the configured limit |
| `error_rate` | Error rate exceeds its configured limit after at least 20 requests |
| `token_utilization` | Average estimated tokens per query exceed the configured limit |
| `retrieval_recall` | Latest Recall@k falls below the configured minimum |
| `retrieval_recall_drift` | Recall falls below the configured baseline by more than the allowed drift |

Alerts are exposed through:

```text
GET /alerts
```

The response includes the active alert count and alert details, including severity, rule, observed value, threshold, and message.

These are application-level threshold checks. External notifications, paging, and automated incident-management integrations are not included unless added separately.

## Google Cloud Integrations

### Vertex AI

The application includes an optional Vertex AI generation path.

To enable it:

1. Configure the required Google Cloud project and model settings.
2. Enable the appropriate Vertex AI services.
3. Authenticate using a supported Google Cloud identity.
4. Grant the minimum required permissions.
5. Submit a RAG query and inspect `generation_mode` in the response.

The default extractive fallback does not prove that Vertex AI generation is configured or working.

### BigQuery

The application includes an export endpoint:

```text
POST /export/bigquery
```

The export implementation depends on the configuration and credentials expected by the project.

To validate it:

1. Configure the Google Cloud project and BigQuery destination.
2. Authenticate the application.
3. Grant the required dataset and table permissions.
4. Call the export endpoint.
5. Verify the exported records in BigQuery.

Cloud export should be considered optional until a successful export has been observed.

## Docker Deployment

The repository includes a `Dockerfile` and `docker-compose.yml`.

Build and start the application with:

```bash
docker compose up --build
```

Check the container logs if startup fails, then open:

```text
http://127.0.0.1:8000/docs
```

Ensure that the container has access to the required configuration and document files.

For production deployment, use persistent or shared metrics storage, managed credentials, suitable resource limits, health checks, and centralized logging.

## Testing

Run the automated tests from the project root:

```powershell
python -m pytest -q
```

The repository includes API tests in `tests/test_api.py`.

After running the suite, review the test output and resolve any failures. Automated tests and cloud integration tests should be reported separately when their environments differ.

## Initial Local Validation

The following results were observed during the initial local run on October 28, 2026.

| Check | Observed result |
|---|---|
| FastAPI startup | Successful |
| Embedding model initialization | Completed |
| `POST /query` | HTTP 200 |
| Retrieved chunks | 3 |
| Query latency | 51.594 ms |
| Estimated input tokens | 143 |
| Estimated output tokens | 122 |
| Generation mode | `extractive_fallback` |

The test question asked how the telemetry engine monitors latency and triggers alerts. The highest-scoring retrieved chunk discussed p95 latency and threshold alerts.

The returned retrieval scores were approximately 0.660, 0.489, and 0.474. These are similarity scores, not probabilities of correctness.

**Important:** The latency above is one observed query measurement, not p95 latency across a representative workload. It should not be presented as a production benchmark.

The local query test verifies the basic retrieval and response path. Metrics endpoint behavior, threshold-triggering scenarios, evaluation quality, Vertex AI generation, and BigQuery export must be validated independently before claiming successful end-to-end operation for those features.

## Current Limitations

- Metrics and event history are held in process-local memory.
- Metrics reset when the application restarts.
- Multiple application replicas do not share a common telemetry store.
- Token counts are estimated.
- Retrieval quality depends on the size and quality of the evaluation dataset.
- The default answer path may use extractive fallback rather than generated synthesis.
- Prometheus metrics are exposed, but a full Prometheus server and dashboard are not configured by this API alone.
- BigQuery export and Vertex AI generation require cloud configuration and credentials.
- Production readiness requires further testing of concurrency, persistence, security, failure recovery, and deployment behavior.

## Future Improvements

1. Add persistent telemetry storage and shared metrics for multi-replica deployments.
2. Separate RAG query latency from general HTTP request latency.
3. Add richer retrieval-quality and drift evaluation over a representative dataset.
4. Integrate Prometheus scraping and Grafana dashboards.
5. Add structured logs, distributed tracing, and request correlation IDs.
6. Add configurable notification integrations for critical alerts.
7. Validate Vertex AI generation and BigQuery export in a configured Google Cloud environment.
8. Add load testing and measure p50/p95 latency under increasing concurrency.
9. Strengthen authentication, rate limiting, input validation, and production secret management.

## Summary

This project demonstrates the core components of an observable RAG service: semantic retrieval, an API layer, latency and error monitoring, token-utilization estimates, retrieval evaluation, and threshold-based alerts.

It provides a foundation for exploring how cloud-native AI services can be monitored and evaluated beyond answer generation alone. The current implementation is a local demonstrator, with cloud deployment and production-grade observability remaining separate validation and engineering steps.
