# Cloud-Native RAG Service Architecture

A small cloud-native retrieval service can expose a FastAPI HTTP API, run inside a Docker container, embed text with a sentence-transformer model, and use a FAISS vector index for nearest-neighbor retrieval. A RAG orchestration layer assembles retrieved context and optionally calls a hosted model such as Vertex AI Gemini.

BigQuery can store structured telemetry events for later analysis. A production design should use workload identity or application default credentials, least-privilege IAM, a durable telemetry pipeline, health checks, request timeouts, and secret management. Multiple replicas need shared metrics storage rather than process-local counters.
