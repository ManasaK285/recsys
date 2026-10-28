from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "Cloud-Native Telemetry & RAG Observability Engine"
    host: str = "0.0.0.0"
    port: int = 8000
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    docs_dir: str = "data/docs"
    eval_set_path: str = "data/eval_set.json"
    p95_latency_threshold_ms: float = 1000.0
    error_rate_threshold: float = 0.05
    avg_token_threshold: float = 2500.0
    min_recall_at_k: float = 0.60
    recall_baseline: float = 0.75
    recall_drift_tolerance: float = 0.15
    use_vertex: bool = False
    google_cloud_project: str | None = None
    google_cloud_location: str = "us-central1"
    vertex_model: str = "gemini-2.5-flash"
    bigquery_enabled: bool = False
    bigquery_dataset: str = "rag_observability"
    bigquery_table: str = "telemetry_events"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

settings = Settings()
