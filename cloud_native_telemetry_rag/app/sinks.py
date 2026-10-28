import logging
from app.config import settings

logger = logging.getLogger("observability.bigquery")

class BigQuerySink:
    def __init__(self):
        self.client = None
        self.table_id = None
        if settings.bigquery_enabled and settings.google_cloud_project:
            try:
                from google.cloud import bigquery
                self.client = bigquery.Client(project=settings.google_cloud_project)
                self.table_id = (f"{settings.google_cloud_project}."
                                 f"{settings.bigquery_dataset}.{settings.bigquery_table}")
                self._ensure_table(bigquery)
            except Exception:
                logger.exception("BigQuery initialization failed; export disabled")

    def _ensure_table(self, bigquery):
        dataset_id = f"{settings.google_cloud_project}.{settings.bigquery_dataset}"
        try:
            self.client.get_dataset(dataset_id)
        except Exception:
            dataset = bigquery.Dataset(dataset_id)
            dataset.location = settings.google_cloud_location
            self.client.create_dataset(dataset, exists_ok=True)
        schema = [
            bigquery.SchemaField("timestamp", "TIMESTAMP"),
            bigquery.SchemaField("event_type", "STRING"),
            bigquery.SchemaField("method", "STRING"),
            bigquery.SchemaField("path", "STRING"),
            bigquery.SchemaField("status_code", "INTEGER"),
            bigquery.SchemaField("latency_ms", "FLOAT"),
            bigquery.SchemaField("estimated_input_tokens", "INTEGER"),
            bigquery.SchemaField("estimated_output_tokens", "INTEGER"),
            bigquery.SchemaField("retrieved_document_ids", "STRING"),
            bigquery.SchemaField("k", "INTEGER"),
            bigquery.SchemaField("recall_at_k", "FLOAT"),
        ]
        self.client.create_table(bigquery.Table(self.table_id, schema=schema), exists_ok=True)

    def export_events(self, events):
        if not self.client or not events:
            return {"exported": 0, "enabled": bool(self.client)}
        rows = []
        for event in events:
            row = {key: event.get(key) for key in (
                "timestamp", "event_type", "method", "path", "status_code", "latency_ms",
                "estimated_input_tokens", "estimated_output_tokens", "k", "recall_at_k")}
            row["retrieved_document_ids"] = ",".join(event.get("retrieved_document_ids", []))
            rows.append(row)
        errors = self.client.insert_rows_json(self.table_id, rows)
        if errors:
            logger.error("BigQuery insert errors: %s", errors)
            return {"exported": 0, "errors": errors}
        return {"exported": len(rows), "enabled": True}

sink = BigQuerySink()
