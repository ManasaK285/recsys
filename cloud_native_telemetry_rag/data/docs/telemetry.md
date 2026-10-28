# Telemetry and Service Level Indicators

Operational telemetry helps operators understand reliability and performance. Common indicators include request rate, error rate, and latency percentiles such as p50 and p95. The p95 latency is the value below which 95 percent of observed request durations fall. A latency threshold can trigger a warning when the service is slower than the intended service level objective.

Token utilization is an approximate measure of the amount of input and output text processed by a language model. Exact token counts depend on the model tokenizer and provider usage metadata. Monitoring token use helps identify unexpectedly long prompts and rising inference costs.

A production deployment should export metrics to a durable monitoring platform and alert through an incident-management channel. In-memory metrics are useful for local demonstrations but reset when the process restarts.
