FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .
RUN mkdir -p /app/data /app/outputs
EXPOSE 8000
CMD ["uvicorn", "signalscope.api:app", "--host", "0.0.0.0", "--port", "8000"]
