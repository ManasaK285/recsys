FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir .
EXPOSE 8000 8501
CMD ["uvicorn", "bargaining_lab.api:app", "--host", "0.0.0.0", "--port", "8000"]
