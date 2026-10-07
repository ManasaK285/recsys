FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python run_pipeline.py
EXPOSE 8501 8000
# Dashboard: docker run -p 8501:8501 curriculumguard
# API:       docker run -p 8000:8000 curriculumguard uvicorn api:app --host 0.0.0.0 --port 8000
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
