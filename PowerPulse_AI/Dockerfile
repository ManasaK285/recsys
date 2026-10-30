FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p data
EXPOSE 8000 8501
CMD ["sh","-c","python -m src.generate_data --days 21 --devices 80 --seed 42 && (uvicorn src.api:app --host 0.0.0.0 --port 8000 &) && streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501"]
