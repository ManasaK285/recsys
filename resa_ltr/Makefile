install:
	python -m pip install -r requirements.txt

run:
	python run.py

test:
	pytest -q

api:
	uvicorn src.api:app --host 0.0.0.0 --port 8000
