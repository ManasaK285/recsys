install:
	pip install -e .

api:
	uvicorn api.main:app --reload

dashboard:
	streamlit run ui/dashboard.py

demo:
	python -m harness.demo

test:
	pytest

lint:
	ruff check .

typecheck:
	mypy . --ignore-missing-imports
