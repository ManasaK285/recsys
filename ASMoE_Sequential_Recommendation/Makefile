smoke:
	python -m src.train --config configs/smoke.yaml --output outputs_smoke
train:
	python -m src.train --config configs/default.yaml --output outputs_v2
test:
	pytest -q
app:
	streamlit run app.py
