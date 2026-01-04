.PHONY: setup data eda train test lint format api docker

setup:
	python -m pip install -r requirements.txt

data:
	python scripts/download_data.py --out data/raw/heart.csv

eda:
	python scripts/run_eda.py --data data/raw/heart.csv --outdir reports/figures

train:
	python -m src.train --data data/raw/heart.csv --outdir artifacts --experiment heart_disease --seed 42

test:
	pytest -q

lint:
	ruff check .

format:
	black .

api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000

docker:
	docker build -t heart-api:latest .
