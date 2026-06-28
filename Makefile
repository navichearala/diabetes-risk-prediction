# Convenience targets for the Diabetes Risk Prediction project.
# Run `make help` to list available commands.

.PHONY: help install eda train predict test lint clean mlflow-ui all

help:
	@echo "Targets:"
	@echo "  install    Install dependencies"
	@echo "  eda        Generate EDA visualizations"
	@echo "  train      Train all models + log to MLflow"
	@echo "  predict    Run a sample inference"
	@echo "  test       Run the pytest suite"
	@echo "  lint       Run ruff + black --check"
	@echo "  mlflow-ui  Launch the MLflow tracking UI"
	@echo "  clean      Remove caches and generated artifacts"
	@echo "  all        install -> lint -> test -> train"

install:
	pip install -r requirements.txt

eda:
	python src/eda.py

train:
	python src/train.py

predict:
	python src/predict.py

test:
	pytest -q

lint:
	ruff check src tests
	black --check src tests

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache .ruff_cache
	find . -name "*.pyc" -delete

all: install lint test train
