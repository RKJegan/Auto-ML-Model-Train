.PHONY: run test lint format clean install

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

run:
	streamlit run app.py

api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v --tb=short

lint:
	flake8 src/ api/ scripts/
	mypy src/ api/

format:
	black src/ api/ scripts/ app.py
	isort src/ api/ scripts/ app.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache build dist *.egg-info

docker-build:
	docker-compose build

docker-run:
	docker-compose up -d
