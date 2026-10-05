# Development Guide

## Setup

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements-dev.txt
```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

```bash
black src/ api/ scripts/ app.py
isort src/ api/ scripts/ app.py
flake8 src/ api/ scripts/
```
