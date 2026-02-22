# Testing & CI/CD Commands

## FIRST TIME: Install Dependencies
```bash
# From root folder
pip install -r requirements.txt
```
This installs pytest, flake8, pylint, tensorflow, pillow, dvc and other libraries relevant for the image pipeline.

## Run Unit Tests
```bash
cd cats_dogs_code
pytest -v
```

## Run Tests with Coverage
```bash
pytest -v --cov=src --cov-report=html
# View: htmlcov/index.html
```

## Run Linting
```bash
# Flake8
flake8 src tests

# Pylint
pylint src
```

## All Checks (as per CI/CD)
```bash
# 1. Lint
flake8 src tests --count --select=E9,F63,F7,F82 --show-source --statistics

# 2. Test
pytest tests/ -v --cov=src --cov-report=html

# 3. Check coverage
# Open htmlcov/index.html
```

## Expected Results
- Unit tests should exercise the image preprocessing and CNN utilities
- Coverage should remain high (~95%)
- Lint: No critical errors or style violations
