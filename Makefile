# CDSS Automation Makefile

VENV = backend/venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

.PHONY: help install ingest eval serve test clean frontend ui

help:
	@echo "Available commands:"
	@echo "  make install  - Install backend dependencies"
	@echo "  make ingest   - Ingest symptom-disease & PMC literature into Qdrant"
	@echo "  make eval     - Run MedQA benchmark evaluation suite"
	@echo "  make serve    - Start FastAPI REST API server (UI at http://localhost:8000)"
	@echo "  make frontend - Start Angular dev server on port 4200"
	@echo "  make ui       - Open browser to http://localhost:8000"
	@echo "  make test     - Run pytest test suite"

install:
	$(PIP) install -r backend/requirements.txt

ingest:
	PYTHONPATH=backend/src $(PYTHON) -m cdss.corpus.ingest --max-pubmed 50

eval:
	PYTHONPATH=backend/src $(PYTHON) -m cdss.eval.benchmark_suite --n 50

eval-quick:
	PYTHONPATH=backend/src $(PYTHON) -m cdss.eval.benchmark_suite --n 10

serve:
	PYTHONPATH=backend/src $(PYTHON) -m uvicorn cdss.api.server:app --host 0.0.0.0 --port 8000 --reload

frontend:
	cd frontend && npm start

ui:
	open http://localhost:8000

test:
	PYTHONPATH=backend/src $(PYTHON) -m pytest tests/

