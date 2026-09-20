# Makefile for arXiv Digest Agent

.PHONY: install test demo lint clean help

# Default target
help:
	@echo "Available targets:"
	@echo "  make install   - Install dependencies"
	@echo "  make test      - Run pytest"
	@echo "  make demo      - Run demo (requires Ollama or API keys)"
	@echo "  make lint      - Run ruff linter"
	@echo "  make clean     - Clean cache and data directories"

# Install dependencies
install:
	pip install -e ".[groq,gemini]"

# Install without optional deps
install-minimal:
	pip install -e .

# Run tests
test:
	python -m pytest

# Run with coverage
test-cov:
	python -m pytest --cov=agent --cov-report=term-missing

# Lint
lint:
	ruff check src tests

# Format
format:
	ruff format src tests

# Run demo: digest 2401.12345, then in-paper ask, then off-topic ask
# Uses LLM_PROVIDER from the environment
demo:
	python -m agent.cli digest 2401.12345 --auto --no-qa --json-out .data/briefing_check.json
	python -m agent.cli ask 2401.12345 "What datasets did they evaluate on?" --verbose
	python -m agent.cli ask 2401.12345 "What does this paper say about the 2026 World Cup?" --verbose

# Run demo with specific paper (auto mode, no QA)
demo-paper:
	python -m agent.cli digest 2401.12345 --auto --no-qa --provider ollama

# Clean generated files
clean:
	rm -rf .data __pycache__ src/__pycache__ src/agent/__pycache__ src/agent/nodes/__pycache__ src/agent/services/__pycache__ tests/__pycache__ .pytest_cache .ruff_cache *.egg-info

# Create .env from example if not exists
env:
	@if not exist .env (copy .env.example .env && echo "Created .env from .env.example") else (echo ".env already exists")

# Show dependency tree
deps:
	pip list --local