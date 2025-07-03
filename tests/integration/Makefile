# Makefile for OpenWebUI Adaptive Memory Integration Tests

.PHONY: help test test-setup test-integration test-all test-performance clean install

# Default target
help:
	@echo "Available targets:"
	@echo "  help              - Show this help message"
	@echo "  install           - Install test dependencies"
	@echo "  test-setup        - Run setup validation tests"
	@echo "  test-integration  - Run integration tests (default scenario)"
	@echo "  test-all          - Run all integration test scenarios"
	@echo "  test-performance  - Run performance tests"
	@echo "  test-specific     - Run specific test (use TEST=test_name)"
	@echo "  clean             - Clean test artifacts"
	@echo "  lint              - Run linting on test code"

# Install dependencies
install:
	@echo "Installing test dependencies..."
	pip install pytest pytest-asyncio aiohttp httpx sentence-transformers
	pip install -r ../../requirements.txt

# Run setup validation
test-setup:
	@echo "Running setup validation tests..."
	python -m pytest test_setup_validation.py -v

# Run integration tests with default scenario
test-integration:
	@echo "Running integration tests (happy path scenario)..."
	python run_integration_tests.py

# Run all scenarios
test-all:
	@echo "Running all integration test scenarios..."
	python run_integration_tests.py --all-scenarios

# Run performance tests
test-performance:
	@echo "Running performance tests..."
	python run_integration_tests.py --performance

# Run specific test
test-specific:
	@echo "Running specific test: $(TEST)"
	python -m pytest $(TEST) -v -s

# Clean test artifacts
clean:
	@echo "Cleaning test artifacts..."
	rm -rf __pycache__ .pytest_cache
	rm -rf logs/* results/* test_data/*
	rm -f test_recordings.json
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Run linting
lint:
	@echo "Running linting on test code..."
	python -m flake8 . --max-line-length=100 --ignore=E203,W503
	python -m black --check --diff .

# Quick smoke test
smoke:
	@echo "Running smoke tests..."
	python -m pytest test_setup_validation.py::TestSetupValidation::test_openwebui_mock_basic_functionality -v

# Debug mode tests
debug:
	@echo "Running tests in debug mode..."
	python run_integration_tests.py --debug

# CI/CD target
ci: install test-setup test-integration
	@echo "CI/CD tests completed"

# Development target
dev: clean install smoke test-integration
	@echo "Development testing completed"