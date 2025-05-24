
.PHONY: test test-unit test-integration test-performance test-coverage clean

# Tests
test:
	python -m pytest tests/ -v

test-unit:
	python -m pytest tests/test_core/ tests/test_features/ tests/test_filters/ tests/test_utils/ -v

test-integration:
	python -m pytest tests/test_integration/ -v --timeout=60

test-performance:
	python -m pytest tests/test_performance/ -v --benchmark-only

test-coverage:
	python -m pytest tests/ --cov=lorentzian_trading --cov-report=html --cov-report=term

test-fast:
	python -m pytest tests/ -x -v --disable-warnings

# Nettoyage
clean:
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
