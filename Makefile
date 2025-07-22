# Makefile for Python cleanup

.PHONY: clean

clean:
	@echo "Removing __pycache__ and *.pyc files..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "Cleanup complete ✅"
