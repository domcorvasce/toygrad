.PHONY: test

test:
	@echo "Running tests..."
	@python3 toygrad/test_gradient.py -v

lint:
	@echo "Running linter..."
	@black .
