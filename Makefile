.PHONY: test

test:
	@echo "Running tests..."
	@python3 -m unittest discover -s toygrad -p "test_*.py" -v

lint:
	@echo "Running linter..."
	@black .
