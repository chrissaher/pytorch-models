.PHONY: test

# make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = src

# Run tests for the library
test:
	python -m pytest -n auto --dist=loadfile -v ./tests/
