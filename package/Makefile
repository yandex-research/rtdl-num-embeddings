.PHONY: default clean doctest lint pre-commit typecheck

PACKAGE_ROOT = rtdl_num_embeddings.py

default:
	echo "Hello, World!"

clean:
	find . -type f -name "*.py[co]" -delete -o -type d -name __pycache__ -delete
	rm -rf .ipynb_checkpoints
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf dist

doctest:
	xdoctest $(PACKAGE_ROOT)
	python test_code_blocks.py README.md

lint:
	ruff check .
	ruff format --check

# The order is important.
pre-commit: clean lint doctest typecheck

typecheck:
	mypy $(PACKAGE_ROOT)
