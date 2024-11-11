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

docs:
	pixi run --environment dev docs

doctest:
	pixi run --environment test doctest

lint:
	pixi run --environment test lint

# The order is important.
pre-commit: clean lint doctest typecheck

typecheck:
	pixi run --environment test typecheck
