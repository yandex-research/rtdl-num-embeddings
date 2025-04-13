.PHONY: default clean doctest lint pre-commit typecheck

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
	uv run pdoc -d google --math --no-include-undocumented rtdl_num_embeddings.py

doctest:
	uv run xdoctest rtdl_num_embeddings.py
	uv run python test_code_blocks.py README.md

lint:
	uv run typos rtdl_num_embeddings.py
	uv run ruff check rtdl_num_embeddings.py
	uv run ruff format --check rtdl_num_embeddings.py

# The order is important.
pre-commit: clean lint doctest typecheck

typecheck:
	uv run mypy rtdl_num_embeddings.py
