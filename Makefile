.PHONY: install lint test build

install:
	uv sync

lint:
	uv run ruff check packages/
	uv run ruff format --check packages/
	uv run mypy packages/

test:
	uv run pytest packages/

build:
	uv build --package ci-core
	uv build --package ci-article-review
	uv build --package ci-web-intel
