.PHONY: install setup lint test build

install:
	uv sync

setup:
	uv sync
	uv run python -m ci_article_review.setup

lint:
	uv run ruff check packages/
	uv run ruff format --check packages/
	uv run mypy packages/

test:
	uv run pytest packages/

build:
	uv build --package ci-core
	uv build --package ci-article-review
	uv build --package ci-style-profile
