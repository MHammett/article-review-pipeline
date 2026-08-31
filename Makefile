.PHONY: install setup lint test test-fast build

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

# Same suite minus the inherently wall-clock-bound tests — for a tight inner
# loop. `make test` is what has to be green; see README "The `slow` marker".
test-fast:
	uv run pytest packages/ -m "not slow"

build:
	uv build --package ci-core
	uv build --package ci-article-review
	uv build --package ci-style-profile
