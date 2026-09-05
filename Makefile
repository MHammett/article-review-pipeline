.PHONY: install setup lint test test-fast build

install:
	uv sync

setup:
	uv sync
	uv run python -m ci_article_review.setup

# mypy is pointed at each package's src/ rather than at packages/ wholesale:
# the three identically-named `tests` packages collide and it aborts with
# "Duplicate module named tests" before type-checking anything at all. CI
# (.github/workflows/ci.yml) and the pre-commit hook scope it to src/ for the
# same reason — keep all three in agreement.
lint:
	uv run ruff check packages/
	uv run ruff format --check packages/
	uv run mypy packages/ci-core/src
	uv run mypy packages/ci-article-review/src
	uv run mypy packages/ci-style-profile/src

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
