.PHONY: install setup lint test test-fast build

install:
	uv sync

setup:
	uv sync
	uv run python -m ci_article_review.setup

# mypy gets each package's `src` explicitly rather than the whole tree. All
# three packages ship a top-level `tests` package, so `mypy packages/` aborts
# with "Duplicate module named tests" having checked nothing at all — it is a
# silent no-op, not a warning. CI (.github/workflows/ci.yml) and the
# pre-commit mypy hook scope it the same way. `wildcard` so a fourth package
# is covered without editing this line.
MYPY_TARGETS := $(wildcard packages/*/src)

lint:
	uv run ruff check packages/
	uv run ruff format --check packages/
	uv run mypy $(MYPY_TARGETS)

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
