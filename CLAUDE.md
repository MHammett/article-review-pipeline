# Working in this repo

**Multiple Claude Code sessions can be active in this checkout at the same time** (different tools/windows/tasks Mike is running concurrently). `HEAD` and the git stash stack are both single, repo-wide state — not scoped to a session or even to a `git worktree`.

**Always start any git work in a dedicated worktree — never edit, branch, commit, or stash directly in this main checkout,** even for what looks like a one-line change:

```bash
git worktree list                                  # see what's already isolated
git worktree add ../content-intelligence-<slug> -b <branch>
```

This has caused real collisions more than once: a branch-switch by one session moved `HEAD` out from under another session mid-task (2026-08-09), and a `git stash` from one session was popped into a different session's working tree with conflicts (2026-08-09, 2026-08-11) — because the stash stack and `HEAD` are shared across every worktree of the repo, not scoped to one. Being in a worktree protects against the branch-switch problem but **not** the stash-collision one.

**Avoid `git stash` entirely when there's any other way to achieve the goal** — e.g. `git show <ref>:<path>` to inspect another ref's version of a file without touching the working tree. If `git stash` is unavoidable, never chain `push`/`pop` in one command; run `git stash list` immediately before *and* after any stash operation to confirm nothing unexpected was touched.

**A new worktree needs its own `uv sync` before its tests mean anything.** The venv's editable install resolves `ci_article_review`/`ci_core` to whichever checkout ran `uv sync` last — usually the main checkout — so a fresh worktree's tests can silently run against the *main checkout's* source, not the worktree's edits, and still show green. After creating a worktree:

```bash
uv sync
uv run python -c "import ci_article_review; print(ci_article_review.__file__)"   # must print a path under THIS worktree
```

**Before treating a PR as done, recheck its mergeability, not just its status at open time:**

```bash
gh pr view <n> --json mergeable,mergeStateStatus,statusCheckRollup
```

This repo sees frequent concurrent PRs — a branch that opened cleanly can go `CONFLICTING` by the time you come back to it if other PRs landed on `main` in the meantime.
