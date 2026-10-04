# AGENTS.md

Working rules for this fork of [antirez/ds4](https://github.com/antirez/ds4),
for AI coding agents and human contributors. It covers how work lands; the
engine's own goals, quality rules and layout are in [AGENT.md](AGENT.md), which
comes from upstream and wins on anything about the code itself.

## Trunk-only development

`main` is the only long-lived branch. It holds upstream `main` plus this fork's
work, and every change is committed directly to it.

- **No pull requests, no feature branches.** Review happens as commits land,
  not at a merge gate. A short-lived local branch is fine for an experiment, but
  it lands on `main` (or is dropped) quickly rather than accumulating
  divergence. Do not create branches on the remote.
- **Upstream comes in by merge.** Fetch `upstream` and merge `upstream/main`
  (or an upstream pull request the fork depends on) into `main`, resolving
  conflicts in the merge commit. Never rebase or rewrite commits that may have
  been pushed.
- **Keep `main` buildable.** Each commit builds for the backends it touches and
  passes the checks that cover it (unit tests, the model-backed comparisons
  under `tests/`, `./ds4_test --server` for server changes). A change spanning
  several files lands as one self-contained commit, not a series with broken
  intermediate states.
- **Stage by explicit path.** More than one agent session may share a working
  tree and its index; commit with `git commit -- <paths>` or `git add <paths>`,
  never `git add -A`.
- **Pushing is the maintainer's call.** Do not push unless asked.

## Semantic commits

Subjects follow `type(scope): subject`:

- **type**: `feat` (new capability), `fix` (wrong behaviour), `perf` (same
  output, faster or smaller), `refactor` (same behaviour, different structure),
  `test`, `docs`, `build`, `chore`.
- **scope**: the subsystem, e.g. `rocm`, `metal`, `cuda`, `server`, `cli`,
  `gguf-tools`, `tests`, or a model family such as `kolibri`, `glm`, `qwen4`.
  Several scopes are comma-separated: `feat(kolibri,rocm): ...`.
- **subject**: lowercase, no trailing period, says what the change does or what
  is now true ("decode reads each KV row once per KV head"), not how it was
  done.

The body explains why, and carries the evidence: for `perf`, before and after
numbers from the same benchmark; for anything touching inference, the
correctness check that shows the output did not drift (see AGENT.md:
correctness before speed). Measurement noise and the machine they came from
belong there too.

AI assistance is recorded with a `Co-Authored-By:` trailer, not in the files.

Commits made before this file existed use plain `scope: subject` subjects; they
are not rewritten.
