# Dakota1890 Repo2RLEnv environments

This directory contains Repo2RLEnv/Harbor-style RL environments generated from the Dakota1890 GitHub repository history.

## What Repo2RLEnv adds

Hugging Face's `Repo2RLEnv` turns merged GitHub PRs into verifiable coding-agent tasks in the Harbor format. For Dakota1890 this is a second RL-environment track alongside the existing Dakota grammar/verifiers environment:

- existing track: language/grammar rewards from Dakota 1890 source material
- this track: software-engineering rewards from real Dakota1890 PRs, scored against oracle patches

The generated tasks are agent-facing coding tasks. A model edits a checkout of Dakota1890 at the PR base commit, and the verifier scores the resulting `git diff` against the merged PR's oracle patch.

## Generated environment

`dakota1890-prdiff/` was generated with the stable `pr_diff` pipeline:

```bash
PATH=/tmp/gh-latest-arm64/bin:$PATH uvx --from repo2rlenv repo2rlenv --no-ui generate \
  --repo HarleyCoops/Dakota1890 \
  --pipeline pr_diff \
  --pipeline-opt limit=8 \
  --pipeline-opt max_files_per_pr=25 \
  --pipeline-opt min_loc_changed=3 \
  --out repo2rlenv/dakota1890-prdiff \
  --org HarleyCoops \
  --dataset-name dakota1890-prdiff
```

Current local result:

- candidates: 4 merged PRs
- emitted tasks: 3
- skipped tasks: 1 test-only PR
- reward kind: `diff_similarity`
- generated task IDs:
  - `HarleyCoops__Dakota1890-15`
  - `HarleyCoops__Dakota1890-16`
  - `HarleyCoops__Dakota1890-18`

## Validate

```bash
PATH=/tmp/gh-latest-arm64/bin:$PATH uvx --from repo2rlenv repo2rlenv validate repo2rlenv/dakota1890-prdiff
```

Expected result:

```text
✓ all 3 tasks valid
```

## Important local prerequisite

The Ubuntu `gh` currently installed in this WSL environment is too old for Repo2RLEnv's `pr_diff` query because it does not expose the `baseRefOid` JSON field. The working generation command above prepends a locally downloaded GitHub CLI v2.93.0 arm64 binary at `/tmp/gh-latest-arm64/bin/gh`.

If `/tmp/gh-latest-arm64` is gone, reinstall a recent arm64 `gh` binary or update system `gh` before regenerating.

## Running with Harbor

Repo2RLEnv emits Harbor-shaped task directories. Once Docker and Harbor are available, the oracle should score 1.0:

```bash
uv tool install harbor
harbor run -p repo2rlenv/dakota1890-prdiff -a oracle --env docker -n 1
```

For a coding agent, swap `-a oracle` for an agent adapter such as `codex`, `claude-code`, `openhands`, or `hermes` and pass that adapter's provider credentials through Harbor's `--ae` flags. The verifier's optional LLM judge can receive its key through `--ve`; without it, the deterministic reward components still run.

## Publishing later

Do not publish automatically unless you intend to create/update a Hugging Face dataset. When ready:

```bash
repo2rlenv push repo2rlenv/dakota1890-prdiff HarleyCoops/dakota1890-prdiff
```

For `pr_diff`, publishing creates the Hub dataset card and Harbor registry metadata; it does not need the heavyweight bootstrap-image flow used by `pr_runtime`.
