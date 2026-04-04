# Step 0 Legacy Archive

This directory holds Dakota-adjacent material that was removed from the active repository surface during the step-0 audit.

## What lives here

- superseded setup and launch guides that pointed at stale counts, OpenRouter, or renamed paths
- legacy extraction scripts that duplicated the maintained Dakota entrypoints
- manual or obsolete tests that no longer reflected the packaged environment
- tracked root-level debug logs and one-off publish bundles that are not needed for local reproducibility

## Why archive instead of delete

The repository is both a technical artifact and a research narrative. These files still show dead ends, earlier assumptions, and historical implementation choices, so they are preserved for context. They are no longer the canonical instructions for running Dakota1890.

Use the root-level step-0 docs instead:

- `REPO_MAP.md`
- `PIPELINE.md`
- `SETUP.md`
- `PIPELINE_AUDIT.md`
- `VALIDATION_REPORT.md`
