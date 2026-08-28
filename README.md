# SPATHI benchmarks

Reproducible experiments for comparing [SPATHI](https://github.com/AdrianSeguraOrtiz/SPATHI)
with inference tools orchestrated by
[ANDREA](https://github.com/AdrianSeguraOrtiz/ANDREA).

This is intentionally a third repository. It owns experiment design, pinned source
revisions, container adapters and generated reports without coupling those concerns to
either scientific package.

## Current scope

Version `v1` starts with one runnable end-to-end smoke profile:

1. ANDREA simulates a small dynamic single-cell dataset with dyngen.
2. ANDREA runs GENIE3 in `group_emulated` mode and SPATHI as a `group_native`
   external Docker tool in the same inference plan.
3. The benchmark validates the canonical `run_report.json` handoff.
4. ANDREA evaluates both predictions against the same group-specific truth.
5. A compact, provenance-enriched metrics table is written under the run directory.

`reduced` and `full` are defined as planned profiles in
[`experiments/v1/design.md`](experiments/v1/design.md); they are deliberately not
executable until their simulator/tool matrices have been preflighted and frozen.

## Quick start

The default layout is:

```text
Salonica/
├── ANDREA/
├── SPATHI/
└── SPATHI-benchmarks/
```

From this repository:

```bash
python -m pip install -e '.[dev]'
scripts/run_all.sh smoke
```

The command creates an immutable run directory under `work/v1/smoke/`. To choose a
stable identifier, pass it as the second argument:

```bash
scripts/run_all.sh smoke local-check
```

Every stage can also be invoked independently with the same profile and run ID:

```bash
scripts/00_preflight.sh smoke local-check
scripts/05_build_spathi_image.sh smoke local-check
scripts/10_generate_data.sh smoke local-check
scripts/20_run_inference.sh smoke local-check
scripts/30_validate_handoff.sh smoke local-check
scripts/40_evaluate.sh smoke local-check
scripts/50_summarize.sh smoke local-check
```

Set `ANDREA_REPO`, `SPATHI_REPO` or `WORK_ROOT` to override the sibling paths and
work root. Stages never delete or silently overwrite an existing result.

## Why there is no integration script

The smoke inference is deliberately hybrid: SPATHI is registered through ANDREA's
external-tool contract and runs beside catalog tools. ANDREA therefore owns the merged
network and canonical `run_report.json`; inventing another integration format here would
duplicate that contract and risk dataset-identity errors. The benchmark only validates
the handoff and enriches evaluation metrics with experiment provenance.

## Repository layout

```text
containers/spathi/       ANDREA `/io` adapter and reproducible image
experiments/v1/          design, profiles and native ANDREA request JSON
schemas/                 benchmark-owned JSON Schemas
scripts/                 thin stage orchestrators
src/spathi_bench/        validation, state and summarization helpers
tests/                    unit and adapter-contract tests
work/                     generated data and logs (ignored)
```

Source revisions are fixed in `repos.lock.json`; the image base, dependencies and
adapter-file SHA-256 digests are fixed in `images.lock.json`. Each concrete run also
records the built local image ID.
