# SPATHI benchmarks

Reproducible experiments for comparing [SPATHI](https://github.com/AdrianSeguraOrtiz/SPATHI)
with inference tools orchestrated by
[ANDREA](https://github.com/AdrianSeguraOrtiz/ANDREA).

This is a third, independent repository. It owns experiment design, source and image
locks, the ANDREA adapter, immutable intermediate artifacts and compact reports without
coupling those concerns to either scientific package.

## What is runnable now

`smoke` is the only runnable profile. It verifies the complete contract with one small
dyngen dataset, GENIE3 in `group_emulated` mode and SPATHI in `group_native` mode. Each
tool is executed and evaluated independently so its result can be reused independently.
The smoke has no statistical power and its SPATHI configuration is not eligible for
scientific selection.

`reduced` and `full` remain explicit planned profiles. They will not run until their
dataset partitions and complete ANDREA request matrices are committed. The bounded
SPATHI calibration engine is implemented, but scientific calibration must wait for a
proper `development` panel; it must never use the smoke or test panel.

## Setup and first run

The expected checkout layout is:

```text
Salonica/
├── ANDREA/
├── SPATHI/
└── SPATHI-benchmarks/
```

Use one Python environment for the benchmark CLI and the locked ANDREA checkout:

```bash
python -m pip install -e ../ANDREA -e '.[dev]'
scripts/doctor.sh
scripts/run_all.sh smoke smoke-001
```

Experiments require all three repositories to be at their locked revisions and clean.
The benchmark repository must also be committed and clean because every run freezes its
orchestrator commit and all run-defining inputs.

The default paths can be overridden with `ANDREA_REPO`, `SPATHI_REPO`, `WORK_ROOT` and
`BENCH_PYTHON`. `ANDREA_PYTHON` selects the interpreter used for `python -m andrea`; it
defaults to the interpreter running `spathi-bench`.

## Resume and stage control

Stages are idempotent. Repeating the same profile/run ID resumes from its recorded state
and verifies every reused artifact. A failed inference or evaluation leaves already
published per-tool artifacts in the shared cache, so the next attempt does not repeat
them.

```bash
scripts/run_all.sh smoke smoke-001 --resume
scripts/run_all.sh smoke smoke-001 --from infer
scripts/run_all.sh smoke smoke-001 --until generate
```

Valid stage names are `preflight`, `image`, `generate`, `infer`, `validate`, `evaluate`
and `summarize`. Individual scripts under `scripts/` expose the same stages. A per-run
filesystem lock rejects concurrent mutation of the same run ID.

Inference and evaluation misses can run concurrently. `resources.infer_network` defines
`max_parallel_runs`; `max_cores` and `max_ram_gb` apply to each concurrent ANDREA
invocation, so profile authors must budget their product explicitly.
Catalog images use a host-wide lease per mutable Docker tag: matching immutable digests
share the lease and remain parallel, while conflicting benchmark revisions wait rather
than retagging an image underneath an active container.

## What is reused after a SPATHI change

Artifacts live in a shared content-addressed store under `work/v1/artifacts/`. Their
keys include the inputs that can change their contents:

| Change | Generated datasets | Catalog baselines | SPATHI inference | Evaluation |
| --- | --- | --- | --- | --- |
| SPATHI commit/config/image | reuse | reuse | recompute | recompute only for changed SPATHI results |
| catalog tool image/params | reuse | recompute affected tool | reuse | recompute affected tool |
| simulator request/image | recompute affected dataset | recompute downstream | recompute downstream | recompute downstream |
| ANDREA commit or host environment | recompute | recompute | recompute | recompute |

To benchmark a new SPATHI commit, update `repos.lock.json`, the SPATHI entry in
`images.lock.json`, its commit-specific image tag in the custom-tool requests, and any
changed build-input hashes. Commit those changes and start a new run ID. The pipeline
will build the new SPATHI image but recover unchanged generated data, catalog inference
and catalog evaluations as cache hits. It is therefore unnecessary to execute the other
tools again merely because SPATHI changed.

The run state records hit counts and the concrete SPATHI image ID. Catalog containers
are pulled by immutable repository digest. SPATHI is passed to ANDREA by exact local
image ID, eliminating mutable-tag ambiguity for the external tool.

## SPATHI calibration

The search contract in `experiments/v1/spathi/search-space.json` has three bounded
development stages:

1. convergence: 4 configurations × 5 algorithmic seeds;
2. structure: 7 configurations × 3 seeds around one parent;
3. estimator: at most 5 configurations × 3 seeds around at most two parents.

Generate the first native ANDREA request bundle with:

```bash
spathi-bench calibration-plan \
  --repo-root . \
  --stage convergence \
  --output-dir experiments/v1/spathi/generated/convergence
```

After adding that bundle to a committed `purpose=calibration` profile and running it,
rank the complete balanced stage from its `reports/summary.json`:

```bash
spathi-bench calibration-rank \
  --repo-root . \
  --stage convergence \
  --manifest experiments/v1/spathi/generated/convergence/calibration-manifest.json \
  --summary work/v1/<profile>/<run>/reports/summary.json \
  --output experiments/v1/spathi/generated/convergence-shortlist.json
```

Pass that shortlist to the next `calibration-plan` with `--parents`. After estimator
ranking, use the same command with `--stage validation` and the 2–3 finalist shortlist;
then rank that validation run with `calibration-rank --stage validation`. Freeze its
winner explicitly, review the diff and commit both files:

```bash
spathi-bench calibration-freeze \
  --repo-root . \
  --ranking experiments/v1/spathi/generated/validation-ranking.json \
  --output experiments/v1/spathi/selected-config.json
```

Selection maximizes directed group-level AUPR with equal weight per group, seed and
dataset, followed by stability, wall time, peak memory when available, and simplicity.
It rejects incomplete batches, unequal datasets/seeds, different evaluable group sets,
mixed roles and all test rows. A ranking embeds the calibration rows needed to recompute
its scores and binds them to the exact SPATHI source/build identity and local image ID.
The frozen selection is hash-linked to that committed ranking; editing a score, winner,
configuration or source build invalidates the test gate.

Development, validation and test are separate runnable profile purposes. Test data
cannot even be generated by a runnable profile until `selected-config.json` contains a
fully evidenced frozen choice; a SPATHI test batch may contain only that selected
configuration. Dataset generation tasks are content-fingerprinted globally across
development, validation and test; identical effective tasks and shared external inputs
cannot cross partitions. The CLI never freezes a winner as a side effect of ranking:
`calibration-freeze` is a separate, explicit and reviewable operation.

## Output layout

```text
work/v1/
├── artifacts/                   immutable shared dataset/inference/evaluation objects
└── <profile>/<run-id>/
    ├── input_snapshot/          frozen committed requests, schemas and locks
    ├── preflight/
    ├── provenance/
    ├── logs/
    ├── handoff-validation.json
    ├── state.json
    └── reports/
        ├── metrics.csv          raw ANDREA metrics plus benchmark provenance
        └── summary.json         partitioned aggregates and calibration-ready rows
```

The benchmark reads canonical manifests rather than discovering timestamped outputs.
There is no extra integration script: ANDREA owns `run_report.json` and the merged
network, while this repository validates that handoff and enriches its evaluation.

See [`experiments/v1/design.md`](experiments/v1/design.md) for the scientific design,
information regimes, simulator panel and leakage controls.
