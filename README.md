# SPATHI benchmarks

Reproducible experiments for comparing [SPATHI](https://github.com/AdrianSeguraOrtiz/SPATHI)
with inference tools orchestrated by
[ANDREA](https://github.com/AdrianSeguraOrtiz/ANDREA).

This is a third, independent repository. It owns experiment design, source and image
locks, the ANDREA adapter, immutable intermediate artifacts and compact reports without
coupling those concerns to either scientific package.

## What is runnable now

The committed `smoke` profile verifies the complete contract with one small dyngen
dataset, GENIE3 in `group_emulated` mode and SPATHI in `group_native` mode. It has no
statistical power and is never eligible for scientific selection. The scientific
profiles are materialized automatically below `work/`. The default `full` calibration
begins with four tree-count configurations over 15 development datasets and five
algorithmic seeds. A paired `reduced` budget keeps the same search and validation
stages but uses five datasets per role and two algorithmic seeds per stage.

The frozen development and validation panels each contain twelve dyngen datasets plus
three scMultiSim datasets. scMultiSim uses a different committed regulatory network in
each partition. The held-out test design contains 50 datasets: twenty dyngen, ten
scMultiSim, five SERGIO, ten BoolODE and five GeneSPIDER2. Dyngen spans 200--500 cells
and 70--130 genes; scMultiSim crosses low balanced rewiring with high rewiring, UMI noise
and a 12-cell minority population; BoolODE crosses clean and dropout-processed fixed-GRN
conditions. SERGIO, BoolODE and GeneSPIDER2 are invariant-network controls. All simulator images
are pinned by repository digest and every scenario/run pair passes ANDREA planning.

Every later calibration profile is created only after ranking its predecessor;
pre-generating it would violate the bounded search. `reduced` and `full` are immutable
dataset templates. The high-level workflows materialize the adaptive calibration stages
and the selected `reduced-v1` or `full-v1` profile below `work/`, where their contents
are frozen by the normal provenance pipeline. Test data therefore remain inaccessible
until configuration selection is complete.

## Setup

The expected checkout layout is:

```text
Salonica/
├── ANDREA/
├── SPATHI/
└── SPATHI-benchmarks/
```

Use one Python environment for the benchmark CLI and the locked ANDREA checkout:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ../ANDREA -e '.[dev]'
```

Experiments require all three repositories to be at their locked revisions and clean.
The benchmark repository must also be committed and clean because every run freezes its
orchestrator commit and all run-defining inputs. Adaptive rankings, generated profiles
and the selected configuration live below ignored `work/`; calibration does not require
intermediate Git commits.

No shell exports or environment activation are required. The scripts discover sibling
`ANDREA` and `SPATHI` checkouts and automatically use `.venv/bin/python` when it exists.
Nonstandard installations may override `ANDREA_REPO`, `SPATHI_REPO` or `BENCH_PYTHON`;
high-level adaptive artifacts deliberately remain below the repository's ignored
`work/`. `ANDREA_PYTHON` selects the interpreter used for `python -m andrea`; it defaults
to the interpreter running `spathi-bench`.
Preflight fingerprints both interpreters separately. The statistics stage refuses to
reuse a run if the benchmark interpreter has changed, even when `ANDREA_PYTHON` points
to a different, unchanged environment.

## Natural workflow

The complete scientific workflow has two commands: calibrate SPATHI and compare its
frozen configuration on the held-out benchmark.

```bash
scripts/calibrate.sh
scripts/compare.sh
```

`scripts/calibrate.sh` executes the tree-count, structure and estimator searches plus
independent validation, then freezes one global SPATHI configuration. `scripts/compare.sh` uses that
selection and runs the `full` held-out benchmark by default. For a cheaper development
workflow before spending the full budget, use the matching reduced pair:

```bash
scripts/calibrate.sh --profile reduced
scripts/compare.sh --profile reduced
```

The reduced calibration uses a five-dataset subset of each development/validation
panel and two seeds in every stage. The reduced comparison uses the development panel;
it is a regression run and cannot support claims about held-out performance. A reduced
selection cannot be passed to the full comparison, or vice versa.

Both entrypoints resume automatically. If Docker, the terminal or a tool fails, run the
same command again; completed datasets, individual tool runs, evaluations and adaptive
rankings are verified and reused. The default calibration ID is derived from the locked
repository identities. An explicit ID is useful for naming a study:

```bash
scripts/calibrate.sh --profile reduced --run-id quick-v1
scripts/compare.sh --profile reduced --calibration-run-id quick-v1
scripts/calibrate.sh --run-id study-v1
scripts/compare.sh --calibration-run-id study-v1
```

Concurrent mutation of the same workflow is rejected by a filesystem lock. A different
SPATHI or benchmark commit receives a different default workflow identity rather than
silently reusing an incompatible selection.

`doctor.sh` is optional because normal preflight performs the required gates. It is a
non-destructive way to diagnose Python, Docker, repository locks and image availability
before a long run:

```bash
scripts/doctor.sh
```

Inference and evaluation misses can run concurrently. `resources.infer_network` defines
`max_parallel_runs`; `max_cores` and `max_ram_gb` apply to each concurrent ANDREA
invocation, so profile authors must budget their product explicitly.
Catalog images use a host-wide lease per mutable Docker tag: matching immutable digests
share the lease and remain parallel, while conflicting benchmark revisions wait rather
than retagging an image underneath an active container.

## What is reused after a SPATHI change

Artifacts live in a shared content-addressed store under `work/v1/artifacts/`. Their
keys include the inputs that can change their contents:

| Change | Generated datasets | Catalog runs | SPATHI runs | Joint view/evaluation/exports |
| --- | --- | --- | --- | --- |
| SPATHI commit/config/image | reuse | reuse | recompute | recompute joint results; reserialize ZIPs |
| catalog tool image/params | reuse | recompute affected tool | reuse | recompute affected dataset/regime |
| simulator request/image | recompute affected dataset | recompute downstream | recompute downstream | recompute downstream |
| ANDREA commit or host environment | recompute | recompute | recompute | recompute |

To benchmark a new SPATHI commit, update `repos.lock.json`, the SPATHI entry in
`images.lock.json` and any changed build-input hashes, then commit those source changes.
Running `scripts/calibrate.sh` creates a new workflow identity for that revision. The
pipeline builds the new SPATHI image but recovers unchanged generated data, catalog
inference and catalog evaluations as cache hits. It is therefore unnecessary to execute
the other tools again merely because SPATHI changed.

The run state records hit counts and the concrete SPATHI image ID. Catalog containers
are pulled by immutable repository digest. SPATHI is passed to ANDREA by exact local
image ID, eliminating mutable-tag ambiguity for the external tool.

Joint projections and ZIP artifacts also bind the SPATHI-benchmarks commit that produced
them. A change in integration or serialization code therefore cannot reuse output from
an older implementation; this does not invalidate simulated data or tool executions.

## ANDREA-compatible handoffs

`scripts/run_all.sh ... --until export` produces deterministic `analysis.zip` files
whose member lists and virtual paths come from the locked ANDREA checkout's official
resolvers. SPATHI-benchmarks serializes them, but does not redefine their contracts.

For each dataset there is a generate-data truth bundle. For every individual inference
and evaluation there is a corresponding bundle, and each dataset/regime also has a
joint inference and joint evaluation bundle. The joint inference report contains all
tool IDs in one source, which is what allows ANDREA `compare-networks` to calculate
GENIE3↔SPATHI distances. It records its source reports and hashes and explicitly marks
`integration.monolithic_execution=false`.

The ANDREA GUI can upload the inference `analysis.zip` directly to
`evaluate-inference`; CLI users extract it and pass `run_report.json`. An evaluation
`analysis.zip` contains the evaluation report. The compare-networks GUI accepts the
joint inference ZIP for the networks and the joint evaluation ZIP for metrics; its CLI
likewise consumes the extracted report paths through a comparison request. The original
individual evaluations remain the only inputs to the scientific summary, so metrics are
not duplicated.

## Statistical comparison

Every runnable profile predeclares its statistical comparison families: exact
configuration IDs, SPATHI control, dataset roles, information regime, network levels,
metrics, stratification and minimum independent block count. This prevents a result from
silently relabelling a comparison family or choosing configurations after seeing test
scores. The declared candidate-space class remains a reviewed design attribute; at run
time the benchmark additionally requires identical ANDREA candidate-universe counts for
every paired context. Directed AUPR is the confirmatory endpoint for the primary `group_only`
family; other requested metrics and levels are reported as exploratory analyses.

The hierarchy is fixed:

1. average equally over evaluable group contexts within one execution;
2. average algorithmic-seed executions within one configuration and dataset;
3. use each independent `dataset_id` once as a paired block.

Cells, edges, groups and algorithmic seeds are never treated as independent replicates.
Role, regime and level are mandatory strata; profiles may additionally separate splits,
conditions, simulators or generation jobs. Two dataset IDs with the same simulator seed
cannot become two blocks in any resulting stratum, so shared conditions must be split or
redesigned explicitly.
Configuration hashes may not change under one ID, duplicate algorithmic seeds are
rejected, and deterministic or differently replicated tools are still reduced to one
equal-weight score per dataset. Every tool must have the same contexts, candidate universe
and complete dataset panel. Missing or
non-evaluable values are not imputed and do not trigger pairwise deletion; the family is
reported as `incomplete_panel`. Numerically evaluated empty networks remain valid scores.

For at least three tools the omnibus test is Friedman with midranks, tie correction,
Kendall's W and an exact or seeded within-block permutation p-value. Post-hoc contrasts
compare the predeclared SPATHI control with every rival using a two-sided paired Wilcoxon
signed-rank permutation test. Holm correction controls family-wise error. With two tools,
Friedman is not applicable and the paired Wilcoxon contrast is used directly. Reports
also include paired differences, Hodges–Lehmann estimates, rank-biserial effects,
win/tie/loss counts and a dataset-level bootstrap interval whose confidence level follows
the predeclared alpha. The Markdown report exposes the omnibus and post-hoc evidence as
well as the machine-readable JSON and CSV files.

`not_testable` is a successful analytical outcome, not a pipeline failure. In particular,
the smoke always produces that status. `reduced-v1` is intended for development; claims
from `full-v1` require the held-out complete panel, its predeclared Holm families and a
reported power/sensitivity analysis; fifty datasets are not treated as an automatic
guarantee of power.

## SPATHI calibration

The search contract has three bounded development stages followed by one validation
stage. The candidate space is identical in both budgets; only the amount of evidence
changes:

| Stage | `reduced` | `full` |
| --- | ---: | ---: |
| tree-count (`convergence`) | 4 configurations × 2 seeds × 5 datasets = 40 | 4 × 5 × 15 = 300 |
| structure | 7 × 2 × 5 = 70 | 7 × 3 × 15 = 315 |
| estimator (maximum) | 10 × 2 × 5 = 100 | 10 × 3 × 15 = 450 |
| validation (maximum) | 3 finalists × 2 × 5 = 30 | 3 × 3 × 15 = 135 |
| **Maximum total** | **240** | **1,200** |

The three development stages reference exactly the same generation jobs, so simulated
data are generated once and recovered from the content-addressed cache thereafter.
Every stage still performs normal evaluation and exports the official ANDREA
`analysis.zip` handoffs. It stops at `summarize` because `calibration-rank`, not the
generic Friedman/Holm stage, owns configuration selection.

`scripts/calibrate.sh` performs that entire sequence, including stage materialization,
ranking and final freezing. It defaults to `--profile full`; `--profile reduced` changes
the evidence budget without changing candidates, objective or stage order. Adaptive
request bundles, rankings and the selected configuration are stored together under
`work/v1/workflows/calibration/<budget>/<flow-id>/`; the workflow state hashes each
transition and binds the selected configuration to that budget.
Nothing from one adaptive stage has to be copied into the source tree or committed
before the next stage can run.

Selection maximizes directed group-level AUPR with equal weight per group, seed and
dataset, followed by stability, wall time, peak memory when available, and simplicity.
The first stage is therefore bounded tuning of `n_estimators`, not a formal plateau or
one-standard-error convergence test.
Every manifest freezes canonical run IDs, parameters and execution hashes. Each ranking
is content-linked to its predecessor, and freezing recursively validates the complete
convergence → structure → estimator → validation chain. Missing runs, extra finalists,
unequal panels, role mixing, changed topologies and altered source/build identities are
rejected.

After calibration, `scripts/compare.sh` materializes the matching selected-SPATHI
profile below `work/v1/workflows/comparison/` and executes generation, inference,
evaluation, ANDREA-compatible exports, summarization and statistics. The default is the
held-out `full` profile; `scripts/compare.sh --profile reduced` selects the development
regression profile instead. In either case it requires a completed calibration with the
same profile name.

The full primary directed family compares SPATHI with GENIE3, GRNBoost2,
inferCSN-without-pseudotime, pySCENIC-GRN, scMINER with the common TF list and TIGRESS.
ARACNe3 is added only to the exploratory topology family because its catalog contract is
undirected. Separate paired families compare SPATHI against both SimiC order variants on
linear trajectories and against inferCSN with and without entropy-based pseudotime
filtering on compatible datasets. Prior-assisted tools remain excluded until a
separately fingerprinted non-leaky prior panel exists.

An order-permuted SimiC control is deliberately not synthesized. SimiC receives the
simulator-owned `column_phenotypes.tsv`, and the current ANDREA contracts expose no order
override or derived-dataset identity. SPATHI-benchmarks also has no provenance-preserving
dataset-derivation stage. Altering the file in place would reuse the wrong dataset
fingerprint, so this negative control stays blocked until one of those contracts exists.

The planned `reduced` and `full` source templates themselves remain non-runnable by
design. The high-level workflow creates a content-bound `*-v1` profile only after it has
validated the calibration selection.

## Low-level commands (advanced)

The public entrypoints above are the normal interface. For contract debugging, a single
committed profile can still be run by explicit stage, and the materializers remain
available directly:

```bash
scripts/run_all.sh smoke smoke-001
scripts/run_all.sh smoke smoke-001 --from infer --until summarize
.venv/bin/python -m spathi_bench.cli calibration-plan --help
.venv/bin/python -m spathi_bench.cli calibration-rank --help
.venv/bin/python -m spathi_bench.cli calibration-freeze --help
.venv/bin/python -m spathi_bench.cli benchmark-plan --help
```

The low-level stages are `preflight`, `image`, `generate`, `infer`, `validate`,
`integrate`, `evaluate`, `export`, `summarize` and `statistics`. They are idempotent, but
manual callers are responsible for passing the correct adaptive parent rankings and
profile files; the high-level workflows enforce those links automatically.

## Output layout

```text
work/v1/
├── artifacts/                   immutable shared objects, including joint views and ZIPs
├── workflows/
│   ├── calibration/<budget>/<flow-id>/
│   │                           adaptive requests, rankings and selected-config.json
│   └── comparison/<profile>/<flow-id>/
│                               selected benchmark profile and workflow state
└── <profile>/<run-id>/
    ├── input_snapshot/          frozen source and adaptive requests, schemas and locks
    ├── preflight/
    ├── provenance/
    ├── logs/
    ├── handoff-validation.json
    ├── state.json
    ├── exports/
    │   ├── manifest.json
    │   ├── datasets/<dataset>/analysis.zip
    │   ├── inference/<dataset>/<regime>/
    │   │   ├── individual/<run-id>/analysis.zip
    │   │   └── joint/analysis.zip
    │   └── evaluation/<dataset>/<regime>/
    │       ├── individual/<run-id>/analysis.zip
    │       └── joint/analysis.zip
    └── reports/
        ├── metrics.csv          raw ANDREA metrics plus benchmark provenance
        ├── summary.json         partitioned aggregates and calibration-ready rows
        └── statistics/
            ├── statistics.json methods, blocks, exclusions, tests and effects
            ├── rankings.csv     mean score and mean rank per configuration
            ├── pairwise.csv     Wilcoxon contrasts and Holm-adjusted p-values
            └── report.md        compact human-readable conclusions
```

Every ZIP has an adjacent `analysis.zip.manifest.json` with archive and member hashes.
The benchmark reads canonical manifests rather than discovering timestamped outputs.
ANDREA owns the individual `run_report.json` files and merged networks;
SPATHI-benchmarks owns only the explicitly identified joint projection and validates it
against ANDREA before publication.

See [`experiments/v1/design.md`](experiments/v1/design.md) for the scientific design,
information regimes, simulator panel and leakage controls.
