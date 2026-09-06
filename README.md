# SPATHI benchmarks

Reproducible experiments for calibrating [SPATHI](https://github.com/AdrianSeguraOrtiz/SPATHI),
justifying its computational policies, comparing its frozen production configuration with
gene-regulatory-network inference tools orchestrated by
[ANDREA](https://github.com/AdrianSeguraOrtiz/ANDREA), and testing it across a prospectively
declared public synthetic scale grid.

This is an independent repository. SPATHI owns the method, ANDREA owns the standardized
simulation, inference and evaluation contracts, and SPATHI-benchmarks owns the experimental
design, phase separation, seed ledger, provenance, reusable artifacts and study-level reports.

## Scientific workflow

The publication workflow is deliberately sequential:

| Phase | Protected data | Decision or evidence produced |
| --- | --- | --- |
| Calibration | `development`, then `validation` | one frozen exact SPATHI group policy |
| Efficiency justification | `efficiency_validation`; conditional `efficiency_safety` | one frozen production policy |
| Dimensional transfer | `dimensional_transfer` | reduced operational checks or full post-freeze descriptive one-axis scale contrasts |
| Held-out comparison | `test` | confirmatory dynamic-network accuracy and exploratory analyses |
| Scalability | public synthetic scale grid | completion, attributable failures and resource evidence across declared engineering stress levels |

The five natural commands are:

```bash
scripts/calibrate.sh
scripts/justify-efficiency.sh
scripts/dimensional-transfer.sh
scripts/compare.sh
scripts/scalability.sh
```

They use the `full` budget by default. `full` is the evidence budget intended for publication;
`reduced` is a seed-disjoint descriptive budget for development and end-to-end verification:

```bash
scripts/calibrate.sh --profile reduced
scripts/justify-efficiency.sh --profile reduced
scripts/dimensional-transfer.sh --profile reduced
scripts/compare.sh --profile reduced
scripts/scalability.sh --profile reduced
```

Do not mix budgets. A reduced calibration can feed only the reduced efficiency study, and a
reduced production policy can feed only reduced dimensional-transfer, comparison and scalability
workflows. Reduced results must not be presented as substitutes for the full analyses.

All high-level workflows are content-bound and resumable. Repeating the same command verifies and
reuses completed work. If several compatible upstream workflows exist, select one explicitly with
`--calibration-run-id` or `--efficiency-run-id` as indicated by the command's help.

## Setup

The expected checkout layout is:

```text
/path/to/workspace/
├── ANDREA/
├── SPATHI/
└── SPATHI-benchmarks/
```

Create one environment for the benchmark CLI and the locked ANDREA checkout:

```bash
cd SPATHI-benchmarks
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -c constraints/benchmark.txt -e ../ANDREA -e '.[dev]'
```

No activation or shell export is required. The scripts use `.venv/bin/python` when present and
discover the sibling repositories automatically. Nonstandard layouts may set `ANDREA_REPO`,
`SPATHI_REPO`, `WORK_ROOT`, `BENCH_PYTHON` or `ANDREA_PYTHON`. `WORK_ROOT`, when set, must still
resolve below this repository's ignored `work/` tree so runtime material cannot escape the
provenance boundary.

Run the optional diagnostic before a long experiment:

```bash
scripts/doctor.sh
```

`doctor.sh` does not install, build or execute an experiment. It checks Python, the ANDREA CLI,
the SPATHI public API, Docker and Buildx, repository cleanliness, locked revisions, image
definitions and all committed design contracts. Normal workflow preflight repeats the mandatory
gates.

The benchmark, ANDREA and SPATHI repositories must be clean and at the revisions recorded in
`repos.lock.json`. Container inputs and catalog images are fixed by `images.lock.json`.

## What each command does

### `calibrate.sh`

```bash
scripts/calibrate.sh [--profile full|reduced] [--run-id ID] [--report-only]
```

Calibration executes six bounded stages: `geometry`, `estimator`, `interaction`,
`estimator_scale_transfer`, `convergence` and `validation`. The scale-transfer gate crosses the six
declared `max_features` rules over at most three interaction finalists on an independent 1,000-TF,
2,200-gene development subpanel; it does not add an unchanged inherited candidate and advances at
most two configurations. The other development stages use the ordinary development panel, and
validation chooses one finalist on an independent validation panel. The objective is directed
AUPR, with deterministic tie-breakers for stability, simplicity and canonical configuration ID.
Runtime and memory remain reported evidence but do not select a scientific configuration;
efficiency is justified later on its dedicated panel.

Target eligibility remains `all` and adaptive trees remain disabled during calibration. These
cost-saving policies are not silently selected as ordinary hyperparameters. The result is
`selected-config.json`: an exact scientific configuration, not yet the configuration authorized
for the external-tool comparison.

`bandwidth=auto` also remains fixed: it adapts to each dataset's observed distance scale, whereas
an absolute width calibrated on a small synthetic panel would not transfer meaningfully across
heterogeneous expression geometries.

The completed workflow and its interactive report are written below:

```text
work/v1/workflows/calibration/<budget>/<flow-id>/
work/v1/reports/<budget>/<flow-id>/calibration.html
```

`--report-only` rebuilds presentation from one completed workflow without executing scientific
stages.

### `justify-efficiency.sh`

```bash
scripts/justify-efficiency.sh [--profile full|reduced] [--calibration-run-id ID]
```

This phase uses data that are disjoint from calibration and test. It compares the calibrated exact
arm with three predeclared policies: automatic target eligibility, adaptive tree stopping, and
their combination. A policy is adoptable only if it passes accuracy non-inferiority, target and
truth-edge coverage where applicable, complete execution, and a strict runtime-benefit gate.
Development may nominate fewer than three policies. If it nominates none, the held-out step is a
reference-only execution audit: it computes no empty contrast or multiplicity adjustment and
freezes exact SPATHI for every downstream phase.

Only the full budget calculates one-sided confidence bounds and gate decisions. Its selected
passing policy is first provisional: before it can be frozen, it must also pass the separate
pre-freeze efficiency-safety veto described below. If no formal candidate passes, or the sole
provisional candidate fails safety, the exact arm is frozen in `production-config.json`.
The reduced budget reports point estimates and observed completeness only: it emits no confidence
interval, inferential pass/fail field or adoption claim, and freezes the exact arm. That file is
the only SPATHI configuration accepted by downstream dimensional-transfer, comparison and
scalability workflows.

Candidate selection is deliberately conservative. Any formally bound execution failure in a
candidate arm invalidates the global relative selection and retains the exact policy; the workflow
does not silently choose among the surviving candidates. A failure in the exact/reference arm
makes the comparison uninterpretable and aborts the decision instead of being recorded as evidence
for a candidate.

The runner is serial and resumes only a valid prefix of its frozen execution order. After a crash,
the next invocation verifies and preserves that prefix and executes only the missing suffix; it
does not delete valid cache entries. A non-prefix or identity-inconsistent state is rejected as
corrupt rather than silently normalized. The final evidence, decision and report are published
atomically before downstream phases may consume them.

The absolute AUPR margin (`0.002`) and coverage thresholds (`0.50` of targets and `0.90` of truth
edges) are frozen, author-defined engineering tolerances, not empirically validated biological
equivalence thresholds. Full conclusions are conditional on the declared synthetic panel and
fixed algorithm-seed ensemble. Runtime is analysed as a paired log ratio and reported as a
hierarchically weighted geometric-mean ratio. Coverage must satisfy the aggregate confidence-bound
gate and the frozen observed guardrails for every dataset and every simulator-condition stratum.

`experiments/v1/spathi/efficiency-design-sensitivity.json` prospectively records deterministic
normal-location scenario calculations for the full design. They expose how conclusions depend on
assumed nuisance variation; they are not an empirical power guarantee and do not validate
small-sample bootstrap coverage.

Only when the formal full analysis nominates a candidate, the workflow compares that candidate
with exact SPATHI on 18 new scMultiSim datasets: three balanced 3,000-gene, 100-TF, 300-cell
datasets; three 500-cell single-group controls; three 300-cell controls with group sizes
10/40/250; three balanced 10,000-cell datasets; three 3,000-gene, 1,500-TF datasets; and three
15,000-gene, 100-TF datasets. Their generation seeds
and four distinct regulatory topologies are protected from every other phase. The safety
topologies share no directed edge with one another or with the dimensional-transfer topologies.
All pairs use one common algorithm seed and a separately derived counterbalanced execution order.

This safety step is a conservative descriptive veto, not another search or statistical claim. It
uses no p-values or confidence intervals and never tries a fallback candidate. Authorization
requires complete execution, the frozen per-dataset coverage safeguards, global and per-condition
mean and median AUPR loss no worse than `0.002`, no individual loss worse than `0.01`, and a global
median wall-time ratio below `1.0`. A candidate failure retains exact; an exact/reference failure
aborts because there is no interpretable paired contrast.

Outputs are written below:

```text
work/v1/workflows/efficiency/<budget>/<flow-id>/
├── plan.json
├── evidence.json
├── results/
├── efficiency-safety/       # full only, and only when one candidate is provisional
├── production-config.json
└── workflow.json

work/v1/reports/<budget>/<flow-id>/efficiency.html
work/v1/reports/<budget>/<flow-id>/efficiency-report.manifest.json
```

### `dimensional-transfer.sh`

```bash
scripts/dimensional-transfer.sh [--profile full|reduced] [--efficiency-run-id ID]
```

This post-freeze phase runs only the same-budget production policy on truth-bearing scMultiSim
data. The reduced budget is deliberately cheap: one dataset for each of two moderate conditions
(`g3000_tf100_c300` and `g100_tf10_c300`) plus the single-group and extreme-imbalance routing
controls. It omits the 15,000-gene, 1,500-TF and 10,000-cell endpoints and therefore emits
`contrasts=[]`; it is an operational check, not scale-transfer evidence.

The full budget varies genes, TFs and cells one axis at a time and provides three descriptive
one-axis scale contrasts. The TF contrast also changes the regulatory topology and which TFs
regulate targets, so it does not identify a pure causal effect of candidate-TF count. Neither
budget reselects the policy, enters the confirmatory external-tool ranking or justifies biological
biological realism. If the production policy differs from the exact calibrated arm, the two
predeclared scale anchors receive an exact-policy diagnostic; that diagnostic does not turn the
high-dimensional endpoints into exact-versus-production comparisons.

Outputs are written below:

```text
work/v1/workflows/dimensional-transfer/<budget>/<flow-id>/
work/v1/reports/<budget>/<flow-id>/dimensional_transfer.html
work/v1/reports/<budget>/<flow-id>/dimensional-transfer-report.manifest.json
```

### `compare.sh`

```bash
scripts/compare.sh [--profile full|reduced] [--efficiency-run-id ID] [--report-only]
```

Comparison materializes the frozen production policy only after a compatible efficiency workflow
has completed. It runs the declared catalog baselines and SPATHI on held-out test datasets,
validates every ANDREA handoff, constructs joint per-dataset views, evaluates the networks, exports
ANDREA-compatible ZIP files, summarizes metrics and performs the predeclared paired non-parametric
analyses.

The full confirmatory family asks whether SPATHI improves directed AUPR on datasets with genuinely
group-specific regulatory truth when all methods receive expression, group labels and the same TF
candidate list. Invariant-network stability, topology and the order-aware SimiC result are kept in
separate exploratory families. pySCENIC's default GRNBoost2 step is compared with the direct
GRNBoost2 wrapper only as an implementation sensitivity, not as an independent rival. V1 has no
pseudotime-aware family: the integrated inferCSN CRAN core consumes no pseudotime and exposes only
the `none` and `max` sift modes.

Two additional exploratory all-gene families use only the held-out linear dyngen conditions. In
`full`, directed inference compares SPATHI with kSReCNI and MetaSEM, while topology also
includes CeSpGRN without a prior, CLR, LIONESS, PIDC and PPCOR. The reduced operational counterpart
keeps MetaSEM for directed analysis and CeSpGRN, CLR and MetaSEM for topology. Those competitors
infer with every gene as a possible regulator; ANDREA then projects all predictions onto the same
TF-restricted evaluation universe used for SPATHI. They therefore measure the consequence of an
all-gene inference scope and never enter the primary rank.

The reduced comparison is strictly descriptive. It validates exact panel completion and reports
paired scores and effects, but emits no Friedman or Wilcoxon p-value, no bootstrap confidence
interval and no statistical winner claim.

Outputs and the interactive report are written below:

```text
work/v1/workflows/comparison/<budget>/<flow-id>/
work/v1/<budget>-v1/<flow-id>/
work/v1/reports/<budget>/<flow-id>/comparison.html
```

`--report-only` changes only the presentation layer of one completed comparison.

### `scalability.sh`

```bash
scripts/scalability.sh [--profile full|reduced] [--efficiency-run-id ID]
```

Scalability runs the frozen production policy on deterministically generated expression matrices
whose cell, gene, TF and group dimensions span a rounded, author-defined public stress grid. It records
completion and computational cost without reopening calibration or efficiency selection.

Each dimensional shape runs separately so its resources remain attributable. The workflow records
generation, ANDREA, SPATHI and harness wall times; sampled peak RSS for the SPATHI container process
tree; expression, artifact and SPATHI-output sizes; requested, eligible and ineligible targets;
requested and completed models; and fitted trees. A completed measurement may be resumed and
verified, but fresh scale evidence cannot be reconstructed from a pre-existing inference cache hit.
Another container using the same SPATHI image is rejected while RSS is sampled, because its memory
could not be attributed unambiguously.

An attributable SPATHI execution failure is a valid scalability outcome, never a successful
completion. OOM, timeout, tool and distinguishable infrastructure failures are retained per
dataset with the exact resource limits, elapsed time, available RSS samples, last valid progress
snapshot and a content-verified command log. The panel still publishes its report with
`all_completed=false`, and continues with later shapes. This exception applies only when ANDREA
produced a failed report matching the exact requested dataset, run and tool origin; a missing,
ambiguous, tampered or identity-inconsistent report aborts publication as a contract error.

The scale-stress generator is computational rather than biological. Its dropout, latent structure,
noise and regulatory model are explicit public synthetic scenarios. Consequently, this phase supports statements about
feasibility and resource use only; it does not contribute to the confirmatory network-accuracy
claim.

The generator first creates deterministic non-negative counts-like values, then applies its
declared public preparation contract. For every cell, its library size is summed
across every generated gene and the emitted value is
`log1p(10000 * raw_value / per_cell_library_size)`. SPATHI therefore receives prepared expression,
not raw counts-like generator output. The raw working matrix is a unique disk-backed temporary,
processed in bounded gene blocks and removed on success or failure. The request, benchmark
manifest, per-dataset provenance and scalability results all record this exact transformation;
validation rejects any normalization or target-sum mismatch.

Every request contains its complete `scale_grid` with provenance
`author_defined_public_synthetic_grid`. Dataset roles distinguish declared scale cases from
synthetic controls. The full panel covers separate cell-, gene- and regulator-heavy cases, two
joint workloads and an invariant-network control. The reduced panel contains two moderate
development workloads. Per-dataset runtime materializations narrow the grid to the one shape they
execute while retaining the same public preparation and seed contracts.

Outputs are written below:

```text
work/v1/workflows/scalability/<budget>/<flow-id>/
├── materialized/<dataset>/
├── measurements/<dataset>.json
├── scalability-results.json
└── workflow.json

work/v1/scalability_<budget>_<dataset>/<flow-id>/
work/v1/reports/<budget>/<flow-id>/scalability.html
work/v1/reports/<budget>/<flow-id>/scalability-report.manifest.json
```

## Interactive reports

All five phases generate an English, self-contained HTML report automatically. Plotly, styles and
report data are embedded, so no server or network access is required. A provenance manifest beside
each HTML file separates frozen scientific inputs from renderer inputs and records their hashes.
Reports present existing evidence; they do not rerank candidates, recompute tests or change a
production decision.

Efficiency and scalability publish their report bundles transactionally. Their `workflow.json`
remains in `publishing` while the HTML and provenance manifest are being written and changes to
`complete` only after both canonical paths and SHA-256 hashes have been recorded under `reports`.
On resume, missing report members are rebuilt deterministically; an existing member whose bytes no
longer match the frozen bundle is rejected rather than silently overwritten.

The calibration report explains every bounded stage and its parameter values. The efficiency
report shows the complete exact and candidate policies, accuracy, coverage, completion and
resource endpoints. Full reports add non-inferiority intervals and gate decisions; reduced reports
label point estimates as descriptive and never render missing bounds as failures.
The comparison report shows declared analysis families, paired dataset scores, statistical tests,
runtimes and links to the ANDREA-compatible ZIP files.
The dimensional-transfer report shows the budget-specific conditions, routing controls and the
limited exact-policy anchor diagnostic. Full reports show three descriptive one-axis scale
contrasts; reduced reports explicitly show that no scale contrast was estimated.
The scalability report shows declared dimensional coverage, per-dataset completion, target and
model accounting, timings, memory, fitted trees, artifact sizes and attributable failures without
presenting those data as biological-accuracy evidence.

## Evidence budgets and panels

The reduced budgets are seed-disjoint operational subsets used to exercise the workflow at
tolerable cost; they do not necessarily preserve every full-budget scientific question or analysis
family. Full is the publication evidence budget and is inferential only where explicitly declared.
Reduced datasets are not prefixes or reused subsets of full datasets.

| Phase | `reduced` | `full` |
| --- | --- | --- |
| Ordinary calibration development stages | 5 datasets, 2 algorithmic seeds per stage | 15 datasets, 5 algorithmic seeds per stage |
| Calibration estimator scale-transfer | 1 dataset, 2 algorithmic seeds | 3 datasets, 3 algorithmic seeds |
| Calibration validation | 5 datasets, 2 algorithmic seeds | 15 datasets, 5 algorithmic seeds |
| Efficiency validation | 5 datasets, 2 algorithmic seeds | 25 datasets (5 per stratum), 5 algorithmic seeds |
| Pre-freeze efficiency safety | not run | 18 datasets (3 in each of 6 conditions), 1 common paired algorithm seed |
| Dimensional transfer | 4 moderate/routing conditions, 1 dataset each; 1 algorithmic seed; no scale contrasts | 7 conditions; 3 datasets for the 3,000-gene anchor and 1 for each other condition; 1 algorithmic seed |
| Held-out comparison | 10 datasets; 3 algorithmic seeds per stochastic configuration and 1 per deterministic configuration | 50 datasets; 3 algorithmic seeds per stochastic configuration and 1 per deterministic configuration |
| Scalability | 2 dimensional stress cases | 6 dynamic scale cases plus 1 invariant control |

The committed v1 execution ceiling is eight logical CPU cores and 48 GB RAM across all concurrent
runs. Resource values passed to one ANDREA inference are per-run quotas, so the profile validator
also checks `max_parallel_runs × quota` against the global ceiling. Individual phases may impose a
lower ceiling. Every result records the observed host and concrete limits; changing them defines a
new computational study rather than silently mixing cost measurements.

Dyngen and scMultiSim provide dynamic group-specific truth for calibration and the main comparison.
SERGIO, BoolODE and GeneSPIDER2 provide invariant-network controls. The full held-out panel contains
30 dynamic datasets and 20 invariant controls. The reduced comparison contains one independent
dataset from each of the same ten conditions.

For the ordinary scMultiSim panels, role-exclusive topology means that the complete committed graph
input cannot be reused in another protected role. Individual directed edges may recur because all
roles deliberately use the same synthetic TF-target universe. This is not model leakage: no fitted
network or edge-level result crosses a partition, and only the globally selected SPATHI policy is
carried forward. A stricter explicit contract keeps the distinct efficiency-safety topologies
edge-disjoint from one another and keeps their union edge-disjoint from dimensional transfer.

The committed smoke profile is separate from both budgets. It checks the full ANDREA contract with
one tiny dyngen dataset, GENIE3 and SPATHI, but has no statistical power:

```bash
scripts/run_all.sh smoke smoke-001
```

## Reproducibility

`experiments/v1/seeds/seed-plan.json` is the single v1 seed ledger. A fixed master seed and
SHA-256 domain separation derive positive seeds from semantic coordinates rather than file order.
The ledger separates budget, dataset role and stochastic stream:

- `generation` for simulator instances;
- `augmentation` for deterministic synthetic derivations, including scale data;
- `inference` for algorithmic repetitions;
- `statistical` for permutations and bootstrap procedures; and
- `execution_order` for deterministic counterbalancing.

The `statistical` stream is allocated only to the full held-out comparison and the full
efficiency study. Smoke and reduced workflows have descriptive contracts, so they neither reserve
statistical seeds nor carry permutation, bootstrap, alpha or multiplicity parameters.

Every stochastic request records its seed and derivation coordinates. The committed design can be
audited independently of a run:

```bash
.venv/bin/spathi-bench audit-reproducibility \
  --repo-root "$PWD" \
  --output work/v1/reproducibility-audit.json
```

The audit verifies declared allocations, simulator manifests, inference metadata and scale-stress
requests. It also validates and hashes the prospective efficiency design-sensitivity contract and
its deterministic operating-characteristic table, then emits schema-validated evidence.

## Artifact reuse after a SPATHI change

Immutable artifacts live in `work/v1/artifacts/`. Their keys include every input that can affect
their content. Updating the pinned SPATHI commit requires a new calibration, efficiency decision,
SPATHI inference and downstream joint evaluation, because a policy selected for another
implementation cannot be promoted. It does not require rerunning unchanged simulators or catalog
tools: those artifacts remain cache hits when their own inputs, images, ANDREA revision and
environment are unchanged.

| Change | Simulated data | Catalog inference | SPATHI inference | Dependent joint outputs |
| --- | --- | --- | --- | --- |
| SPATHI source/config/image | reuse | reuse | recompute | recompute |
| One catalog image or request | reuse | recompute affected tool | reuse | recompute affected views |
| Simulator request or image | recompute affected data | recompute downstream | recompute downstream | recompute |
| ANDREA revision/environment | recompute | recompute | recompute | recompute |

The state files report cache hits and bind the concrete SPATHI image ID. Repeating an interrupted
workflow never discovers outputs by timestamps; it validates canonical manifests and content
hashes.

## ANDREA-compatible outputs

For ordinary inference/evaluation pipelines, exports live below the run root:

```text
exports/
├── manifest.json
├── datasets/<dataset>/analysis.zip
├── inference/<dataset>/<regime>/
│   ├── individual/<run-id>/analysis.zip
│   └── joint/analysis.zip
└── evaluation/<dataset>/<regime>/
    ├── individual/<run-id>/analysis.zip
    └── joint/analysis.zip
```

Individual inference ZIPs can be passed to ANDREA `evaluate-inference`. Joint inference and joint
evaluation ZIPs contain all methods for one dataset and regime and can be loaded together by
ANDREA `compare-networks`. SPATHI-benchmarks creates the joint projection but preserves ANDREA's
report and archive contracts. Every ZIP has an adjacent manifest containing archive and member
hashes.

## Statistical comparison

The independent unit is a simulated dataset. Group contexts are averaged within one execution;
algorithmic seeds are averaged within configuration and dataset; then each dataset contributes one
equally weighted paired block from the complete declared panel. Cells, groups, edges and
algorithmic seeds never inflate the replicate count.

For three or more methods, the analysis uses a tie-corrected Friedman omnibus test with Kendall's W
and exact or seeded within-block permutations. Prespecified SPATHI-versus-rival contrasts use paired
Wilcoxon signed-rank tests followed by Holm correction. Reports also include paired differences,
Hodges-Lehmann estimates, rank-biserial effects, win/tie/loss counts and bootstrap intervals that
resample replicates within fixed simulator-by-condition design strata. A
missing member invalidates the family instead of being removed by pairwise deletion; a numerically
evaluated empty network remains a valid zero-edge prediction. The high-level comparison workflow
cannot finish unless every predeclared confirmatory primary analysis has status `tested`.

Each comparison declares `expected_blocks`, the exact cardinality of its frozen dataset panel.
This is a completeness contract, not a power calculation or evidence that the sample size is
adequate for every effect. Any mismatch is rejected before analysis. Reduced comparisons are
descriptive only; the Friedman, Wilcoxon, Holm and bootstrap procedures run only for full analyses.
The smoke report is contract-only. Both smoke and reduced reports retain paired point estimates
for diagnostics but contain no inferential seed contract or winner claim.

See [`experiments/v1/design.md`](experiments/v1/design.md) for the estimands, partition rules,
efficiency gates, scale interpretation and leakage controls.
