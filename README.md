# SPATHI article benchmark

This repository defines and runs the reproducible synthetic study used to select a SPATHI
configuration, compare the selected method with other network-inference tools, and measure
computational scalability. SPATHI contains the method itself; ANDREA provides the simulator,
inference, evaluation, and resource-outcome contracts; this repository owns the experimental
design, orchestration, statistical analysis, and portable reports.

There is one workflow and exactly three scientific phases:

1. **Calibration** selects one SPATHI configuration without privileging a reference candidate.
2. **Comparison** measures held-out accuracy inside formally defined competitor families.
3. **Scalability** measures runtime, memory, parallel scaling, and the feasibility frontier.

The single source of truth is
[`experiments/article-study.json`](experiments/article-study.json). All generated datasets are
synthetic. Application datasets, annotations, identifiers, and derived results do not belong in
this repository.

## Installation

Keep the three checkouts next to one another:

```text
workspace/
├── ANDREA/
├── SPATHI/
└── SPATHI-benchmarks/
```

Then install the editable projects into one environment:

```bash
cd SPATHI-benchmarks
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install \
  -c constraints/benchmark.txt \
  -e ../ANDREA \
  -e ../SPATHI \
  -e '.[dev]'
.venv/bin/spathi-bench doctor
```

`doctor` is read-only. It checks Python, Docker, disk space, both sibling checkouts, the article
design, the imports needed by the workflow, and every external route against the current ANDREA
ToolSpec catalog. Catalog drift in delivery mode, declared inputs or parameters, seed parameter,
thread cap, or directedness fails before a study starts. The command does not install, build, or
execute anything. Use `doctor --json` for machine-readable output.

## One command to run, one ID to resume

Start a study with a stable identifier:

```bash
.venv/bin/spathi-bench run --profile smoke   --id smoke-1
.venv/bin/spathi-bench run --profile reduced --id reduced-1
.venv/bin/spathi-bench run --profile full    --id article-full-1
```

Resume the same study after interruption with:

```bash
.venv/bin/spathi-bench run --id reduced-1
```

The study ID identifies its immutable scientific design, frozen machine-resource budget, SQLite
journal, and content-addressed artifacts. Completed tasks are reused only when their scientific
request, dependency evidence, and executor contract are unchanged. A Git revision change is
recorded in provenance and reported as a warning; it does not by itself discard valid results.
If a planned scientific request or explicit executor contract changes, that task and its
dependants are invalidated and recomputed. Any incompatible implementation change must therefore
bump the explicit executor/workflow contract in the same commit; those contracts are propagated
through every task and artifact identity. A source-only revision change remains resumable by
design, is retained per attempt, and must be reviewed as mixed-revision provenance before
publication.
Each planned task is also bound to the SHA-256 identity and declared Docker image of every ANDREA
ToolSpec or SimulatorSpec it consumes. Changing a catalog definition while retaining the same
image tag therefore invalidates exactly the affected task branch. The executor checks the binding
again when the task starts, so an edit made after planning cannot enter an attempt unnoticed.
This makes compatible orchestration or reporting fixes resumable across commits without silently
mixing incompatible evidence. A change that adds, removes, or rewires tasks changes the frozen DAG
fingerprint and requires a new study ID.

Useful read-only commands are:

```bash
# Inspect the deterministic DAG and detected capacity without running it.
.venv/bin/spathi-bench plan --profile reduced
.venv/bin/spathi-bench plan --profile reduced --json

# List all studies or inspect one study.
.venv/bin/spathi-bench status
.venv/bin/spathi-bench status --id reduced-1
.venv/bin/spathi-bench status --id reduced-1 --json

# Rebuild every portable report from journalled evidence; no inference is repeated.
.venv/bin/spathi-bench report --id reduced-1
```

The workflow detects usable CPU affinity/cgroup quota and available/cgroup-limited RAM. Optional
ceilings make a run coexist with other jobs:

```bash
.venv/bin/spathi-bench run \
  --profile reduced \
  --id shared-reduced \
  --max-cpus 24 \
  --max-memory-gb 128
```

Parallel-safe calibration and comparison tasks share the declared capacity. Formal scalability
measurements run in isolation so wall time and peak memory remain attributable. Each tool receives
its declared thread budget; the scheduler never oversubscribes the frozen study capacity. When a
comparison matrix lies beyond every measured ANDREA cost-profile size, the small profiling
container limit is not extrapolated as a memory model. Instead, ANDREA assigns one conservative
machine-relative memory slot per standard eight-thread worker and lets its wave planner reduce
concurrency to remain within the frozen host budget. Explicit per-run memory requests remain
authoritative.
External simulator generation uses the largest CPU/RAM tier measured by ANDREA that fits the host
(RAM: 8, 16, or 32 GiB; parallel simulator CPUs: 1, 2, 4, or 8). Hosts with less than 8 GiB of
available benchmark memory are rejected before launching a simulator. The chunked in-repository
scalability generator uses a separate memmap-aware estimate. Generation therefore participates
honestly in the rolling admission policy instead of being treated as negligible work.

## Profiles

The three profiles use the same code paths and scientific contracts. They differ only in coverage
and replication.

| Profile | Calibration | Held-out comparison | Scalability | Intended use and qualitative duration |
| --- | --- | --- | --- | --- |
| `smoke` | C1 plus compact X0 transfer; 1 replicate and 1 algorithm seed per split; 35 fits | T1, 1 replicate | P-S; 1 ladder replicate; 1 and standard-thread points | Contract and installation check; minutes to a few hours, depending on image builds |
| `reduced` | C1–C7 plus X1 transfer; 1 replicate and 1 algorithm seed per split; 221 fits | 10 primary scenarios plus D1 deployment transfer; 11 instances | P-S through P-XL; 1 ladder replicate; P-M at 1, standard, and all threads; 1-hour task limit | Development-grade end-to-end evidence; normally hours to roughly a day on a capable many-core host |
| `full` | C1–C7 plus X1–X2 transfer; 2 moderate development/validation replicates, 3 validation seeds, and 1 transfer replicate/seed; 554 fits | 30 primary instances plus D1–D2 deployment transfer; 32 instances | P-S through P-XL; 3 ladder and 3 P-M thread-scaling replicates; 4-hour task limit | Publication study; multiple days, with a target envelope of about one week on a large server |

Durations are planning ranges, not promises. They include simulator and container startup costs and
depend on which external tools reach their time or memory boundaries. `plan` records the exact DAG
and capacity for the current machine before execution. Per-physical-task limits are 30 minutes in
smoke, 1 hour in reduced, and—only in full—2 hours for comparison and 4 hours for scalability.

## Phase 1 — calibration

Calibration tests a finite, predeclared 27-candidate design. Every candidate has the same status,
development data, seed budget, and advancement rule; there is no reference configuration and no
candidate receives protected treatment.

| Condition | Simulator and property | Cells | Genes | TFs | Groups |
| --- | --- | ---: | ---: | ---: | ---: |
| C1 | dyngen, linear and clean | 300 | 100 | 10 | 4 |
| C2 | dyngen, branching, elevated kinetic noise | 600 | 400 | 40 | 7 |
| C3 | scMultiSim, dynamic and balanced | 300 | 100 | 10 | 3 |
| C4 | scMultiSim, dynamic and imbalanced | 800 | 400 | 40 | 3 |
| C5 | scMultiSim, 50% regulator fraction | 600 | 600 | 300 | 3 |
| C6 | scMultiSim, dynamic, strong group skew (20/180/600 cells) | 800 | 400 | 40 | 3 |
| C7 | scMultiSim, dynamic, rare-group stress (2/98/700 cells) | 800 | 400 | 40 | 3 |

These are prospective design dimensions. scMultiSim, SERGIO, BoolODE and the benchmark-owned
scale generator must reproduce their declared dimensions exactly. dyngen may add transcription
factors required by its backbone, and its declared group count denotes backbone milestones rather
than guaranteed dominant cell assignments. For dyngen, cells remain exact; TF/gene expansion is
validated against the requested target and housekeeping counts, and the observed group count must
contain at least two occupied milestones without exceeding the declared backbone milestones. Every
report shows the observed dimensions and any accepted requested-to-observed deviation.

All calibration conditions contain multiple biological groups. Single-group data require a
different SPATHI weighting contract, so they cannot fairly select the policy later frozen for the
multi-group comparison and are deliberately outside this study. C5 keeps half of all genes as
candidate regulators while bounding the 27-way screening cost; it tests regulator-density stress,
not the large-matrix boundary reserved for scalability.

The 27 candidates form a deterministic, balanced pairwise covering array. Every declared value
appears, marginal level counts differ by at most one, and every two-parameter value combination
appears at least once. This supports main-effect and two-factor screening without pretending to be
the astronomically larger full factorial. It covers:

- weighting mode, PCA or expression distance space, 20/50/100 components, and optional distance
  standardisation;
- cosine or Euclidean distance, Gaussian or exponential kernel, bandwidth scales 0.5/1/2, and
  optional group-size correction;
- Extra-Trees or random forest, `max_features` in `sqrt`/0.25/0.5/1.0, leaf sizes 1/2/5,
  maximum depths 2/4/8/unlimited, and minimum weighted leaf fractions 0/0.05/0.1/0.2.

The remaining model fields are declared, not silently ignored. `bandwidth=auto` defines the
data-adaptive bandwidth and `bandwidth_scale` is the calibrated control; `pca_svd_solver=auto`
selects the numerically appropriate solver; and `bootstrap=null` preserves the canonical choice of
the selected tree family. Threads, backend, seed and report generation are operational controls.
No target list is supplied because the study models the usual unknown-target setting, and no
centroid-confidence weights are used because the synthetic generators do not emit an independent,
biologically meaningful annotation-confidence measurement.

All 27 candidates first receive 50 trees on development data. The same declared primary metric and
tie-breakers advance four candidates. Those four are fitted once to 250 trees on independent
moderate validation data and scored at exact 25/50/100/250-tree prefixes. They then receive the
same treatment on a separate high-dimensional transfer panel:

| Transfer condition | Purpose | Cells | Genes | TFs | Group cells |
| --- | --- | ---: | ---: | ---: | --- |
| X0 | Compact smoke-path check | 300 | 400 | 40 | 3 / 27 / 270 |
| X1 | High-dimensional transfer | 2,000 | 5,000 | 500 | 20 / 180 / 1,800 |
| X2 | Near-real-scale, rare-group transfer | 5,000 | 15,000 | 1,000 | 10 / 490 / 4,500 |

X0 is used only by `smoke`, X1 by `reduced` and `full`, and X2 only by `full`. These are generic,
deterministically seeded synthetic scale-stress models with known dynamic truth. They contain no
application-cohort data, identifiers, annotations, or derived statistics.

The winner is chosen from a predeclared score that gives equal weight to the moderate-validation
panel and the near-real transfer panel; conditions are averaged before panel weighting so extra
replicates cannot silently change the estimand. Directed AUPR defines a 2% practical-equivalence
band, followed by between-condition stability and declared tree count. All four prefixes come from
one shared 250-tree physical fit per candidate, dataset and seed. Stability uses the same panel
weights as accuracy, so the larger moderate panel cannot dominate that tie-breaker either. The
workflow therefore records each shared forest cost
once and does not invent a per-prefix runtime. This is a deterministic selection rule, not a
statistical non-inferiority test. Operational settings such as thread count and random seed are
controlled by the study rather than tuned.

The calibration report shows candidate score distributions, the accuracy–cost plane,
dataset-level consistency, moderate exact-prefix convergence, transfer-panel convergence, the
panel-balanced final ranking, every parameter vector, the recorded decisions, and exceptions.

## Phase 2 — held-out comparison

The selected SPATHI policy is frozen before any comparison dataset is evaluated.

| Dataset | Panel | Simulator and role | Cells | Genes | TFs | Group cells |
| --- | --- | --- | ---: | ---: | ---: | --- |
| T1 | Primary accuracy | dyngen linear, clean | 300 | 100 | 10 | 4 occupied milestones |
| T2 | Primary accuracy | dyngen linear, noisy | 600 | 400 | 40 | 4 occupied milestones |
| T3 | Primary accuracy | dyngen branching, clean | 600 | 400 | 40 | 7 occupied milestones |
| T4 | Primary accuracy | dyngen branching, elevated kinetic noise | 800 | 400 | 40 | 7 occupied milestones |
| T5 | Primary accuracy | scMultiSim dynamic, balanced | 1,000 | 1,000 | 100 | 3 balanced groups |
| T6 | Primary accuracy | scMultiSim dynamic, larger | 3,000 | 3,000 | 500 | 3 balanced groups |
| T7 | Primary accuracy | scMultiSim dynamic, strong skew | 1,000 | 1,000 | 100 | 25 / 225 / 750 |
| T8 | Primary accuracy | scMultiSim dynamic, rare-group stress | 1,000 | 1,000 | 100 | 5 / 95 / 900 |
| I1 | Primary accuracy | SERGIO invariant-network control | 360 | 100 | 10 | 9 groups |
| I2 | Primary accuracy | BoolODE invariant-network control | 300 | 19 | 18 | 3 groups |
| D1 | Deployment transfer | scMultiSim dynamic, high-dimensional and imbalanced | 2,000 | 5,000 | 500 | 20 / 180 / 1,800 |
| D2 | Deployment transfer | generic dynamic scale stress, near-real dimensions | 5,000 | 15,000 | 1,000 | 10 / 490 / 4,500 |

Competitors use the defaults of their corresponding ANDREA catalog integrations. Only seeds,
threads, delivery mode, and explicitly stated no-prior/information-contract switches are supplied;
no competitor is tuned on study outcomes.

| Family | Methods | Why it is separate |
| --- | --- | --- |
| A · information-matched directed TF-to-target | SPATHI, GENIE3, GRNBoost2, inferCSN, TIGRESS, scMTNI independent/no-prior | All receive expression, groups, and the same TF list, without order, lineage, or prior. This is the primary Friedman plus all-pairs Wilcoxon–Holm family. |
| B · ordered-state directed | SPATHI, SimiC | SimiC receives valid state order that SPATHI ignores; only datasets with an explicit ordered-state contract enter this secondary family. |
| C · lineage-aware directed | SPATHI, scMTNI joint-lineage/no-prior | scMTNI receives valid simulator lineage that SPATHI ignores; only datasets with a trustworthy lineage contract enter this separate family. |
| D · unrestricted-regulator fitting | SPATHI, MetaSEM, kScReNI | MetaSEM and kScReNI fit over all genes whereas SPATHI uses the TF list. All are scored on the same TF-to-target candidate space, but the unequal fitting search space keeps this secondary analysis separate. |
| E · undirected topology | direction-collapsed SPATHI, PIDC, CLR, PPCOR, ARACNe3, CeSpGRN, LIONESS | Direction is unavailable for every method, so all outputs are direction-collapsed and scored on the same TF-to-target topology candidate space in a separate family. |

Eligibility is derived from each dataset's actual input capabilities. Statistical tests use only
complete paired blocks inside one family. No method from an information-richer or directionally
different family is pooled into the primary ranking. The report offers family-specific interactive
accuracy distributions, paired-score heatmaps, execution coverage, complete audit tables, and all
failures or exclusions.

The publication estimand is declared prospectively as an equal mixture of the ten primary
scenarios T1–T8, I1 and I2. `full` therefore contributes exactly three independently seeded
datasets per primary scenario; every seeded dataset is one paired block, so cheap scenarios receive
no extra statistical weight. Within each family and metric, only primary blocks completed by every
eligible member enter Friedman and paired Wilcoxon–Holm inference. T7 and T8 make robustness to
strong and extreme population imbalance part of that formal comparison instead of an anecdotal
appendix.

D1 and D2 answer a different question: whether each contract-eligible method remains accurate and
computationally viable as the synthetic problem approaches deployment dimensions. Every eligible
method is submitted under the same profile-level resource budget. ANDREA's empirical infeasibility,
timeout, memory-limit and failure outcomes are retained, not silently discarded. These datasets are
reported descriptively because forcing incompletely observed large-scale blocks into Friedman–Holm
would change the estimand toward whichever methods happen to finish. D1 is present in `reduced` and
`full`; D2 is publication-only. The report separates this transfer evidence from the primary panel
and also exposes per-group accuracy against observed prevalence, including minority, intermediate
and majority groups. Runtime and memory comparisons remain isolated in Phase 3, where their
measurement contract is controlled.

The scMultiSim datasets T5–T8 and D1 request both the simulator-derived ordered phenotypes and its
lineage tree. SimiC and joint-lineage scMTNI therefore participate in the imbalanced and
high-dimensional scenarios inside their respective information-richer families. D2 exposes only
groups and a TF list, so those two routes are correctly ineligible there rather than being supplied
with an invented order or lineage; independent/no-prior scMTNI remains eligible.

## Phase 3 — scalability

Every contract-eligible method starts on the same ordered synthetic ladder:

| Scale | Cells | Genes | TFs | Groups |
| --- | ---: | ---: | ---: | ---: |
| P-S | 300 | 100 | 10 | 3 |
| P-M | 1,000 | 1,000 | 100 | 3 |
| P-L | 3,000 | 5,000 | 500 | 3 |
| P-XL | 10,000 | 20,000 | 1,500 | 3 |

P-XL is a public synthetic engineering workload with real-scale dimensions; it does not encode or
approximate any private cohort. Methods advance breadth-first by scale. ANDREA may classify a run
as empirically time-infeasible before launch when a valid cost profile supports that conclusion;
otherwise it is attempted under the profile's hard physical timeout. Completion with warnings,
timeout, memory limit, and runtime failure are preserved as results rather than hidden exclusions.
Once ANDREA reports one of those non-completion outcomes for a method, its remaining larger-scale
measurements are recorded as `not_attempted`, with the originating ANDREA status and reason; they
are never relabelled as additional timeouts. In `full`, later replicates are likewise targets only
while that method keeps advancing through the ordered scout chain. This bounds predictable cost
without hard-coded "fast" and "slow" tool panels.

The same P-M realization is reused for the 1-thread, standard-thread (`min(8, available)`), and
all-available-thread measurements. Size-ladder and thread-scaling timings are isolated. The report
shows log-scale runtime, peak memory, parallel scaling, and a method-by-scale feasibility frontier.

## Seeds and reproducibility

Generation and algorithm seeds are derived from the committed master seed plus semantic
coordinates (phase, dataset, split, replicate, method, and algorithm replicate). Scheduling order,
worker count, and resume timing therefore do not change them. Reduced and full replication is
declared in the design rather than inferred from files on disk.

Each study records:

- the complete design fingerprint and frozen resource plan;
- scientific source identities and every observed repository revision;
- task requests, dependency digests, attempts, results, and failures, with every attempt linked to
  the exact source revision active when it started;
- the effective ToolSpec/SimulatorSpec hash and immutable container identity observed by every
  attempt; distinct definitions sharing one image are retained separately rather than collapsed;
- ANDREA plans, run reports, evaluation evidence, and resource outcomes;
- hashes for published artifacts and report deliverables.

## Portable reports

Successful execution, or `spathi-bench report`, writes one self-contained directory:

```text
deliverables/<study-id>/
├── article-report.html
├── calibration.html
├── comparison.html
├── scalability.html
├── results-calibration.csv
├── results-comparison.csv
├── results-scalability.csv
├── results-outcomes.csv
├── provenance.json
└── SHA256SUMS
```

Every HTML file embeds its CSS, JavaScript, Plotly runtime, data, and provenance. It can be copied
or opened independently without the work directory, a web server, network access, or relative
assets. The combined report and each phase report are disposable views of journalled evidence;
regenerating them never changes a scientific result.

Runtime state and large artifacts live under `work/studies/<study-id>/`; portable deliverables live
under `deliverables/<study-id>/`. Both locations are ignored by Git.

## Development checks

```bash
.venv/bin/pytest
.venv/bin/ruff check src tests containers/spathi
.venv/bin/python -m pip wheel . --no-deps -w /tmp/spathi-bench-wheel
```

The wheel includes the report template and front-end assets used to build standalone HTML files.
It is a development and packaging check, not a standalone distribution of the study: the canonical
design and Docker adapter remain repository-owned inputs. Run the benchmark from this checkout (or
point an installed CLI to it with `--project`) so those inputs and both sibling repositories are
available.
