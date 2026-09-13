# SPATHI benchmarks

Reproducible synthetic experiments for calibrating SPATHI, justifying computational shortcuts,
comparing the frozen method with ANDREA catalog tools, and measuring scale and runtime.
SPATHI implements the method; ANDREA provides simulation, inference and evaluation contracts;
this repository defines the experiment and owns its orchestration and reports.

## Install once

Keep the three checkouts together:

```text
workspace/
├── ANDREA/
├── SPATHI/
└── SPATHI-benchmarks/
```

From `SPATHI-benchmarks`:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -c constraints/benchmark.txt -e ../ANDREA -e ../SPATHI -e '.[dev]'
.venv/bin/spathi-bench doctor
```

The benchmark executable discovers this checkout and the sibling repositories. No environment
exports or shell wrappers are needed. All examples below use `.venv/bin/spathi-bench`; after
activating the environment, `spathi-bench` is equivalent. ANDREA runs in this environment and
inference tools run in Docker. Installing SPATHI here also supplies the dependencies needed for
local qualification on a fresh machine. Qualification uses a sibling project's `.venv` when one
exists; otherwise it uses this interpreter. The Docker SPATHI image is built from the exact source
revision pinned by the study's image contract and the committed adapter. Docker, permission to
access its daemon, and Buildx must be available.

`doctor` checks prerequisites, repository and image contracts, and committed experimental inputs.
It does not install software, build images or run experiments. Use `doctor --json` for complete
diagnostic details. `repos.lock.json`, `images.lock.json` and `constraints/benchmark.txt` describe
the intended software environment. Keep scientific inputs committed before starting a new study.

## Run, resume and inspect

```bash
.venv/bin/spathi-bench qualify
.venv/bin/spathi-bench run smoke --id first-smoke
.venv/bin/spathi-bench plan reduced
.venv/bin/spathi-bench run reduced --id pilot
```

`qualify` verifies local scientific contracts using real SPATHI output and ANDREA evaluation on
small known inputs. It covers diagnostic interpretation and independent metric agreement.
`smoke` exercises Docker simulation, both inference adapters, integration, evaluation and exports.
Neither has statistical power, and the simple smoke is not a physical execution of all six
scientific phases. New reduced/full studies run the local contract qualification before proceeding.

One ID identifies the entire study, including all phases. If omitted, a new ID is generated and
printed. A new `run` with no profile uses `reduced`. An existing ID resumes its saved profile:

```bash
.venv/bin/spathi-bench run --id pilot
.venv/bin/spathi-bench status
.venv/bin/spathi-bench status pilot
.venv/bin/spathi-bench report pilot
```

IDs have 1–96 characters: lowercase letters, digits, hyphens or underscores, starting with a
letter or digit. Examples: `pilot`, `reduced-sep12`, `article_full_01`.

`status` lists studies; `status ID` shows phase progress and failures. `report ID` rebuilds the
study entry page and recovers missing phase reports from saved evidence, without rerunning inference
or selecting another configuration. Published phase bundles are immutable: recovering them requires
the same renderer; this command does not overwrite earlier plots with a different rendering revision.
Add `--json` to `status` or `plan` for structured details.

The study journal carries upstream calibration and efficiency identities internally. There is no
need to find a calibration ID, delete other studies, or remember which phase failed. A completed
phase remains associated with its original software and inputs. Resuming after a commit change
emits and records a warning and allows continuation under the same study ID; it does not rewrite
the provenance of prior results. The experimental settings, seeds, dataset contracts,
`repos.lock.json`, `images.lock.json` and host dependency constraints remain frozen for that ID;
changing these normally requires a new study. The explicit pending-generation repair below is a
narrow exception; it cannot replace already generated data. Hash and artifact-integrity checks still apply. For a
publication run, freeze the software first or explicitly disclose any mixed-revision continuation.

### Correct a failed simulator request without discarding the study

Ordinary `run --id` never changes frozen scientific inputs. If a committed correction changes
simulator parameters **before that request has published any dataset**, stop the study and apply
an explicit, documented amendment:

```bash
.venv/bin/spathi-bench amend-inputs pilot --reason "Correct an unsupported simulator integration step"
.venv/bin/spathi-bench run --id pilot
```

With PM2, first `pm2 stop PROCESS_NAME`, run `amend-inputs`, and then `pm2 restart PROCESS_NAME`
if its saved command already runs this study ID. The amendment does not launch simulations or
inference: it checks completed phase receipts and frozen inputs, rejects changes to published
datasets, refreshes only affected simulation plans, and preserves the prior requests, manifests,
plans and journal in `work/studies/ID/amendments/`. Its reason and before/after checksums are
recorded in the study journal and its warning appears in the study report. If interrupted while
applying an amendment, repeat the same `amend-inputs` command before resuming.

Only parameters in existing simulator-run requests may change: seeds, run identities, replicate
counts, scenarios, selection settings, resource policies and image locks remain protected. An
accompanying update of the ANDREA commit in `repos.lock.json` and the catalog matrix is accepted
only when it matches the observed clean ANDREA checkout; previous frozen locks and cached
artifact identities are retained. This is an explicit mixed-revision correction, not permission
to retune a benchmark after observing its results. Amendments to a published dataset or a completed
generation phase require a new study instead.

A checkout advancing does not silently upgrade the study's SPATHI container. The image contract
still requests its exact locked source revision, even if the checkout HEAD is newer; the build
extracts that revision and records the concrete image identity. Local qualification checks the
observed local SPATHI source, so its revision is recorded separately. Update the declared image
contract and start a new study when the intention is to benchmark a changed scientific method.
The pinned source commit must still exist in the local Git checkout for an image rebuild. The
study index lists checkout revisions per invocation; each inference artifact's image identity
records the software actually executed in its container.

A new ID creates a separate study directory and its own artifact store. Reuse happens within that
study when its ID is resumed. To repeat the entire experiment from the beginning, choose a new ID;
there is no need to delete other studies. Deleting a study directory discards both its evidence
and its reusable computation. Keep it until the desired reports and analysis archives have been saved.
Older phase-oriented directories under `work/v1/` are not automatically adopted or migrated into
the named-study interface. Their existing reports remain available at their original paths;
the new workflow starts in `work/studies/<id>/` and does not interpret old phase IDs as study IDs.

The machine is detected automatically, including CPU affinity/cgroup limits, topology and RAM.
Optional ceilings are useful on shared hosts:

```bash
.venv/bin/spathi-bench run reduced --id shared-host --cores 16 --ram-gb 64
```

These ceilings are total study capacity, not a promise that every individual tool will use that
many threads. The resource plan gives each task its quota and records actual usage. Calibration
and accuracy-only repetitions can share the machine; designated timing measurements run alone.
`--project PATH`, `--andrea PATH` and `--spathi PATH` are optional overrides for unusual layouts.
Output stays in the project's ignored `work/` tree.

For PM2, give the process the same stable study ID so an ordinary restart resumes it:

```bash
pm2 start .venv/bin/spathi-bench --name spathi-pilot --interpreter none \
  --no-autorestart --time -- run reduced --id pilot
pm2 logs spathi-pilot --lines 100
```

After pulling a compatible orchestration fix into the editable checkout:

```bash
pm2 restart spathi-pilot
```

The saved command already contains the complete workflow and its ID. Do not keep a previous PM2
process pointing at a phase-specific command. A changed dependency constraint still requires the
installation command above to be rerun.

## The six scientific phases

| Phase | Purpose and frozen output | `reduced` | `full` |
| --- | --- | --- | --- |
| 1. Calibration | Select an exact conditional SPATHI policy by directed AUPR; write `selected-config.json` | 8 ordinary development + 1 high-TF development + 8 independent validation datasets; 431 physical fits / 527 logical results | 24 + 3 + 24 datasets; 1,762 physical fits / 2,482 logical results |
| 2. Efficiency | Develop target-eligibility/adaptive-tree candidates, independently compare one nominee with exact SPATHI, freeze `production-config.json` | 5 development datasets, 75 fits; 5 validation datasets × 2 algorithm seeds; descriptive estimates, exact policy retained | 15 development datasets, 285 fits; 25 validation datasets × 5 seeds; non-inferiority gates and conditional 18-dataset safety veto |
| 3. Dimensional transfer | Inspect frozen-policy accuracy across dimensions and routing controls, without tuning | 4 moderate/control datasets, 1 seed; no scale contrasts | 9 datasets across 7 conditions, 1 seed; 3 descriptive one-axis contrasts |
| 4. Held-out comparison | Compare frozen SPATHI with catalog defaults on independent test data | 10 datasets; descriptive paired scores and ranks | 60 datasets: 40 dynamic + 20 invariant controls; predeclared confirmatory and exploratory families |
| 5. Scalability | Measure SPATHI feasibility across a public synthetic stress grid | 2 shapes; 64-target scout before possible all-target fit | 7 shapes; 128-target scout before possible all-target fit |
| 6. Performance | Measure isolated software cost and parallel scaling | 4 methods × 1 dataset × 1/8 threads; 8 measured runs + 8 warm-ups | 6 methods × 2 datasets × 1/2/4/8 threads × 3 repetitions; 144 measured runs + 48 warm-ups |

`reduced` is for development and inspection: it has no inferential winner claim, bootstrap interval
or non-inferiority adoption decision. `full` is the publication budget, with inference only where
the committed design permits it. Reduced and full use independent generation and algorithm seeds;
one is not a prefix of the other's data. Do not combine their results into a single analysis.

The source of truth is [experiments/v1/design.md](experiments/v1/design.md), with executable
contracts in [experiments/v1/profiles](experiments/v1/profiles), the
[seed ledger](experiments/v1/seeds/seed-plan.json), and the phase-specific files linked below.
Times depend strongly on the selected forest, host, Docker images and simulator workloads.
`plan` describes scheduled work and available resource evidence; a reduced run completing within
one day remains an estimate until that exact environment and design have been measured end to end.

### Calibration: which SPATHI configurations compete?

The ordinary development and validation panels contain four dyngen conditions and four scMultiSim
conditions. Dyngen requests linear baseline (200 cells), linear noisy (500), bifurcating baseline
(300) and bifurcating noisy (400), with 10 requested TFs and respectively 50/90/70/80 target genes
and 10/30/20/10 housekeeping genes. Realized dimensions and group counts come from the simulator
manifest. scMultiSim supplies 80-gene/300-cell dynamic, imbalanced and single-group cases, plus a
400-gene/600-cell case. Its high-TF development panel has 2,200 genes, 1,000 TFs and 300 cells.

The [bounded search](experiments/v1/spathi/search-space.json) advances candidates through:

| Stage | Predeclared choices |
| --- | --- |
| Geometry | 16 initial configurations spanning PCA/full-expression distance; 20/50/100 PCA components; auto/full/randomized SVD; none/standard scaling; cosine/Euclidean distance; Gaussian/exponential kernel; bandwidth scale 0.5/1/2; cell-distance, group-distance and cell-distance-group-anchored weighting; none/cap-to-target group correction |
| Estimator | Up to 3 geometries × 12 forest choices: Extra-Trees/random forest; `max_features` = integer `1`, `sqrt`, `log2`, fraction `0.25`, `0.5`, `1.0`; leaf size 1/2/5; unlimited/20 depth; valid bootstrap choices |
| Interaction | Up to 3 finalists × 6 explicitly crossed forest/regularization settings |
| High-TF transfer | Up to 3 finalists × all 6 `max_features` rules on the independent 1,000-TF panel; outcome-blind 256-target scout, then all targets |
| Convergence | Up to 2 finalists × 250/500/750/1,000 trees, obtained as exact prefixes of one 1,000-tree fit per dataset and seed |
| Validation | Choose one of the two finalists on independent validation data |

This is a finite, constrained design, not an exhaustive Cartesian product and not a claim to test
every value of continuous parameters. Every public SPATHI parameter is accounted for in
[parameter-accountability.json](experiments/v1/spathi/parameter-accountability.json).
Integer `max_features=1` means one candidate per split; float `1.0` means all predictors.

The objective averages groups within a dataset, then algorithm seeds, then datasets. Ties use
stability, declared simplicity and canonical configuration ID. Runtime does not select calibration
parameters. Reduced uses one scout and one confirmation seed and two convergence/validation seeds;
full uses one scout, two confirmation seeds and five convergence/validation seeds.

Calibration keeps `target_eligibility=all`, `adaptive_trees=false` and `bandwidth=auto`.
Annotation-confidence weights and application-specific target lists are excluded. A single-group
dataset follows the frozen cell-distance/Euclidean/no-group-correction branch; all other learned
parameters are shared. No per-test-dataset tuning is allowed.

### Efficiency: which shortcuts are justified?

The [development grid](experiments/v1/spathi/efficiency-development.json) contains exact SPATHI
and six candidates:

| Component | Three settings |
| --- | --- |
| Automatic eligibility: detected cells / detected fraction / weighted detected fraction / weighted ESS | `10 / .005 / .005 / 5`; `20 / .01 / .01 / 10`; `50 / .02 / .02 / 20` |
| Adaptive trees: minimum / step / tolerance / patience | `200 / 25 / .005 / 3`; `100 / 50 / .01 / 2`; `100 / 100 / .02 / 2` |

Development uses independent instances of the four dyngen conditions and the 80-gene/300-cell
dynamic scMultiSim condition. It first nominates one setting per component, then confirms exact
SPATHI, both nominees and their combination. Only the designated timing seed supplies cost:
those runs have identical quotas and run alone in a deterministic counterbalanced order.
Held-back accuracy seeds run concurrently. Their incidental timings are excluded from selection.
The subsequent formal validation retains isolated runs for all its declared paired resource
observations. Runtime evidence therefore keeps its original estimand and resource context.

Validation uses new datasets from the same five strata. Full requires AUPR non-inferiority
(absolute margin 0.002), complete execution, runtime benefit, and relevant target/truth coverage
(0.50/0.90 lower-bound thresholds plus observed safeguards). Omitted targets remain in the full
evaluation universe as minimum-score predictions. If development nominates nothing, validation is
an exact-only execution audit. Reduced reports estimates and retains exact SPATHI.

A provisional full candidate must also survive 18 new scMultiSim safety datasets: 3 replicates of
each 3,000-gene/100-TF/300-cell anchor, 100-gene/10-TF/500-cell single group,
100-gene/10-TF/300-cell 10/40/250 imbalance, 100-gene/10-TF/10,000-cell endpoint,
3,000-gene/1,500-TF/300-cell endpoint and 15,000-gene/100-TF/300-cell endpoint.
The [formal contract](experiments/v1/spathi/efficiency-study.json) defines all gates.
The safety step is a descriptive veto; it never searches for another candidate.

### Dimensional transfer: does the frozen policy transfer?

All cases use truth-bearing scMultiSim. Reduced includes `(genes, TFs, cells)` equal to
`(3000, 100, 300)` and `(100, 10, 300)`, plus the 500-cell single-group and 10/40/250 imbalance
controls. Full adds `(15000, 100, 300)`, `(3000, 1500, 300)` and `(100, 10, 10000)`; its
3,000-gene anchor has three replicates, with one at each other condition.

The production policy remains frozen. If it differs from the calibrated exact policy, declared
anchors also receive exact-policy diagnostics. Full reports separate gene-, TF- and cell-axis
contrasts; the TF contrast also changes regulatory topology and is not a causal isolation of TF
count. Reduced has no scale contrast. Neither profile contributes these data to the external
accuracy ranking or uses them for selection.

### Comparison: data and competitors

| Simulator condition | Requested scale | Reduced datasets | Full datasets |
| --- | --- | ---: | ---: |
| Dyngen linear baseline/noisy | 200/500 cells; 10 TFs + 50/90 targets + 10/30 housekeeping genes | 1 each | 5 each |
| Dyngen bifurcating baseline/noisy | 300/400 cells; 10 TFs + 70/80 targets + 20/10 housekeeping genes | 1 each | 5 each |
| scMultiSim low balanced/high imbalanced rewiring | 80 genes, 240/300 cells; high case has technical noise and a small group | 1 each | 5 each |
| scMultiSim medium balanced/imbalanced | 400 genes, 600/800 cells | — | 5 each |
| SERGIO invariant | 100 genes, 9 bins × 40 cells | 1 | 5 |
| BoolODE invariant clean/dropout | GSD network preset, 300 cells, 3 clusters, integration step 0.01 | 1 each | 5 each |
| GeneSPIDER2 invariant | 80 genes, 4 perturbation replicates per gene | 1 | 5 |

Dyngen/scMultiSim provide group-specific truth; the remaining simulators are invariant-network
controls. Simulators may realize dimensions differently from their requests; reports start with
actual dataset dimensions and identities. Ordinary scMultiSim graph inputs are role-exclusive;
safety and dimensional-transfer topology contracts impose additional edge-disjointness.

All catalog tools use the locked ANDREA defaults corresponding to their libraries. Seeds and
input/execution contracts are supplied for reproducibility and a common information set; no
competitor is tuned against benchmark outcomes. Inclusion and blocked reasons are recorded in the
[catalog matrix](experiments/v1/inference/catalog-matrix.json).

| Analysis family | `reduced` | `full` |
| --- | --- | --- |
| Group-only directed accuracy and invariant stability | Frozen SPATHI, GENIE3, GRNBoost2, inferCSN | Same + TIGRESS |
| Dynamic topology | — | Frozen SPATHI, GENIE3, GRNBoost2, inferCSN, TIGRESS, ARACNe3 |
| GRNBoost implementation sensitivity | — | Direct GRNBoost2 vs pySCENIC's default GRN step |
| Order-aware, linear dyngen only | — | Frozen SPATHI vs SimiC |
| All-gene directed, linear dyngen only | Frozen SPATHI vs MetaSEM | Same + kSReCNI |
| All-gene topology, linear dyngen only | Frozen SPATHI, CeSpGRN without a prior, CLR, MetaSEM | Same + kSReCNI, LIONESS, PIDC, PPCOR |

Stochastic configurations use three algorithm seeds; deterministic or internally fixed-seed
methods have one declared execution. All-gene competitors infer over all genes, then ANDREA
projects their predictions onto the common TF-restricted evaluation space. They remain separate
from the primary family. inferCSN uses ANDREA `group_emulated`: one independent default core call
per declared group, with no pseudotime or hidden state windows. V1 has no pseudotime-aware family.
Prior-assisted tools are catalogued but are not executed as primary competitors.

Full's confirmatory family is directed AUPR on the 40 dynamic test datasets. Dataset-level
aggregation prevents cells, groups and algorithm seeds from becoming artificial independent
replicates. Friedman omnibus tests and paired nonparametric post-hoc tests with Holm correction
are applied only under their declared contracts. Other metrics/families retain their exploratory
labels. Reduced shows paired effects and ranks without p-values or winner claims.

### Scalability: which large matrices?

The [public synthetic generator](experiments/v1/simulations/scale_stress) produces counts-like
expression with declared latent structure, noise, dropout, imbalance and rewiring, then applies
`log1p(10000 * value / full-cell-library-size)`. These are engineering workloads, not a validated
model of a particular biological cohort.

| Profile/case | Genes | TFs | Cells |
| --- | ---: | ---: | ---: |
| Reduced baseline | 5,000 | 250 | 1,000 |
| Reduced moderate | 10,000 | 500 | 3,000 |
| Full baseline | 5,000 | 250 | 1,000 |
| Full cell-heavy | 5,000 | 250 | 10,000 |
| Full gene-heavy | 25,000 | 250 | 1,000 |
| Full TF-heavy | 5,000 | 1,500 | 1,000 |
| Full joint balanced | 25,000 | 1,500 | 10,000 |
| Full joint imbalanced | 20,000 | 1,250 | 5,000 |
| Full invariant control | 20,000 | 1,250 | 5,000 |

Only frozen SPATHI runs here. Each shape begins with an outcome-blind hashed target scout.
The committed resource boundary permits an all-target fit only within 80% of the 12-hour reduced
or 24-hour full envelope and 80% of allocated RAM. The envelope is enforced as a physical timeout.
A reached boundary stops downstream shapes in its declared progression lane. Completion, timeout,
OOM and attributable tool failures remain evidence; malformed contracts or missing telemetry
abort the experiment. Accuracy claims do not come from this phase.

### Performance: how fast are the tools under comparable resources?

Reduced uses one scMultiSim 400-gene/600-cell dataset, frozen SPATHI, GENIE3, GRNBoost2 and
inferCSN at 1 and 8 threads. Full adds an 80-gene/300-cell dataset, ARACNe3 and TIGRESS, uses
1/2/4/8 threads, and repeats each measured condition three times. One warm-up per condition is
retained but excluded from summaries. Each system repetition uses the same algorithm seed.

The entire invocation runs alone with nested, topology-aware CPU affinity, and requests full
ANDREA outputs. Canonical wall time includes preparation, scheduling, merging and output writing.
Memory is the recorded overlap upper bound of physical-container peaks, excluding ANDREA's host
process. Reports distinguish unavailable block-I/O counters from measured output bytes.
Speedup and parallel efficiency require complete, semantically equal networks across the thread
endpoints and repetitions; otherwise they show an explicit unavailable reason. These runs cannot
select a policy or enter accuracy rankings.

## Results and reports

The central entry point for a study is:

```text
work/studies/<id>/
├── study.json
├── reports/index.html
└── v1/
    ├── artifacts/                     # immutable datasets/inferences/evaluations
    ├── workflows/<phase>/<profile>/   # phase evidence and frozen decisions
    └── reports/<profile>/<phase-id>/  # self-contained English phase reports
```

Open `work/studies/<id>/reports/index.html`. It links every phase that has published a report.
Individual HTML reports embed Plotly and their data; they need no server or internet connection.
The entry page and artifact links require the corresponding local directory structure to remain
available. Each report shows its dataset base, profile, exact configurations, claim scope and
provenance. The calibration report exposes candidate parameter values, convergence prefixes and
selection. Efficiency distinguishes descriptive estimates from formal gates. Transfer separates
routing controls from scale contrasts. Comparison labels each information regime. Scalability and
performance show measured resource endpoints and failures, with no biological-accuracy claim.

Repeated scikit-learn parallel-configuration warnings are grouped in the console; the full original
messages remain in command logs. Scientific warnings are not suppressed. In particular, external
groups can contribute more aggregate weight when `group_size_correction=none`; this describes the
tested weighting configuration, not a failed fit or a reason to discard the run.

Plots present the canonical saved results. Rebuilding HTML must not rerank candidates, recalculate
statistical decisions or silently replace corrupted scientific evidence. Report manifests record
both evidence and renderer inputs so a presentation fix remains distinguishable from a scientific
change.

ANDREA-compatible exports from ordinary inference/evaluation pipelines have this structure:

```text
exports/
├── manifest.json
├── datasets/<dataset>/analysis.zip
├── inference/<dataset>/<regime>/individual/<run-id>/analysis.zip
├── inference/<dataset>/<regime>/joint/analysis.zip
├── evaluation/<dataset>/<regime>/individual/<run-id>/analysis.zip
└── evaluation/<dataset>/<regime>/joint/analysis.zip
```

Inference archives can be passed to ANDREA `evaluate-inference`. Joint views contain all methods
for one dataset and regime and preserve ANDREA's contracts for its GUI and `compare-networks`.
Each archive has integrity metadata. Internal calibration artifacts need not duplicate every full
visualization export; the public comparison produces the archives used to inspect results.

Comparison panels can share the same selected SPATHI execution. All batch references remain in
the study state for attribution, but identical executions appear only once per dataset and regime
in joint views, metric rows and analysis exports. Reuse requires matching artifact identities and
scientific metadata; a repeated run ID with different results or parameters is an error. Different
seeds and information regimes remain separate observations.

## Reproducibility, failures and development

The seed ledger derives positive seeds from semantic coordinates using SHA-256 domain separation.
Separate streams cover simulation, augmentation, inference, execution order and, only for the
inferential full analyses, statistical resampling. SPATHI then derives its own per-group/target
model seeds; those are deliberately different from its caller seed. Unestimable targets remain
explicitly accounted for in diagnostics and never disappear from coverage denominators.

Both evaluation paths use the declared candidate space. Simulator truth outside the common
TF-by-gene space is excluded with counts and examples, while genes outside the declared universe
remain an error. Missing predicted edges receive the contract's minimum score. An empty valid
network is distinct from a failed inference. Failed inference preserves logs, diagnostics and
available telemetry and cannot masquerade as a successful cache artifact.

Source, dataset, parameter, image and artifact hashes remain attached to the computations that
produced them. Machine quotas and observed resources are recorded separately. An operator-accepted
commit warning on resume is not evidence of numerical equivalence; it is a traceable decision to
continue. Existing scientific files must still pass integrity checks. For deliberate scientific
changes, use a new study ID and freeze the revised design before examining held-out outcomes.

Tests and static checks:

```bash
.venv/bin/python -m pytest
.venv/bin/ruff check src tests containers/spathi
.venv/bin/spathi-bench qualify
```

The repository contains public synthetic inputs and experimental contracts only. Do not add
patient data, identifiers, private annotation files or paths to restricted research material. Runtime datasets,
networks, archives and logs stay under ignored `work/`; compact public reports may be archived
separately when appropriate.
