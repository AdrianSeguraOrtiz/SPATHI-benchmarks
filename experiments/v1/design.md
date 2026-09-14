# SPATHI benchmark v1

## Scope and research questions

The primary question is whether a frozen SPATHI calibrated policy recovers group-specific
regulatory networks more accurately than competing methods when every method receives the same
expression data, cell groups and transcription-factor candidate list.

The design also answers four separate questions:

1. Does SPATHI remain competitive with methods that receive group order, even though SPATHI does
   not use that additional information?
2. How does the frozen calibrated policy's synthetic accuracy transfer as cells, genes and TFs
   increase separately, including its single-group and extreme-imbalance routing branches?
3. Can the frozen calibrated policy complete inference across a prospectively declared public
   synthetic scale grid, and what resources does it consume?
4. How do frozen SPATHI and catalog-default competitors scale from one to eight assigned threads
   under isolated, identical full-output executions?

These are not pooled into one claim. Dynamic-network accuracy is confirmatory only in the full
held-out comparison. Invariant
network behaviour, topology and information-advantaged methods are exploratory. Dimensional
transfer and all reduced-budget accuracy results are strictly descriptive. The
scale-stress and performance panels supply computational evidence only.

## Sequential design and freeze points

The experiment has five ordered phases:

| Order | Phase | Data role | Input state | Frozen output |
| ---: | --- | --- | --- | --- |
| 1 | Calibration | `development`, then `validation` | predeclared exact search space | `selected-config.json` |
| 2 | Dimensional transfer | `dimensional_transfer` | frozen calibrated policy | reduced operational checks or full descriptive one-axis scale contrasts |
| 3 | External-tool comparison | `test` | the same frozen calibrated policy | accuracy and stability evidence |
| 4 | Scalability | `scalability` | the same frozen calibrated policy | completion, attributable-failure and resource evidence |
| 5 | Performance | `performance` | the same frozen calibrated policy and catalog defaults | isolated cross-tool runtime, memory, output-size and parallel-scaling evidence |

Calibration cannot inspect dimensional-transfer, test, scalability or performance results.
The complete conditional group policy, SPATHI commit and ranking-chain evidence are frozen in
`selected-config.json` after independent validation. Every downstream phase consumes that same
artifact from the completed, same-budget calibration workflow. No downstream result can change
the selected parameters, selection objective or external-tool ranking.

## Protected dataset partitions

The assignment unit is an independently generated dataset instance, never a cell, group, edge or
algorithmic seed. Instances are assigned before inference and stratified by simulator, topology,
condition and group class. Related data and external simulator inputs cannot cross roles.

| Role | Permitted use | Forbidden use |
| --- | --- | --- |
| `development` | bounded calibration search and shortlisting | final selection, external comparison |
| `validation` | choose one calibration finalist | change the search space, external comparison |
| `dimensional_transfer` | reduced operational checks or full post-freeze descriptive one-axis scale contrasts | calibration or external-tool ranking |
| `test` | one comparison of the frozen calibrated policy with competitors | any configuration decision |
| `scalability` | completion and resource measurement across the public synthetic scale grid | biological-accuracy or method-ranking claims |
| `performance` | cross-tool computational and parallel-scaling measurement on fixed public synthetic inputs | truth evaluation, calibration or method-ranking claims |

The reduced and full budgets have disjoint generation and algorithmic seeds. Reduced is not a
subset of full and does not spend any full dataset early.

The scientific panels are:

| Role | `reduced` | `full` |
| --- | ---: | ---: |
| ordinary calibration development | 4 dyngen + 4 scMultiSim conditions = 8 | (4 dyngen + 4 scMultiSim conditions) × 3 = 24 |
| estimator scale-transfer (`development` sub-split) | 1 scMultiSim | scMultiSim × 3 = 3 |
| validation | 4 dyngen + 4 scMultiSim conditions = 8 | (4 dyngen + 4 scMultiSim conditions) × 3 = 24 |
| held-out dynamic test | 4 dyngen + 2 scMultiSim = 6 | 4 dyngen × 5 + 4 scMultiSim × 5 = 40 |
| held-out invariant test | SERGIO + 2 BoolODE + GeneSPIDER2 = 4 | SERGIO × 5 + 2 BoolODE × 5 + GeneSPIDER2 × 5 = 20 |
| performance | 1 fixed medium scMultiSim dataset | 1 fixed small + 1 fixed medium scMultiSim dataset |

Ordinary development, estimator scale-transfer and validation use distinct
scMultiSim regulatory topologies.
For the ordinary panels, distinct means that the complete topology input is role-exclusive; it does
not require pairwise edge-disjoint supports. Some edges may recur because the protected roles share
the same synthetic TF-target universe, but no inferred network, truth edge or fitted model crosses
roles. Dimensional-transfer topologies remain separate from all calibration and test inputs.
All protected roles use distinct simulator seeds. Full test contains 60 independent datasets;
reduced test contains ten independent descriptive counterparts.

## Evidence budgets

`reduced` preserves phase order and the contracts for each included operational subset while using
fewer independent datasets and algorithmic repetitions. It does not preserve every full-budget
scientific question or external-comparison family. It exists to catch contract, runtime and
integration failures at tolerable cost. Its comparison analyses produce point
estimates only: no confidence intervals, hypothesis tests, inferential pass/fail labels or winner
claims. It selects one provisional calibrated policy for its downstream phases.

`full` is the publication budget:

- 24 ordinary development datasets: the multi-fidelity geometry, estimator and interaction stages
  use one scout seed and two confirmation seeds, while convergence uses five; 24 independent
  validation datasets use five seeds, and three independent 1,000-TF scale-transfer datasets use
  three;
- nine dimensional-transfer datasets: three independent instances at the 3,000-gene anchor and one
  instance at each of the other six predeclared conditions;
- 60 held-out test datasets (40 dynamic and 20 invariant controls) with three paired algorithmic
  seeds per stochastic configuration;
- seven full-budget scale-stress cases, including one invariant-network control, for scalability;
  and
- 144 isolated full-output performance measurements plus 48 excluded warm-ups over two fixed
  public synthetic datasets, six methods and four thread endpoints.

The scientific design has no machine-specific CPU or RAM ceiling. At startup, the execution layer
detects affinity, cgroup quotas, available RAM, CPU topology and workspace storage, applies any
operator ceilings or reserves, and freezes the resulting runtime plan. A deterministic rolling
scheduler admits tasks only while their explicit CPU and RAM claims fit that plan. ANDREA limits
remain per-run quotas. Every result records detected capacity, requested ceilings, resolved task
resources and scheduler utilization; formal timing comparisons are stratified by their frozen
resource endpoint and never silently pooled across different runtime plans.

The number of algorithmic seeds does not increase the biological sample size. Repetitions estimate
and average algorithmic variation within an independent dataset.

The `smoke` profile belongs to neither budget. It validates the end-to-end software contract on one
tiny dyngen instance and must never enter selection or comparison evidence.

## Phase 1: calibration

### Inputs and conditional policy

Calibration uses only dynamic dyngen and scMultiSim datasets. No candidate receives a target list,
regulatory prior, group order, pseudotime or centroid-confidence weights. Calibration `bandwidth=auto` rule. Every requested target is considered
for fitting, subject only to numerical safety checks; each model uses a fixed-size forest.

Automatic bandwidth is fixed because it adapts to the observed distance scale of each dataset.
Calibrating an absolute width on the small development panel would create a non-transferable
quantity across heterogeneous expression geometries.

The selected object is one conditional group policy rather than unrelated per-dataset winners.
Shared parameters apply to all datasets. For a single observed group, the policy forces
`weight_mode=cell-distance`, `distance_metric=euclidean` and `group_size_correction=none`. For two
or more groups, the calibrated multi-group branch applies. Routing depends only on group count.

### Bounded stages

`experiments/v1/spathi/search-space.json` defines the complete search. Geometry, estimator and
convergence include the unchanged inherited configuration alongside only the listed candidates.
Interaction and estimator scale-transfer use complete, explicit crossed variants so an override
cannot accidentally reproduce an inherited configuration. The `base_configuration` is the current
first-release SPATHI default policy with automatic bootstrap resolved to `false` for Extra-Trees;
it is an incumbent candidate, not a privileged reference:

1. `geometry` compares PCA or expression space, component count, PCA solver (`auto`, `full` or
   `randomized`), standardization, Euclidean or cosine distance, Gaussian or exponential kernel,
   weight mode and group-size correction.
2. `estimator` compares Extra-Trees and Random Forest choices, feature subsampling,
   `min_samples_leaf`, depth and valid bootstrap combinations.
3. `interaction` tests six complete, predeclared estimator interactions over three retained
   geometry identities rather than a full Cartesian grid. Its upstream shortlist must retain three
   configurations that remain distinct after the five estimator parameters overwritten by this
   cross are removed from their identities.
4. `estimator_scale_transfer` crosses all six declared `max_features` rules over at most three
   inherited interaction structures on an independent scMultiSim panel with 1,000 TFs, 2,200 genes
   and 300 cells. It has no unchanged inherited candidate. Full uses three datasets and three
   algorithmic seeds; reduced uses one dataset and two algorithmic seeds. At most two
   configurations advance. Its upstream shortlist analogously retains three interaction structures
   that remain distinct when `max_features` is excluded. These preregistered diversity constraints
   preserve the exact candidate cardinality without silently deduplicating effective configurations.
5. `convergence` selects a fixed tree count among 250, 500, 750 and 1000 trees.
   One physical 1,000-tree fit per parent, dataset and seed emits four
   exact forest-prefix networks through SPATHI's public prefix API; each prefix remains a separate
   ANDREA-compatible logical run for integration and evaluation. Reduced therefore contains 32
   physical convergence fits and 128 logical results; full contains 240 physical fits and 960
   logical results.
6. `validation` evaluates exactly two development finalists on the independent validation panel
   and freezes one policy.

The first five stages use `development`, but only the scale-transfer stage can consume its dedicated
split and condition; all other search stages consume the ordinary development panel. Validation
uses `validation` and advances exactly one.

### Calibration objective

Directed AUPR is averaged equally across group contexts within an execution, across algorithmic
seeds within dataset and configuration, and across datasets. Candidates are ordered by:

1. higher mean directed AUPR over equally weighted datasets;
2. lower AUPR dispersion across algorithmic seeds;
3. fewer changes from the first-release `base_configuration`; and
4. canonical configuration identifier as a deterministic final tie-breaker.

Wall time and peak memory are retained in every ranking artifact and report for transparency, but
they are not calibration tie-breakers. Computational cost is measured separately in scalability
and in the isolated cross-tool performance phase.

Every stage manifest records the complete parameter map, configuration and execution hashes,
expected datasets and algorithmic seeds, SPATHI commit, image ID and build fingerprint. Rankings
reject missing or extra runs, unequal panels, role mixing and a changed upstream shortlist. The
frozen selection recursively validates the whole ranking chain. Across every calibration stage,
the reduced budget is exactly 431 physical fits and 527 logical results; the publication full budget
is exactly 1,762 physical fits and 2,482 logical results. Candidate expansion rejects duplicate
effective configurations instead of silently shrinking these preregistered cardinalities. These
totals never conflate convergence fan-out
outputs with forests actually fitted.

Centroid confidence weights are deliberately absent from the calibration. The committed simulators
do not provide a justified, calibrated model of an external annotation-confidence measurement
process. Applying arbitrary synthetic confidence scores would tune to an invented signal. The v1
primary policy therefore uses uniform centroid contributions; no accuracy claim about confidence
weighting is made.

## Phase 2: post-freeze dimensional transfer

The dimensional-transfer phase asks a different question from computational scalability:
how the already frozen calibrated policy's synthetic accuracy behaves as one input
dimension increases at a time. It uses scMultiSim datasets with dynamic group-specific truth and
never selects parameters, changes the policy or contributes observations to the confirmatory
external-tool comparison.

The full panel has two balanced three-group anchors and three descriptive one-axis scale contrasts:

- genes: 3,000 to 15,000 at 100 TFs and 300 cells;
- candidate TFs: 100 to 1,500 at 3,000 genes and 300 cells; and
- cells: 300 to 10,000 at 100 genes and 10 TFs.

It also contains a 500-cell single-group routing control and a three-group control with group sizes
10, 40 and 250. These controls exercise policy branches; they are not dimensional contrasts. No
condition combines the maximum gene, TF and cell counts, so the phase cannot estimate their joint
interaction or claim biological realism. The two anchor families also use different
regulatory-core sizes and are interpreted only within their declared scale pairs. In addition,
the TF contrast changes the regulatory topology and which TFs regulate targets between anchor and
endpoint; it therefore cannot causally identify a pure effect of the number of candidate TFs.

The reduced budget is a low-cost operational subset with one independent dataset for each of
`g3000_tf100_c300`, `g100_tf10_c300`, `g100_tf10_c500_single_group` and
`g100_tf10_c300_groups_10_40_250`. It omits all three large scale endpoints and records
`contrasts=[]`; it cannot answer any of the full-budget scale questions. The full budget contains
three independent datasets for the 3,000-gene, 100-TF, 300-cell anchor and one for each other
condition. These sparse, deliberately bounded replicates provide publication-descriptive transfer
evidence, not confirmatory inference or a power guarantee. Cells, contexts and edges do not become
independent replicates.

## Phase 3: held-out external comparison

### Primary estimand

The confirmatory full analysis uses only the 40 held-out dynamic dyngen and scMultiSim datasets in
the TF-restricted `group_only` regime. Every method receives:

- the same expression dataset;
- the same cell-to-group assignment; and
- the same simulator-generated `tf_list` as its regulator candidate set.

SPATHI is `group_native`: it may use all cells to produce all group networks. Catalog methods in
`group_emulated` mode receive each group subset independently. This difference is the method being
tested, not an assertion that their physical input matrices are identical.

No primary method receives a regulatory prior, true edges, lineage order or pseudotime. No target
list is supplied, reflecting the intended real-data use where targets are not known in advance.

The primary endpoint is directed group-level AUPR. Directed AUROC is secondary within the dynamic
family. Topology accuracy is exploratory. Exact batches and statistical families are committed in
`experiments/v1/inference/plans/full-v1.json`; the seed-disjoint descriptive counterpart is
`experiments/v1/inference/plans/reduced-v1.json`.

### Executed configurations

The full directed `group_only` family contains:

- frozen calibrated SPATHI;
- GENIE3 with the common TF list;
- GRNBoost2 with the common TF list;
- inferCSN with `sift_method=none`;
- TIGRESS with the common TF list.

ARACNe3 is included only in the exploratory topology family because its catalog output is
undirected. pySCENIC-GRN is executed with its default GRNBoost2 backend only in a separate
implementation-sensitivity contrast against the direct GRNBoost2 wrapper; it cannot count as an
independent rival in the primary ranking. The reduced descriptive family contains SPATHI, GENIE3,
GRNBoost2 and inferCSN.

Separate all-gene-inference families are exploratory and use only the two held-out linear dyngen
conditions. The full directed family compares SPATHI with kSReCNI and MetaSEM; the full
topology family additionally includes expression-only CeSpGRN, CLR, LIONESS, PIDC and PPCOR. The
reduced directed counterpart keeps SPATHI and MetaSEM, while reduced topology keeps SPATHI,
CeSpGRN, CLR and MetaSEM. These competitors infer with all genes as possible regulators, but ANDREA
projects their predictions onto the same TF-restricted evaluation universe as SPATHI. Consequently,
the families assess the effect of unequal inference scope and cannot enter the primary rank.

The separate `order_aware` family compares SPATHI with SimiC in its sole upstream-default mode
(`similarity=true`) on compatible linear dyngen trajectories. SimiC consumes ordered phenotypes,
so this information-advantaged comparison is exploratory and cannot enter the primary rank. V1
defines no pseudotime-aware family because no integrated competitor consumes pseudotime through a
supported catalog contract. The integrated inferCSN CRAN core receives no pseudotime and exposes
only `none` and `max` as sift modes.

scMTNI is excluded because its current multigroup input carries truth-derived lineage gain/loss
rates. Inferelator3, scRegulate and other required-prior runs are excluded because a simulator prior
would leak regulatory truth. scMINER is excluded because its default built-in species drivers do not
match synthetic identifiers; changing `driver_source` would tune a scientific parameter. scING is
blocked rather than dimensionally retuned when its upstream defaults cannot execute the declared
panel. Methods unable to restrict candidate regulators to `tf_list` are not mixed with the primary
TF-restricted family. An order-permutation control is not fabricated by editing a simulator
artifact in place; such a derivation would require its own fingerprinted dataset contract.

### Invariant controls

BoolODE uses its upstream default integration step of `0.01` (simulation time `8`).
This aligns the generated time grid with its native one-cell-per-trajectory sampler.
Both clean and dropout panels simulate 300 trajectories, cluster them into three groups,
and export 300 cells; the integration step is not an output-downsampling parameter.

SERGIO, BoolODE and GeneSPIDER2 generate groups or trajectories while keeping the true regulatory
network invariant. They answer whether a method invents group-to-group reconfiguration. Their
dedicated exploratory endpoint is `invariant_edge_stability`.

For each invariant dataset, `k` is the common number of directed truth edges. Each inferred group
network contributes its top `k` directed edges; stability is the mean pairwise intersection size
divided by `k`. Empty predictions score zero. This metric measures consistency, not correctness,
and never replaces accuracy against truth.

### Statistical inference

The independent block is one independently simulated dataset. Within each configuration and
dataset, group contexts are averaged equally and then algorithmic executions are averaged equally.
Every complete dataset then receives equal weight in the predeclared panel. This panel-average
estimand deliberately reflects the declared mixture of simulator conditions; it does not pretend
that unequal simulator or condition counts are an equal-simulator estimand.

Cells, edges, groups and algorithmic seeds never increase the inferential sample size. Distinct
dataset IDs sharing a simulator seed are rejected. Each configuration ID has one immutable
parameter hash, duplicate algorithmic seeds are rejected, and every paired member must expose the
same contexts, dataset fingerprint, source and target gene sets, candidate-edge universe,
truth-edge universe and ANDREA candidate-universe counts.

Missing or non-evaluable members invalidate the complete family; there is no imputation or
pairwise deletion. An empty inferred network that ANDREA evaluates numerically remains a valid
zero-edge observation. Exploratory results can never mask an untestable primary endpoint: the
comparison workflow fails closed unless every confirmatory primary analysis is `tested`.

Every comparison declares `expected_blocks`, the exact number of independently simulated datasets
in its frozen panel after its generation-job filter. The value is a cardinality and completeness
contract, not a minimum sample-size heuristic, power calculation or guarantee of detectable
effects. Any mismatch is a design error and stops analysis. The reduced comparison always has
status `descriptive_only`; Friedman, Wilcoxon, Holm and bootstrap calculations are not run and no
ranking is labelled a statistical winner.

With at least three methods, the omnibus test is a tie-corrected Friedman test with midranks,
Kendall's W and an exact or preseeded within-block label-permutation p-value. Predeclared
SPATHI-versus-rival contrasts use two-sided paired Wilcoxon signed-rank tests, followed by Holm
step-down family-wise correction. With two methods, only the paired contrast applies.

Reports accompany p-values with mean ranks, medians, Hodges-Lehmann paired differences,
rank-biserial effects, win/tie/loss counts and paired dataset-level bootstrap intervals. Bootstrap
resampling keeps simulator-by-condition design strata fixed and resamples independent replicates
within each stratum. SPATHI is described as the statistical winner only if it is uniquely
top-ranked and every required directional contrast is significant after correction.

## Phase 4: public synthetic scalability

The scalability phase uses a rounded, author-defined public grid selected to cover the intended
application-scale envelope. It retains no source samples, identifiers, exact source dimensions or
biological observations. The full grid spans 1,000, 5,000 and 10,000 cells; 5,000, 20,000 and 25,000
genes; and 250, 1,250 and 1,500 candidate TFs. It contains separate axis-heavy cases, two joint
workloads and an invariant-network control. The reduced grid contains two moderate development
workloads. Neither panel supplies a target list. The requests are committed as
`experiments/v1/simulations/scale_stress/scale-full.json` and
`experiments/v1/simulations/scale_stress/scale-reduced.json`.

Each request declares the complete `scale_grid` it covers with provenance
`author_defined_public_synthetic_grid`. Every dataset has a unique `design_role` identifying either
a synthetic scale case or a methodological control. When the workflow materializes one dataset at
a time, it narrows the grid to that dataset's one cell, gene and TF level while retaining the same
preparation and deterministic seed contracts.

The generator writes ANDREA-compatible expression, groups, TF list, truth network and manifests in
bounded gene blocks. It explicitly models dynamic or invariant wiring, active-TF and regulated-
target fractions, low-detection targets, dropout, library-size variation, latent factors and noise.
Those values are public synthetic scenario definitions, not retained source measurements.

The counts-like values produced by that synthetic model are not handed directly to inference.
Before publication, the generator sums every generated gene for each cell and emits
`log1p(10000 * raw_value / per_cell_library_size)` under the declared
`library-size-log1p`, `target_sum=10000` contract. It uses a unique
disk-backed float32 working matrix and bounded float64 normalization blocks, and removes the
working file on every exit path. The request, benchmark manifest, generator provenance and final
scalability results expose the contract. Request validation rejects normalization or target-sum
drift and requires every scale axis to enumerate exactly the values used by its datasets.

The scalability workflow does not evaluate its known synthetic truth or rank inference methods.
That truth makes the generated
artifact explicit and auditable, but it cannot support a biological accuracy claim or join the
external-tool comparison. The full phase supports publication claims about completion and measured
computational resources on this machine and software stack. It does not establish runtimes for
every machine or biological fidelity to a real population.

Each shape first runs an outcome-blind target scout selected by SHA-256 over the algorithm seed,
dataset ID and gene ID: 64 targets in reduced and 128 in full. The scout and all-target stages use
the same frozen calibrated policy and the same algorithm seed. The preregistered contract fixes a
prospective per-endpoint wall envelope of 12 hours in reduced and 24 hours in full. An all-target
run is attempted only when the scout's linear target-count projection is at most 80% of that wall
envelope and its observed peak memory is at most 80% of the RAM actually allocated to the
container. Neither truth, accuracy, expression values nor inferred edges may enter this decision.

Datasets belong to ordered progression lanes. Once an attempted stage reaches a preregistered
resource boundary, every later stage in that lane is recorded exactly as
`not_attempted_after_preregistered_resource_boundary`; it is not treated as a failure or a
completion. The committed boundary, target-scout rule, lane order, wall envelopes, fractions,
decisions and evidence dataset are retained in the results and interactive report.

The recorded endpoints are immutable generation time, SPATHI core time and ANDREA end-to-end wall
time; canonical ANDREA container CPU time and the overlap upper bound formed from physical-container
cgroup memory peaks (excluding the ANDREA host process); expression, dataset, inference and
SPATHI-output bytes; requested-target counts; model completion; and fitted-tree count. The
12/24-hour envelope is also enforced by ANDREA as the physical-task timeout. Inference must be fresh
for resource attribution. Measurements are resumable only when their digest is already bound to the
exact stage-specific runtime state.

An OOM, timeout or SPATHI tool error is retained only when ANDREA has written a failed report that
matches the exact requested dataset, stage, run and SPATHI origin. Missing or ambiguous reports,
infrastructure failures, identity mismatches, corrupted artifacts, invalid progress, incomplete
canonical telemetry or inconsistent provenance abort the workflow. Such failures never become a
resource boundary or a publishable SPATHI endpoint.

## Phase 5: formal computational performance

Performance is deliberately separate from scalability. Scalability asks whether frozen SPATHI can
complete increasingly large shapes; performance compares methods and thread endpoints on the same
fixed inputs. The phase never reads synthetic truth, evaluates accuracy, selects a method or
changes the calibrated policy. Its budget-specific generation seeds are disjoint from every other
phase, although the public simulator scenario and dimensional definitions are deliberately reused.

The reduced operational panel uses one medium scMultiSim dataset and measures SPATHI, GENIE3,
GRNBoost2 and inferCSN at one and eight threads once. The full panel uses one small and one medium
scMultiSim dataset and measures SPATHI, ARACNe3, GENIE3, GRNBoost2, inferCSN and TIGRESS at 1, 2, 4
and 8 threads with three independent system repetitions. Every repetition retains one common
algorithm seed. pySCENIC and order-aware configurations are outside the primary panel.

One excluded warm-up precedes each dataset–method–thread condition. Measured blocks follow a
committed hash-ranked rotate-and-reverse order so method, thread count and run position are not
silently aligned. Each task is a fresh, full-output ANDREA execution with explicit threads and a
CPU set resolved from the immutable runtime plan. CPU sets are nested and allocate one logical CPU
per physical core before SMT siblings; tasks run strictly one at a time. Completed task directories
are immutable and content-validated on resume, which is recovery rather than an
inference cache.

Wall time uses ANDREA's canonical execution-level measurement, including input verification and
preparation, scheduling, merging, and requested output materialization. It is therefore an end-to-end software measurement, not a kernel-only
microbenchmark; this same boundary is applied to every method even when their delivery mode differs.

The report retains wall time, CPU time, the overlap upper bound formed from physical-container cgroup
memory peaks (excluding the ANDREA host process), output bytes, completion, attributable failure,
OOM evidence and semantic equality of the canonical normalized network across threads and system
repetitions. Speedup and parallel efficiency are published only when every measured repetition and
thread endpoint completes and all canonical networks are identical; otherwise both values are null
and the reason is explicit. ANDREA v1 exposes exact output bytes but no block-device read/write
counters, so those I/O fields are explicitly unavailable. Full results support only machine- and
software-stack-specific computational claims.

## Reproducibility contract

`experiments/v1/seeds/seed-plan.json` is the authoritative seed ledger. It uses master seed
`20260905` and SHA-256 domain-separated derivation to produce portable positive 31-bit seeds from:

- budget (`smoke`, `reduced` or `full`);
- protected dataset role;
- stochastic stream; and
- semantic coordinates such as simulator run, replicate, method, stage or dataset ID.

The streams are `generation`, `augmentation`, `inference` and `statistical`.
Seed derivation is independent of manifest ordering, so adding or reordering an unrelated task
cannot change existing seeds. Contiguous simulator seed blocks are allocated explicitly; other
task seeds are derived directly and stored with their coordinates.

Only the full held-out comparison allocates the `statistical` stream.
Smoke is contract-only and every reduced analysis is descriptive: their contracts omit alpha,
permutation and bootstrap counts, multiplicity rules and statistical seeds.

The reproducibility audit verifies every committed simulator allocation, inference request and
scale-stress seed derivation. It rejects
stale derived values, duplicate or overlapping generation blocks, undeclared namespaces,
request/metadata disagreement and cross-budget reuse.

`repos.lock.json` fixes ANDREA and SPATHI source revisions. `images.lock.json` fixes catalog images
by digest and fixes the SPATHI base image, source revision, Python dependencies and adapter hashes.
Every run snapshots its design inputs, concrete image IDs and Python environment. Scientific
artifacts and joint projections are content-addressed; no stage selects outputs by modification
time.

## ANDREA handoffs and result integrity

Simulation, catalog inference and SPATHI inference are separate immutable artifacts. ANDREA owns
the individual `run_report.json`, merged networks and evaluation contract. SPATHI-benchmarks passes
individual inference reports unchanged to `evaluate-inference`, then builds an explicitly
non-monolithic joint view for each dataset and information regime.

The exported individual and joint `analysis.zip` files use ANDREA's official virtual paths and
member contracts. Individual inference ZIPs can be reevaluated by ANDREA. Joint inference and
evaluation ZIPs can be used together in `compare-networks`. Joint views retain source report paths,
hashes and tool identities; they never claim the methods ran in one monolithic command.

The shared artifact store keys generation by simulator request and image, and inference by dataset,
method parameters, seed, environment and concrete image. A SPATHI change invalidates SPATHI and
dependent joint artifacts while leaving unchanged simulator and catalog artifacts reusable. Phase
workflow identities bind all frozen upstream evidence, preventing a result from consuming a policy
selected for another implementation or budget.

A failed inference attempt is never published under the `inference` artifact kind. Before its
private staging directory is removed, the harness instead binds the validated ANDREA failure report,
command and container logs, last progress snapshot, and available SPATHI diagnostic sidecars into a
content-addressed `inference-failure` evidence artifact. That operational evidence remains explicitly
failed and is ineligible for integration, evaluation, ranking, or successful-inference cache reuse.

Scalability and performance use a two-phase report publication contract.
`workflow.json` is first
written with status `publishing`; the deterministic HTML and provenance manifest are then published
as immutable files; only after both have been verified are their canonical paths and SHA-256 hashes
stored under `reports` and the workflow atomically changed to `complete`. A resume may reconstruct a
missing bundle member from frozen inputs, but it must reject an existing member with different
bytes. The report manifest intentionally excludes `workflow.json` from its hashed scientific inputs
to avoid a circular digest while retaining a navigational link to the final state.

## Interpretation rules

Only the complete full workflow may support manuscript conclusions, and each conclusion must retain
its declared scope:

- directed AUPR on dynamic held-out `group_only` data: confirmatory external-tool accuracy;
- post-freeze dimensional transfer: publication-descriptive synthetic accuracy along separate
  cell, gene and TF axes, without policy selection or a joint-scale accuracy claim;
- invariant edge stability, topology, order-aware and pySCENIC implementation-sensitivity analyses:
  exploratory;
- all-gene-inference directed and topology families: exploratory unequal-inference-scope analyses;
- reduced comparison: descriptive only, without inferential winner claims;
- public synthetic scalability: computational feasibility and measured resources only; and
- full performance: isolated cross-tool computational and parallel-scaling evidence for the
  recorded host and software stack only; reduced performance remains descriptive.

No test result may change the calibrated policy. No truth-derived prior or lineage rate
may enter a primary inference input. No failure may silently narrow a family. No synthetic
dimensional stress result may be described as biological realism.
