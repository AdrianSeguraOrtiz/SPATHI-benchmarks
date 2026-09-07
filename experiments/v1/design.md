# SPATHI benchmark v1

## Scope and research questions

The primary question is whether a frozen SPATHI production policy recovers group-specific
regulatory networks more accurately than competing methods when every method receives the same
expression data, cell groups and transcription-factor candidate list.

The design also answers four separate questions:

1. Can target eligibility or adaptive tree stopping reduce SPATHI cost without an unacceptable
   loss of directed AUPR or network coverage?
2. Does SPATHI remain competitive with methods that receive group order, even though SPATHI does
   not use that additional information?
3. How does the frozen production policy's synthetic accuracy transfer as cells, genes and TFs
   increase separately, including its single-group and extreme-imbalance routing branches?
4. Can the frozen production policy complete inference across a prospectively declared public
   synthetic scale grid, and what resources does it consume?

These are not pooled into one claim. Dynamic-network accuracy is confirmatory only in the full
held-out comparison. Efficiency has its own full confirmatory non-inferiority study. Invariant
network behaviour, topology and information-advantaged methods are exploratory. Dimensional
transfer and all reduced-budget accuracy and efficiency results are strictly descriptive. The
scale-stress panel supplies computational evidence only.

## Sequential design and freeze points

The experiment has five ordered phases:

| Order | Phase | Data role | Input state | Frozen output |
| ---: | --- | --- | --- | --- |
| 1 | Calibration | `development`, then `validation` | predeclared exact search space | `selected-config.json` |
| 2 | Efficiency justification | `efficiency_validation`; conditional `efficiency_safety` | frozen exact calibration winner | `production-config.json` |
| 3 | Dimensional transfer | `dimensional_transfer` | frozen production policy | reduced operational checks or full descriptive one-axis scale contrasts |
| 4 | External-tool comparison | `test` | the same frozen production policy | accuracy and stability evidence |
| 5 | Scalability | `scalability` | the same frozen production policy | completion, attributable-failure and resource evidence |

Calibration cannot inspect efficiency-validation, efficiency-safety, dimensional-transfer, test
or scalability results. Formal efficiency selection cannot inspect the safety panel until it has
nominated its sole provisional candidate; safety may only authorize that candidate or retain exact.
Efficiency justification cannot inspect dimensional-transfer or test results and cannot change the
scientific calibration objective.
Dimensional transfer and comparison accept only the production policy frozen by a completed,
same-budget efficiency workflow. Neither can change that policy. Scalability uses that same policy
and cannot feed back into calibration, efficiency selection, dimensional-transfer interpretation
or the external-tool ranking.

The exact calibration winner has `target_eligibility=all` and `adaptive_trees=false`. It becomes
the exact arm in the efficiency study. The full efficiency decision either retains that arm or
adopts one predeclared efficiency policy. That decision, its complete group policy, the SPATHI
commit and all evidence hashes are frozen in `production-config.json` before any
dimensional-transfer or test data are used.

## Protected dataset partitions

The assignment unit is an independently generated dataset instance, never a cell, group, edge or
algorithmic seed. Instances are assigned before inference and stratified by simulator, topology,
condition and group class. Related data and external simulator inputs cannot cross roles.

| Role | Permitted use | Forbidden use |
| --- | --- | --- |
| `development` | bounded calibration search and shortlisting | final selection, efficiency decision, external comparison |
| `validation` | choose one calibration finalist | change the search space, efficiency decision, external comparison |
| `efficiency_validation` | paired non-inferiority and resource gates | calibration or external-tool ranking |
| `efficiency_safety` | descriptive exact-versus-provisional adoption veto | calibration, candidate search, formal non-inferiority or external-tool ranking |
| `dimensional_transfer` | reduced operational checks or full post-freeze descriptive one-axis scale contrasts | calibration, efficiency selection or external-tool ranking |
| `test` | one comparison of the frozen production policy with competitors | any configuration or efficiency decision |
| `scalability` | completion and resource measurement across the public synthetic scale grid | biological-accuracy or method-ranking claims |

The reduced and full budgets have disjoint generation and algorithmic seeds. Reduced is not a
subset of full and does not spend any full dataset early.

The scientific panels are:

| Role | `reduced` | `full` |
| --- | ---: | ---: |
| ordinary calibration development | 4 dyngen + 4 scMultiSim conditions = 8 | (4 dyngen + 4 scMultiSim conditions) × 3 = 24 |
| estimator scale-transfer (`development` sub-split) | 1 scMultiSim | scMultiSim × 3 = 3 |
| validation | 4 dyngen + 4 scMultiSim conditions = 8 | (4 dyngen + 4 scMultiSim conditions) × 3 = 24 |
| efficiency development | 4 dyngen conditions + 1 scMultiSim = 5 | (4 dyngen conditions + 1 scMultiSim) × 3 = 15 |
| efficiency validation | 4 dyngen conditions + 1 scMultiSim = 5 | 4 dyngen conditions × 5 + scMultiSim × 5 = 25 |
| pre-freeze efficiency safety | not run | 6 scMultiSim conditions × 3 = 18 |
| held-out dynamic test | 4 dyngen + 2 scMultiSim = 6 | 4 dyngen × 5 + 2 scMultiSim × 5 = 30 |
| held-out invariant test | SERGIO + 2 BoolODE + GeneSPIDER2 = 4 | SERGIO × 5 + 2 BoolODE × 5 + GeneSPIDER2 × 5 = 20 |

Ordinary development, estimator scale-transfer, validation and efficiency validation use distinct
scMultiSim regulatory topologies.
For the ordinary panels, distinct means that the complete topology input is role-exclusive; it does
not require pairwise edge-disjoint supports. Some edges may recur because the protected roles share
the same synthetic TF-target universe, but no inferred network, truth edge or fitted model crosses
roles. A stronger explicit contract keeps the distinct efficiency-safety topologies edge-disjoint
from one another and keeps their union edge-disjoint from dimensional transfer.
All protected roles use distinct simulator seeds. Full test contains 50 independent datasets;
reduced test contains ten independent descriptive counterparts.

## Evidence budgets

`reduced` preserves phase order and the contracts for each included operational subset while using
fewer independent datasets and algorithmic repetitions. It does not preserve every full-budget
scientific question or external-comparison family. It exists to catch contract, runtime and
integration failures at tolerable cost. Its comparison and efficiency analyses produce point
estimates only: no confidence intervals, hypothesis tests, inferential pass/fail labels or winner
claims. Its efficiency workflow always emits a descriptive production file containing the exact
calibrated policy.

`full` is the publication budget:

- 24 ordinary development datasets and 24 validation datasets with five algorithmic seeds per
  ordinary calibration stage, plus three independent 1,000-TF scale-transfer datasets with three
  algorithmic seeds;
- 15 efficiency-development datasets with three algorithmic seeds per candidate;
- 25 efficiency-validation datasets (five in each of five strata) with five algorithmic seeds per arm;
- 18 efficiency-safety datasets (three in each of six protected routing or dimensional conditions)
  with one common paired algorithm seed;
- nine dimensional-transfer datasets: three independent instances at the 3,000-gene anchor and one
  instance at each of the other six predeclared conditions;
- 50 held-out test datasets with three paired algorithmic seeds per stochastic configuration; and
- seven full-budget scale-stress cases, including one invariant-network control, for scalability.

The v1 host budget is capped at eight logical CPU cores and 48 GB RAM across concurrent work, with
stricter per-phase limits where declared. ANDREA inference limits are per-run quotas; therefore the
validated contract requires `max_parallel_runs × max_cores <= 8` and
`max_parallel_runs × max_ram_gb <= 48`. Runtime evidence records both these limits and the observed
host. A change to the resource ceiling is a new experimental design and cannot be pooled silently
with v1 costs.

The number of algorithmic seeds does not increase the biological sample size. Repetitions estimate
and average algorithmic variation within an independent dataset.

The `smoke` profile belongs to neither budget. It validates the end-to-end software contract on one
tiny dyngen instance and must never enter selection, non-inferiority or comparison evidence.

## Phase 1: calibration

### Inputs and conditional policy

Calibration uses only dynamic dyngen and scMultiSim datasets. No candidate receives a target list,
regulatory prior, group order, pseudotime or centroid-confidence weights. The following parameters
remain fixed throughout calibration:

- `pca_svd_solver=auto`;
- `bandwidth=auto`;
- `target_eligibility=all`; and
- `adaptive_trees=false`.

Automatic bandwidth is fixed because it adapts to the observed distance scale of each dataset.
Calibrating an absolute width on the small development panel would create a non-transferable
quantity across heterogeneous expression geometries.

The selected object is one conditional group policy rather than unrelated per-dataset winners.
Shared parameters apply to all datasets. For a single observed group, the policy forces
`weight_mode=cell-distance`, `distance_metric=euclidean` and `group_size_correction=none`. For two
or more groups, the calibrated multi-group branch applies. Routing depends only on group count.

### Bounded stages

`experiments/v1/spathi/search-space.json` defines the complete search. Every stage except the
dedicated estimator scale-transfer cross includes the unchanged inherited configuration and only
the listed candidates. Its `base_configuration` is the current first-release SPATHI default policy
with automatic bootstrap resolved to `false` for Extra-Trees; it is an incumbent candidate, not a
privileged reference:

1. `geometry` compares PCA or expression space, component count, standardization, Euclidean or
   cosine distance, Gaussian or exponential kernel, weight mode and group-size correction.
2. `estimator` compares Extra-Trees and Random Forest choices, feature subsampling,
   `min_samples_leaf`, depth and valid bootstrap combinations.
3. `interaction` tests a small predeclared set of estimator interactions rather than a full
   Cartesian grid.
4. `estimator_scale_transfer` crosses all six declared `max_features` rules over at most three
   inherited interaction structures on an independent scMultiSim panel with 1,000 TFs, 2,200 genes
   and 300 cells. It has no unchanged inherited candidate. Full uses three datasets and three
   algorithmic seeds; reduced uses one dataset and two algorithmic seeds. At most two
   configurations advance.
5. `convergence` selects a fixed tree-count ceiling among the inherited count and 500, 750 or 1000
   trees. Adaptive stopping is still disabled.
6. `validation` evaluates at most three development finalists on the independent validation panel
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
they are not calibration tie-breakers. Computational efficiency is decided only in the later,
independent efficiency-justification phase.

Every stage manifest records the complete parameter map, configuration and execution hashes,
expected datasets and algorithmic seeds, SPATHI commit, image ID and build fingerprint. Rankings
reject missing or extra runs, unequal panels, role mixing and a changed upstream shortlist. The
frozen selection recursively validates the whole ranking chain.

Centroid confidence weights are deliberately absent from the calibration. The committed simulators
do not provide a justified, calibrated model of an external annotation-confidence measurement
process. Applying arbitrary synthetic confidence scores would tune to an invented signal. The v1
primary policy therefore uses uniform centroid contributions; no accuracy claim about confidence
weighting is made.

## Phase 2: efficiency justification

Automatic target eligibility and adaptive trees are computational policies, not ordinary
calibration parameters. They are evaluated only after the exact policy is frozen and on an
independent `efficiency_validation` panel. The formal efficiency-validation contract is committed in
`experiments/v1/spathi/efficiency-study.json`.

The exact arm is compared with three predeclared candidates:

| Candidate | Change from exact policy |
| --- | --- |
| Target eligibility | automatic eligibility with at least 20 detected cells, detected fraction 0.01, weighted detected fraction 0.01 and weighted effective sample size 10 |
| Adaptive trees | minimum 100 trees, batches of 50, tolerance 0.01 and patience 2, with calibrated `n_estimators` as the hard ceiling |
| Combined | the exact union of the two policies above |

The full requested target universe is primary. A target omitted by an eligibility policy is scored
as an empty minimum-score prediction, so reducing target coverage cannot manufacture a favourable
AUPR. Accuracy on the retained-target intersection is diagnostic only. TF predictor hashes must
remain identical between paired arms.

For every candidate, adoption requires all of the following:

- the multiplicity-adjusted one-sided lower confidence bound for paired
  `AUPR(candidate) - AUPR(exact)` is at least `-0.002`;
- every scheduled run and every requested target-group model completes;
- the multiplicity-adjusted one-sided upper confidence bound for the paired wall-time ratio is
  strictly below `1.0`;
- target eligibility and the combined policy retain a lower-bound target fraction of at least
  `0.50` and truth-edge fraction of at least `0.90`, while every dataset and every
  simulator-condition mean independently meet the same observed guardrails; and
- adaptive trees alone retain the exact target and predictor universes, while the combined arm
  reuses exactly the target set produced by eligibility alone.

These confirmatory gates nominate at most one provisional candidate; they do not yet authorize the
published production policy. If no formal candidate passes, the exact arm is retained and no
safety run is needed. Otherwise, that sole candidate and the exact arm enter a separate pre-freeze
descriptive safety veto. No failed candidate can be replaced by the next formal runner-up.

Development can also produce no finalist for any component. In that case the held-out efficiency
step records a reference-only execution audit, performs no non-inferiority test or multiplicity
adjustment, freezes the exact calibrated policy, and allows the remaining phases to continue.

The safety panel contains three independently generated scMultiSim datasets in each of six
protected conditions:

- a balanced moderate problem with 3,000 genes, 100 TFs, 300 cells and groups 100/100/100;
- the 100-gene, 10-TF, 500-cell single-group branch;
- the same small feature space with 300 cells split 10/40/250;
- the 100-gene, 10-TF high-cell case with 10,000 cells split 3,333/3,333/3,334;
- the 3,000-gene, 1,500-TF dimensional endpoint with 300 cells split 100/100/100; and
- the 15,000-gene, 100-TF dimensional endpoint with 300 cells split 100/100/100.

All 18 generation seeds are disjoint from every other phase. The two added dimensional endpoints
each use their own topology, and none of the four distinct safety topology files is reused outside
this panel. The four safety topologies share no directed edge with one another or with any
dimensional-transfer topology. Exact and provisional arms
use one common algorithm seed within and across these datasets, and one independent execution-order
seed counterbalances their frozen serial order. The independent unit remains the generated dataset,
not a group, edge or cell.

Safety has no p-values, confidence intervals or second optimization opportunity. Authorization
requires all 18 pairs and all three pairs per condition to complete; candidate-minus-exact AUPR
mean and median of at least `-0.002` globally and in every condition; no individual dataset below
`-0.01`; every dataset to retain its candidate-specific target and truth-edge coverage threshold;
and a global median candidate/exact wall-time ratio below `1.0`. Any failed gate or attributable
candidate execution retains exact. A reference/exact failure makes the safety contrast
uninterpretable and aborts it rather than favouring the candidate.

The `0.002` absolute AUPR margin is the largest mean accuracy loss the authors accept for a purely
computational optimization. The `0.50` target and `0.90` truth-edge coverage thresholds are likewise
author-defined engineering safeguards. None is an empirically validated biological-equivalence
threshold. Therefore any positive full-budget decision is conditional on this predeclared
synthetic panel, its fixed algorithm-seed ensemble and these explicit value judgements; it does not
establish equivalence on real data without known ground truth.

The three candidate decisions use an intersection-union rule: accuracy, coverage, completion and
runtime must all pass. Bonferroni allocates the family-wise `alpha=0.05` across the three possible
candidate adoptions; mandatory gates within one candidate are not treated as independent claim
opportunities. Confirmatory bounds use a 20,000-resample fixed-stratum hierarchical bootstrap.
Algorithmic seeds are averaged within dataset; datasets are weighted equally within condition,
conditions within simulator, and simulators within the panel. The full design has 25 independent
datasets but an effective sample size of 16 under these unequal hierarchical weights. This is a
declared design property, not a power guarantee. Runs are serial and candidate order is
deterministically counterbalanced by dataset and seed so no arm always occupies the same temporal
position.

Execution is resumable only at the frozen-order prefix boundary. A crash preserves every completed
prefix member and the next invocation executes only the missing suffix, still serially and without
deleting valid cache entries. A non-prefix or identity-inconsistent state is rejected as corruption,
not normalized into an apparently valid resume. Evidence, results, production policy and report
are then published atomically so downstream phases cannot observe a partially frozen decision.

The analysis reports paired wall time, peak memory, output size, processed targets, requested
models and fitted trees. Runtime is formed as a paired log ratio within algorithm seed, averaged
within dataset, and exponentiated after hierarchical aggregation; hence the endpoint is a
hierarchically weighted geometric-mean candidate/exact ratio. The formal full analysis nominates
the passing candidate with the lowest observed geometric-mean runtime ratio, breaking ties by
candidate ID. This tie-break does not claim a significant speed difference between passing
candidates. The safety veto then either authorizes that one candidate or retains exact. The reduced
workflow reports descriptive estimates only, freezes exact and cannot authorize adoption.

The failure rule is deliberately conservative. Any formally bound execution failure in a
candidate arm invalidates the global relative selection and retains the exact policy; the workflow
does not choose opportunistically among only the surviving candidates. A failure in the
exact/reference arm makes every paired candidate contrast uninterpretable and aborts the decision.

Before full outcomes are inspected,
`experiments/v1/spathi/efficiency-design-sensitivity.json` freezes an outcome-free,
deterministic normal-location scenario grid for AUPR differences and runtime log ratios. It makes
the assumed variance dependence and multiplicity penalty reproducible. Its probabilities are
assumption-conditional operating characteristics, not empirical power estimates, and they do not
validate percentile-bootstrap coverage with five replicates per stratum.

Evidence is reconstructed from immutable artifacts rather than trusted from self-declared result
fields. It verifies SPATHI identity, input fingerprints, the absence of truth-bearing inference
inputs, context-specific target eligibility, complete directed TF-by-target candidate-space counts
minus self-edges, and the original simulator truth.

## Phase 3: post-freeze dimensional transfer

The dimensional-transfer phase asks a deliberately narrower question than either efficiency or
scalability: how the already frozen production policy's synthetic accuracy behaves as one input
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

If efficiency freezes a policy different from the calibrated exact arm, the exact arm is run only
as a diagnostic at the two scale anchors (`g3000_tf100_c300` and `g100_tf10_c300`). This checks
the local policy contrast without reopening selection. It does not make either expensive endpoint
an exact-versus-production experiment and cannot justify adoption outside the formal efficiency
panel.

## Phase 4: held-out external comparison

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

- frozen production SPATHI;
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

## Phase 5: public synthetic scalability

The scalability phase uses a rounded, author-defined grid declared independently of any private or
external dataset. The full grid spans 1,000, 5,000 and 10,000 cells; 5,000, 20,000 and 25,000
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
Those values are public scenario definitions rather than estimates from external observations.

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

Each shape runs separately under the same frozen production policy. The recorded endpoints are
generation, ANDREA, SPATHI and harness wall time; peak RSS sampled from the SPATHI container process
tree; expression, dataset, inference and SPATHI-output bytes; target eligibility counts; model
completion; and fitted-tree count. Inference must be fresh for resource attribution. Once a
measurement exists it can be resumed and verified, but a bare inference cache hit without its RSS
measurement is not accepted as scale evidence. The sampler also rejects a concurrent container
using the same SPATHI image, because peak RSS would no longer be attributable to one run.

An execution failure is retained as a scalability outcome only when ANDREA has written a failed
report that matches the exact requested dataset, run and SPATHI origin. The harness classifies an
OOM, timeout, tool error or distinguishable infrastructure error from that validated evidence and
records the configured limits, elapsed time, any available RSS samples, the last valid progress
snapshot, diagnostic log tail and a content-addressed command log. The failed dataset is never
labelled completed and contributes no model-completion or accuracy observation. Later shapes may
continue, and the panel may publish with `all_completed=false` so a computational boundary remains
visible rather than disappearing through complete-case filtering.

This exception is intentionally narrow. A missing or ambiguous failure report, dataset/run/tool
identity mismatch, corrupted artifact, invalid progress contract or inconsistent provenance is an
infrastructure or scientific-contract failure and aborts the workflow. It is not transformed into
a publishable endpoint. No inference wall-time limit is implied unless one is explicitly recorded;
a timeout can be classified only when the validated execution evidence distinguishes it.

## Reproducibility contract

`experiments/v1/seeds/seed-plan.json` is the authoritative seed ledger. It uses master seed
`20260905` and SHA-256 domain-separated derivation to produce portable positive 31-bit seeds from:

- budget (`smoke`, `reduced` or `full`);
- protected dataset role;
- stochastic stream; and
- semantic coordinates such as simulator run, replicate, method, stage or dataset ID.

The streams are `generation`, `augmentation`, `inference`, `statistical` and `execution_order`.
Seed derivation is independent of manifest ordering, so adding or reordering an unrelated task
cannot change existing seeds. Contiguous simulator seed blocks are allocated explicitly; other
task seeds are derived directly and stored with their coordinates.

Only the full held-out comparison and full efficiency study allocate the `statistical` stream.
Smoke is contract-only and every reduced analysis is descriptive: their contracts omit alpha,
permutation and bootstrap counts, multiplicity rules and statistical seeds.

The reproducibility audit verifies every committed simulator allocation, inference request and
scale-stress seed derivation. It also validates and hashes the prospective efficiency
design-sensitivity contract and its deterministic operating-characteristic output. It rejects
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

Efficiency and scalability use a two-phase report publication contract. `workflow.json` is first
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
- full efficiency non-inferiority and cost gates: confirmatory production-policy decision;
- post-freeze dimensional transfer: publication-descriptive synthetic accuracy along separate
  cell, gene and TF axes, without policy selection or a joint-scale accuracy claim;
- invariant edge stability, topology, order-aware and pySCENIC implementation-sensitivity analyses:
  exploratory;
- all-gene-inference directed and topology families: exploratory unequal-inference-scope analyses;
- reduced efficiency and comparison workflows: descriptive only, without inferential winner or
  adoption claims; and
- public synthetic scalability: computational feasibility and measured resources only.

No test result may change calibration or efficiency policy. No truth-derived prior or lineage rate
may enter a primary inference input. No failure may silently narrow a family. No synthetic
dimensional stress result may be described as biological realism.
