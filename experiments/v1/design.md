# SPATHI benchmark v1

## Question

Does SPATHI recover group-specific regulatory networks more accurately than methods
given the same expression matrix, group labels and common TF candidate set? As a
secondary question, how does it compare with methods that receive extra lineage,
pseudotime or prior information that SPATHI deliberately ignores?

## Primary estimand

The primary comparison is the TF-restricted stratum of `group_only`. Every method in
that stratum has access only to:

- the same underlying expression dataset;
- the group assignment for every cell; and
- the same simulator-generated `tf_list`.

`group_native` methods such as SPATHI can use all cells while producing every group
network; `group_emulated` methods receive one group subset at a time. This is the
methodological contrast being tested, not a claim that their physical input matrices
are identical.

No method receives a regulatory prior, true edges, lineage order or pseudotime in the
primary ranking. Catalog methods that cannot restrict candidate regulators to `tf_list`
are classified as potential all-gene exploratory methods in
`inference/catalog-matrix.json`, but v1 does not execute or statistically analyse that
stratum. They are never pooled with the primary ranking.

Metrics are computed per group against the simulator's group-specific truth. Summary
means first average contexts within each dataset and then give datasets equal weight, so
a simulator that emits more groups cannot dominate. Directed evaluations are emphasized;
topology metrics are also retained because some catalog tools are undirected. Empty
inferred networks remain valid zero-edge predictions.

The confirmatory endpoint is directed group-level AUPR in the TF-restricted
`group_only` stratum. AUROC, F1 at the truth-edge count, EPR at the truth-edge count and
topology analyses are explicitly exploratory unless a later version predeclares a joint
multiplicity family before generating test data.

## Statistical inference

The unit of replication is an independently simulated dataset. Group contexts are
averaged equally within an execution, and repeated algorithmic seeds are then averaged
within a configuration and dataset. Cells, edges, groups and algorithmic seeds cannot
increase the inferential sample size. Conditions derived from a shared simulated object
must either be one block or remain in separately predeclared strata.

Role, regime and level are mandatory strata; a profile may additionally separate split,
condition, simulator and generation job, or combine independent simulators in one panel.
Within any resulting stratum, distinct dataset IDs sharing the same simulator seed are
rejected as a duplicated simulation unit. Each configuration
ID has one immutable parameter hash, repeated seed identities are rejected, and each
paired context must expose the same ANDREA candidate-universe counts. Different tools may
use different numbers of algorithmic repetitions—including one deterministic run—because
those repetitions are averaged and never increase the number of dataset blocks.

Each profile freezes exact statistical families alongside its simulation and inference
requests. A family identifies the control and competitor configuration IDs, dataset
role, information regime, network levels, metrics, stratification and minimum block
count. Confirmatory test profiles may contain only the globally frozen SPATHI
configuration; calibration candidates and ablations cannot enter that ranking.

Inference requires a complete paired panel and identical group-context sets for every
configuration and algorithmic execution. No missing metric is imputed, and a failed or
non-evaluable member invalidates the confirmatory family instead of reducing it by
pairwise deletion. Empty inferred networks that ANDREA evaluates numerically remain
valid observations.

With three or more tools, the omnibus comparison is a tie-corrected Friedman test over
dataset blocks. Its p-value comes from exact label permutations when feasible and from a
predeclared seeded Monte Carlo permutation otherwise; Kendall's W accompanies the
p-value. The post-hoc family consists of two-sided paired Wilcoxon signed-rank contrasts
between SPATHI and each predeclared rival, followed by Holm step-down correction. With
two tools, only the paired contrast applies. Average ranks are descriptive: SPATHI is
called a winner only when it is uniquely top-ranked and every required directional
contrast remains significant after correction. Otherwise the report states that it is
top-ranked but not statistically distinguishable, or that it is not top-ranked.

P-values are accompanied by median and Hodges–Lehmann paired differences,
rank-biserial effects, win/tie/loss counts and paired dataset-level bootstrap intervals
at confidence level `1 - alpha`.
The smoke emits `not_testable` because it has one dataset. The final full-panel replicate
count must be justified by simulation-based power for its predeclared Holm family; the
nominal five-seed design is not itself a guarantee of adequate power.

## Information regimes

The four regime files under `inference/regimes/` are labels for information access, not
necessarily disjoint tool families.

| Regime | Additional information | Role |
| --- | --- | --- |
| `group_only` | none beyond expression, groups and TF list | primary ranking |
| `order_aware` | lineage or ordered group phenotype | secondary comparison |
| `pseudotime_aware` | cell-level pseudotime | secondary comparison |
| `prior_assisted` | regulatory prior | assisted upper bound only |

Important contract corrections:

- scMTNI cannot yet enter a clean `order_aware` comparison. Its current multigroup mode
  needs `lineage_tree`, whose simulator-produced gain/loss rates are computed from true
  group GRNs. The native input is therefore leaky. scMTNI remains blocked until a
  justified neutral-rate lineage dataset is derived and assigned a new fingerprint.
- Both SimiC variants are `order_aware`, including `similarity=false`, because SimiC
  always consumes ordered column phenotypes.
- inferCSN with `sift_method=none` is a `group_only` control; `entropy` is
  `pseudotime_aware`.
- scRegulate has no expression-only mode in its locked ToolSpec. It appears only in
  `prior_assisted` with `prior_source=provided_prior`.
- Inferelator3 and any run using a truth-derived prior are never included in the
  primary ranking. Group-specific oracle priors and truth-derived lineage rates are
  excluded even from the global-prior assisted stratum.

An order-permutation control must be a newly derived dataset with a newly computed
fingerprint. Mutating an extra file underneath an existing dataset manifest is forbidden.
SimiC comparisons use linear trajectories or a separately identified branch path;
silently linearizing a branching trajectory into one total order is forbidden.

## Simulation panel

The principal dynamic benchmarks are dyngen and scMultiSim because they can provide
group-varying regulatory truth. SERGIO, BoolODE and GeneSPIDER2 are invariant-network
controls: groups or trajectories may vary while the true network does not. Their v1
role is to measure ordinary group-level inference accuracy against invariant truth;
an explicit cross-group reconfiguration metric is not yet implemented.

scMultiSim uses `grn_source=input_tsv` with three committed 100-edge regulatory networks:
one each for development, validation and test. Their contents and seeds are disjoint, so
no bundled or external topology crosses a protected partition. Its image and the images
for every test control are pinned by repository digest. SERGIO uses its own preset and is
included only as an invariant control, not as the source of a scMultiSim topology.

## Profiles

### smoke (runnable)

- dyngen, one seed, 30 cells and a small gene universe;
- GENIE3 `group_emulated` and SPATHI `group_native`;
- 25 trees per method;
- full generation → independently cached inference → handoff validation → independently
  cached joint projection → individual and joint evaluation → official ANDREA analysis
  exports.

This profile checks contracts, not statistical conclusions.
At the locked seed, dyngen creates 42 genes, 35 TF candidates and seven groups, including
two groups with only two cells. That deliberately tiny and unrepresentative geometry is
acceptable only for this contract smoke; it must not be reused as scientific evidence.

### reduced (planned)

- twelve dyngen and three scMultiSim development datasets;
- three independent seeds per condition and role-specific scMultiSim topology;
- a `group_only` regression among selected SPATHI, GENIE3, GRNBoost2 and
  inferCSN-without-pseudotime;
- moderate gene/cell counts suitable for development and regression testing.

### full (planned)

- 50 independently seeded test datasets: 20 dyngen, ten scMultiSim, five SERGIO,
  ten BoolODE and five GeneSPIDER2;
- dyngen topology/noise conditions spanning 200--500 cells and 70--130 genes;
- scMultiSim low-rewiring balanced populations versus high-rewiring, UMI-noisy
  populations with one predeclared 12-cell minority group;
- clean versus dropout-processed BoolODE invariant-network controls;
- SERGIO, BoolODE and GeneSPIDER2 fixed-network controls;
- `group_only`, `order_aware` and `pseudotime_aware`, plus negative controls;
- `prior_assisted` remains explicitly blocked until a separately fingerprinted,
  non-leaky prior dataset exists.

The `reduced` and `full` files are non-runnable dataset templates. Their catalog requests
and exact dataset selectors are committed under `inference/`; after validation freezes
one SPATHI configuration, the comparison workflow materializes `reduced-v1` or `full-v1`
below `work/` with the selected custom-tool requests. A profile is never silently narrowed after failures:
included, blocked and failed combinations remain visible in provenance.

The full TF-restricted directed family contains selected SPATHI, GENIE3, GRNBoost2,
inferCSN without pseudotime, pySCENIC-GRN, scMINER with `custom_tf_list` and TIGRESS.
ARACNe3 joins only the exploratory `topology` family because its catalog output is
undirected. The order-aware family contains SPATHI and both SimiC similarity variants on
the two linear dyngen jobs. The pseudotime-aware family contains SPATHI and paired
inferCSN `none`/`entropy` variants on dyngen, scMultiSim and BoolODE jobs that provide
pseudotime. Batch and statistical selectors prevent these methods from leaking onto
incompatible simulations.

The requested SimiC order-permutation negative control is not expressible safely in the
current contracts. ANDREA's SimiC wrapper consumes the simulator-emitted
`column_phenotypes.tsv`; neither `generate-data` nor `infer-network` accepts an order
override, and SPATHI-benchmarks has no fingerprinted dataset-derivation stage. Mutating
that file under an existing dataset would violate dataset identity. The control therefore
remains explicitly blocked until ANDREA exposes a derived-dataset contract or this
repository gains a provenance-preserving derivation stage; no fake simulator parameter is
used in its place.

## Calibration and data partitions

The smoke configuration is a contract check only. Scientific SPATHI configuration uses
three bounded development stages declared in `spathi/search-space.json`: a tree-count
sweep (historically identified as `convergence`), one-factor structural variants and a
small estimator shortlist. The first stage tunes `n_estimators`; it is not interpreted as
a formal plateau or one-standard-error convergence test. The `full` and `reduced`
budgets use exactly the same candidates, objective and stage order. Every candidate is
repeated over the algorithmic seeds predeclared for its budget. The primary objective is
directed AUPR, averaged equally over groups, then seeds, then datasets. Ties are resolved
by stability, wall time, peak memory when measured, and distance from the reference
configuration.

Dataset instances—not cells—are assigned once to mutually exclusive `development`,
`validation` and `test` roles, stratified by simulator, topology and condition. Related
datasets stay together. Search stages see development only; validation chooses one
global finalist; test compares the frozen choice with external methods once and can
never alter it.

The frozen v1 panel is:

| Role | dyngen | scMultiSim | invariant controls | Total |
| --- | ---: | ---: | ---: | ---: |
| development | 4 conditions × 3 | 1 topology × 3 | 0 | 15 |
| validation | 4 conditions × 3 | 1 topology × 3 | 0 | 15 |
| test | 4 conditions × 5 | 1 topology × 2 stress conditions × 5 | SERGIO/GeneSPIDER2 × 5 plus BoolODE × 2 conditions × 5 | 50 |

Full calibration uses all 15 development and all 15 validation datasets. Reduced
calibration uses one replicate from each of the four dyngen conditions plus one
scMultiSim dataset in each role (five per role), and two algorithmic seeds in every
stage. Its maximum execution count is 240 rather than 1,200. It is intended for rapid
operational feedback; only the full selection followed by the held-out full comparison
supports the primary scientific analysis.

`scripts/calibrate.sh --profile full|reduced` materializes every executable stage
profile below `work/`. Later
profiles appear only after the preceding content-linked ranking has been validated. The
profiles, request bundles, rankings and frozen selection are copied into each run's
immutable input snapshot, so adaptive calibration needs no intermediate Git commits.
The generic Friedman/Holm phase is marked not applicable to calibration and selection
profiles; their authoritative selector is `calibration-rank`.
The frozen selection, manifests, ranking chain and workflow path all record the chosen
budget. `compare.sh` accepts only the calibration with the same profile, preventing a
reduced selection from being presented as a full result.

This separation is enforced structurally:

- each runnable profile purpose maps to exactly one dataset role;
- every inference batch names that role and, where needed, an exact compatible
  `generation_jobs` subset; statistical families use the same selectors;
- a runnable test profile requires an evidenced, frozen `selected-config.json`;
- a SPATHI test batch may contain only `selection_role=selected`;
- summaries never aggregate across role, split or condition; and
- calibration ranking rejects missing candidates, unbalanced seed/dataset panels and
  every row outside its expected development or validation role;
- effective simulator tasks are content-fingerprinted globally, and identical tasks or
  shared external inputs cannot cross development, validation and test; and
- every ranking embeds auditable rows, exact execution hashes and SPATHI build provenance;
  parent rankings are linked by content hash and reconciled against each child shortlist;
  and a frozen winner recursively validates the complete chain back to convergence.

The high-level calibration workflow materializes deterministic ANDREA request bundles,
proposes bounded shortlists and freezes the validation winner. Low-level planning,
ranking and freezing commands remain available for diagnosis. Derived files are
content-addressed runtime evidence; version control remains reserved for the scientific
design, source inputs and schemas.

## Reproducibility and handoff

`repos.lock.json` fixes the ANDREA and SPATHI revisions. `images.lock.json` fixes catalog
images by repository digest and fixes the SPATHI base image, source revision, Python
dependencies and adapter-file SHA-256 digests. A run records its concrete local SPATHI
image ID and fingerprints the Python environment used to execute ANDREA. A changed
orchestration environment therefore cannot silently reuse an old artifact.

The benchmark consumes paths from `benchmark-manifest.json`; it never selects outputs
with timestamp sorting. SPATHI runs through ANDREA's external-tool contract. Catalog and
SPATHI runs are submitted separately so each dataset/tool/configuration has an immutable,
content-addressed result and can be invalidated independently. ANDREA still owns
`merged_network_raw.csv` and `run_report.json`; each single-tool report is passed
unchanged to `evaluate-inference`.

The shared artifact store is keyed by the relevant simulator request and image,
ANDREA revision/environment, dataset artifact, tool parameters, algorithmic seed and
concrete tool image. Changing SPATHI invalidates only SPATHI inference and its downstream
evaluation; unchanged simulated data and catalog baselines remain reusable. A joint
inference view is keyed by the ordered source-artifact identities within one dataset and
information regime, so a source-artifact change invalidates only that view, its joint
evaluation and dependent analysis ZIPs. The view preserves source network rows, records
source hashes and never claims to be a monolithic execution. Both the projection and
analysis ZIP keys bind the SPATHI-benchmarks producer commit, deliberately invalidating
all such transformations when their implementation revision changes. Cache misses may
execute concurrently within an explicit per-profile resource limit, while state is
published only after the deterministic ordered result set is complete.
