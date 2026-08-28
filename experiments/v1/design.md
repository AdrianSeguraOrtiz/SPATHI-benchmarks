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
remain in a separately labelled all-gene exploratory stratum; they are never pooled with
the primary ranking. `inference/catalog-matrix.json` records this decision for every
tool in the locked ANDREA catalog.

Metrics are computed per group against the simulator's group-specific truth. Summary
means first average contexts within each dataset and then give datasets equal weight, so
a simulator that emits more groups cannot dominate. Directed metrics are primary;
topology metrics are also retained because some catalog tools are undirected. Empty
inferred networks remain valid zero-edge predictions.

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
controls: groups or trajectories may vary while the true network does not, testing
whether a method invents reconfiguration.

scMultiSim and SERGIO conditions must not all share the same bundled topology. At least
one scMultiSim panel will use an independently supplied network. scMultiSim is not part
of the initial smoke because its image reconstruction still needs a separately verified,
pinned build; this does not affect the dyngen smoke.

## Profiles

### smoke (runnable)

- dyngen, one seed, 30 cells and a small gene universe;
- GENIE3 `group_emulated` and SPATHI `group_native`;
- 25 trees per method;
- full generation → inference → handoff validation → evaluation path.

This profile checks contracts, not statistical conclusions.
At the locked seed, dyngen creates 42 genes, 35 TF candidates and seven groups, including
two groups with only two cells. That deliberately tiny and unrepresentative geometry is
acceptable only for this contract smoke; it must not be reused as scientific evidence.

### reduced (planned)

- dyngen and scMultiSim;
- three independent seeds per condition;
- `group_only`, `order_aware` and `pseudotime_aware` regimes;
- every eligible catalog route is accounted for by the frozen catalog matrix, with
  TF-restricted and all-gene strata kept separate;
- moderate gene/cell counts suitable for development and regression testing.

### full (planned)

- dynamic benchmarks plus SERGIO, BoolODE and GeneSPIDER2 controls;
- at least five independent seeds per condition;
- topology, noise, dropout, imbalance, small-group and rewiring factors;
- all four information regimes, negative controls and SPATHI ablations.

The reduced and full profiles stay `planned` until every referenced simulator/tool
combination passes ANDREA preflight and its exact request JSON is committed. A profile
is never silently narrowed after failures: included, blocked and failed combinations
must remain visible in the final provenance.

## Reproducibility and handoff

`repos.lock.json` fixes the ANDREA and SPATHI revisions. `images.lock.json` fixes the
SPATHI base image, source revision, Python dependencies and adapter-file SHA-256
digests. A run records its concrete local image ID in `state.json`.

The benchmark consumes paths from `benchmark-manifest.json`; it never selects outputs
with timestamp sorting. SPATHI runs through ANDREA's external-tool contract in the same
plan as catalog methods. Consequently, ANDREA owns `merged_network_raw.csv` and
`run_report.json`; the latter is passed unchanged to `evaluate-inference`.
