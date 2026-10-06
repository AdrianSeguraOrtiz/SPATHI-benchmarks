# Protocol provenance

Scientific snapshot: `AdrianSeguraOrtiz/SPATHI-benchmarks` at
`9baa6bb8efb8e5b96b74bd9c9702a2c45de1ab75`.

The figure is a visual reconstruction of the preceding approved infographic.
It does not claim to represent a new live run or the latest repository state.
The source material was inspected through the GitHub connector in this conversation.

## Source mapping

- [Committed study design](https://github.com/AdrianSeguraOrtiz/SPATHI-benchmarks/blob/9baa6bb8efb8e5b96b74bd9c9702a2c45de1ab75/experiments/article-study.json):
  Full-profile scenario lists, replications, candidate parameter sets, families,
  routes, resource limits and selection policy.
- [Workflow](https://github.com/AdrianSeguraOrtiz/SPATHI-benchmarks/blob/9baa6bb8efb8e5b96b74bd9c9702a2c45de1ab75/src/spathi_bench/study_workflow.py):
  Calibration dependencies, winner freeze, held-out comparison, isolated
  scalability measurements, thread-point reuse and crash-resumable execution.
- [Executor](https://github.com/AdrianSeguraOrtiz/SPATHI-benchmarks/blob/9baa6bb8efb8e5b96b74bd9c9702a2c45de1ab75/src/spathi_bench/executor.py):
  Exact-prefix evaluation, condition- and panel-weighted selection, evaluation
  metrics, resource outcomes and propagated `not_attempted` decisions.
- [Convergence runner](https://github.com/AdrianSeguraOrtiz/SPATHI-benchmarks/blob/9baa6bb8efb8e5b96b74bd9c9702a2c45de1ab75/containers/spathi/run_convergence.py):
  One shared physical forest job with four exact logical prefixes; no independent
  runtime attribution to each prefix.
- [Paired statistics](https://github.com/AdrianSeguraOrtiz/SPATHI-benchmarks/blob/9baa6bb8efb8e5b96b74bd9c9702a2c45de1ab75/src/spathi_bench/article_statistics.py):
  Complete blocks, Friedman, two-sided all-pairs Wilcoxon and Holm correction.
- [Repository README](https://github.com/AdrianSeguraOrtiz/SPATHI-benchmarks/blob/9baa6bb8efb8e5b96b74bd9c9702a2c45de1ab75/README.md):
  Scientific scope, information contracts, outcome interpretation and deliverables.

Organizational reference for this directory:
[SPATHI's graphical abstract directory](https://github.com/AdrianSeguraOrtiz/SPATHI/tree/main/docs/graphical_abstract).
This packaging example is not the scientific source of the benchmark.

## Counts displayed in the approved figure

- **74 seeded synthetic datasets** = 14 calibration-development + 14 independent
  moderate-validation + 2 calibration-transfer + 32 held-out comparison + 12
  size-ladder datasets. Thread scaling reuses the matching medium ladder data.
- **554 calibration fits** = 378 screening jobs (27 × 7 × 2) + 168 validation
  jobs (4 × 7 × 2 × 3) + 8 transfer jobs (4 × 2 × 1 × 1).
  These are network-level fitting jobs. They are not individual target-gene
  forests and are not the number of logical exact-prefix outputs.
- **32 held-out instances** = three seeds for each of ten primary scenarios,
  plus one instance of each of the two deployment-transfer scenarios.
- **16 routes** correspond to 15 distinct tool names; scMTNI appears under two
  separate information contracts. Families are overlapping analysis groups,
  not five additional executions of SPATHI.
- **492 eligible route × dataset requests** are prospective requests before
  empirical resource-admission and completion outcomes: 288 for the six
  three-replicate primary scenarios admitting all 16 routes, 90 for the two
  branching scenarios admitting 15, 84 for the two invariant controls admitting
  14, and 16 + 14 for the two deployment instances.
- **192 planned ladder measurements** = four scales × three replicates × 16
  routes. These are planned slots, not a promise of 192 completed physical
  executions. After non-completion, blocked subsequent slots are not attempted.

## Interpretive boundaries

All scatter points, forest symbols, grids and metric icons are schematic.
Neither their coordinates nor their size/color differences encode experimental
scores. Increasing tree symbols illustrate forest-size evaluation, not a claim
that the selected tree-count prefixes have different maximum depths.

The finalists' exact 25/50/100/250-tree results share the corresponding
250-tree fitting job. The full inference still contains separate models for
groups and target genes. The final selection averages conditions before giving
50% weight to moderate validation and 50% to transfer. The 2% band is relative
to the best score, followed by the declared stability and tree-count rule; it
is not a statistical non-inferiority test.

The figure's primary statistical pathway uses only complete paired primary
blocks. Deployment-scale data are descriptive. The order-aware and
lineage-aware routes are admitted only when their actual input contracts are
available. AUPR, AUROC and F1-at-truth-count do not imply signed or causal edges.

The standard thread point is `min(8, available cores)`. The all-core request is
capped by each route's declared maximum. Coincident effective points are not
new measurements. The medium standard-thread ladder measurement is reused.
The 4-hour limit applies per physical scalability task; the full comparison
uses its separately declared 2-hour physical limit. Completion with warnings
can advance; a genuine non-completion blocks that route's remaining scout chain.
