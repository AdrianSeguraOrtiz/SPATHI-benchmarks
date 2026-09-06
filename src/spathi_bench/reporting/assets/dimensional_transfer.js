(function () {
  "use strict";

  const R = window.SpathiReport;
  const {
    PALETTE,
    addStatCards,
    append,
    boot,
    createTable,
    csvButton,
    element,
    formatInteger,
    formatNumber,
    formatSeconds,
    humanize,
    notice,
    plot,
    query,
    renderHeader,
    renderLinks,
  } = R;

  function conditionLabel(value) {
    return value.replace(/^g/, "").replace("_tf", " genes · ").replace("_c", " TF · ") + " cells";
  }

  function renderScope(data) {
    notice(query("#scope-notice"), data.scope.interpretation, "warning", "Bounded descriptive scope");
    addStatCards(query("#overview-cards"), [
      { label: "Completed datasets", value: `${formatInteger(data.summary.datasets_completed)} / ${formatInteger(data.summary.datasets_requested)}`, detail: "All declared truth-bearing datasets" },
      { label: "Group contexts", value: formatInteger(data.summary.contexts_evaluated), detail: "Directed group-specific evaluations" },
      { label: "Frozen algorithm seed", value: formatInteger(data.algorithm_seed.value), detail: "One policy; generation provides replication" },
      { label: "Claim", value: "Descriptive transfer", detail: "No ranking, selection, or non-inferiority test" },
    ]);
  }

  function renderDimensions(rows) {
    const maximumTf = Math.max(...rows.map((row) => row.transcription_factors));
    plot(query("#shape-chart"), [{
      type: "scatter",
      mode: "markers+text",
      x: rows.map((row) => row.cells),
      y: rows.map((row) => row.genes),
      text: rows.map((row) => conditionLabel(row.id)),
      textposition: "top center",
      customdata: rows.map((row) => [row.transcription_factors, row.adjusted_aupr, row.replicates]),
      marker: {
        size: rows.map((row) => 14 + 28 * Math.sqrt(row.transcription_factors / maximumTf)),
        color: rows.map((row) => row.adjusted_aupr),
        coloraxis: "coloraxis",
        opacity: 0.88,
        line: { color: "#ffffff", width: 1 },
      },
      hovertemplate: "<b>%{text}</b><br>Cells: %{x:,}<br>Genes: %{y:,}<br>TFs: %{customdata[0]:,}<br>Adjusted AUPR: %{customdata[1]:.4f}<br>Datasets: %{customdata[2]}<extra></extra>",
    }], {
      xaxis: { title: "Cells (count, log scale)", type: "log" },
      yaxis: { title: "Genes (count, log scale)", type: "log" },
      coloraxis: {
        colorscale: [[0, "#dbeafe"], [1, "#1d4ed8"]],
        colorbar: { title: { text: "Adjusted<br>AUPR" } },
      },
      showlegend: false,
    });
  }

  function renderAccuracy(rows) {
    const labels = rows.map((row) => conditionLabel(row.id));
    plot(query("#accuracy-chart"), [
      { type: "bar", name: "Raw AUPR", x: labels, y: rows.map((row) => row.directed_aupr), marker: { color: "#93c5fd" }, hovertemplate: "%{x}<br>Raw AUPR: %{y:.4f}<extra></extra>" },
      { type: "bar", name: "Adjusted AUPR", x: labels, y: rows.map((row) => row.adjusted_aupr), marker: { color: "#1d4ed8" }, hovertemplate: "%{x}<br>Adjusted AUPR: %{y:.4f}<extra></extra>" },
    ], {
      barmode: "group",
      xaxis: { title: "Dimensional-transfer condition", tickangle: -20 },
      yaxis: { title: "AUPR", rangemode: "tozero" },
      legend: { orientation: "h", y: -0.35 },
    });
  }

  function renderConditions(rows) {
    const container = query("#condition-table");
    createTable(container, [
      { label: "Condition", value: (row) => row.id },
      { label: "Role", value: (row) => humanize(row.analysis_role) },
      { label: "Genes", value: (row) => row.genes, format: formatInteger, numeric: true },
      { label: "TFs", value: (row) => row.transcription_factors, format: formatInteger, numeric: true },
      { label: "Cells", value: (row) => row.cells, format: formatInteger, numeric: true },
      { label: "Datasets", value: (row) => row.replicates, format: formatInteger, numeric: true },
      { label: "Raw AUPR", value: (row) => row.directed_aupr, format: (value) => formatNumber(value, { maximumFractionDigits: 5 }), numeric: true },
      { label: "Adjusted AUPR", value: (row) => row.adjusted_aupr, format: (value) => formatNumber(value, { maximumFractionDigits: 5 }), numeric: true },
      { label: "AUROC", value: (row) => row.directed_auroc, format: (value) => formatNumber(value, { maximumFractionDigits: 5 }), numeric: true },
      { label: "F1 @ truth count", value: (row) => row.f1_at_truth_count, format: (value) => formatNumber(value, { maximumFractionDigits: 5 }), numeric: true },
      { label: "Median wall time", value: (row) => row.median_wall_time_seconds, format: formatSeconds, numeric: true },
    ], rows);
    query("#condition-table-actions").appendChild(csvButton("Download condition summary", rows, "dimensional-transfer-conditions.csv"));
  }

  function renderContrasts(rows) {
    if (rows.length === 0) {
      R.empty(query("#contrast-chart"), "No scale contrast in the reduced budget", "Run the full budget to estimate the three descriptive one-axis scale contrasts.");
      return;
    }
    plot(query("#contrast-chart"), [{
      type: "bar",
      x: rows.map((row) => humanize(row.id.replace("_scale_contrast", ""))),
      y: rows.map((row) => row.estimate),
      marker: { color: rows.map((row) => row.estimate >= 0 ? "#2563eb" : "#7f1d1d") },
      hovertemplate: "%{x}<br>Endpoint − anchor: %{y:+.5f}<extra></extra>",
    }], {
      xaxis: { title: "Expanded dimension" },
      yaxis: { title: "Difference in mean adjusted AUPR", zeroline: true, zerolinewidth: 2 },
      showlegend: false,
    });
  }

  function renderRuntime(rows) {
    plot(query("#runtime-chart"), [{
      type: "bar",
      x: rows.map((row) => conditionLabel(row.id)),
      y: rows.map((row) => row.median_wall_time_seconds),
      marker: { color: "#2563eb" },
      hovertemplate: "%{x}<br>Median SPATHI wall time: %{y:.2f} s<extra></extra>",
    }], {
      xaxis: { title: "Dimensional-transfer condition", tickangle: -20 },
      yaxis: { title: "Wall time (seconds)", rangemode: "tozero" },
      showlegend: false,
    });
  }

  function renderDatasets(rows) {
    createTable(query("#dataset-table"), [
      { label: "Dataset", value: (row) => row.dataset_id },
      { label: "Condition", value: (row) => row.condition_id },
      { label: "Generation seed", value: (row) => row.generation_seed, format: formatInteger, numeric: true },
      { label: "Adjusted AUPR", value: (row) => row.adjusted_aupr, format: (value) => formatNumber(value, { maximumFractionDigits: 5 }), numeric: true },
      { label: "Candidate prevalence", value: (row) => row.candidate_prevalence, format: (value) => formatNumber(value, { maximumFractionDigits: 6 }), numeric: true },
      { label: "Wall time", value: (row) => row.wall_time_seconds, format: formatSeconds, numeric: true },
    ], rows);
  }

  function renderEndpointAdmission(feasibility) {
    const resources = feasibility.resource_ceiling;
    addStatCards(query("#admission-summary"), [
      { label: "CPU ceiling", value: `${formatInteger(resources.max_cores)} logical cores`, detail: "Global v1 execution contract" },
      { label: "RAM ceiling", value: `${formatInteger(resources.max_ram_gb)} GB`, detail: "Global v1 execution contract" },
      { label: "Execution", value: resources.serial_execution ? "Serial" : "Concurrent", detail: "Resource attribution mode" },
    ]);
    append(
      query("#endpoint-admission"),
      element("h3", {}, "Admission rule"),
      element("p", {}, feasibility.endpoint_admission),
      element("h3", {}, "Joint-dimension rule"),
      element("p", {}, feasibility.joint_condition_rule),
    );
  }

  function renderPolicy(data) {
    const policy = data.production.policy;
    addStatCards(query("#policy-summary"), [
      { label: "Policy", value: policy.policy_id, detail: `SHA-256 ${policy.policy_sha256.slice(0, 12)}…` },
      { label: "Budget", value: humanize(data.production.budget), detail: humanize(data.production.evidence_role) },
      { label: "Selection feedback", value: data.design.selection_feedback ? "Allowed" : "Forbidden", detail: "Results cannot alter calibration or efficiency decisions" },
      { label: "Confirmatory ranking", value: data.design.confirmatory_ranking_eligible ? "Eligible" : "Excluded", detail: "Scale contrasts and operational checks are descriptive" },
    ]);
    const limitationList = element("ul", { className: "plain-list" });
    data.limitations.forEach((item) => append(limitationList, element("li", {}, item)));
    append(query("#limitations"), element("h3", {}, "Interpretation limits"), limitationList);
    append(query("#comparison-reuse"), element("h3", {}, "External-comparison separation"), element("p", {}, data.scope.comparison_reuse));
  }

  function renderExactDiagnostic(diagnostic) {
    const status = diagnostic.status;
    addStatCards(query("#exact-summary"), [
      { label: "Status", value: humanize(status), detail: "Post-freeze and descriptive only" },
      { label: "Paired seed", value: formatInteger(diagnostic.paired_algorithm_seed), detail: "Shared by exact and production within a dataset" },
      { label: "Exact reference", value: diagnostic.exact_policy.source_policy_id, detail: `SHA-256 ${diagnostic.exact_policy.policy_sha256.slice(0, 12)}…` },
      { label: "Production", value: diagnostic.production_policy.policy_id, detail: `SHA-256 ${diagnostic.production_policy.policy_sha256.slice(0, 12)}…` },
    ]);
    notice(
      query("#exact-notice"),
      diagnostic.interpretation,
      status === "paired_descriptive" ? "warning" : "success",
      status === "paired_descriptive" ? "No selection feedback" : "Duplicate execution avoided",
    );
    if (status !== "paired_descriptive") {
      R.empty(query("#exact-chart"), "Production is exact", "All policy-identity deltas are exactly zero.");
      createTable(query("#exact-table"), [
        { label: "Metric", value: (row) => row.metric },
        { label: "Production − exact", value: (row) => row.value, format: (value) => formatNumber(value, { maximumFractionDigits: 5 }), numeric: true },
      ], Object.entries(diagnostic.identity_delta).map(([metric, value]) => ({ metric: humanize(metric), value })));
      return;
    }
    const pairs = diagnostic.pairs;
    const labels = pairs.map((row) => `${row.condition_id} · ${row.dataset_id}`);
    plot(query("#exact-chart"), [
      { type: "bar", name: "Adjusted AUPR", x: labels, y: pairs.map((row) => row.delta_production_minus_exact.adjusted_aupr), marker: { color: "#1d4ed8" } },
      { type: "bar", name: "Target coverage", x: labels, y: pairs.map((row) => row.delta_production_minus_exact.target_coverage), marker: { color: "#60a5fa" } },
      { type: "bar", name: "Truth-edge coverage", x: labels, y: pairs.map((row) => row.delta_production_minus_exact.truth_edge_coverage), marker: { color: "#0f766e" } },
    ], {
      barmode: "group",
      xaxis: { title: "Paired dataset", tickangle: -20 },
      yaxis: { title: "Production minus exact", zeroline: true, zerolinewidth: 2 },
      legend: { orientation: "h", y: -0.4 },
    });
    createTable(query("#exact-table"), [
      { label: "Dataset", value: (row) => row.dataset_id },
      { label: "Condition", value: (row) => row.condition_id },
      { label: "Δ adjusted AUPR", value: (row) => row.delta_production_minus_exact.adjusted_aupr, format: (value) => formatNumber(value, { maximumFractionDigits: 5 }), numeric: true },
      { label: "Δ target coverage", value: (row) => row.delta_production_minus_exact.target_coverage, format: (value) => formatNumber(value, { maximumFractionDigits: 5 }), numeric: true },
      { label: "Δ truth-edge coverage", value: (row) => row.delta_production_minus_exact.truth_edge_coverage, format: (value) => formatNumber(value, { maximumFractionDigits: 5 }), numeric: true },
      { label: "Δ wall time", value: (row) => row.delta_production_minus_exact.wall_time_seconds, format: formatSeconds, numeric: true },
      { label: "Δ fitted trees", value: (row) => row.delta_production_minus_exact.fitted_estimators, format: formatInteger, numeric: true },
    ], pairs);
  }

  boot((data) => {
    renderHeader(data, {
      badges: [humanize(data.workflow.profile), humanize(data.workflow.evidence_role), "Frozen policy"],
    });
    renderScope(data);
    renderDimensions(data.conditions);
    renderAccuracy(data.conditions);
    renderConditions(data.conditions);
    renderContrasts(data.contrasts);
    renderExactDiagnostic(data.exact_production_diagnostic);
    renderRuntime(data.conditions);
    renderDatasets(data.datasets);
    renderEndpointAdmission(data.feasibility);
    renderPolicy(data);
    renderLinks(query("#report-links"), data.links, "Artifacts and reusable ANDREA ZIPs");
  });
})();
