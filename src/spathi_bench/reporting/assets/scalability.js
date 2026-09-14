(function () {
  "use strict";

  const R = window.SpathiReport;
  const {
    addStatCards, boot, clear, createTable, csvButton, element, formatInteger,
    formatNumber, formatSeconds, humanize, notice, plot, query, renderHeader,
    renderKeyValues, renderLinks, shortId,
  } = R;

  function bytes(value) {
    if (value === null || value === undefined || value === "") return "—";
    const number = Number(value);
    if (!Number.isFinite(number)) return "—";
    if (number >= 1024 ** 3) return `${formatNumber(number / (1024 ** 3), { maximumFractionDigits: 2 })} GiB`;
    if (number >= 1024 ** 2) return `${formatNumber(number / (1024 ** 2), { maximumFractionDigits: 1 })} MiB`;
    if (number >= 1024) return `${formatNumber(number / 1024, { maximumFractionDigits: 1 })} KiB`;
    return `${formatInteger(number)} B`;
  }

  function fraction(numerator, denominator) {
    const top = Number(numerator);
    const bottom = Number(denominator);
    return Number.isFinite(top) && Number.isFinite(bottom) && bottom > 0 ? top / bottom : null;
  }

  function rangeText(range) {
    if (!range || range.minimum === undefined || range.maximum === undefined) return "—";
    return range.minimum === range.maximum ? formatInteger(range.minimum) : `${formatInteger(range.minimum)}–${formatInteger(range.maximum)}`;
  }

  function parameterText(value) {
    if (value === null) return "null";
    if (typeof value === "boolean") return value ? "true" : "false";
    if (Array.isArray(value)) return value.join(", ");
    if (value && typeof value === "object") return JSON.stringify(value);
    return String(value);
  }

  function datasetLabel(row) {
    return row.dataset_id.replace(/^scale_stress_/, "").replace(/_/g, " ");
  }

  function attemptLabel(row) {
    return `${datasetLabel(row)} · ${row.stage === "scout" ? "scout" : "all targets"}`;
  }

  function attempts(rows) {
    return rows.flatMap((row) => [row.scout, row.all_targets].filter(Boolean));
  }

  function andreaMeasurement(row) {
    return (row.inference || row.observations).andrea_measurement;
  }

  function andreaExecutionMeasurement(row) {
    return (row.inference || row.observations).andrea_execution_measurement;
  }

  function renderDatasets(datasets, boundary) {
    const container = query("#dataset-overview");
    clear(container);
    const summary = datasets.summary;
    const cards = element("div", { className: "stat-grid dataset-overview-cards" });
    addStatCards(cards, [
      { label: "Datasets", value: formatInteger(summary.dataset_count), detail: "Deterministic dimensional stressors" },
      { label: "Total cells", value: formatInteger(summary.total_cells), detail: `Per dataset: ${rangeText(summary.cells)}` },
      { label: "Genes", value: rangeText(summary.genes), detail: "All-target universe" },
      { label: "Groups", value: rangeText(summary.groups), detail: "Cell groups per dataset" },
      { label: "Transcription factors", value: rangeText(summary.transcription_factors), detail: "Candidate regulators per dataset" },
      { label: "Scout targets", value: formatInteger(boundary.target_scout_size), detail: "Outcome-blind SHA-256 selection" },
    ]);
    container.appendChild(cards);
    const table = element("div", { className: "card table-wrap dataset-overview-table" });
    createTable(table, [
      { label: "Dataset", value: (row) => row.dataset_id },
      { label: "Endpoint status", value: (row) => humanize(row.status) },
      { label: "Lane(s)", value: (row) => row.lane_ids.join(", ") },
      { label: "Scout decision", value: (row) => humanize(row.decisions.scout.decision) },
      { label: "Scout reason", value: (row) => humanize(row.decisions.scout.reason) },
      { label: "Scout outcome", value: (row) => row.scout ? humanize(row.scout.status) : "Not attempted" },
      { label: "All-target decision", value: (row) => humanize(row.decisions.all_targets.decision) },
      { label: "All-target reason", value: (row) => humanize(row.decisions.all_targets.reason) },
      { label: "Cells", value: (row) => row.dimensions.cells, format: formatInteger, numeric: true },
      { label: "Genes", value: (row) => row.dimensions.genes, format: formatInteger, numeric: true },
      { label: "TFs", value: (row) => row.dimensions.transcription_factors, format: formatInteger, numeric: true },
      { label: "Generation seed", value: (row) => row.generation_seed, format: formatInteger, numeric: true },
      { label: "Algorithm seed", value: (row) => row.algorithm_seed, format: formatInteger, numeric: true },
    ], datasets.rows);
    container.appendChild(table);
  }

  function renderOverview(summary, rows, boundary) {
    const largest = rows.reduce((best, row) => row.dimensions.cells * row.dimensions.genes > best.dimensions.cells * best.dimensions.genes ? row : best);
    addStatCards(query("#overview-cards"), [
      { label: "All-target completion", value: `${formatInteger(summary.datasets_completed)} / ${formatInteger(summary.datasets_requested)}`, detail: `${formatInteger(summary.datasets_failed)} failed · ${formatInteger(summary.datasets_not_attempted)} prospectively skipped` },
      { label: "Scouts completed", value: `${formatInteger(summary.scouts_completed)} / ${formatInteger(summary.scouts_attempted)}`, detail: `${formatInteger(boundary.target_scout_size)} targets per attempted scout` },
      { label: "Largest declared matrix", value: `${formatInteger(largest.dimensions.cells)} × ${formatInteger(largest.dimensions.genes)}`, detail: largest.dataset_id },
      { label: "Total attempted time", value: formatSeconds(summary.total_inference_wall_time_seconds), detail: "Canonical ANDREA end-to-end wall time across both stages" },
      { label: "Maximum container memory", value: bytes(summary.maximum_peak_memory_bytes), detail: `${formatInteger(summary.attempts_with_peak_memory)} attempted stage(s) with telemetry` },
      { label: "Prospective wall limit", value: formatSeconds(boundary.prospective_wall_time_limit_seconds), detail: `${formatNumber(boundary.maximum_projected_wall_time_fraction * 100)}% projection boundary` },
    ]);
  }

  function renderDimensions(rows) {
    const completed = rows.filter((row) => row.status === "completed");
    const other = rows.filter((row) => row.status !== "completed");
    const maximumTf = Math.max(...rows.map((row) => row.dimensions.transcription_factors));
    const traces = [];
    if (completed.length) traces.push({
      type: "scatter", mode: "markers", name: "All targets completed",
      x: completed.map((row) => row.dimensions.cells), y: completed.map((row) => row.dimensions.genes), text: completed.map((row) => row.dataset_id),
      customdata: completed.map((row) => [row.dimensions.transcription_factors, row.all_targets.inference.spathi_wall_time_seconds]),
      marker: { size: completed.map((row) => 11 + 21 * Math.sqrt(row.dimensions.transcription_factors / maximumTf)), color: completed.map((row) => row.all_targets.inference.spathi_wall_time_seconds), coloraxis: "coloraxis", opacity: 0.86 },
      hovertemplate: "<b>%{text}</b><br>Cells: %{x:,}<br>Genes: %{y:,}<br>TFs: %{customdata[0]:,}<br>All-target SPATHI: %{customdata[1]:.2f} s<extra></extra>",
    });
    if (other.length) traces.push({
      type: "scatter", mode: "markers", name: "Failed or prospectively skipped",
      x: other.map((row) => row.dimensions.cells), y: other.map((row) => row.dimensions.genes), text: other.map((row) => row.dataset_id), customdata: other.map((row) => humanize(row.status)),
      marker: { size: 17, symbol: "x", color: "#b91c1c" },
      hovertemplate: "<b>%{text}</b><br>Cells: %{x:,}<br>Genes: %{y:,}<br>Status: %{customdata}<extra></extra>",
    });
    plot(query("#dimensions-chart"), traces, {
      xaxis: { title: "Cells (count)" }, yaxis: { title: "Genes (count)" },
      coloraxis: { colorscale: [[0, "#dbeafe"], [1, "#1d4ed8"]], colorbar: { title: { text: "All-target<br>SPATHI seconds" } } },
      legend: { orientation: "h", y: -0.2 },
    });
  }

  function renderRuntime(rows) {
    if (!rows.length) return notice(query("#runtime-chart"), "warning", "No completed attempts", "Runtime components are unavailable.");
    const labels = rows.map(attemptLabel);
    const series = [
      ["Generation", (row) => row.generation.wall_time_seconds, "#94a3b8"],
      ["SPATHI core", (row) => row.inference.spathi_wall_time_seconds, "#2563eb"],
      ["ANDREA end-to-end", (row) => row.inference.andrea_execution_measurement.wall_time_seconds, "#0f8c84"],
    ];
    plot(query("#runtime-chart"), series.map(([name, accessor, color]) => ({ type: "bar", name, x: labels, y: rows.map(accessor), marker: { color } })), {
      barmode: "group", xaxis: { title: "Dataset and preregistered stage", tickangle: -25 }, yaxis: { title: "Wall time (seconds)", rangemode: "tozero" },
    });
  }

  function renderMemoryAndDisk(rows) {
    if (!rows.length) {
      notice(query("#memory-chart"), "warning", "No completed attempts", "Memory is unavailable.");
      return notice(query("#disk-chart"), "warning", "No completed attempts", "Output footprints are unavailable.");
    }
    const labels = rows.map(attemptLabel);
    plot(query("#memory-chart"), [
      { type: "bar", name: "Canonical container peak", x: labels, y: rows.map((row) => andreaMeasurement(row).peak_memory_bytes / (1024 ** 3)), marker: { color: "#2563eb" } },
      { type: "bar", name: "SPATHI heap estimate", x: labels, y: rows.map((row) => row.inference.estimated_peak_heap_with_rough_trees_bytes / (1024 ** 3)), marker: { color: "#93c5fd" } },
    ], { barmode: "group", xaxis: { title: "Dataset and preregistered stage", tickangle: -25 }, yaxis: { title: "Memory (GiB)", rangemode: "tozero" } });
    const disk = [["Expression matrix", (row) => row.generation.expression_bytes, "#60a5fa"], ["SPATHI output", (row) => row.inference.spathi_output_bytes, "#2563eb"], ["Dataset artifact", (row) => row.generation.dataset_artifact_bytes, "#14b8a6"], ["Inference artifact", (row) => row.inference.inference_artifact_bytes, "#7c3aed"]];
    plot(query("#disk-chart"), disk.map(([name, accessor, color]) => ({ type: "bar", name, x: labels, y: rows.map((row) => accessor(row) / (1024 ** 2)), marker: { color } })), {
      barmode: "group", xaxis: { title: "Dataset and preregistered stage", tickangle: -25 }, yaxis: { title: "Disk footprint (MiB)", rangemode: "tozero" },
    });
  }

  function renderWorkAndCompletion(rows) {
    if (!rows.length) {
      notice(query("#work-chart"), "warning", "No completed attempts", "Execution counts are unavailable.");
      return notice(query("#completion-chart"), "warning", "No completed attempts", "Completion fractions are unavailable.");
    }
    const labels = rows.map(attemptLabel);
    plot(query("#work-chart"), [["Requested targets", "targets_requested", "#60a5fa"], ["Completed models", "models_completed", "#0f8c84"], ["Fitted trees", "trees_fitted", "#7c3aed"]].map(([name, field, color]) => ({ type: "bar", name, x: labels, y: rows.map((row) => row.inference[field]), marker: { color } })), {
      barmode: "group", xaxis: { title: "Dataset and preregistered stage", tickangle: -25 }, yaxis: { title: "Execution count (log scale)", type: "log" },
    });
    plot(query("#completion-chart"), [
      { type: "bar", name: "Completed models", x: labels, y: rows.map((row) => fraction(row.inference.models_completed, row.inference.models_requested)), marker: { color: "#0f8c84" } },
    ], { barmode: "group", xaxis: { title: "Dataset and preregistered stage", tickangle: -25 }, yaxis: { title: "Fraction of requested work", range: [0, 1.03], tickformat: ".0%" } });
  }

  function renderAttemptTable(rows) {
    const columns = [
      { label: "Dataset", value: (row) => row.dataset_id }, { label: "Stage", value: (row) => humanize(row.stage) }, { label: "Status", value: (row) => humanize(row.status) },
      { label: "Targets", value: (row) => row.dimensions.targets, format: formatInteger, numeric: true },
      { label: "Completed models", value: (row) => row.inference ? row.inference.models_completed : null, format: formatInteger, numeric: true },
      { label: "Trees", value: (row) => row.inference ? row.inference.trees_fitted : null, format: formatInteger, numeric: true },
      { label: "ANDREA end-to-end time", value: (row) => andreaExecutionMeasurement(row).wall_time_seconds, format: formatSeconds, numeric: true },
      { label: "CPU time", value: (row) => andreaMeasurement(row).cpu_time_seconds, format: formatSeconds, numeric: true },
      { label: "Peak memory", value: (row) => andreaMeasurement(row).peak_memory_bytes, format: bytes, numeric: true },
      { label: "Telemetry", value: (row) => humanize(andreaMeasurement(row).telemetry.status) },
      { label: "Seed", value: (row) => row.algorithm_seed, format: formatInteger, numeric: true },
    ];
    createTable(query("#dataset-table"), columns, rows);
    const actions = query("#dataset-table-actions"); clear(actions);
    actions.appendChild(csvButton("Export attempts CSV", "spathi-scalability-attempts.csv", columns, rows));
  }

  function renderFailures(rows) {
    const section = query("#failure-section");
    const failures = rows.filter((row) => row.status === "failed");
    if (!failures.length) { section.hidden = true; return; }
    section.hidden = false;
    notice(query("#failure-notice"), "warning", "Attributable SPATHI failures", "Only OOM, timeout, or tool failures are retained. Infrastructure and contract failures abort the workflow.");
    createTable(query("#failure-table"), [
      { label: "Dataset", value: (row) => row.dataset_id }, { label: "Stage", value: (row) => humanize(row.stage) },
      { label: "Kind", value: (row) => humanize(row.failure.failure_kind) }, { label: "Tool exit", value: (row) => row.failure.tool_exit_code, format: formatInteger, numeric: true },
      { label: "ANDREA end-to-end time", value: (row) => andreaExecutionMeasurement(row).wall_time_seconds, format: formatSeconds, numeric: true },
      { label: "Peak memory", value: (row) => andreaMeasurement(row).peak_memory_bytes, format: bytes, numeric: true },
      { label: "Enforced task timeout", value: (row) => row.execution_limits.timeout_seconds, format: formatSeconds, numeric: true },
      { label: "Message", value: (row) => row.failure.message }, { label: "Tool log tail", value: (row) => row.diagnostics.tool_log_tail || "Unavailable" },
      { label: "Command log", value: (row) => row.diagnostics.command_log.path },
    ], failures);
  }

  function renderPolicy(selection) {
    addStatCards(query("#policy-summary"), [
      { label: "Policy", value: selection.policy.policy_id, detail: `SHA-256 · ${shortId(selection.policy.policy_sha256, 18)}` },
      { label: "Evidence role", value: humanize(selection.evidence_role), detail: `Budget: ${humanize(selection.budget)}` },
      { label: "Policy status", value: humanize(selection.status), detail: `SPATHI · ${shortId(selection.policy.spathi_commit, 14)}` },
    ]);
    const branches = selection.policy.group_policy.branches;
    const names = new Set(); Object.values(branches).forEach((branch) => Object.keys(branch.parameters).forEach((name) => names.add(name)));
    const rows = Object.entries(branches).map(([branchId, branch]) => ({ branch_id: branchId, condition: branch.condition, parameters: branch.parameters }));
    createTable(query("#policy-table"), [{ label: "Branch", value: (row) => humanize(row.branch_id) }, { label: "Condition", value: (row) => parameterText(row.condition) }, ...Array.from(names).sort().map((name) => ({ label: humanize(name), value: (row) => parameterText(row.parameters[name]) }))], rows);
  }

  function renderProvenance(environment, links, notes, boundary) {
    renderKeyValues(query("#environment"), "Recorded execution environment", [
      { label: "Operating system", value: `${environment.system} ${environment.kernel_release}` }, { label: "Machine", value: environment.machine }, { label: "CPU", value: environment.cpu_model },
      { label: "Logical CPUs", value: environment.logical_cpus, format: formatInteger }, { label: "Host memory", value: environment.host_memory_bytes, format: bytes },
      { label: "Assigned CPU threads", value: environment.cpu_threads, format: formatInteger }, { label: "Assigned RAM", value: environment.memory_bytes, format: bytes },
      { label: "CPU set", value: environment.cpuset_cpus.join(", ") }, { label: "Measurement", value: humanize(environment.measurement_contract) },
      { label: "Scout size", value: boundary.target_scout_size, format: formatInteger }, { label: "Wall projection boundary", value: `${formatNumber(boundary.maximum_projected_wall_time_fraction * 100)}% of ${formatSeconds(boundary.prospective_wall_time_limit_seconds)}` },
      { label: "Memory boundary", value: `${formatNumber(boundary.maximum_observed_memory_fraction * 100)}% of assigned RAM` }, { label: "SPATHI image", value: environment.spathi_image_id },
    ]);
    renderLinks(query("#report-links"), "Canonical artifacts", links);
    const card = query("#study-notes"); clear(card); card.appendChild(element("h3", { text: "Interpretation notes" }));
    const list = element("ul", { className: "link-list" }); notes.forEach((item) => list.appendChild(element("li", { text: item }))); card.appendChild(list);
  }

  boot((data) => {
    renderHeader(data.workflow, "scalability");
    const summary = data.summary;
    const incomplete = summary.datasets_failed > 0 || summary.datasets_not_attempted > 0;
    notice(query("#status-banner"), incomplete ? "warning" : "success", incomplete ? "Scale study complete with incomplete all-target endpoints" : "Scale study complete", `${formatInteger(summary.datasets_completed)} completed, ${formatInteger(summary.datasets_failed)} failed, and ${formatInteger(summary.datasets_not_attempted)} prospectively skipped out of ${formatInteger(summary.datasets_requested)} declared shapes.`);
    renderDatasets(data.datasets, data.scope.resource_boundary);
    notice(query("#scope-notice"), "warning", "Interpretation boundary", data.scope.interpretation);
    renderOverview(summary, data.datasets.rows, data.scope.resource_boundary);
    const allAttempts = attempts(data.datasets.rows);
    const completedAttempts = allAttempts.filter((row) => row.status === "completed");
    renderFailures(allAttempts); renderDimensions(data.datasets.rows); renderRuntime(completedAttempts); renderMemoryAndDisk(completedAttempts); renderWorkAndCompletion(completedAttempts); renderAttemptTable(allAttempts); renderPolicy(data.selection); renderProvenance(data.environment, data.links, data.notes, data.scope.resource_boundary);
  });
}());
