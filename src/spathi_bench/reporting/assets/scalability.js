(function () {
  "use strict";

  const R = window.SpathiReport;
  const {
    PALETTE,
    addStatCards,
    append,
    boot,
    clear,
    createTable,
    csvButton,
    element,
    formatInteger,
    formatNumber,
    formatSeconds,
    humanize,
    notice,
    pill,
    plot,
    query,
    renderHeader,
    renderKeyValues,
    renderLinks,
    shortId,
  } = R;

  function bytes(value) {
    if (value === null || value === undefined || value === "") return "—";
    const number = Number(value);
    if (!Number.isFinite(number)) return "—";
    if (number >= 1024 ** 3) {
      return `${formatNumber(number / (1024 ** 3), { maximumFractionDigits: 2 })} GiB`;
    }
    if (number >= 1024 ** 2) {
      return `${formatNumber(number / (1024 ** 2), { maximumFractionDigits: 1 })} MiB`;
    }
    if (number >= 1024) {
      return `${formatNumber(number / 1024, { maximumFractionDigits: 1 })} KiB`;
    }
    return `${formatInteger(number)} B`;
  }

  function percent(numerator, denominator) {
    const top = Number(numerator);
    const bottom = Number(denominator);
    if (!Number.isFinite(top) || !Number.isFinite(bottom) || bottom <= 0) return "—";
    return `${formatNumber((top / bottom) * 100, { maximumFractionDigits: 2 })}%`;
  }

  function fraction(numerator, denominator) {
    const top = Number(numerator);
    const bottom = Number(denominator);
    if (!Number.isFinite(top) || !Number.isFinite(bottom) || bottom <= 0) return null;
    return top / bottom;
  }

  function rangeText(range) {
    if (!range || range.minimum === undefined || range.maximum === undefined) return "—";
    if (range.minimum === range.maximum) return formatInteger(range.minimum);
    return `${formatInteger(range.minimum)}–${formatInteger(range.maximum)}`;
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

  function renderDatasets(datasets) {
    const container = query("#dataset-overview");
    clear(container);
    const summary = datasets.summary;
    const cards = element("div", { className: "stat-grid dataset-overview-cards" });
    addStatCards(cards, [
      { label: "Datasets", value: formatInteger(summary.dataset_count), detail: "Deterministic dimensional stressors" },
      { label: "Total cells", value: formatInteger(summary.total_cells), detail: `Per dataset: ${rangeText(summary.cells)}` },
      { label: "Genes", value: rangeText(summary.genes), detail: "Generated features per dataset" },
      { label: "Groups", value: rangeText(summary.groups), detail: "Cell groups per dataset" },
      { label: "Transcription factors", value: rangeText(summary.transcription_factors), detail: "Candidate regulators per dataset" },
      { label: "Targets", value: rangeText(summary.targets), detail: "Requested target universe" },
    ]);
    container.appendChild(cards);

    const tableCard = element("div", { className: "card table-wrap dataset-overview-table" });
    createTable(tableCard, [
      { label: "Dataset", value: (row) => row.dataset_id },
      { label: "Status", value: (row) => humanize(row.status) },
      { label: "Failure", value: (row) => row.failure ? humanize(row.failure.failure_kind) : "—" },
      { label: "Network mode", value: (row) => humanize(row.network_mode) },
      { label: "Policy branch", value: (row) => humanize(row.policy_branch) },
      { label: "Cells", value: (row) => row.dimensions.cells, format: formatInteger, numeric: true },
      { label: "Genes", value: (row) => row.dimensions.genes, format: formatInteger, numeric: true },
      { label: "Groups", value: (row) => row.dimensions.groups, format: formatInteger, numeric: true },
      { label: "TFs", value: (row) => row.dimensions.transcription_factors, format: formatInteger, numeric: true },
      { label: "Targets", value: (row) => row.dimensions.targets, format: formatInteger, numeric: true },
      { label: "Generation seed", value: (row) => row.generation_seed, format: formatInteger, numeric: true },
      { label: "Algorithm seed", value: (row) => row.algorithm_seed, format: formatInteger, numeric: true },
    ], datasets.rows);
    container.appendChild(tableCard);
  }

  function renderOverview(summary, rows) {
    const largest = rows.reduce((best, row) => (
      row.dimensions.cells * row.dimensions.genes > best.dimensions.cells * best.dimensions.genes
        ? row
        : best
    ));
    addStatCards(query("#overview-cards"), [
      {
        label: "Completion",
        value: `${formatInteger(summary.datasets_completed)} / ${formatInteger(summary.datasets_requested)}`,
        detail: summary.all_completed
          ? "All declared shapes completed"
          : `${formatInteger(summary.datasets_failed)} endpoint(s) failed`,
      },
      {
        label: "Largest matrix",
        value: `${formatInteger(largest.dimensions.cells)} × ${formatInteger(largest.dimensions.genes)}`,
        detail: largest.dataset_id,
      },
      {
        label: "Total inference time",
        value: formatSeconds(summary.total_inference_wall_time_seconds),
        detail: "Harness wall time across completed and failed attempts",
      },
      {
        label: "Maximum peak RSS",
        value: bytes(summary.maximum_peak_rss_bytes),
        detail: `${formatInteger(summary.datasets_with_peak_rss)} dataset(s) yielded RSS samples`,
      },
      {
        label: "SPATHI output",
        value: bytes(summary.total_spathi_output_bytes),
        detail: "Sum across completed datasets",
      },
      {
        label: "Trees fitted",
        value: formatInteger(summary.total_trees_fitted),
        detail: "Total estimator volume",
      },
    ]);
  }

  function renderDimensions(rows) {
    const completed = rows.filter((row) => row.status === "completed");
    const failed = rows.filter((row) => row.status === "failed");
    const maximumTf = Math.max(...rows.map((row) => row.dimensions.transcription_factors));
    const branches = Array.from(new Set(completed.map((row) => row.policy_branch)));
    const symbols = { single_group: "diamond", multi_group: "circle" };
    const traces = branches.map((branch) => {
      const subset = completed.filter((row) => row.policy_branch === branch);
      return {
        type: "scatter",
        mode: "markers",
        name: humanize(branch),
        x: subset.map((row) => row.dimensions.cells),
        y: subset.map((row) => row.dimensions.genes),
        text: subset.map((row) => row.dataset_id),
        customdata: subset.map((row) => [
          row.dimensions.transcription_factors,
          row.dimensions.groups,
          row.inference.spathi_wall_time_seconds,
          row.network_mode,
        ]),
        marker: {
          size: subset.map((row) => 11 + 21 * Math.sqrt(row.dimensions.transcription_factors / maximumTf)),
          sizemode: "diameter",
          symbol: symbols[branch] || "circle",
          color: subset.map((row) => row.inference.spathi_wall_time_seconds),
          coloraxis: "coloraxis",
          opacity: 0.86,
          line: { color: "#ffffff", width: 1 },
        },
        hovertemplate:
          "<b>%{text}</b><br>Cells: %{x:,}<br>Genes: %{y:,}" +
          "<br>TFs: %{customdata[0]:,}<br>Groups: %{customdata[1]:,}" +
          "<br>SPATHI: %{customdata[2]:.2f} s<br>Network: %{customdata[3]}<extra></extra>",
      };
    });
    if (failed.length) {
      traces.push({
        type: "scatter",
        mode: "markers",
        name: "Failed endpoint",
        x: failed.map((row) => row.dimensions.cells),
        y: failed.map((row) => row.dimensions.genes),
        text: failed.map((row) => row.dataset_id),
        customdata: failed.map((row) => [
          row.dimensions.transcription_factors,
          row.dimensions.groups,
          humanize(row.failure.failure_kind),
          row.network_mode,
        ]),
        marker: {
          size: failed.map((row) => 11 + 21 * Math.sqrt(row.dimensions.transcription_factors / maximumTf)),
          sizemode: "diameter",
          symbol: "x",
          color: "#b91c1c",
          opacity: 0.95,
        },
        hovertemplate:
          "<b>%{text}</b><br>Cells: %{x:,}<br>Genes: %{y:,}" +
          "<br>TFs: %{customdata[0]:,}<br>Groups: %{customdata[1]:,}" +
          "<br>Failure: %{customdata[2]}<br>Network: %{customdata[3]}<extra></extra>",
      });
    }
    plot(query("#dimensions-chart"), traces, {
      xaxis: { title: "Cells (count)" },
      yaxis: { title: "Genes (count)" },
      coloraxis: {
        colorscale: [[0, "#dbeafe"], [1, "#1d4ed8"]],
        colorbar: { title: { text: "SPATHI<br>seconds" } },
      },
      legend: { title: { text: "Policy branch" }, orientation: "h", y: -0.2 },
    });
  }

  function renderRuntime(rows) {
    if (!rows.length) {
      notice(query("#runtime-chart"), "warning", "No completed endpoints", "Runtime components are unavailable because every declared endpoint failed.");
      return;
    }
    const labels = rows.map(datasetLabel);
    const series = [
      ["Generation", (row) => row.generation.wall_time_seconds, "#94a3b8"],
      ["SPATHI core", (row) => row.inference.spathi_wall_time_seconds, "#2563eb"],
      ["ANDREA wrapper", (row) => row.inference.andrea_wall_time_seconds, "#0f8c84"],
      ["Inference harness", (row) => row.inference.harness_wall_time_seconds, "#7c3aed"],
    ];
    plot(query("#runtime-chart"), series.map(([name, accessor, color]) => ({
      type: "bar",
      name,
      x: labels,
      y: rows.map(accessor),
      marker: { color },
      hovertemplate: `<b>%{x}</b><br>${name}: %{y:.2f} s<extra></extra>`,
    })), {
      barmode: "group",
      xaxis: { title: "Synthetic dimensional dataset", tickangle: -25 },
      yaxis: { title: "Wall time (seconds)", rangemode: "tozero" },
    });
  }

  function renderMemoryAndDisk(rows) {
    if (!rows.length) {
      notice(query("#memory-chart"), "warning", "No completed endpoints", "Completed-run memory estimates are unavailable.");
      notice(query("#disk-chart"), "warning", "No completed endpoints", "Completed-run output footprints are unavailable.");
      return;
    }
    const labels = rows.map(datasetLabel);
    plot(query("#memory-chart"), [
      {
        type: "bar",
        name: "Observed peak RSS",
        x: labels,
        y: rows.map((row) => row.inference.peak_rss_bytes / (1024 ** 3)),
        marker: { color: "#2563eb" },
        hovertemplate: "<b>%{x}</b><br>Observed peak RSS: %{y:.3f} GiB<extra></extra>",
      },
      {
        type: "bar",
        name: "SPATHI heap estimate",
        x: labels,
        y: rows.map((row) => row.inference.estimated_peak_heap_with_rough_trees_bytes / (1024 ** 3)),
        marker: { color: "#93c5fd" },
        hovertemplate: "<b>%{x}</b><br>Heap estimate: %{y:.3f} GiB<extra></extra>",
      },
    ], {
      barmode: "group",
      xaxis: { title: "Synthetic dimensional dataset", tickangle: -25 },
      yaxis: { title: "Memory (GiB)", rangemode: "tozero" },
    });

    const diskSeries = [
      ["Expression matrix", (row) => row.generation.expression_bytes, "#60a5fa"],
      ["SPATHI output", (row) => row.inference.spathi_output_bytes, "#2563eb"],
      ["Dataset artifact", (row) => row.generation.dataset_artifact_bytes, "#14b8a6"],
      ["Inference artifact", (row) => row.inference.inference_artifact_bytes, "#7c3aed"],
    ];
    plot(query("#disk-chart"), diskSeries.map(([name, accessor, color]) => ({
      type: "bar",
      name,
      x: labels,
      y: rows.map((row) => accessor(row) / (1024 ** 2)),
      marker: { color },
      hovertemplate: `<b>%{x}</b><br>${name}: %{y:.2f} MiB<extra></extra>`,
    })), {
      barmode: "group",
      xaxis: { title: "Synthetic dimensional dataset", tickangle: -25 },
      yaxis: { title: "Disk footprint (MiB)", rangemode: "tozero" },
    });
  }

  function renderWorkAndCompletion(rows) {
    if (!rows.length) {
      notice(query("#work-chart"), "warning", "No completed endpoints", "Model and tree counts are unavailable.");
      notice(query("#completion-chart"), "warning", "No completed endpoints", "Completion fractions are unavailable.");
      return;
    }
    const labels = rows.map(datasetLabel);
    const work = [
      ["Requested targets", "targets_requested", "#60a5fa"],
      ["Completed models", "models_completed", "#0f8c84"],
      ["Fitted trees", "trees_fitted", "#7c3aed"],
    ];
    plot(query("#work-chart"), work.map(([name, field, color]) => ({
      type: "bar",
      name,
      x: labels,
      y: rows.map((row) => row.inference[field]),
      marker: { color },
      hovertemplate: `<b>%{x}</b><br>${name}: %{y:,}<extra></extra>`,
    })), {
      barmode: "group",
      xaxis: { title: "Synthetic dimensional dataset", tickangle: -25 },
      yaxis: { title: "Execution count (log scale)", type: "log" },
    });

    plot(query("#completion-chart"), [
      {
        type: "bar",
        name: "Eligible targets",
        x: labels,
        y: rows.map((row) => fraction(row.inference.targets_eligible, row.inference.targets_requested)),
        marker: { color: "#2563eb" },
        hovertemplate: "<b>%{x}</b><br>Eligible targets: %{y:.2%}<extra></extra>",
      },
      {
        type: "bar",
        name: "Completed models",
        x: labels,
        y: rows.map((row) => fraction(row.inference.models_completed, row.inference.models_requested)),
        marker: { color: "#0f8c84" },
        hovertemplate: "<b>%{x}</b><br>Completed models: %{y:.2%}<extra></extra>",
      },
    ], {
      barmode: "group",
      xaxis: { title: "Synthetic dimensional dataset", tickangle: -25 },
      yaxis: { title: "Fraction of requested work", range: [0, 1.03], tickformat: ".0%" },
    });
  }

  function renderDatasetTable(rows) {
    const columns = [
      { label: "Dataset", value: (row) => row.dataset_id },
      { label: "Status", value: (row) => humanize(row.status) },
      { label: "Failure", value: (row) => row.failure ? humanize(row.failure.failure_kind) : "—" },
      { label: "Mode", value: (row) => humanize(row.network_mode) },
      { label: "Branch", value: (row) => humanize(row.policy_branch) },
      { label: "Cells", value: (row) => row.dimensions.cells, format: formatInteger, numeric: true },
      { label: "Genes", value: (row) => row.dimensions.genes, format: formatInteger, numeric: true },
      { label: "Groups", value: (row) => row.dimensions.groups, format: formatInteger, numeric: true },
      { label: "TFs", value: (row) => row.dimensions.transcription_factors, format: formatInteger, numeric: true },
      { label: "Eligible targets", value: (row) => row.inference ? row.inference.targets_eligible : null, format: formatInteger, numeric: true },
      { label: "Target eligibility", value: (row) => row.inference ? percent(row.inference.targets_eligible, row.inference.targets_requested) : "—", numeric: true },
      { label: "Models", value: (row) => row.inference ? `${formatInteger(row.inference.models_completed)} / ${formatInteger(row.inference.models_requested)}` : "—", numeric: true },
      { label: "Trees", value: (row) => row.inference ? row.inference.trees_fitted : null, format: formatInteger, numeric: true },
      { label: "Threads", value: (row) => row.inference ? `${formatInteger(row.inference.threads_effective)} / ${formatInteger(row.inference.threads_available)}` : "—", numeric: true },
      { label: "SPATHI time", value: (row) => row.inference ? row.inference.spathi_wall_time_seconds : null, format: formatSeconds, numeric: true },
      { label: "Harness time", value: (row) => row.inference ? row.inference.harness_wall_time_seconds : row.observations.harness_wall_time_seconds, format: formatSeconds, numeric: true },
      { label: "Peak RSS", value: (row) => row.inference ? row.inference.peak_rss_bytes : row.observations.peak_rss_bytes, format: bytes, numeric: true },
      { label: "SPATHI output", value: (row) => row.inference ? row.inference.spathi_output_bytes : null, format: bytes, numeric: true },
      { label: "RSS samples", value: (row) => row.inference ? row.inference.rss_samples : row.observations.rss_samples, format: formatInteger, numeric: true },
      { label: "Generation seed", value: (row) => row.generation_seed, format: formatInteger, numeric: true },
      { label: "Algorithm seed", value: (row) => row.algorithm_seed, format: formatInteger, numeric: true },
    ];
    createTable(query("#dataset-table"), columns, rows);
    const actions = query("#dataset-table-actions");
    clear(actions);
    actions.appendChild(csvButton("Export measurements CSV", "spathi-scalability-measurements.csv", columns, rows));
  }

  function renderFailures(rows) {
    const section = query("#failure-section");
    const failures = rows.filter((row) => row.status === "failed");
    if (!failures.length) {
      section.hidden = true;
      return;
    }
    section.hidden = false;
    notice(
      query("#failure-notice"),
      "warning",
      "The panel completed with failed endpoints",
      "Failures count as evidence that the declared shape did not complete under the recorded policy and resource limits. They are never counted as successful runs.",
    );
    createTable(query("#failure-table"), [
      { label: "Dataset", value: (row) => row.dataset_id },
      { label: "Kind", value: (row) => humanize(row.failure.failure_kind) },
      { label: "Stage", value: (row) => humanize(row.failure.stage) },
      { label: "Orchestrator exit", value: (row) => row.failure.orchestrator_exit_code, format: formatInteger, numeric: true },
      { label: "Tool exit", value: (row) => row.failure.tool_exit_code, format: formatInteger, numeric: true },
      { label: "Harness time", value: (row) => row.observations.harness_wall_time_seconds, format: formatSeconds, numeric: true },
      { label: "Tool time", value: (row) => row.observations.tool_wall_time_seconds, format: formatSeconds, numeric: true },
      { label: "Peak RSS", value: (row) => row.observations.peak_rss_bytes, format: bytes, numeric: true },
      { label: "RSS samples", value: (row) => row.observations.rss_samples, format: formatInteger, numeric: true },
      { label: "Progress", value: (row) => row.diagnostics.progress ? `${formatInteger(row.diagnostics.progress.percent)}% · ${humanize(row.diagnostics.progress.phase)}` : "Unavailable" },
      { label: "Parallel runs", value: (row) => row.execution_limits.max_parallel_runs, format: formatInteger, numeric: true },
      { label: "RAM limit", value: (row) => `${formatNumber(row.execution_limits.max_ram_gb)} GiB`, numeric: true },
      { label: "Cores", value: (row) => row.execution_limits.max_cores, format: formatInteger, numeric: true },
      { label: "Planner", value: (row) => humanize(row.execution_limits.planner) },
      { label: "Planner limit", value: (row) => formatSeconds(row.execution_limits.planner_time_limit_seconds), numeric: true },
      { label: "Inference limit", value: (row) => row.execution_limits.inference_wall_time_limit_seconds === null ? "Not configured" : formatSeconds(row.execution_limits.inference_wall_time_limit_seconds), numeric: true },
      { label: "Message", value: (row) => row.failure.message },
      { label: "Tool log tail", value: (row) => row.diagnostics.tool_log_tail || "Unavailable" },
      { label: "Command log", value: (row) => row.diagnostics.command_log.path },
    ], failures);
  }

  function renderPolicy(production) {
    addStatCards(query("#policy-summary"), [
      { label: "Policy", value: production.policy.policy_id, detail: `SHA-256 · ${shortId(production.policy.policy_sha256, 18)}` },
      { label: "Decision", value: humanize(production.decision), detail: "Outcome frozen by efficiency justification" },
      { label: "Evidence role", value: humanize(production.evidence_role), detail: `Budget: ${humanize(production.budget)}` },
      { label: "Policy status", value: humanize(production.status), detail: `SPATHI · ${shortId(production.policy.spathi_commit, 14)}` },
    ]);

    const branches = production.policy.group_policy.branches;
    const parameterNames = new Set();
    Object.values(branches).forEach((branch) => {
      Object.keys(branch.parameters).forEach((name) => parameterNames.add(name));
    });
    const ordered = Array.from(parameterNames).sort();
    const rows = Object.entries(branches).map(([branchId, branch]) => ({
      branch_id: branchId,
      condition: branch.condition,
      parameters: branch.parameters,
    }));
    createTable(query("#policy-table"), [
      { label: "Branch", value: (row) => humanize(row.branch_id) },
      { label: "Condition", value: (row) => parameterText(row.condition) },
      ...ordered.map((name) => ({
        label: humanize(name),
        value: (row) => parameterText(row.parameters[name]),
      })),
    ], rows);
  }

  function renderProvenance(environment, links, notes) {
    renderKeyValues(query("#environment"), "Recorded execution environment", [
      { label: "Operating system", value: `${environment.system} ${environment.kernel_release}` },
      { label: "Machine", value: environment.machine },
      { label: "CPU", value: environment.cpu_model },
      { label: "Logical CPUs", value: environment.logical_cpus, format: formatInteger },
      { label: "Host memory", value: environment.host_memory_bytes, format: bytes },
      { label: "Maximum cores", value: environment.max_cores, format: formatInteger },
      { label: "Maximum RAM budget", value: `${formatNumber(environment.max_ram_gb)} GiB` },
      { label: "RSS method", value: humanize(environment.rss_measurement_method) },
      { label: "RSS sample interval", value: `${formatNumber(environment.rss_sample_interval_seconds)} s` },
      { label: "SPATHI image", value: environment.spathi_image_id },
      { label: "ANDREA environment", value: environment.andrea_environment_fingerprint },
      { label: "Benchmark environment", value: environment.benchmark_environment_fingerprint },
      { label: "Docker environment", value: environment.docker_runtime_environment_fingerprint },
    ]);
    renderLinks(query("#report-links"), "Canonical artifacts", links);
    const notesCard = query("#study-notes");
    clear(notesCard);
    notesCard.appendChild(element("h3", { text: "Interpretation notes" }));
    const list = element("ul", { className: "link-list" });
    notes.forEach((item) => list.appendChild(element("li", { text: item })));
    notesCard.appendChild(list);
  }

  boot((data) => {
    renderHeader(data.workflow, "scalability");
    const allCompleted = data.summary.all_completed;
    notice(
      query("#status-banner"),
      allCompleted ? "success" : "warning",
      allCompleted ? "Scale study complete" : "Scale study complete with failed endpoints",
      `${formatInteger(data.summary.datasets_completed)} of ${formatInteger(data.summary.datasets_requested)} declared dimensional shapes completed with the frozen production policy; ${formatInteger(data.summary.datasets_failed)} failed.`,
    );
    renderDatasets(data.datasets);
    notice(query("#scope-notice"), "warning", "Interpretation boundary", data.scope.interpretation);
    renderOverview(data.summary, data.datasets.rows);
    renderFailures(data.datasets.rows);
    renderDimensions(data.datasets.rows);
    const completed = data.datasets.rows.filter((row) => row.status === "completed");
    renderRuntime(completed);
    renderMemoryAndDisk(completed);
    renderWorkAndCompletion(completed);
    renderDatasetTable(data.datasets.rows);
    renderPolicy(data.production);
    renderProvenance(data.environment, data.links, data.notes);
  });
}());
