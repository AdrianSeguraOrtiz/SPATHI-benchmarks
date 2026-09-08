(function () {
  "use strict";

  const R = window.SpathiReport;
  const {
    PALETTE, addStatCards, boot, createTable, csvButton, formatInteger,
    formatNumber, formatSeconds, humanize, notice, plot, query,
    renderHeader, renderKeyValues, renderLinks,
  } = R;

  function bytes(value) {
    if (value === null || value === undefined) return "—";
    const n = Number(value);
    if (!Number.isFinite(n)) return "—";
    if (n >= 1024 ** 3) return `${formatNumber(n / 1024 ** 3, { maximumFractionDigits: 2 })} GiB`;
    if (n >= 1024 ** 2) return `${formatNumber(n / 1024 ** 2, { maximumFractionDigits: 1 })} MiB`;
    return `${formatNumber(n / 1024, { maximumFractionDigits: 1 })} KiB`;
  }

  function series(rows, field) {
    const keys = Array.from(new Set(rows.map((row) => `${row.configuration_id}|${row.size_class}`)));
    return keys.map((key, index) => {
      const [configuration, size] = key.split("|");
      const subset = rows.filter((row) => row.configuration_id === configuration && row.size_class === size)
        .sort((left, right) => left.threads - right.threads);
      return {
        type: "scatter", mode: "lines+markers", name: `${configuration} · ${size}`,
        x: subset.map((row) => row.threads), y: subset.map((row) => row[field]),
        marker: { color: PALETTE[index % PALETTE.length] },
        line: { color: PALETTE[index % PALETTE.length], width: 3 },
        connectgaps: false,
        hovertemplate: `<b>${configuration}</b> · ${size}<br>Threads: %{x}<br>Value: %{y:.3g}<extra></extra>`,
      };
    });
  }

  function renderCharts(rows, threads) {
    plot(query("#runtime-chart"), series(rows, "wall_time_median_seconds"), {
      xaxis: { title: "Assigned threads", tickmode: "array", tickvals: threads },
      yaxis: { title: "Median wall time (seconds)", type: "log" },
      legend: { orientation: "h", y: -0.25 },
    });
    const speedup = series(rows, "speedup_vs_one_thread");
    speedup.push({ type: "scatter", mode: "lines", name: "Ideal", x: threads, y: threads,
      line: { color: "#64748b", dash: "dash", width: 2 }, hoverinfo: "skip" });
    plot(query("#speedup-chart"), speedup, {
      xaxis: { title: "Assigned threads", tickmode: "array", tickvals: threads },
      yaxis: { title: "Speedup (×)", rangemode: "tozero" },
      legend: { orientation: "h", y: -0.25 },
    });
    plot(query("#efficiency-chart"), series(rows, "parallel_efficiency"), {
      xaxis: { title: "Assigned threads", tickmode: "array", tickvals: threads },
      yaxis: { title: "Parallel efficiency (speedup / threads)", rangemode: "tozero" },
      legend: { orientation: "h", y: -0.25 },
    });
    plot(query("#memory-chart"), series(rows, "container_peak_overlap_upper_bound_max_bytes").map((trace) => ({
      ...trace, y: trace.y.map((value) => value === null ? null : value / 1024 ** 3),
    })), {
      xaxis: { title: "Assigned threads", tickmode: "array", tickvals: threads },
      yaxis: { title: "Container-memory overlap upper bound (GiB)", rangemode: "tozero" },
      legend: { orientation: "h", y: -0.25 },
    });
  }

  boot((data) => {
    if (data.schema_version !== "1.0" || data.report_type !== "performance") {
      throw new Error("Unsupported SPATHI performance report payload.");
    }
    renderHeader(data.workflow, "performance", { status: "complete" });
    const totalCells = data.datasets.reduce((sum, item) => sum + item.cells, 0);
    addStatCards(query("#dataset-cards"), [
      { label: "Datasets", value: formatInteger(data.datasets.length), detail: data.datasets.map((item) => item.size_class).join(" + ") },
      { label: "Cells represented", value: formatInteger(totalCells), detail: "Fixed synthetic inputs" },
      { label: "Methods", value: formatInteger(new Set(data.aggregates.map((item) => item.configuration_id)).size), detail: "Selected SPATHI policy plus catalog defaults" },
      { label: "Thread endpoints", value: data.protocol.thread_counts.join(" / "), detail: "Frozen CPU affinity" },
    ]);
    renderKeyValues(query("#protocol"), "Execution protocol", [
      { label: "Output profile", value: data.protocol.output_profile },
      { label: "Inference cache", value: data.protocol.inference_cache },
      { label: "Concurrency", value: humanize(data.protocol.concurrency) },
      { label: "System repetitions", value: data.protocol.system_repetitions },
      { label: "Warm-ups", value: `${data.warmups.count} (${data.warmups.completed} completed; excluded)` },
      { label: "Algorithm seed policy", value: humanize(data.protocol.repetition_semantics) },
      { label: "Memory source", value: data.protocol.memory_measurement_source },
      { label: "Memory semantics", value: "Overlap upper bound from physical-container cgroup peaks; ANDREA host excluded" },
    ]);
    addStatCards(query("#overview-cards"), [
      { label: "Measured completion", value: `${formatInteger(data.summary.measured_runs_completed)} / ${formatInteger(data.summary.measured_runs_requested)}`, detail: `${formatInteger(data.summary.measured_runs_failed)} attributable failure(s)` },
      { label: "OOM observations", value: formatInteger(data.summary.oom_runs), detail: "From container telemetry" },
      { label: "Equivalent conditions", value: `${formatInteger(data.summary.equivalence_confirmed)} / ${formatInteger(data.summary.equivalence_conditions)}`, detail: `${formatInteger(data.summary.equivalence_failed)} failed; ${formatInteger(data.summary.equivalence_unavailable)} unavailable` },
      { label: "Output contract", value: "Full", detail: "CSV, reports, and graph exports retained" },
    ]);
    if (!data.summary.all_measured_runs_completed) {
      notice(query("#completion-notice"), "warning", "Incomplete endpoints", "Attributable failures remain in the results and speedup is withheld for incomplete conditions.");
    }
    if (data.summary.equivalence_failed > 0) {
      notice(query("#completion-notice"), "warning", "Semantic drift detected", "Speedup and parallel efficiency are withheld wherever canonical normalized networks differ across thread counts or system repetitions.");
    }
    renderCharts(data.aggregates, data.protocol.thread_counts);
    notice(query("#memory-notice"), "info", "Memory metric scope", "This metric is the overlap-aware upper bound formed from physical-container cgroup peaks. It excludes the ANDREA host process, so it is neither whole-process peak RAM nor total host memory use.");
    createTable(query("#equivalence-table"), [
      { label: "Dataset", value: (row) => row.dataset_id },
      { label: "Configuration", value: (row) => row.configuration_id },
      { label: "Completed", value: (row) => `${row.runs_completed} / ${row.runs_expected}` },
      { label: "Equivalent", value: (row) => row.equivalent_across_threads_and_repetitions === null ? "Unavailable" : row.equivalent_across_threads_and_repetitions ? "Yes" : "No" },
      { label: "Scaling metrics", value: (row) => row.equivalent_across_threads_and_repetitions === true ? "Published" : "Withheld" },
      { label: "Distinct networks", value: (row) => row.distinct_networks, numeric: true },
    ], data.summary.equivalence);
    const tableRows = data.measurements.map((row) => ({
      sequence: row.sequence_index, dataset: row.dataset.dataset_id, size: row.dataset.size_class,
      configuration: row.configuration_id, threads: row.threads, repetition: row.system_repetition,
      status: row.status, wall: row.measurement.wall_time_seconds,
      cpu: row.measurement.cpu_time_seconds, memory: row.measurement.peak_memory_bytes,
      output: row.measurement.output_bytes, oom: row.oom_observed ? "yes" : "no",
    }));
    const exportColumns = [
      { label: "Sequence", key: "sequence" }, { label: "Dataset", key: "dataset" },
      { label: "Size", key: "size" }, { label: "Configuration", key: "configuration" },
      { label: "Threads", key: "threads" }, { label: "Repetition", key: "repetition" },
      { label: "Status", key: "status" }, { label: "Wall seconds", key: "wall" },
      { label: "CPU seconds", key: "cpu" }, { label: "Container-memory upper-bound bytes", key: "memory" },
      { label: "Output bytes", key: "output" }, { label: "OOM", key: "oom" },
    ];
    query("#measurement-actions").appendChild(csvButton("Download visible measurements", "performance-measurements-visible.csv", exportColumns, tableRows));
    createTable(query("#measurement-table"), [
      { label: "#", value: (row) => row.sequence, numeric: true },
      { label: "Dataset", value: (row) => row.dataset },
      { label: "Configuration", value: (row) => row.configuration },
      { label: "Threads", value: (row) => row.threads, numeric: true },
      { label: "Rep", value: (row) => row.repetition, numeric: true },
      { label: "Status", value: (row) => humanize(row.status) },
      { label: "Wall", value: (row) => formatSeconds(row.wall), numeric: true },
      { label: "CPU", value: (row) => formatSeconds(row.cpu), numeric: true },
      { label: "Container-memory upper bound", value: (row) => bytes(row.memory), numeric: true },
      { label: "Output", value: (row) => bytes(row.output), numeric: true },
      { label: "OOM", value: (row) => row.oom },
    ], tableRows);
    notice(query("#io-notice"), "warning", "Disk I/O unavailable", "ANDREA v1 records exact output bytes but not block-device read/write counters; the JSON and CSV therefore expose I/O as explicitly unavailable.");
    const runtime = data.execution_environment.runtime_plan;
    renderKeyValues(query("#environment"), "Runtime plan", [
      { label: "Execution class", value: runtime.execution_class },
      { label: "Host logical CPUs", value: formatInteger(runtime.detected.logical_cpus_host) },
      { label: "Effective CPUs", value: formatInteger(runtime.detected.effective_cpu_threads) },
      { label: "Allowed CPU set", value: runtime.detected.allowed_cpus.join(", ") },
      { label: "CPU capacity", value: `${formatInteger(runtime.capacity.cpu_threads)} threads` },
      { label: "Host RAM", value: bytes(runtime.detected.host_memory_total_bytes) },
      { label: "Effective available RAM", value: bytes(runtime.detected.effective_memory_bytes) },
      { label: "RAM capacity", value: bytes(runtime.capacity.memory_bytes) },
      { label: "System", value: `${data.execution_environment.andrea.system} · ${data.execution_environment.andrea.machine}` },
      { label: "ANDREA commit", value: data.execution_environment.andrea_commit },
      { label: "SPATHI image", value: data.execution_environment.spathi_image_id },
    ]);
    renderLinks(query("#report-links"), "Canonical artifacts", data.links);
    renderKeyValues(query("#study-notes"), "Interpretation notes", data.notes.map((note, index) => ({ label: `${index + 1}`, value: note })));
  });
}());
