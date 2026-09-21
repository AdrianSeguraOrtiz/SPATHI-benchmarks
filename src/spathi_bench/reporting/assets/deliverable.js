(function () {
  "use strict";

  const COLORS = ["#0f8c84", "#2563eb", "#7c3aed", "#db6b22", "#d9467a", "#64748b", "#0891b2", "#65a30d"];
  const PHASE_COPY = {
    calibration: {
      title: "Configuration selection",
      kicker: "Phase 1",
      lead: "All declared candidates receive the same development budget; finalists are selected and confirmed on independent data.",
      scope: "Configuration results are selection evidence, not a comparison with external methods."
    },
    comparison: {
      title: "Held-out accuracy comparison",
      kicker: "Phase 2",
      lead: "The frozen SPATHI policy is compared within predeclared information and output-contract families.",
      scope: "Rankings are only meaningful inside one competitor family and on complete paired blocks."
    },
    scalability: {
      title: "Computational scalability",
      kicker: "Phase 3",
      lead: "Runtime, memory, parallel scaling and evidence-based stopping are measured along the common synthetic size ladder.",
      scope: "A timeout or feasibility stop is a computational result; it is never converted into an accuracy value."
    }
  };

  function el(tag, options) {
    const node = document.createElement(tag);
    const opts = options || {};
    if (opts.className) node.className = opts.className;
    if (opts.text !== undefined) node.textContent = String(opts.text);
    if (opts.id) node.id = opts.id;
    if (opts.attrs) Object.entries(opts.attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
    return node;
  }

  function append(parent) {
    Array.from(arguments).slice(1).flat(Infinity).forEach((child) => {
      if (child !== null && child !== undefined) parent.appendChild(child instanceof Node ? child : document.createTextNode(String(child)));
    });
    return parent;
  }

  function number(value) {
    if (value === null || value === undefined || value === "" || typeof value === "boolean") return NaN;
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : NaN;
  }

  function first(row, keys, fallback) {
    for (const key of keys) if (row[key] !== undefined && row[key] !== null && row[key] !== "") return row[key];
    return fallback;
  }

  function unique(values) { return Array.from(new Set(values.filter((value) => value !== null && value !== undefined && value !== ""))); }
  function median(values) {
    const numeric = values.map(number).filter(Number.isFinite).sort((a, b) => a - b);
    if (!numeric.length) return NaN;
    const middle = Math.floor(numeric.length / 2);
    return numeric.length % 2 ? numeric[middle] : (numeric[middle - 1] + numeric[middle]) / 2;
  }
  function mean(values) {
    const numeric = values.map(number).filter(Number.isFinite);
    return numeric.length ? numeric.reduce((total, value) => total + value, 0) / numeric.length : NaN;
  }
  function standardDeviation(values) {
    const numeric = values.map(number).filter(Number.isFinite);
    if (!numeric.length) return NaN;
    const center = mean(numeric);
    return Math.sqrt(numeric.reduce((total, value) => total + ((value - center) ** 2), 0) / numeric.length);
  }
  function human(value) { return String(value === undefined || value === null ? "—" : value).replace(/[_-]+/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase()); }
  function compact(value) {
    const numeric = number(value);
    if (!Number.isFinite(numeric)) return value === undefined || value === null || value === "" ? "—" : String(value);
    if (Math.abs(numeric) >= 1000000) return numeric.toExponential(2);
    if (numeric !== 0 && Math.abs(numeric) < .0001) return numeric.toExponential(3);
    return numeric.toLocaleString("en-US", { maximumFractionDigits: 4 });
  }
  function seconds(value) {
    const numeric = number(value);
    if (!Number.isFinite(numeric)) return "—";
    if (numeric < 60) return `${compact(numeric)} s`;
    if (numeric < 3600) return `${compact(numeric / 60)} min`;
    return `${compact(numeric / 3600)} h`;
  }
  function bytes(value) {
    const numeric = number(value);
    if (!Number.isFinite(numeric)) return "—";
    const gib = numeric / (1024 ** 3);
    return gib >= 1 ? `${compact(gib)} GiB` : `${compact(numeric / (1024 ** 2))} MiB`;
  }

  function statusTone(value) {
    const status = String(value || "").toLowerCase();
    if (status === "not_attempted") return "not-attempted";
    if (["completed", "complete", "success", "selected", "eligible"].includes(status)) return "success";
    if (["timeout", "estimated_infeasible_time", "memory_limit", "warning", "pending", "completed_with_warnings", "insufficient_complete_blocks", "missing_methods", "no_data"].includes(status)) return "warning";
    if (["failed", "error", "runtime_failure", "invalid"].includes(status)) return "error";
    return "neutral";
  }

  function badge(value) { return el("span", { className: `status status--${statusTone(value)}`, text: human(value || "unknown") }); }

  function stat(label, value, detail) {
    const card = el("article", { className: "stat" });
    append(card, el("p", { className: "stat__label", text: label }), el("p", { className: "stat__value", text: value }));
    if (detail) card.appendChild(el("p", { className: "stat__detail", text: detail }));
    return card;
  }

  function sectionHeading(kicker, title, help) {
    const heading = el("div", { className: "section__heading" });
    const copy = el("div");
    append(copy, el("p", { className: "kicker", text: kicker }), el("h2", { text: title }));
    append(heading, copy, help ? el("p", { className: "muted", text: help }) : null);
    return heading;
  }

  function plotCard(title, explanation, wide) {
    const card = el("article", { className: `plot-card${wide ? " plot-card--wide" : ""}` });
    const head = el("div", { className: "plot-card__head" });
    append(head, el("h3", { text: title }), el("p", { text: explanation }));
    const plot = el("div", { className: "plot", attrs: { role: "img", "aria-label": title } });
    append(card, head, plot);
    return { card, plot };
  }

  function renderTable(container, rows, preferred) {
    if (!rows.length) { container.appendChild(el("div", { className: "empty", text: "No records were emitted for this section." })); return; }
    const keys = unique(rows.flatMap((row) => Object.keys(row)));
    const columns = [...(preferred || []).filter((key) => keys.includes(key)), ...keys.filter((key) => !(preferred || []).includes(key)).sort()].slice(0, 20);
    const wrap = el("div", { className: "table-wrap" });
    const table = el("table");
    const header = el("tr");
    columns.forEach((key) => header.appendChild(el("th", { text: human(key) })));
    const thead = el("thead"); thead.appendChild(header); table.appendChild(thead);
    const tbody = el("tbody");
    rows.slice(0, 2000).forEach((row) => {
      const tr = el("tr");
      columns.forEach((key) => {
        const cell = el("td");
        const value = row[key];
        if (key === "status") cell.appendChild(badge(value));
        else if (value && typeof value === "object") cell.textContent = JSON.stringify(value);
        else cell.textContent = compact(value);
        tr.appendChild(cell);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody); wrap.appendChild(table); container.appendChild(wrap);
    if (rows.length > 2000) container.appendChild(el("p", { className: "muted", text: `Showing the first 2,000 of ${rows.length.toLocaleString("en-US")} embedded records.` }));
  }

  function metricRows(rows) {
    const expanded = [];
    rows.forEach((row) => {
      const direct = number(first(row, ["value", "metric_value", "score"], NaN));
      if (row.metric && Number.isFinite(direct)) expanded.push({ ...row, metric: String(row.metric), value: direct });
      if (row.metrics && typeof row.metrics === "object") Object.entries(row.metrics).forEach(([metric, value]) => {
        const numeric = number(value);
        if (Number.isFinite(numeric)) expanded.push({ ...row, metric, value: numeric });
      });
    });
    return expanded;
  }

  function outcomeGroup(status) {
    const value = String(status || "completed").toLowerCase();
    if (["complete", "completed", "completed_with_warnings", "success"].includes(value)) return "completed";
    if (["failed", "error", "runtime_failure", "invalid"].includes(value)) return "failed";
    return "stopped";
  }

  function chooseMetric(rows, summary) {
    const metrics = unique(rows.map((row) => row.metric));
    const requested = summary.primary_metric;
    if (requested && metrics.includes(requested)) return requested;
    for (const preferred of ["aupr_directed", "aupr", "AUPR"]) if (metrics.includes(preferred)) return preferred;
    return metrics[0] || null;
  }

  function plot(div, traces, layout) {
    const settings = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d"] };
    Plotly.newPlot(div, traces, { margin: { l: 64, r: 20, t: 28, b: 72 }, paper_bgcolor: "#fff", plot_bgcolor: "#fff", font: { family: "Inter, system-ui, sans-serif", color: "#27364c" }, hoverlabel: { namelength: -1 }, ...layout }, settings);
  }

  function datasetRows(phase) {
    const declared = phase.summary.datasets;
    if (Array.isArray(declared)) return declared.map((row) => {
      const deviations = row.dimension_deviations && typeof row.dimension_deviations === "object" ? row.dimension_deviations : {};
      const deviationText = Object.entries(deviations).map(([axis, value]) => `${human(axis)} ${compact(value.requested)} → ${compact(value.observed)}`).join("; ");
      return {
        ...row,
        dataset_id: first(row, ["dataset_id", "id"], null),
        simulator: first(row, ["simulator", "generator"], null),
        dataset_role: first(row, ["dataset_role", "role"], null),
        cells: first(row, ["cells"], row.dimensions && row.dimensions.cells),
        genes: first(row, ["genes"], row.dimensions && row.dimensions.genes),
        tf_count: first(row, ["tf_count"], row.dimensions && row.dimensions.transcription_factors),
        groups: first(row, ["groups"], row.dimensions && row.dimensions.groups),
        dimension_contract: row.observed_dimensions ? (deviationText || "Exact match") : "Planned; not materialized"
      };
    });
    const map = new Map();
    phase.rows.forEach((row) => {
      const id = first(row, ["dataset_id", "dataset", "instance_id"], null);
      if (!id || map.has(id)) return;
      map.set(id, { dataset_id: id, simulator: row.simulator, dataset_role: row.dataset_role || row.role, cells: row.cells, genes: row.genes, tf_count: row.tf_count || row.tfs, groups: row.groups, seed: row.seed });
    });
    return Array.from(map.values());
  }

  function renderDatasetContext(root, phase) {
    const section = el("section", { className: "section table-card dataset-context" });
    append(section, sectionHeading(
      "Dataset context",
      "Synthetic evidence represented in this phase",
      "Simulator, scientific role, observed dimensions and seed are shown before any result plot. Any accepted simulator-specific difference from the prospective design is stated explicitly."
    ));
    renderTable(section, datasetRows(phase), ["dataset_id", "simulator", "dataset_role", "cells", "genes", "tf_count", "groups", "dimension_contract", "seed"]);
    root.appendChild(section);
  }

  function renderEvidenceTables(root, phase) {
    const section = el("section", { className: "section" });
    section.appendChild(sectionHeading("Evidence", "Decisions and exceptions", "Every displayed record is embedded in this HTML."));
    const evidenceGrid = el("div", { className: "grid section" });
    const decisions = el("article", { className: "table-card" }); append(decisions, el("h3", { text: "Recorded decisions" }));
    renderTable(decisions, phase.decisions, ["decision", "selected", "reason", "metric", "value"]);
    const failures = el("article", { className: "table-card" }); append(failures, el("h3", { text: "Failures and feasibility stops" }));
    renderTable(failures, phase.failures, ["method_id", "dataset_id", "scale_id", "status", "reason"]);
    append(evidenceGrid, decisions, failures); section.appendChild(evidenceGrid); root.appendChild(section);
  }

  function calibrationPlots(root, phase) {
    const candidateId = (row) => first(row, ["configuration_id", "candidate_id", "configuration"], "unknown");
    const treeCount = (row) => number(first(row, ["n_estimators"], row.parameters && row.parameters.n_estimators));
    const summaries = Array.isArray(phase.summary.stage_summaries) ? phase.summary.stage_summaries : [];
    const validationPrefixes = unique(summaries.flatMap((summary) => Array.isArray(summary.prefixes) ? summary.prefixes.map(number).filter(Number.isFinite) : []));
    const finalistDecision = phase.decisions.find((row) => row.decision === "advance_equal_candidates") || {};
    const finalists = Array.isArray(finalistDecision.selected) ? finalistDecision.selected.map(String) : [];
    const winnerDecision = phase.decisions.find((row) => row.decision === "selected_configuration") || {};
    const winnerId = winnerDecision.selected === undefined ? null : String(winnerDecision.selected);
    const winnerTrees = number(winnerDecision.n_estimators);
    const threshold = number(winnerDecision.practical_equivalence_threshold);
    const metrics = metricRows(phase.rows);
    const metric = chooseMetric(metrics, phase.summary);
    const primary = metrics.filter((row) => row.metric === metric);
    const explicitStage = (row) => String(first(row, ["calibration_stage", "stage", "dataset_role", "role"], "")).toLowerCase();
    const isValidation = (row) => {
      const stage = explicitStage(row);
      if (stage.includes("valid")) return true;
      if (stage.includes("develop") || stage.includes("screen")) return false;
      const trees = treeCount(row);
      const runtime = number(first(row, ["duration_seconds", "wall_seconds", "runtime_seconds"], NaN));
      return finalists.includes(String(candidateId(row))) && validationPrefixes.includes(trees) && !Number.isFinite(runtime);
    };
    const validationRows = primary.filter(isValidation);
    const screeningRows = primary.filter((row) => !isValidation(row));
    const screeningTrees = unique(screeningRows.map(treeCount).filter(Number.isFinite)).sort((a, b) => a - b);
    const candidateIds = unique(screeningRows.map(candidateId)).sort();

    const screening = el("section", { className: "section" });
    screening.appendChild(sectionHeading(
      "Calibration stage 1",
      `Equal-budget screening${screeningTrees.length === 1 ? ` at ${screeningTrees[0]} trees` : ""}`,
      "All declared configurations are evaluated on the development blocks with the same forest size and seed budget; no candidate is a reference or control."
    ));
    const screeningGrid = el("div", { className: "grid" });
    const accuracy = plotCard("Screening accuracy", `${human(metric || "metric")} distribution for every candidate using development data only.`, true);
    if (screeningRows.length) {
      const traces = candidateIds.map((candidate, index) => ({
        type: "box",
        name: candidate,
        y: screeningRows.filter((row) => candidateId(row) === candidate).map((row) => row.value),
        boxpoints: "all",
        jitter: .25,
        pointpos: 0,
        marker: { color: COLORS[index % COLORS.length], size: 5 },
        line: { color: COLORS[index % COLORS.length] }
      }));
      plot(accuracy.plot, traces, { yaxis: { title: human(metric), rangemode: "tozero" }, xaxis: { title: "Candidate configuration", tickangle: -35 }, showlegend: false });
    } else accuracy.plot.appendChild(el("div", { className: "empty", text: "No development-screening observations are available." }));

    const pareto = plotCard("Screening accuracy–runtime", "Development-only mean accuracy and mean per-run wall time. Validation prefixes are deliberately excluded because they share one physical forest fit.", false);
    const aggregates = candidateIds.map((candidate) => {
      const rows = screeningRows.filter((row) => candidateId(row) === candidate);
      const time = mean(rows.map((row) => first(row, ["duration_seconds", "wall_seconds", "runtime_seconds"], NaN)));
      return { candidate, accuracy: mean(rows.map((row) => row.value)), time, observations: rows.length };
    }).filter((row) => Number.isFinite(row.time) && Number.isFinite(row.accuracy));
    if (aggregates.length) plot(pareto.plot, [{
      type: "scatter",
      mode: "markers+text",
      x: aggregates.map((row) => row.time),
      y: aggregates.map((row) => row.accuracy),
      text: aggregates.map((row) => row.candidate),
      customdata: aggregates.map((row) => [row.observations]),
      textposition: "top center",
      marker: { size: 11, color: "#2563eb" },
      hovertemplate: "%{text}<br>Mean wall time: %{x:.3g} s<br>Mean score: %{y:.4f}<br>Observations: %{customdata[0]}<extra></extra>"
    }], { xaxis: { title: "Mean screening wall time per run (seconds)", type: "log" }, yaxis: { title: `Mean ${human(metric)}` }, showlegend: false });
    else pareto.plot.appendChild(el("div", { className: "empty", text: "Per-run screening telemetry is not available." }));

    const consistency = plotCard("Screening consistency", "Mean development score by candidate and independent dataset block; validation data are not used here.", false);
    const developmentDatasets = unique(screeningRows.map((row) => first(row, ["dataset_id", "dataset"], "unknown"))).sort();
    if (candidateIds.length && developmentDatasets.length) {
      const z = candidateIds.map((candidate) => developmentDatasets.map((dataset) => {
        const values = screeningRows.filter((row) => candidateId(row) === candidate && first(row, ["dataset_id", "dataset"], "unknown") === dataset).map((row) => row.value);
        const value = mean(values);
        return Number.isFinite(value) ? value : null;
      }));
      plot(consistency.plot, [{ type: "heatmap", x: developmentDatasets, y: candidateIds, z, colorscale: "Blues", colorbar: { title: human(metric) }, hoverongaps: false }], { xaxis: { title: "Development dataset block", tickangle: -30 }, yaxis: { title: "Candidate configuration", automargin: true } });
    } else consistency.plot.appendChild(el("div", { className: "empty", text: "No paired development evidence is available." }));
    append(screeningGrid, accuracy.card, pareto.card, consistency.card);
    screening.appendChild(screeningGrid);
    root.appendChild(screening);

    const advancement = el("section", { className: "section table-card" });
    append(advancement, sectionHeading(
      "Calibration stage 2",
      "Finalist advancement",
      "The predeclared primary metric, screening runtime and between-block stability reduce the equal candidate set to the configurations evaluated on independent validation data."
    ));
    const advancementRows = phase.rows.filter((row) => Number.isFinite(number(row.mean_aupr)) && Object.prototype.hasOwnProperty.call(row, "mean_duration_seconds"));
    renderTable(advancement, advancementRows.map((row, index) => ({
      screening_rank: index + 1,
      configuration_id: candidateId(row),
      mean_aupr: row.mean_aupr,
      stability_sd: row.stability_sd,
      mean_duration_seconds: row.mean_duration_seconds,
      advanced: finalists.includes(String(candidateId(row)))
    })), ["screening_rank", "configuration_id", "mean_aupr", "stability_sd", "mean_duration_seconds", "advanced"]);
    root.appendChild(advancement);

    const validation = el("section", { className: "section" });
    validation.appendChild(sectionHeading(
      "Calibration stage 3",
      "Exact-prefix validation",
      "Each finalist is fit once as a deterministic maximum-size forest on independent validation blocks. Exact tree prefixes are scored from that shared fit, so runtime is not assigned to individual prefix points."
    ));
    const validationCard = plotCard("Validation convergence", `${human(metric || "metric")} versus the exact categorical tree prefix; points show the mean and population standard deviation across validation observations.`, true);
    const validatedCandidates = unique(validationRows.map(candidateId)).sort();
    const prefixes = unique([...validationPrefixes, ...validationRows.map(treeCount).filter(Number.isFinite)]).sort((a, b) => a - b);
    if (validationRows.length && prefixes.length) {
      const traces = validatedCandidates.map((candidate, index) => {
        const points = prefixes.map((trees) => {
          const values = validationRows.filter((row) => candidateId(row) === candidate && treeCount(row) === trees).map((row) => row.value);
          return { trees, mean: mean(values), sd: standardDeviation(values), observations: values.length };
        }).filter((point) => Number.isFinite(point.mean));
        return {
          type: "scatter",
          mode: "lines+markers",
          name: candidate,
          x: points.map((point) => String(point.trees)),
          y: points.map((point) => point.mean),
          error_y: { type: "data", array: points.map((point) => point.sd), visible: true },
          customdata: points.map((point) => [point.sd, point.observations]),
          hovertemplate: "Candidate: %{fullData.name}<br>Trees: %{x}<br>Mean score: %{y:.4f}<br>SD: %{customdata[0]:.4f}<br>Observations: %{customdata[1]}<extra></extra>",
          marker: { color: COLORS[index % COLORS.length], size: 8 },
          line: { color: COLORS[index % COLORS.length], width: 2.5 }
        };
      });
      plot(validationCard.plot, traces, { xaxis: { title: "Exact tree prefix", type: "category", categoryorder: "array", categoryarray: prefixes.map(String) }, yaxis: { title: `Mean validation ${human(metric)}`, rangemode: "tozero" }, legend: { orientation: "h", y: -0.28 } });
    } else validationCard.plot.appendChild(el("div", { className: "empty", text: "No independent exact-prefix validation observations are available." }));
    validation.appendChild(validationCard.card);
    root.appendChild(validation);

    const selection = el("section", { className: "section" });
    selection.appendChild(sectionHeading(
      "Calibration stage 4",
      "Final selection and practical-equivalence band",
      "All finalist-prefix combinations remain candidates. The declared tolerance forms a practical-equivalence band; stability and then tree count break ties inside that band. This is a deterministic selection rule, not a statistical non-inferiority test."
    ));
    const selectionRows = phase.rows.filter((row) => Number.isFinite(number(row.mean_aupr)) && Number.isFinite(number(row.n_estimators)) && !Object.prototype.hasOwnProperty.call(row, "mean_duration_seconds"));
    const orderedSelection = [...selectionRows].sort((left, right) => number(right.mean_aupr) - number(left.mean_aupr));
    const selectionCard = plotCard("Validated candidate-prefix ranking", "Mean validation accuracy for every finalist and exact tree prefix. The shaded region marks the recorded practical-equivalence band.", true);
    if (orderedSelection.length) {
      const labels = orderedSelection.map((row) => `${candidateId(row)} · ${compact(row.n_estimators)} trees`);
      const values = orderedSelection.map((row) => number(row.mean_aupr));
      const best = Math.max(...values);
      const colors = orderedSelection.map((row) => candidateId(row) === winnerId && number(row.n_estimators) === winnerTrees ? "#0f8c84" : "#93a4bb");
      const shapes = Number.isFinite(threshold) ? [{ type: "rect", xref: "paper", x0: 0, x1: 1, yref: "y", y0: threshold, y1: best, fillcolor: "rgba(15, 140, 132, .12)", line: { width: 0 }, layer: "below" }, { type: "line", xref: "paper", x0: 0, x1: 1, yref: "y", y0: threshold, y1: threshold, line: { color: "#0f8c84", dash: "dash" } }] : [];
      plot(selectionCard.plot, [{
        type: "bar",
        x: labels,
        y: values,
        marker: { color: colors },
        customdata: orderedSelection.map((row) => [row.stability_sd, Number.isFinite(threshold) && number(row.mean_aupr) >= threshold ? "yes" : "no", candidateId(row) === winnerId && number(row.n_estimators) === winnerTrees ? "selected" : "candidate"]),
        hovertemplate: "%{x}<br>Mean validation score: %{y:.4f}<br>Stability SD: %{customdata[0]:.4f}<br>Inside practical band: %{customdata[1]}<br>Status: %{customdata[2]}<extra></extra>"
      }], { xaxis: { title: "Candidate configuration and exact tree prefix", type: "category", tickangle: -40 }, yaxis: { title: `Mean validation ${human(metric)}`, rangemode: "tozero" }, shapes, showlegend: false });
    } else selectionCard.plot.appendChild(el("div", { className: "empty", text: "The final candidate-prefix ranking is not yet available." }));
    selection.appendChild(selectionCard.card);
    const ranking = el("article", { className: "table-card" });
    append(ranking, el("h3", { text: "Final selection ledger" }), el("p", { className: "muted", text: Number.isFinite(threshold) ? `Recorded practical-equivalence threshold: ${compact(threshold)}.` : "The practical-equivalence threshold is not yet available." }));
    renderTable(ranking, orderedSelection.map((row, index) => ({
      validation_rank: index + 1,
      configuration_id: candidateId(row),
      n_estimators: row.n_estimators,
      mean_aupr: row.mean_aupr,
      stability_sd: row.stability_sd,
      within_practical_band: Number.isFinite(threshold) ? number(row.mean_aupr) >= threshold : null,
      selected: candidateId(row) === winnerId && number(row.n_estimators) === winnerTrees
    })), ["validation_rank", "configuration_id", "n_estimators", "mean_aupr", "stability_sd", "within_practical_band", "selected"]);
    selection.appendChild(ranking);
    root.appendChild(selection);

    const parameterRows = [];
    const seenCandidates = new Set();
    screeningRows.forEach((row) => {
      const candidate = candidateId(row);
      if (!candidate || candidate === "unknown" || seenCandidates.has(candidate) || !row.parameters || typeof row.parameters !== "object") return;
      seenCandidates.add(candidate);
      const declared = { ...row.parameters };
      delete declared.random_seed;
      delete declared.n_estimators;
      parameterRows.push({ configuration_id: candidate, ...declared });
    });
    const parameters = el("section", { className: "section table-card" });
    append(parameters, sectionHeading(
      "Declared inputs",
      "Complete candidate parameter vectors",
      "Stage-controlled tree counts and execution seeds are reported in their stages, not as candidate parameters. Every configuration below entered screening with the same budget."
    ));
    renderTable(parameters, parameterRows.sort((left, right) => String(left.configuration_id).localeCompare(String(right.configuration_id))), ["configuration_id", "weight_mode", "distance_space", "n_components", "distance_standardization", "distance_metric", "kernel", "bandwidth_scale", "group_size_correction", "tree_method", "max_features", "min_samples_leaf", "max_depth", "min_weight_fraction_leaf"]);
    root.appendChild(parameters);
  }

  function comparisonFamilyDefinitions(phase) {
    const stageSummaries = Array.isArray(phase.summary.stage_summaries) ? phase.summary.stage_summaries : [];
    const sources = [phase.summary.families, phase.summary.competitor_families];
    stageSummaries.forEach((summary) => {
      if (summary && typeof summary === "object") sources.push(summary.families, summary.competitor_families);
    });
    const definitions = new Map();
    sources.forEach((source) => {
      if (!Array.isArray(source)) return;
      source.forEach((raw) => {
        if (!raw || typeof raw !== "object" || raw.id === undefined) return;
        const id = String(raw.id);
        if (!definitions.has(id)) definitions.set(id, raw);
      });
    });
    return Array.from(definitions.values());
  }

  function comparisonFamilyLabel(phase, familyId) {
    const definition = comparisonFamilyDefinitions(phase).find((family) => String(family.id) === String(familyId));
    return definition && definition.label ? String(definition.label) : human(familyId);
  }

  function comparisonCoverageRows(phase) {
    const explicit = phase.summary.execution_coverage;
    const candidates = [];
    if (Array.isArray(explicit)) {
      explicit.forEach((row) => {
        if (row && typeof row === "object") candidates.push(row);
      });
    } else {
      const methodFamilies = new Map();
      phase.rows.forEach((row) => {
        const method = first(row, ["method_id", "tool_id", "method"], null);
        const family = first(row, ["family", "competitor_family"], null);
        if (!method || !family) return;
        if (!methodFamilies.has(String(method))) methodFamilies.set(String(method), new Set());
        methodFamilies.get(String(method)).add(String(family));
      });
      const inferredFamilies = (row) => {
        const declared = first(row, ["family", "competitor_family"], null);
        if (declared) return [String(declared)];
        const method = String(first(row, ["method_id", "tool_id", "method"], ""));
        const observed = methodFamilies.get(method);
        if (observed && observed.size) return Array.from(observed);
        return comparisonFamilyDefinitions(phase)
          .filter((family) => {
            const routes = Array.isArray(family.route_ids) ? family.route_ids : family.methods;
            return Array.isArray(routes) && routes.map(String).includes(method);
          })
          .map((family) => String(family.id));
      };
      [...phase.rows, ...phase.failures].forEach((row) => {
        const dataset = first(row, ["dataset_id", "dataset", "instance_id"], null);
        const method = first(row, ["method_id", "tool_id", "method"], null);
        if (!dataset || !method) return;
        inferredFamilies(row).forEach((family) => candidates.push({
          dataset_id: dataset,
          method_id: method,
          family,
          status: row.status || "completed"
        }));
      });
    }

    const severity = { completed: 0, stopped: 1, failed: 2 };
    const deduplicated = new Map();
    candidates.forEach((row) => {
      const dataset = first(row, ["dataset_id", "dataset", "instance_id"], null);
      const method = first(row, ["method_id", "tool_id", "method"], null);
      const family = first(row, ["family", "competitor_family"], null);
      if (!dataset || !method || !family) return;
      const normalized = { ...row, dataset_id: String(dataset), method_id: String(method), family: String(family), status: String(row.status || "completed") };
      const key = JSON.stringify([normalized.dataset_id, normalized.method_id, normalized.family]);
      const previous = deduplicated.get(key);
      if (!previous || severity[outcomeGroup(normalized.status)] > severity[outcomeGroup(previous.status)]) deduplicated.set(key, normalized);
    });
    return Array.from(deduplicated.values());
  }

  function renderComparisonFamilyDefinitions(root, phase) {
    const section = el("section", { className: "section table-card" });
    append(section, sectionHeading(
      "Comparison contract",
      "Predeclared competitor families",
      "Each family fixes an information contract, output semantics and statistical role before execution. Methods are ranked only within their own family."
    ));
    const definitions = comparisonFamilyDefinitions(phase);
    if (!definitions.length) {
      section.appendChild(el("div", { className: "empty", text: "Formal competitor-family definitions are not available in the study snapshot." }));
    } else {
      renderTable(section, definitions.map((family) => ({
        family_id: family.id,
        label: family.label,
        methods: Array.isArray(family.route_ids) ? family.route_ids.join(", ") : (Array.isArray(family.methods) ? family.methods.join(", ") : "—"),
        statistical_role: human(family.statistical_role),
        rationale: family.rationale
      })), ["family_id", "label", "methods", "statistical_role", "rationale"]);
    }
    const summaries = Array.isArray(phase.summary.stage_summaries) ? phase.summary.stage_summaries : [];
    const estimand = summaries.map((summary) => summary?.estimand).find((value) => value && typeof value === "object");
    if (estimand) {
      append(section,
        el("h3", { text: "Prospectively declared estimand" }),
        el("p", { className: "muted", text: "This contract fixes how scenarios, independent synthetic realizations and missing executions enter the formal comparison." })
      );
      renderTable(section, [estimand], ["scenario_weighting", "sampling_unit", "missing_data_rule"]);
    }
    root.appendChild(section);
  }

  function comparisonPlots(root, phase) {
    const metrics = metricRows(phase.rows);
    const executionCoverage = comparisonCoverageRows(phase);
    const section = el("section", { className: "section" });
    section.appendChild(sectionHeading("Accuracy evidence", "Methods within declared contract families", "Use family-specific blocks; information-asymmetric methods are not pooled into one ranking."));
    const families = unique([
      ...metrics.map((row) => first(row, ["family", "competitor_family"], "unclassified")),
      ...executionCoverage.map((row) => row.family)
    ]);
    const toolbar = el("div", { className: "toolbar" });
    const label = el("label", { className: "control", text: "Competitor family" });
    const select = el("select"); families.forEach((family) => select.appendChild(el("option", { text: comparisonFamilyLabel(phase, family), attrs: { value: family } }))); label.appendChild(select);
    const metricLabel = el("label", { className: "control", text: "Accuracy metric" });
    const metricSelect = el("select"); metricLabel.appendChild(metricSelect);
    append(toolbar, label, metricLabel); section.appendChild(toolbar);
    const grid = el("div", { className: "grid" });
    const accuracy = plotCard("Available-case accuracy distributions (descriptive)", "Selected metric on every held-out synthetic instance observed for each method. Unequal sample sizes are retained for transparency; this plot is descriptive and is not used for paired inference.", true);
    const completeAccuracy = plotCard("Complete-block accuracy distributions (descriptive)", "The same jointly complete dataset blocks used by the paired analysis, displayed descriptively. Friedman and Wilcoxon–Holm inference is reported separately below.", true);
    const heat = plotCard("Paired dataset scores", "Missing cells remain blank and are excluded from complete-block statistics.", false);
    const coverage = plotCard("Execution coverage", "Unique dataset–method executions, including methods with no accuracy row. Multiple metric rows from one execution are counted once.", false);
    append(grid, accuracy.card, completeAccuracy.card, heat.card, coverage.card); section.appendChild(grid); root.appendChild(section);

    function completeBlockRows(family, metric, rows) {
      const analysis = statisticsAnalyses(phase).find((candidate) => candidate.family === family && candidate.metric === metric);
      if (!analysis || !Array.isArray(analysis.complete_blocks) || !Array.isArray(analysis.block_fields)) return [];
      const keys = new Set(analysis.complete_blocks.map((block) => JSON.stringify(analysis.block_fields.map((field) => block[field]))));
      return rows.filter((row) => keys.has(JSON.stringify(analysis.block_fields.map((field) => row[field]))));
    }

    function update() {
      const family = select.value || families[0];
      const familyRows = metrics.filter((row) => first(row, ["family", "competitor_family"], "unclassified") === family);
      const availableMetrics = unique(familyRows.map((row) => row.metric));
      const previousMetric = metricSelect.value;
      metricSelect.textContent = "";
      availableMetrics.forEach((metric) => metricSelect.appendChild(el("option", { text: human(metric), attrs: { value: metric } })));
      metricSelect.value = availableMetrics.includes(previousMetric) ? previousMetric : (availableMetrics.includes(chooseMetric(familyRows, phase.summary)) ? chooseMetric(familyRows, phase.summary) : availableMetrics[0]);
      const metric = metricSelect.value;
      const rows = familyRows.filter((row) => row.metric === metric);
      const methods = unique(rows.map((row) => first(row, ["method_id", "tool_id", "method"], "unknown")));
      const datasets = unique(rows.map((row) => first(row, ["dataset_id", "dataset"], "unknown")));
      plot(accuracy.plot, methods.map((method, index) => ({ type: "box", name: method, y: rows.filter((row) => first(row, ["method_id", "tool_id", "method"], "unknown") === method).map((row) => row.value), boxpoints: "all", jitter: .25, marker: { color: COLORS[index % COLORS.length], size: 5 }, line: { color: COLORS[index % COLORS.length] } })), { yaxis: { title: human(metric), rangemode: "tozero" }, xaxis: { title: "Method", tickangle: -25 }, showlegend: false });
      const pairedRows = completeBlockRows(family, metric, rows);
      const pairedMethods = unique(pairedRows.map((row) => first(row, ["method_id", "tool_id", "method"], "unknown")));
      if (pairedRows.length) {
        plot(completeAccuracy.plot, pairedMethods.map((method, index) => ({ type: "box", name: method, y: pairedRows.filter((row) => first(row, ["method_id", "tool_id", "method"], "unknown") === method).map((row) => row.value), boxpoints: "all", jitter: .25, marker: { color: COLORS[index % COLORS.length], size: 5 }, line: { color: COLORS[index % COLORS.length] } })), { yaxis: { title: human(metric), rangemode: "tozero" }, xaxis: { title: "Method (jointly complete blocks only)", tickangle: -25 }, showlegend: false });
      } else {
        Plotly.purge(completeAccuracy.plot);
        completeAccuracy.plot.textContent = "";
        completeAccuracy.plot.appendChild(el("div", { className: "empty", text: "No complete-block analysis is available for this family and metric." }));
      }
      const z = methods.map((method) => datasets.map((dataset) => {
        const values = rows.filter((row) => first(row, ["method_id", "tool_id", "method"], "unknown") === method && first(row, ["dataset_id", "dataset"], "unknown") === dataset).map((row) => row.value);
        return values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
      }));
      plot(heat.plot, [{ type: "heatmap", x: datasets, y: methods, z, colorscale: "Blues", colorbar: { title: human(metric) }, hoverongaps: false }], { xaxis: { title: "Held-out dataset", tickangle: -30 }, yaxis: { title: "Method", automargin: true } });
      const familyCoverage = executionCoverage.filter((row) => row.family === family);
      const coverageMethods = unique([
        ...methods,
        ...familyCoverage.map((row) => row.method_id)
      ]);
      const tones = ["completed", "stopped", "failed"];
      const traces = tones.map((tone, index) => ({
        type: "bar",
        name: human(tone),
        x: coverageMethods,
        y: coverageMethods.map((method) => familyCoverage.filter((row) =>
          row.method_id === method &&
          outcomeGroup(row.status) === tone
        ).length),
        marker: { color: ["#15803d", "#b45309", "#be123c"][index] }
      }));
      plot(coverage.plot, traces, { barmode: "stack", xaxis: { title: "Method", tickangle: -25 }, yaxis: { title: "Unique dataset–method executions", dtick: 1 } });
    }
    select.addEventListener("change", update);
    metricSelect.addEventListener("change", () => {
      const selectedMetric = metricSelect.value;
      const family = select.value || families[0];
      const rows = metrics.filter((row) => first(row, ["family", "competitor_family"], "unclassified") === family && row.metric === selectedMetric);
      const methods = unique(rows.map((row) => first(row, ["method_id", "tool_id", "method"], "unknown")));
      const datasets = unique(rows.map((row) => first(row, ["dataset_id", "dataset"], "unknown")));
      plot(accuracy.plot, methods.map((method, index) => ({ type: "box", name: method, y: rows.filter((row) => first(row, ["method_id", "tool_id", "method"], "unknown") === method).map((row) => row.value), boxpoints: "all", jitter: .25, marker: { color: COLORS[index % COLORS.length], size: 5 }, line: { color: COLORS[index % COLORS.length] } })), { yaxis: { title: human(selectedMetric), rangemode: "tozero" }, xaxis: { title: "Method", tickangle: -25 }, showlegend: false });
      const pairedRows = completeBlockRows(family, selectedMetric, rows);
      const pairedMethods = unique(pairedRows.map((row) => first(row, ["method_id", "tool_id", "method"], "unknown")));
      if (pairedRows.length) {
        plot(completeAccuracy.plot, pairedMethods.map((method, index) => ({ type: "box", name: method, y: pairedRows.filter((row) => first(row, ["method_id", "tool_id", "method"], "unknown") === method).map((row) => row.value), boxpoints: "all", jitter: .25, marker: { color: COLORS[index % COLORS.length], size: 5 }, line: { color: COLORS[index % COLORS.length] } })), { yaxis: { title: human(selectedMetric), rangemode: "tozero" }, xaxis: { title: "Method (jointly complete blocks only)", tickangle: -25 }, showlegend: false });
      } else {
        Plotly.purge(completeAccuracy.plot);
        completeAccuracy.plot.textContent = "";
        completeAccuracy.plot.appendChild(el("div", { className: "empty", text: "No complete-block analysis is available for this family and metric." }));
      }
      const z = methods.map((method) => datasets.map((dataset) => {
        const values = rows.filter((row) => first(row, ["method_id", "tool_id", "method"], "unknown") === method && first(row, ["dataset_id", "dataset"], "unknown") === dataset).map((row) => row.value);
        return values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
      }));
      plot(heat.plot, [{ type: "heatmap", x: datasets, y: methods, z, colorscale: "Blues", colorbar: { title: human(selectedMetric) }, hoverongaps: false }], { xaxis: { title: "Held-out dataset", tickangle: -30 }, yaxis: { title: "Method", automargin: true } });
    });
    update();
  }

  function statisticsAnalyses(phase) {
    const summaries = Array.isArray(phase.summary.stage_summaries) ? phase.summary.stage_summaries : [];
    return summaries.flatMap((summary) => {
      const statistics = summary && typeof summary.statistics === "object" ? summary.statistics : null;
      return statistics && Array.isArray(statistics.analyses) ? statistics.analyses : [];
    });
  }

  function comparisonStatistics(root, phase) {
    const analyses = statisticsAnalyses(phase);
    const section = el("section", { className: "section" });
    section.appendChild(sectionHeading("Paired inference", "Friedman rankings and Wilcoxon–Holm comparisons", "Each family–metric analysis uses one explicit set of complete paired blocks. No method is designated as a control: every unordered method pair is tested and corrected together."));
    if (!analyses.length) {
      section.appendChild(el("div", { className: "empty card", text: "Paired statistical results are not available yet." }));
      root.appendChild(section);
      return;
    }

    const toolbar = el("div", { className: "toolbar" });
    const familyLabel = el("label", { className: "control", text: "Competitor family" });
    const familySelect = el("select");
    const families = unique(analyses.map((analysis) => analysis.family));
    families.forEach((family) => familySelect.appendChild(el("option", { text: comparisonFamilyLabel(phase, family), attrs: { value: family } })));
    familyLabel.appendChild(familySelect);
    const metricLabel = el("label", { className: "control", text: "Metric" });
    const metricSelect = el("select"); metricLabel.appendChild(metricSelect);
    append(toolbar, familyLabel, metricLabel); section.appendChild(toolbar);

    const panel = el("div", { className: "analysis-panel" }); section.appendChild(panel);
    function renderAnalysis() {
      const family = familySelect.value || families[0];
      const matches = analyses.filter((analysis) => analysis.family === family);
      const previous = metricSelect.value;
      metricSelect.textContent = "";
      unique(matches.map((analysis) => analysis.metric)).forEach((metric) => metricSelect.appendChild(el("option", { text: human(metric), attrs: { value: metric } })));
      if (Array.from(metricSelect.options).some((option) => option.value === previous)) metricSelect.value = previous;
      const analysis = matches.find((candidate) => candidate.metric === metricSelect.value) || matches[0];
      panel.textContent = "";
      if (!analysis) return;

      const status = el("div", { className: "analysis-panel__status" });
      append(status, badge(analysis.status), el("strong", { text: `${comparisonFamilyLabel(phase, analysis.family)} · ${human(analysis.metric)}` }));
      if (analysis.reason) status.appendChild(el("p", { className: "muted", text: analysis.reason }));
      panel.appendChild(status);

      const candidateBlocks = Number(analysis.n_candidate_blocks || 0);
      const completeBlocks = Number(analysis.n_complete_blocks || 0);
      const excludedBlocks = Array.isArray(analysis.excluded_blocks) ? analysis.excluded_blocks : [];
      const blockCoverage = candidateBlocks ? 100 * completeBlocks / candidateBlocks : 0;
      const summary = el("div", { className: "stat-grid" });
      append(summary,
        stat("Declared methods", Array.isArray(analysis.methods) ? analysis.methods.length : 0),
        stat("Candidate blocks", candidateBlocks),
        stat("Complete blocks", completeBlocks, `${compact(blockCoverage)}% retained`),
        stat("Excluded blocks", excludedBlocks.length)
      );
      panel.appendChild(summary);

      const omnibus = analysis.omnibus || {};
      const friedman = el("article", { className: "table-card" });
      append(friedman, el("h3", { text: "Friedman omnibus test" }), el("p", { className: "muted", text: "The same complete paired blocks are used for the omnibus test, ranks and every post-hoc pair." }));
      renderTable(friedman, [{
        status: omnibus.status,
        test: omnibus.test,
        statistic: omnibus.statistic,
        degrees_of_freedom: omnibus.degrees_of_freedom,
        p_value: omnibus.p_value,
        n_blocks: omnibus.n_blocks,
        n_methods: omnibus.n_methods,
        reason: omnibus.reason
      }], ["status", "test", "statistic", "degrees_of_freedom", "p_value", "n_blocks", "n_methods", "reason"]);
      panel.appendChild(friedman);

      const rankings = el("article", { className: "table-card" });
      append(rankings, el("h3", { text: "Symmetric method rankings" }), el("p", { className: "muted", text: analysis.higher_is_better ? "Rank 1 denotes the highest score within each complete block." : "Rank 1 denotes the lowest score within each complete block." }));
      renderTable(rankings, Array.isArray(analysis.rankings) ? analysis.rankings : [], ["method_id", "average_rank", "mean_score_complete", "median_score_complete", "n_complete_blocks", "mean_score_observed", "n_observed"]);
      panel.appendChild(rankings);

      const pairs = el("article", { className: "table-card" });
      append(pairs, el("h3", { text: "All-pairs Wilcoxon tests with Holm correction" }), el("p", { className: "muted", text: `Two-sided paired tests across all unordered method pairs; Holm family-wise error rate α = ${compact(analysis.alpha)}.` }));
      renderTable(pairs, Array.isArray(analysis.pairwise) ? analysis.pairwise : [], ["method_a", "method_b", "status", "n_blocks", "wins_a", "ties", "wins_b", "median_difference_a_minus_b", "p_value", "p_holm", "reject_holm", "method", "reason"]);
      panel.appendChild(pairs);

      const coverageRows = (Array.isArray(analysis.rankings) ? analysis.rankings : []).map((ranking) => ({
        method_id: ranking.method_id,
        observed_blocks: ranking.n_observed,
        candidate_blocks: candidateBlocks,
        observed_coverage_percent: candidateBlocks ? 100 * Number(ranking.n_observed || 0) / candidateBlocks : 0,
        complete_blocks_used: completeBlocks,
        complete_block_coverage_percent: blockCoverage
      }));
      const coverage = el("article", { className: "table-card" });
      append(coverage, el("h3", { text: "Analysis coverage" }), el("p", { className: "muted", text: "Observed coverage is descriptive; inference is restricted to the jointly complete blocks reported below." }));
      renderTable(coverage, coverageRows, ["method_id", "observed_blocks", "candidate_blocks", "observed_coverage_percent", "complete_blocks_used", "complete_block_coverage_percent"]);
      panel.appendChild(coverage);

      const blocks = el("div", { className: "statistics-grid" });
      const complete = el("article", { className: "table-card" });
      append(complete, el("h3", { text: "Complete blocks used" }));
      renderTable(complete, Array.isArray(analysis.complete_blocks) ? analysis.complete_blocks : [], Array.isArray(analysis.block_fields) ? analysis.block_fields : []);
      const excluded = el("article", { className: "table-card" });
      append(excluded, el("h3", { text: "Excluded blocks and reasons" }));
      const excludedRows = excludedBlocks.map((record) => ({ ...(record.block || {}), missing_methods: record.missing_methods, duplicate_methods: record.duplicate_methods, non_finite_methods: record.non_finite_methods }));
      renderTable(excluded, excludedRows, [...(Array.isArray(analysis.block_fields) ? analysis.block_fields : []), "missing_methods", "duplicate_methods", "non_finite_methods"]);
      append(blocks, complete, excluded); panel.appendChild(blocks);
    }
    familySelect.addEventListener("change", renderAnalysis);
    metricSelect.addEventListener("change", renderAnalysis);
    renderAnalysis(); root.appendChild(section);
  }

  function scalabilityPlots(root, phase) {
    const section = el("section", { className: "section" });
    section.appendChild(sectionHeading("Computational evidence", "Common size ladder and parallel scaling", "Tools advance while ANDREA reports feasibility; higher-scale omissions must have an explicit status."));
    const rows = phase.rows;
    const frontierRows = [...phase.rows, ...phase.failures];
    const methods = unique(frontierRows.map((row) => first(row, ["method_id", "tool_id", "method"], null)));
    const runtime = plotCard("Runtime by synthetic scale", "Median wall time on completed size-ladder runs; the declared scale is categorical and wall time is logarithmic.", true);
    const memory = plotCard("Peak memory by synthetic scale", "Median measured maximum resident memory at each declared categorical scale.", false);
    const parallel = plotCard("Parallel scaling", "Runtime against effective threads, with both requested and effective budgets retained in each point.", false);
    const outcomes = plotCard("Feasibility frontier", "The final ANDREA outcome for every declared method–scale pair; not attempted means no process was launched.", true);
    const outcomeLegend = el("div", { className: "outcome-legend", attrs: { "aria-label": "Feasibility outcome legend" } });
    [
      ["#15803d", "Completed"],
      ["#2563eb", "Completed with warnings"],
      ["#f59e0b", "Timeout"],
      ["#b45309", "Estimated infeasible"],
      ["#7c3aed", "Memory limit"],
      ["#be123c", "Failed"],
      ["#94a3b8", "Not attempted — a prior ANDREA resource outcome stopped higher scales"]
    ].forEach(([color, label]) => {
      const item = el("span", { className: "outcome-legend__item" });
      append(item, el("span", { className: "outcome-legend__swatch", attrs: { style: `background:${color}` } }), document.createTextNode(label));
      outcomeLegend.appendChild(item);
    });
    outcomes.card.appendChild(outcomeLegend);
    const grid = el("div", { className: "grid" }); append(grid, runtime.card, memory.card, parallel.card, outcomes.card); section.appendChild(grid); root.appendChild(section);
    const scale = (row) => String(first(row, ["scale_id", "scale", "dataset_id"], "unknown"));
    const duration = (row) => number(first(row, ["duration_seconds", "wall_seconds", "runtime_seconds"], NaN));
    const mem = (row) => number(first(row, ["peak_memory_bytes", "max_rss_bytes", "memory_bytes"], NaN));
    const requestedThreads = (row) => number(first(row, ["requested_threads", "threads"], NaN));
    const effectiveThreads = (row) => number(first(row, ["effective_threads", "threads"], NaN));
    const completed = rows.filter((row) => ["complete", "completed", "completed_with_warnings", "success", ""].includes(String(row.status || "")));
    const declaredScales = datasetRows(phase);
    const scaleMetadata = new Map();
    declaredScales.forEach((record) => {
      const id = String(first(record, ["condition_id", "scale_id", "id", "dataset_id"], "unknown"));
      if (!scaleMetadata.has(id)) scaleMetadata.set(id, record);
    });
    const scales = unique([...declaredScales.map((record) => first(record, ["condition_id", "scale_id", "id"], null)), ...frontierRows.map(scale)]).map(String);
    const dimensions = (records, scaleId) => {
      const row = records.find((record) => Number.isFinite(number(first(record, ["cells", "n_cells"], NaN)))) || records[0] || {};
      const declared = scaleMetadata.get(scaleId) || {};
      return {
        cells: first(row, ["cells", "n_cells"], declared.cells),
        genes: first(row, ["genes", "n_genes"], declared.genes),
        tfs: first(row, ["tf_count", "transcription_factors", "tfs"], declared.tf_count)
      };
    };
    const ladderRows = completed.filter((row) => !row.measurement || row.measurement === "size_ladder");
    const groupedSeries = (method, value) => scales.map((scaleId) => {
      const records = ladderRows.filter((row) => first(row, ["method_id", "tool_id", "method"], "unknown") === method && scale(row) === scaleId && Number.isFinite(value(row)));
      const dims = dimensions(records, scaleId);
      return {
        scale: scaleId,
        value: median(records.map(value)),
        cells: dims.cells,
        genes: dims.genes,
        tfs: dims.tfs,
        requested: unique(records.map(requestedThreads).filter(Number.isFinite)).join(", ") || "—",
        effective: unique(records.map(effectiveThreads).filter(Number.isFinite)).join(", ") || "—",
        replicates: records.length
      };
    }).filter((point) => Number.isFinite(point.value));
    const runtimeTraces = methods.map((method, index) => {
      const selected = groupedSeries(method, duration);
      return { type: "scatter", mode: "lines+markers", name: method, x: selected.map((point) => point.scale), y: selected.map((point) => point.value), customdata: selected.map((point) => [point.cells, point.genes, point.tfs, point.requested, point.effective, point.replicates]), hovertemplate: "Method: %{fullData.name}<br>Scale: %{x}<br>Median wall time: %{y:.4g} s<br>Cells: %{customdata[0]}<br>Genes: %{customdata[1]}<br>TFs: %{customdata[2]}<br>Requested threads: %{customdata[3]}<br>Effective threads: %{customdata[4]}<br>Replicates: %{customdata[5]}<extra></extra>", marker: { color: COLORS[index % COLORS.length] }, line: { color: COLORS[index % COLORS.length] } };
    }).filter((trace) => trace.x.length);
    plot(runtime.plot, runtimeTraces, { xaxis: { title: "Synthetic scale", type: "category", categoryorder: "array", categoryarray: scales }, yaxis: { title: "Median wall time (seconds)", type: "log" } });
    const memoryTraces = methods.map((method, index) => {
      const selected = groupedSeries(method, mem);
      return { type: "scatter", mode: "lines+markers", name: method, x: selected.map((point) => point.scale), y: selected.map((point) => point.value / (1024 ** 3)), customdata: selected.map((point) => [point.cells, point.genes, point.tfs, point.requested, point.effective, point.replicates]), hovertemplate: "Method: %{fullData.name}<br>Scale: %{x}<br>Median peak memory: %{y:.4g} GiB<br>Cells: %{customdata[0]}<br>Genes: %{customdata[1]}<br>TFs: %{customdata[2]}<br>Requested threads: %{customdata[3]}<br>Effective threads: %{customdata[4]}<br>Replicates: %{customdata[5]}<extra></extra>", marker: { color: COLORS[index % COLORS.length] }, line: { color: COLORS[index % COLORS.length] } };
    }).filter((trace) => trace.x.length);
    plot(memory.plot, memoryTraces, { xaxis: { title: "Synthetic scale", type: "category", categoryorder: "array", categoryarray: scales }, yaxis: { title: "Median peak resident memory (GiB)", rangemode: "tozero" } });
    const threadScaleIds = unique(rows.filter((row) => row.measurement === "thread_scaling").map(scale));
    const parallelSource = completed.filter((row) => row.measurement === "thread_scaling" || (row.measurement === "size_ladder" && threadScaleIds.includes(scale(row))));
    const parallelTraces = methods.map((method, index) => {
      const source = parallelSource.filter((row) => first(row, ["method_id", "tool_id", "method"], "unknown") === method && Number.isFinite(effectiveThreads(row)) && Number.isFinite(duration(row)));
      const effective = unique(source.map(effectiveThreads)).sort((a, b) => a - b);
      const selected = effective.map((threads) => {
        const records = source.filter((row) => effectiveThreads(row) === threads);
        const dims = dimensions(records, records.length ? scale(records[0]) : "unknown");
        return { threads, requested: unique(records.map(requestedThreads)).join(", "), duration: median(records.map(duration)), scale: unique(records.map(scale)).join(", "), cells: dims.cells, genes: dims.genes, tfs: dims.tfs, replicates: records.length };
      });
      return { type: "scatter", mode: "lines+markers", name: method, x: selected.map((point) => point.threads), y: selected.map((point) => point.duration), customdata: selected.map((point) => [point.requested, point.threads, point.scale, point.cells, point.genes, point.tfs, point.replicates]), hovertemplate: "Method: %{fullData.name}<br>Requested threads: %{customdata[0]}<br>Effective threads: %{customdata[1]}<br>Median wall time: %{y:.4g} s<br>Scale: %{customdata[2]}<br>Cells: %{customdata[3]}<br>Genes: %{customdata[4]}<br>TFs: %{customdata[5]}<br>Replicates: %{customdata[6]}<extra></extra>", marker: { color: COLORS[index % COLORS.length] }, line: { color: COLORS[index % COLORS.length] } };
    }).filter((trace) => trace.x.length > 1);
    plot(parallel.plot, parallelTraces, { xaxis: { title: "Effective threads", dtick: 1 }, yaxis: { title: "Median wall time (seconds)", type: "log" } });
    const statusCode = { completed: 1, complete: 1, success: 1, eligible: .65, completed_with_warnings: .65, timeout: .3, estimated_infeasible_time: 0, memory_limit: -.65, not_attempted: -.3, failed: -1, error: -1, runtime_failure: -1 };
    const statusText = methods.map((method) => scales.map((scaleId) => {
      const records = frontierRows.filter((row) => (!row.measurement || row.measurement === "size_ladder") && first(row, ["method_id", "tool_id", "method"], "unknown") === method && scale(row) === scaleId);
      const statuses = unique(records.map((record) => String(record.status || "completed")));
      return statuses.length ? statuses.join(", ") : "not_attempted";
    }));
    const z = methods.map((method) => scales.map((scaleId) => {
      const records = frontierRows.filter((row) => (!row.measurement || row.measurement === "size_ladder") && first(row, ["method_id", "tool_id", "method"], "unknown") === method && scale(row) === scaleId);
      return records.length ? Math.min(...records.map((record) => statusCode[String(record.status || "completed")] === undefined ? -.3 : statusCode[String(record.status || "completed")])) : -.3;
    }));
    const outcomeDetails = methods.map((method) => scales.map((scaleId) => {
      const records = frontierRows.filter((row) => (!row.measurement || row.measurement === "size_ladder") && first(row, ["method_id", "tool_id", "method"], "unknown") === method && scale(row) === scaleId);
      const dims = dimensions(records, scaleId);
      return [statusText[methods.indexOf(method)][scales.indexOf(scaleId)], dims.cells, dims.genes, dims.tfs, unique(records.map(requestedThreads).filter(Number.isFinite)).join(", ") || "—", unique(records.map(effectiveThreads).filter(Number.isFinite)).join(", ") || "—"];
    }));
    plot(outcomes.plot, [{ type: "heatmap", x: scales, y: methods, z, customdata: outcomeDetails, hovertemplate: "Method: %{y}<br>Scale: %{x}<br>Outcome: %{customdata[0]}<br>Cells: %{customdata[1]}<br>Genes: %{customdata[2]}<br>TFs: %{customdata[3]}<br>Requested threads: %{customdata[4]}<br>Effective threads: %{customdata[5]}<extra></extra>", zmin: -1, zmax: 1, colorscale: [[0, "#be123c"], [.0874, "#be123c"], [.0875, "#7c3aed"], [.2624, "#7c3aed"], [.2625, "#94a3b8"], [.4249, "#94a3b8"], [.425, "#b45309"], [.5749, "#b45309"], [.575, "#f59e0b"], [.7374, "#f59e0b"], [.7375, "#2563eb"], [.9124, "#2563eb"], [.9125, "#15803d"], [1, "#15803d"]], showscale: false }], { xaxis: { title: "Synthetic scale", type: "category", categoryorder: "array", categoryarray: scales }, yaxis: { title: "Method", automargin: true } });
  }

  function renderPhase(root, name, phase) {
    const copy = PHASE_COPY[name];
    const wrapper = el("article", { className: "phase-report", attrs: { "data-phase": name } });
    const intro = el("section", { className: "section" });
    intro.appendChild(sectionHeading(copy.kicker, copy.title, copy.lead));
    const stats = el("div", { className: "stat-grid" });
    const datasets = datasetRows(phase);
    const entities = name === "calibration" ? unique(phase.rows.map((row) => first(row, ["configuration_id", "candidate_id"], null))) : unique(phase.rows.map((row) => first(row, ["method_id", "tool_id"], null)));
    append(stats,
      stat("Result rows", phase.rows.length.toLocaleString("en-US")),
      stat("Datasets", datasets.length.toLocaleString("en-US")),
      stat(name === "calibration" ? "Candidates" : "Methods", entities.length.toLocaleString("en-US")),
      stat("Exceptions", phase.failures.length.toLocaleString("en-US"))
    );
    append(intro, stats, el("p", { className: "notice notice--info", text: copy.scope })); wrapper.appendChild(intro);
    renderDatasetContext(wrapper, phase);
    if (name === "calibration") calibrationPlots(wrapper, phase);
    if (name === "comparison") { renderComparisonFamilyDefinitions(wrapper, phase); comparisonPlots(wrapper, phase); comparisonStatistics(wrapper, phase); }
    if (name === "scalability") scalabilityPlots(wrapper, phase);
    renderEvidenceTables(wrapper, phase);
    const rows = el("section", { className: "section table-card" });
    append(rows, sectionHeading("Audit table", "Canonical phase rows", "The table is capped for browser responsiveness; the matching CSV export contains every row."));
    renderTable(rows, phase.rows, ["dataset_id", "family", "configuration_id", "method_id", "scale_id", "metric", "value", "status", "measurement", "requested_threads", "effective_threads", "duration_seconds", "peak_memory_bytes"]); wrapper.appendChild(rows);
    root.appendChild(wrapper);
  }

  function start() {
    const source = document.getElementById("report-data");
    const data = JSON.parse(source.textContent || "{}");
    const study = data.study;
    const kinds = Object.keys(data.phases);
    const isArticle = data.report_kind === "article";
    document.getElementById("report-kicker").textContent = isArticle ? "Complete study" : PHASE_COPY[data.report_kind].kicker;
    document.getElementById("report-title").textContent = isArticle ? `Study ${study.id}` : PHASE_COPY[data.report_kind].title;
    document.getElementById("report-lead").textContent = isArticle ? "Configuration selection, held-out accuracy and computational scalability from one frozen study snapshot." : PHASE_COPY[data.report_kind].lead;
    const meta = document.getElementById("study-meta");
    [study.id, `Profile: ${study.profile}`, `Status: ${study.status}`].forEach((value) => meta.appendChild(el("span", { className: "badge", text: value })));
    document.getElementById("scope-notice").textContent = isArticle ? "This global report summarizes all three scientific phases. Each standalone phase HTML embeds the same canonical rows and can be copied independently." : PHASE_COPY[data.report_kind].scope;
    const allRows = kinds.reduce((total, name) => total + data.phases[name].rows.length, 0);
    const allFailures = kinds.reduce((total, name) => total + data.phases[name].failures.length, 0);
    const overview = document.getElementById("overview-stats");
    append(overview, stat("Profile", human(study.profile)), stat("Study status", human(study.status)), stat("Scientific phases", kinds.length), stat("Embedded rows", allRows.toLocaleString("en-US")), stat("Recorded exceptions", allFailures.toLocaleString("en-US")));
    const root = document.getElementById("phase-reports");
    kinds.forEach((name) => renderPhase(root, name, data.phases[name]));
    document.getElementById("resource-json").textContent = JSON.stringify(study.resources || {}, null, 2);
    const provenance = {}; kinds.forEach((name) => { provenance[name] = data.phases[name].provenance; });
    document.getElementById("provenance-json").textContent = JSON.stringify(provenance, null, 2);
  }

  try { start(); }
  catch (error) {
    const main = document.getElementById("report-main");
    const notice = el("section", { className: "notice notice--error" });
    append(notice, el("strong", { text: "The embedded report could not be rendered. " }), document.createTextNode(error && error.message ? error.message : String(error)));
    main.prepend(notice);
    throw error;
  }
}());
