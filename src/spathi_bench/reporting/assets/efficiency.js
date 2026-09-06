(function () {
  "use strict";

  const R = window.SpathiReport;
  const {
    PALETTE,
    addStatCards,
    append,
    badge,
    boot,
    clear,
    createTable,
    csvButton,
    element,
    formatInteger,
    formatNumber,
    humanize,
    notice,
    pill,
    plot,
    query,
    renderDatasetOverview,
    renderHeader,
    renderKeyValues,
    renderLinks,
    shortId,
  } = R;

  const COST_ENDPOINTS = [
    ["Runtime", "runtime"],
    ["Peak container memory", "peak_memory"],
    ["Output size", "output_size"],
    ["Targets processed", "targets_processed"],
    ["Models requested", "models_requested"],
    ["Trees fitted", "trees_fitted"],
  ];

  function percent(value) {
    if (value === null || value === undefined) return "—";
    return `${formatNumber(Number(value) * 100, { maximumFractionDigits: 2 })}%`;
  }

  function signed(value) {
    if (value === null || value === undefined) return "—";
    const number = Number(value);
    if (!Number.isFinite(number)) return "—";
    return `${number > 0 ? "+" : ""}${formatNumber(number, { maximumFractionDigits: 5 })}`;
  }

  function bytes(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "—";
    if (number >= 1024 ** 3) return `${formatNumber(number / (1024 ** 3), { maximumFractionDigits: 2 })} GiB`;
    if (number >= 1024 ** 2) return `${formatNumber(number / (1024 ** 2), { maximumFractionDigits: 1 })} MiB`;
    if (number >= 1024) return `${formatNumber(number / 1024, { maximumFractionDigits: 1 })} KiB`;
    return `${formatInteger(number)} B`;
  }

  function policyLabel(policy) {
    if (policy.role === "exact") return "Exact SPATHI";
    return humanize(policy.component);
  }

  function candidateLabel(comparison) {
    return humanize(comparison.component || comparison.candidate_id);
  }

  function parameterText(value) {
    if (value === null) return "null";
    if (typeof value === "boolean") return value ? "true" : "false";
    if (Array.isArray(value)) return value.join(", ");
    if (value && typeof value === "object") return JSON.stringify(value);
    return String(value);
  }

  function renderPolicies(policies) {
    const cards = query("#policy-cards");
    clear(cards);
    policies.forEach((policy) => {
      const card = element("article", { className: "card efficiency-policy" });
      const heading = element("div", { className: "efficiency-policy__heading" });
      const title = element("div");
      title.appendChild(element("p", { className: "card-kicker", text: humanize(policy.role) }));
      title.appendChild(element("h3", { text: policyLabel(policy) }));
      append(heading, title, pill(policy.decision, humanize(policy.decision)));
      card.appendChild(heading);
      card.appendChild(element("p", { className: "efficiency-policy__rationale", text: policy.rationale }));
      const overrides = policy.execution_overrides || {};
      const values = element("dl", { className: "parameter-grid" });
      const entries = Object.entries(overrides);
      if (!entries.length) {
        const item = element("div", { className: "parameter-item" });
        append(
          item,
          element("dt", { text: "Efficiency overrides" }),
          element("dd", { text: "None — exact calibrated execution" }),
        );
        values.appendChild(item);
      } else {
        entries.forEach(([name, value]) => {
          const item = element("div", { className: "parameter-item" });
          append(
            item,
            element("dt", { text: humanize(name) }),
            element("dd", { text: parameterText(value) }),
          );
          values.appendChild(item);
        });
      }
      card.appendChild(values);
      card.appendChild(element("p", {
        className: "efficiency-policy__hash",
        text: `Policy SHA-256 · ${shortId(policy.policy_sha256, 18)}`,
        title: policy.policy_sha256,
      }));
      cards.appendChild(card);
    });

    const rows = [];
    const parameterNames = new Set();
    policies.forEach((policy) => {
      const branches = (policy.group_policy && policy.group_policy.branches) || {};
      Object.entries(branches).forEach(([branchId, branch]) => {
        const parameters = (branch && branch.parameters) || {};
        Object.keys(parameters).forEach((name) => parameterNames.add(name));
        rows.push({
          policy: policyLabel(policy),
          policy_id: policy.policy_id,
          branch: humanize(branchId),
          condition: parameterText(branch && branch.condition),
          parameters,
        });
      });
    });
    const orderedParameters = Array.from(parameterNames).sort();
    const columns = [
      { label: "Policy", value: (row) => row.policy },
      { label: "Branch", value: (row) => row.branch },
      { label: "Condition", value: (row) => row.condition },
      ...orderedParameters.map((name) => ({
        label: humanize(name),
        value: (row) => parameterText(row.parameters[name]),
      })),
    ];
    createTable(query("#policy-table"), columns, rows);
    const actions = query("#policy-table-actions");
    clear(actions);
    actions.appendChild(csvButton("Export policy CSV", "spathi-efficiency-policies.csv", columns, rows));
  }

  function renderAccuracy(comparisons) {
    if (!comparisons.length) {
      R.empty(query("#accuracy-chart"), "No evaluable candidate", "Development produced no finalist, or a bound execution failure prevented estimation.");
      R.empty(query("#gate-chart"), "No gates computed", "No statistical gate is created without a complete candidate contrast.");
      R.empty(query("#accuracy-table"), "No comparison rows", "Exact SPATHI remains the frozen production policy.");
      clear(query("#accuracy-table-actions"));
      return;
    }
    const inferential = comparisons.every(
      (item) => item.accuracy.inference_status === "confirmatory_interval",
    );
    const labels = comparisons.map(candidateLabel);
    const estimates = comparisons.map((item) => item.accuracy.estimate);
    const lower = comparisons.map((item) => item.accuracy.one_sided_lower_confidence_bound);
    const margin = comparisons[0].accuracy.absolute_noninferiority_margin;
    const intervalTraces = inferential
      ? comparisons.map((item, index) => ({
        type: "scatter",
        mode: "lines",
        x: [labels[index], labels[index]],
        y: [item.accuracy.one_sided_lower_confidence_bound, item.accuracy.estimate],
        line: { color: item.accuracy.passed ? "#15803d" : "#be123c", width: 5 },
        hovertemplate:
          `<b>${labels[index]}</b><br>Estimate: ${signed(item.accuracy.estimate)}` +
          `<br>Lower bound: ${signed(item.accuracy.one_sided_lower_confidence_bound)}<extra></extra>`,
        showlegend: false,
      }))
      : [];
    intervalTraces.push({
      type: "scatter",
      mode: "markers",
      x: labels,
      y: estimates,
      marker: {
        size: 12,
        color: inferential
          ? comparisons.map((item) => item.accuracy.passed ? "#15803d" : "#be123c")
          : comparisons.map(() => "#2563eb"),
        line: { color: "#ffffff", width: 1.5 },
      },
      customdata: lower,
      hovertemplate: inferential
        ? "<b>%{x}</b><br>Estimate: %{y:.5f}<br>Lower bound: %{customdata:.5f}<extra></extra>"
        : "<b>%{x}</b><br>Descriptive estimate: %{y:.5f}<extra></extra>",
      showlegend: false,
    });
    plot(query("#accuracy-chart"), intervalTraces, {
      yaxis: { title: "Directed AUPR difference (candidate − exact)", zeroline: true },
      xaxis: { title: "Efficiency policy" },
      shapes: [
        { type: "line", xref: "paper", x0: 0, x1: 1, y0: -margin, y1: -margin, line: { color: "#b45309", width: 2, dash: "dash" } },
        { type: "line", xref: "paper", x0: 0, x1: 1, y0: 0, y1: 0, line: { color: "#64748b", width: 1 } },
      ],
      annotations: [{ xref: "paper", x: 1, y: -margin, text: `Non-inferiority boundary: −${formatNumber(margin)}`, showarrow: false, xanchor: "right", yanchor: "bottom", font: { color: "#b45309" } }],
    });

    const gateNames = ["Accuracy", "Coverage", "Completion", "Runtime"];
    const gateValue = (item, gate) => {
      if (!inferential) return 0.5;
      if (gate === "Accuracy") return item.accuracy.passed ? 1 : 0;
      if (gate === "Coverage") return item.coverage === null ? null : item.coverage.target_fraction.passed && item.coverage.truth_edge_fraction.passed ? 1 : 0;
      if (gate === "Completion") return item.completion.observed_complete ? 1 : 0;
      return item.cost.runtime.passed ? 1 : 0;
    };
    const z = comparisons.map((item) => gateNames.map((gate) => gateValue(item, gate)));
    const text = z.map((row) => row.map((value) => (
      !inferential ? "DESCRIPTIVE" : value === null ? "N/A" : value ? "PASS" : "FAIL"
    )));
    plot(query("#gate-chart"), [{
      type: "heatmap",
      x: gateNames,
      y: labels,
      z,
      text,
      texttemplate: "%{text}",
      hovertemplate: "<b>%{y}</b><br>%{x}: %{text}<extra></extra>",
      zmin: 0,
      zmax: 1,
      colorscale: inferential
        ? [[0, "#fecdd3"], [0.499, "#fecdd3"], [0.5, "#dcfce7"], [1, "#86efac"]]
        : [[0, "#dbeafe"], [1, "#dbeafe"]],
      showscale: false,
      hoverongaps: false,
    }], {
      xaxis: { title: "Mandatory adoption gate", side: "bottom" },
      yaxis: { title: "Efficiency policy", autorange: "reversed" },
      margin: { l: 145, r: 20, t: 15, b: 60 },
    });

    const columns = [
      { label: "Policy", value: candidateLabel },
      { label: "AUPR difference", value: (row) => row.accuracy.estimate, format: signed, numeric: true },
      { label: "Lower confidence bound", value: (row) => row.accuracy.one_sided_lower_confidence_bound, format: signed, numeric: true },
      { label: "Margin", value: (row) => row.accuracy.absolute_noninferiority_margin, format: formatNumber, numeric: true },
      { label: "Confidence", value: (row) => row.accuracy.confidence_level, format: percent, numeric: true },
      { label: "Accuracy gate", value: (row) => row.accuracy.passed, render: (value) => pill(value === null ? "descriptive_only" : value ? "passed" : "failed") },
      { label: "Retained-target diagnostic", value: (row) => row.intersection_diagnostic && row.intersection_diagnostic.estimate, format: signed, numeric: true },
      { label: "Runtime upper bound", value: (row) => row.cost.runtime.one_sided_upper_confidence_bound, format: formatNumber, numeric: true },
      { label: "Runtime maximum", value: (row) => row.cost.runtime.maximum, format: formatNumber, numeric: true },
      { label: "Final status", value: (row) => row.decision, render: (value) => pill(value) },
    ];
    createTable(query("#accuracy-table"), columns, comparisons);
    const actions = query("#accuracy-table-actions");
    clear(actions);
    actions.appendChild(csvButton("Export result CSV", "spathi-efficiency-noninferiority.csv", columns, comparisons));
  }

  function renderCoverageAndCompletion(comparisons) {
    if (!comparisons.length) {
      R.empty(query("#coverage-chart"), "No coverage result", "No candidate completed its predeclared paired panel.");
      R.empty(query("#completion-chart"), "No completion estimate", "Bound failures are reported separately and retain exact SPATHI.");
      return;
    }
    const eligible = comparisons.filter((item) => item.coverage !== null);
    if (eligible.length) {
      const labels = eligible.map(candidateLabel);
      const traces = [];
      [
        ["Target coverage", "target_fraction", "#2563eb"],
        ["Truth-edge coverage", "truth_edge_fraction", "#0f8c84"],
      ].forEach(([label, endpoint, color]) => {
        const hasIntervals = eligible.every(
          (item) => item.coverage[endpoint].one_sided_lower_confidence_bound !== null,
        );
        const trace = {
          type: "bar",
          name: `${label} estimate`,
          x: labels,
          y: eligible.map((item) => item.coverage[endpoint].estimate),
          marker: { color },
          customdata: eligible.map((item) => [
            item.coverage[endpoint].one_sided_lower_confidence_bound,
            item.coverage[endpoint].minimum,
            item.coverage[endpoint].observed_minimum_dataset,
            item.coverage[endpoint].minimum_each_dataset,
            item.coverage[endpoint].observed_minimum_condition_mean,
            item.coverage[endpoint].minimum_each_condition_mean,
          ]),
          hovertemplate: hasIntervals
            ? `<b>%{x}</b><br>${label}: %{y:.3f}<br>Lower bound: %{customdata[0]:.3f}<br>Mean threshold: %{customdata[1]:.3f}<br>Worst dataset: %{customdata[2]:.3f} (threshold %{customdata[3]:.3f})<br>Worst condition mean: %{customdata[4]:.3f} (threshold %{customdata[5]:.3f})<extra></extra>`
            : `<b>%{x}</b><br>Descriptive ${label.toLowerCase()}: %{y:.3f}<br>Mean threshold (not tested): %{customdata[1]:.3f}<br>Worst dataset: %{customdata[2]:.3f}<br>Worst condition mean: %{customdata[4]:.3f}<extra></extra>`,
        };
        if (hasIntervals) {
          trace.error_y = {
            type: "data",
            symmetric: false,
            array: eligible.map(() => 0),
            arrayminus: eligible.map((item) => (
              item.coverage[endpoint].estimate -
              item.coverage[endpoint].one_sided_lower_confidence_bound
            )),
            color,
            thickness: 2,
            width: 5,
          };
        }
        traces.push(trace);
        traces.push({
          type: "scatter",
          mode: "markers",
          name: `${label} minimum`,
          x: labels,
          y: eligible.map((item) => item.coverage[endpoint].minimum),
          marker: { color, symbol: "line-ew-open", size: 18, line: { color, width: 3 } },
          hovertemplate: `<b>%{x}</b><br>${label} minimum: %{y:.3f}<extra></extra>`,
        });
      });
      plot(query("#coverage-chart"), traces, {
        barmode: "group",
        xaxis: { title: "Eligibility policy" },
        yaxis: { title: "Coverage fraction", range: [0, 1.03], tickformat: ".0%" },
      });
    } else {
      R.empty(query("#coverage-chart"), "Coverage gate not applicable", "No evaluated policy changes target eligibility.");
    }

    const labels = comparisons.map(candidateLabel);
    const completionTraces = [
      ["Exact runs", "reference_run_fraction", "#94a3b8"],
      ["Candidate runs", "candidate_run_fraction", "#2563eb"],
      ["Exact models", "reference_model_fraction", "#64748b"],
      ["Candidate models", "candidate_model_fraction", "#0f8c84"],
    ].map(([name, field, color]) => ({
      type: "bar",
      name,
      x: labels,
      y: comparisons.map((item) => item.completion[field]),
      marker: { color },
      hovertemplate: `<b>%{x}</b><br>${name}: %{y:.2%}<extra></extra>`,
    }));
    plot(query("#completion-chart"), completionTraces, {
      barmode: "group",
      xaxis: { title: "Efficiency policy" },
      yaxis: { title: "Completion fraction", range: [0, 1.03], tickformat: ".0%" },
    });
  }

  function renderCosts(comparisons, resources) {
    if (!comparisons.length) {
      R.empty(query("#cost-chart"), "No paired cost result", "No ratio is reported for an incomplete candidate.");
      R.empty(query("#resource-table"), "No complete paired resources", "See the execution-failure evidence.");
      clear(query("#resource-table-actions"));
      return;
    }
    const labels = comparisons.map(candidateLabel);
    const traces = COST_ENDPOINTS.map(([label, endpoint], index) => {
      const trace = {
        type: "bar",
        name: label,
        x: labels,
        y: comparisons.map((item) => item.cost[endpoint].hierarchical_mean_ratio),
        marker: { color: PALETTE[index % PALETTE.length] },
        hovertemplate: `<b>%{x}</b><br>${label} ratio: %{y:.3f}<extra></extra>`,
      };
      if (endpoint === "runtime") {
        const hasIntervals = comparisons.every(
          (item) => item.cost.runtime.one_sided_upper_confidence_bound !== null,
        );
        if (hasIntervals) {
          trace.error_y = {
            type: "data",
            symmetric: false,
            array: comparisons.map((item) => (
              item.cost.runtime.one_sided_upper_confidence_bound -
              item.cost.runtime.hierarchical_mean_ratio
            )),
            arrayminus: comparisons.map(() => 0),
            color: PALETTE[index % PALETTE.length],
            thickness: 2,
            width: 5,
          };
        }
      }
      return trace;
    });
    plot(query("#cost-chart"), traces, {
      barmode: "group",
      xaxis: { title: "Efficiency policy" },
      yaxis: { title: "Candidate / exact SPATHI ratio (runtime uses a hierarchical geometric mean)", rangemode: "tozero" },
      shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 1, y1: 1, line: { color: "#17243a", width: 2, dash: "dash" } }],
      annotations: [{ xref: "paper", x: 1, y: 1, text: "Equal cost", showarrow: false, xanchor: "right", yanchor: "bottom" }],
    });

    const byId = new Map(comparisons.map((item) => [item.candidate_id, item]));
    const columns = [
      { label: "Policy", value: (row) => candidateLabel(byId.get(row.candidate_id) || row) },
      { label: "Runs", value: (row) => row.candidate_run_count, format: formatInteger, numeric: true },
      { label: "Exact time", value: (row) => row.reference_wall_time_seconds_median, format: (value) => `${formatNumber(value, { maximumFractionDigits: 2 })} s`, numeric: true },
      { label: "Candidate time", value: (row) => row.candidate_wall_time_seconds_median, format: (value) => `${formatNumber(value, { maximumFractionDigits: 2 })} s`, numeric: true },
      { label: "Speedup", value: (row) => byId.get(row.candidate_id).cost.runtime.hierarchical_mean_speedup, format: (value) => `${formatNumber(value, { maximumFractionDigits: 2 })}×`, numeric: true },
      { label: "Exact peak memory", value: (row) => row.reference_peak_memory_bytes_median, format: bytes, numeric: true },
      { label: "Candidate peak memory", value: (row) => row.candidate_peak_memory_bytes_median, format: bytes, numeric: true },
      { label: "Exact output", value: (row) => row.reference_output_bytes_median, format: bytes, numeric: true },
      { label: "Candidate output", value: (row) => row.candidate_output_bytes_median, format: bytes, numeric: true },
      { label: "Exact targets", value: (row) => row.reference_targets_processed_median, format: formatNumber, numeric: true },
      { label: "Candidate targets", value: (row) => row.candidate_targets_processed_median, format: formatNumber, numeric: true },
      { label: "Exact models", value: (row) => row.reference_models_requested_median, format: formatNumber, numeric: true },
      { label: "Candidate models", value: (row) => row.candidate_models_requested_median, format: formatNumber, numeric: true },
      { label: "Exact trees", value: (row) => row.reference_trees_fitted_median, format: formatNumber, numeric: true },
      { label: "Candidate trees", value: (row) => row.candidate_trees_fitted_median, format: formatNumber, numeric: true },
    ];
    createTable(query("#resource-table"), columns, resources);
    const actions = query("#resource-table-actions");
    clear(actions);
    actions.appendChild(csvButton("Export resources CSV", "spathi-efficiency-resources.csv", columns, resources));
  }

  function renderEfficiencySafety(safety) {
    const summaryTarget = query("#efficiency-safety-summary");
    const gatesTarget = query("#efficiency-safety-gates");
    const conditionTarget = query("#efficiency-safety-condition-table");
    const tableTarget = query("#efficiency-safety-table");
    if (!safety || safety.status !== "completed") {
      addStatCards(summaryTarget, [
        { label: "Status", value: "Not required", detail: humanize((safety && safety.reason) || "no provisional candidate") },
      ]);
      notice(
        query("#efficiency-safety-note"),
        "info",
        "Safety bridge not executed",
        "The reduced design is descriptive, or exact SPATHI was retained by the formal confirmatory study.",
      );
      R.empty(conditionTarget, "No safety conditions", "No candidate was eligible for this safeguard.");
      R.empty(gatesTarget, "No safety vetoes applied", "No candidate was eligible for this safeguard.");
      R.empty(tableTarget, "No safety policy pair", "No candidate was eligible for this safeguard.");
      return;
    }
    const decision = safety.decision;
    const summary = safety.summary.global;
    const scales = safety.condition_scales;
    addStatCards(summaryTarget, [
      { label: "Decision", value: humanize(decision.status), detail: humanize(decision.selected_policy_id) },
      { label: "Completed pairs", value: `${formatInteger(summary.completed_dataset_pairs)} / 18`, detail: "All six conditions × three datasets are required" },
      { label: "Safety conditions", value: formatInteger(Object.keys(scales).length), detail: "Scale and grouping transfer safeguards" },
      { label: "Global median ΔAUPR", value: signed(summary.median_aupr_difference), detail: "Candidate − exact" },
      { label: "Global wall-time ratio", value: formatNumber(summary.median_wall_time_ratio), detail: "Median candidate / exact; must be < 1" },
      { label: "Execution failures", value: formatInteger(safety.execution_failures.length), detail: "Any bound tool failure vetoes adoption" },
    ]);
    notice(
      query("#efficiency-safety-note"),
      decision.status === "candidate_authorized" ? "info" : "warning",
      "Descriptive safety boundary",
      safety.interpretation_limit,
    );
    const gateRows = Object.entries(safety.gates).map(([gate, passed]) => ({ gate, passed }));
    createTable(gatesTarget, [
      { label: "Predeclared veto", value: (row) => humanize(row.gate) },
      {
        label: "Outcome",
        value: (row) => (row.passed ? "passed" : "vetoed"),
        render: (value) => pill(value),
      },
    ], gateRows);
    const conditionRows = Object.entries(safety.summary.by_condition).map(([conditionId, values]) => ({
      condition_id: conditionId,
      ...scales[conditionId],
      ...values,
    }));
    createTable(conditionTarget, [
      { label: "Condition", value: (row) => row.condition_id },
      { label: "Genes", value: (row) => row.genes, format: formatInteger, numeric: true },
      { label: "TFs", value: (row) => row.transcription_factors, format: formatInteger, numeric: true },
      { label: "Cells", value: (row) => row.cells, format: formatInteger, numeric: true },
      { label: "Group sizes", value: (row) => row.group_sizes.join(" / ") },
      { label: "Topology", value: (row) => row.scenario_inputs.regulatory_network.files[0].path.split("/").pop() },
      { label: "Lineage tree", value: (row) => row.scenario_inputs.tree_newick.files[0].path.split("/").pop() },
      { label: "Complete", value: (row) => `${row.completed_dataset_pairs} / 3` },
      { label: "Mean ΔAUPR", value: (row) => row.mean_aupr_difference, format: signed, numeric: true },
      { label: "Median ΔAUPR", value: (row) => row.median_aupr_difference, format: signed, numeric: true },
      { label: "Minimum ΔAUPR", value: (row) => row.minimum_aupr_difference, format: signed, numeric: true },
    ], conditionRows);
    const pairedColumns = [
      { label: "Dataset", value: (row) => row.dataset_id },
      { label: "Condition", value: (row) => row.condition_id },
      { label: "Generation seed", value: (row) => row.generation_seed, format: formatInteger, numeric: true },
      { label: "Exact AUPR", value: (row) => row.reference_aupr, format: formatNumber, numeric: true },
      { label: "Candidate AUPR", value: (row) => row.candidate_aupr, format: formatNumber, numeric: true },
      { label: "ΔAUPR", value: (row) => row.aupr_difference, format: signed, numeric: true },
      { label: "Target coverage", value: (row) => row.target_fraction, format: percent, numeric: true },
      { label: "Truth-edge coverage", value: (row) => row.truth_edge_fraction, format: percent, numeric: true },
      { label: "Wall-time ratio", value: (row) => row.wall_time_ratio, format: formatNumber, numeric: true },
    ];
    if (safety.paired_datasets.length) {
      createTable(tableTarget, pairedColumns, safety.paired_datasets);
    } else {
      const failureColumns = [
        { label: "Dataset", value: (row) => row.dataset_id },
        { label: "Policy", value: (row) => row.configuration_id },
        { label: "Role", value: (row) => humanize(row.selection_role) },
        { label: "Status", value: (row) => humanize(row.tool_status) },
        { label: "Failure", value: (row) => row.failure_message },
      ];
      createTable(tableTarget, failureColumns, safety.execution_failures);
    }
  }

  function renderFormalFailures(failures) {
    const section = query("#formal-failures-section");
    if (!failures || !failures.length) return;
    section.hidden = false;
    notice(
      query("#formal-failures-note"),
      "warning",
      "Exact retained fail-safe",
      "At least one predeclared candidate failed. Relative candidate selection is invalidated globally; the exact reference remains the production policy. A failed exact reference would instead abort the study as inconclusive.",
    );
    createTable(query("#formal-failures-table"), [
      { label: "Candidate", value: (row) => row.configuration_id },
      { label: "Dataset", value: (row) => row.dataset_id },
      { label: "Algorithm seed", value: (row) => row.algorithm_seed, format: formatInteger, numeric: true },
      { label: "Tool status", value: (row) => row.tool_status, render: (value) => pill(value) },
      { label: "Failure", value: (row) => row.failure_message },
      { label: "Inference artifact", value: (row) => shortId(row.inference_artifact) },
    ], failures);
  }

  function renderDecision(decision, role) {
    const target = query("#decision-card");
    clear(target);
    const adoption = decision.adoption;
    const production = decision.production;
    const safety = decision.efficiency_safety;
    const tone = production.decision === "candidate_adopted"
      ? "success"
      : adoption.decision === "descriptive_only"
        ? "warning"
        : "info";
    const card = element("article", { className: `selection-hero efficiency-decision__hero efficiency-decision__hero--${tone}` });
    const badges = element("div", { className: "badge-row efficiency-decision__badges" });
    append(badges, badge(role), badge(production.status), badge(production.decision));
    card.appendChild(badges);
    card.appendChild(element("p", { className: "card-kicker", text: "Selected production policy" }));
    card.appendChild(element("h3", { text: humanize(production.policy.policy_id) }));
    card.appendChild(element("p", {
      text: safety && safety.status === "completed" ? safety.decision.reason : adoption.reason,
    }));
    const details = element("dl", { className: "efficiency-decision__details" });
    [
      ["Global decision", humanize(decision.global)],
      ["Formal efficiency decision", humanize(adoption.decision)],
      ["Production decision", humanize(production.decision)],
      ["Selection rule", humanize(adoption.selection_rule)],
      ["Policy SHA-256", production.policy.policy_sha256],
    ].forEach(([label, value]) => {
      append(details, element("dt", { text: label }), element("dd", { text: value, title: value }));
    });
    card.appendChild(details);
    target.appendChild(card);
  }

  function renderProvenance(data) {
    const provenance = data.provenance;
    renderKeyValues(query("#provenance"), "Scientific identity", [
      { label: "Study", value: data.study.id },
      { label: "Evidence role", value: humanize(data.workflow.analysis_role) },
      { label: "Dataset role", value: humanize(data.evidence.dataset_role) },
      { label: "Algorithm seeds", value: data.evidence.algorithm_seeds.join(", ") },
      { label: "SPATHI commit", value: provenance.spathi_commit },
      { label: "SPATHI image", value: provenance.spathi_image_id },
      { label: "Study SHA-256", value: provenance.study && provenance.study.sha256 },
      { label: "Design sensitivity SHA-256", value: provenance.design_sensitivity && provenance.design_sensitivity.sha256 },
      { label: "Panel SHA-256", value: provenance.panel && provenance.panel.sha256 },
      { label: "Selected config SHA-256", value: provenance.selected_config && provenance.selected_config.sha256 },
      { label: "Production policy SHA-256", value: provenance.production_policy && provenance.production_policy.policy_sha256 },
      { label: "Analysis implementation", value: provenance.analysis_implementation_sha256 },
      { label: "Report fingerprint", value: data.report && data.report.fingerprint },
      { label: "Renderer fingerprint", value: data.report && data.report.renderer && data.report.renderer.fingerprint },
    ]);
    renderLinks(query("#report-links"), "Canonical artifacts", data.links);
  }

  function main(data) {
    if (data.schema_version !== "1.0" || data.report_type !== "efficiency") {
      throw new Error("Unsupported SPATHI efficiency report payload.");
    }
    renderHeader(data.workflow, "efficiency", { status: "complete" });
    renderDatasetOverview(query("#dataset-overview"), data.datasets);
    addStatCards(query("#overview-cards"), [
      { label: "Evidence budget", value: humanize(data.workflow.profile), detail: humanize(data.workflow.analysis_role) },
      { label: "Independent datasets", value: formatInteger(data.datasets.summary.dataset_count), detail: `${formatInteger(data.datasets.summary.total_cells)} cells` },
      { label: "Policies evaluated", value: formatInteger(data.policies.length), detail: "Exact SPATHI plus all predeclared candidates" },
      { label: "Algorithm seeds", value: formatInteger(data.evidence.algorithm_seed_count), detail: "Paired within each dataset" },
      { label: "AUPR margin", value: formatNumber(data.study.analysis.absolute_noninferiority_margin), detail: "Predeclared absolute non-inferiority margin" },
      { label: "Full-design effective n", value: formatNumber(data.design_sensitivity.expected_effective_sample_size), detail: `${formatInteger(data.design_sensitivity.expected_independent_datasets)} datasets under hierarchical weighting` },
      { label: "Dataset-group rows", value: formatInteger(data.evidence.paired_row_count), detail: "Paired evidence records" },
      { label: "Decision", value: humanize(data.decision.global), detail: humanize(data.decision.production.status) },
    ]);
    notice(
      query("#interpretation-limit"),
      "info",
      "Interpretation boundary",
      data.study.interpretation_limit,
    );
    notice(
      query("#design-sensitivity-note"),
      "warning",
      "Prospective design sensitivity",
      data.design_sensitivity.required_caveat,
    );
    renderPolicies(data.policies);
    renderFormalFailures(data.execution_failures);
    renderAccuracy(data.comparisons);
    renderCoverageAndCompletion(data.comparisons);
    renderCosts(data.comparisons, data.resources);
    renderEfficiencySafety(data.decision.efficiency_safety);
    renderDecision(data.decision, data.workflow.analysis_role);
    renderProvenance(data);
  }

  boot(main);
}());
