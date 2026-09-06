(function () {
  "use strict";

  const R = window.SpathiReport;

  function asArray(value) {
    if (Array.isArray(value)) return value;
    if (value && typeof value === "object") return Object.values(value);
    return [];
  }

  function firstDefined() {
    for (let index = 0; index < arguments.length; index += 1) {
      if (arguments[index] !== null && arguments[index] !== undefined) return arguments[index];
    }
    return null;
  }

  function analysisField(analysis, field) {
    const stratum = analysis.stratum || {};
    if (field === "comparison") return firstDefined(analysis.comparison_id, analysis.comparison, "comparison");
    if (field === "metric") return firstDefined(analysis.metric, "unknown_metric");
    if (field === "level") return firstDefined(stratum.level, analysis.level, "all_levels");
    if (field === "regime") return firstDefined(stratum.regime, analysis.regime, "all_regimes");
    return null;
  }

  function roleOf(analysis) {
    return firstDefined(analysis.analysis_role, analysis.comparison_role, "unknown");
  }

  function configurationMap(analysis) {
    const map = new Map();
    asArray(analysis.configurations).forEach((configuration) => {
      const id = firstDefined(configuration.configuration_id, configuration.id);
      if (id) map.set(String(id), configuration);
    });
    asArray(analysis.rankings).forEach((ranking) => {
      const id = firstDefined(ranking.configuration_id, ranking.id);
      if (id && !map.has(String(id))) map.set(String(id), ranking);
    });
    return map;
  }

  function configurationLabel(configuration, id, duplicates) {
    const tool = firstDefined(configuration && configuration.tool_id, id, "Method");
    if (duplicates && duplicates.get(String(tool)) > 1) return `${tool} · ${R.shortId(id, 20)}`;
    return String(tool);
  }

  function labelMap(analysis) {
    const configs = configurationMap(analysis);
    const tools = new Map();
    configs.forEach((configuration, id) => {
      const tool = String(firstDefined(configuration.tool_id, id));
      tools.set(tool, (tools.get(tool) || 0) + 1);
    });
    const labels = new Map();
    configs.forEach((configuration, id) => labels.set(id, configurationLabel(configuration, id, tools)));
    return labels;
  }

  function renderStatus(data, analyses) {
    const target = R.query("#status-banner");
    const workflow = data.workflow || {};
    const integrity = data.integrity || {};
    const analysisMode = String(((data.statistics || {}).contract || {}).status || "");
    const incompleteStatuses = new Set([
      "incomplete",
      "incomplete_panel",
      "partially_tested",
      "confirmatory_incomplete",
      "not_testable",
      "failed",
      "error",
    ]);
    const incomplete = analyses.filter((analysis) => incompleteStatuses.has(String(analysis.status || "").toLowerCase())).length;
    const invalidIntegrity = ["failed", "invalid", "error"].includes(String(integrity.status || "").toLowerCase());
    if (invalidIntegrity) {
      R.notice(target, "error", "Integrity not validated", integrity.message || "At least one artifact integrity check failed.");
    } else if (!analyses.length) {
      R.notice(target, "warning", "No analyses", "The comparison does not yet contain statistical analyses.");
    } else if (incomplete || !["complete", "completed"].includes(String(workflow.status || "complete").toLowerCase())) {
      R.notice(target, "warning", "Partial report", `${incomplete} analyses are incomplete or failed. Review their reasons before interpreting results.`);
    } else {
      const message = analysisMode === "inferential"
        ? "Scores, tests, and effect sizes come from the canonical analysis artifacts."
        : "Scores and paired descriptive effects come from the canonical analysis artifacts; no hypothesis test or winner claim was made.";
      R.notice(target, "success", "Comparison complete", message);
    }
  }

  function renderOverview(data, analyses) {
    const statistics = data.statistics || {};
    const summary = statistics.summary || {};
    const blocks = R.unique(analyses.flatMap((analysis) => asArray(analysis.blocks).map((block) => block.dataset_id)));
    const tools = R.unique(analyses.flatMap((analysis) => asArray(analysis.configurations).map((configuration) => configuration.tool_id)));
    const roles = R.unique(analyses.map(roleOf));
    const analyzed = firstDefined(
      summary.n_inferential,
      0,
    ) + firstDefined(summary.n_descriptive, 0) + firstDefined(summary.n_contract_only, 0);
    const inferential = firstDefined(summary.n_inferential, analyses.filter((item) => item.status === "tested").length);
    R.addStatCards(R.query("#overview-cards"), [
      { label: "Analyses", value: R.formatInteger(firstDefined(summary.n_analyses, analyses.length)), detail: `${R.formatInteger(analyzed)} complete; ${R.formatInteger(inferential)} inferential` },
      { label: "Dataset blocks", value: R.formatInteger(blocks.length), detail: "Independent unit of each paired comparison" },
      { label: "Methods", value: R.formatInteger(tools.length || R.unique(analyses.flatMap((analysis) => asArray(analysis.rankings).map((ranking) => ranking.tool_id))).length), detail: "Comparable configurations" },
      { label: "Control wins", value: R.formatInteger(summary.n_control_wins), detail: inferential ? `out of ${R.formatInteger(inferential)} inferential analyses` : "No winner claims in this evidence mode" },
      { label: "Analysis role", value: roles.map(R.humanize).join(" / ") || "—", detail: roles.includes("exploratory") ? "Not confirmatory evidence" : "Held-out partition" },
    ]);
  }

  function buildFilters(analyses, onSelect) {
    const target = R.query("#analysis-filters");
    R.clear(target);
    if (!analyses.length) return;
    const preferred = analyses.find((analysis) => analysisField(analysis, "metric") === "aupr" && analysisField(analysis, "level") === "directed") || analyses[0];
    const state = {
      comparison: analysisField(preferred, "comparison"),
      metric: analysisField(preferred, "metric"),
      level: analysisField(preferred, "level"),
      regime: analysisField(preferred, "regime"),
    };
    const selects = {};

    function subset(until) {
      return analyses.filter((analysis) => {
        if (["metric", "level", "regime"].includes(until) && analysisField(analysis, "comparison") !== state.comparison) return false;
        if (["level", "regime"].includes(until) && analysisField(analysis, "metric") !== state.metric) return false;
        if (until === "regime" && analysisField(analysis, "level") !== state.level) return false;
        return true;
      });
    }

    function values(field) {
      const source = field === "comparison" ? analyses : subset(field);
      return R.unique(source.map((analysis) => analysisField(analysis, field)));
    }

    function chooseValid(field) {
      const available = values(field);
      if (!available.includes(state[field])) state[field] = available[0];
      R.replaceSelectOptions(selects[field], available, state[field]);
    }

    function resolve() {
      const exact = analyses.find((analysis) => ["comparison", "metric", "level", "regime"].every((field) => analysisField(analysis, field) === state[field]));
      onSelect(exact || analyses[0]);
    }

    selects.comparison = R.makeSelect(target, "filter-comparison", "Comparison", values("comparison"), state.comparison, (event) => {
      state.comparison = event.target.value;
      chooseValid("metric");
      chooseValid("level");
      chooseValid("regime");
      resolve();
    });
    selects.metric = R.makeSelect(target, "filter-metric", "Metric", values("metric"), state.metric, (event) => {
      state.metric = event.target.value;
      chooseValid("level");
      chooseValid("regime");
      resolve();
    });
    selects.level = R.makeSelect(target, "filter-level", "Network level", values("level"), state.level, (event) => {
      state.level = event.target.value;
      chooseValid("regime");
      resolve();
    });
    selects.regime = R.makeSelect(target, "filter-regime", "Regime", values("regime"), state.regime, (event) => {
      state.regime = event.target.value;
      resolve();
    });
    resolve();
  }

  function renderAnalysisState(analysis) {
    const target = R.query("#analysis-state");
    const status = String(analysis.status || "unknown").toLowerCase();
    const role = roleOf(analysis);
    if (["incomplete", "incomplete_panel", "partially_tested", "not_testable", "failed", "error"].includes(status)) {
      R.notice(target, status === "failed" || status === "error" ? "error" : "warning", R.statusLabel(status), analysis.reason || "This analysis does not have enough complete blocks for the planned test.");
    } else if (role === "exploratory") {
      R.notice(target, "warning", "Exploratory analysis", "Tests retain their original correction, but this profile must not be interpreted as confirmatory validation or generalized beyond the included datasets.");
    } else {
      R.notice(target, "info", "Confirmatory analysis", "This comparison uses the confirmatory role declared by the experimental design.");
    }
  }

  function renderAnalysisBadges(analysis) {
    const target = R.query("#analysis-badges");
    R.clear(target);
    const omnibus = analysis.omnibus || {};
    target.appendChild(R.badge(analysis.status));
    target.appendChild(R.badge(roleOf(analysis), R.humanize(roleOf(analysis))));
    target.appendChild(R.badge("neutral", `${R.formatInteger(analysis.n_complete_blocks)} blocks`));
    if (Number.isFinite(R.number(omnibus.statistic, NaN))) {
      target.appendChild(R.badge("neutral", `Friedman Q = ${R.formatNumber(omnibus.statistic)}`));
    }
    if (Number.isFinite(R.number(omnibus.p_holm, NaN))) {
      target.appendChild(R.badge("neutral", `p Holm = ${R.formatNumber(omnibus.p_holm)}`));
    } else if (Number.isFinite(R.number(omnibus.p_raw, NaN))) {
      target.appendChild(R.badge("neutral", `p = ${R.formatNumber(omnibus.p_raw)}`));
    }
    if (Number.isFinite(R.number(omnibus.kendalls_w, NaN))) {
      target.appendChild(R.badge("neutral", `Kendall W = ${R.formatNumber(omnibus.kendalls_w)}`));
    }
    if (analysis.conclusion) target.appendChild(R.badge(analysis.conclusion === "control_wins" ? "success" : "warning", R.humanize(analysis.conclusion)));
  }

  function orderedRankings(analysis) {
    return asArray(analysis.rankings).slice().sort((left, right) => R.number(left.mean_rank, Infinity) - R.number(right.mean_rank, Infinity));
  }

  function renderScores(analysis) {
    const target = R.query("#score-chart");
    const rankings = orderedRankings(analysis);
    const labels = labelMap(analysis);
    const blocks = asArray(analysis.blocks);
    if (!rankings.length) {
      R.empty(target, "Scores unavailable", analysis.reason || "The analysis does not contain canonical rankings.");
      return;
    }
    const traces = rankings.map((ranking, index) => {
      const id = String(ranking.configuration_id);
      const values = blocks.map((block) => R.number(block.scores && block.scores[id], NaN)).filter(Number.isFinite);
      if (values.length) {
        return {
          type: "box",
          name: labels.get(id) || id,
          y: values,
          text: blocks.filter((block) => Number.isFinite(R.number(block.scores && block.scores[id], NaN))).map((block) => block.dataset_id),
          boxpoints: "all",
          jitter: 0.22,
          pointpos: 0,
          marker: { color: R.PALETTE[index % R.PALETTE.length], size: 7, opacity: 0.72 },
          line: { color: R.PALETTE[index % R.PALETTE.length] },
          hovertemplate: "<b>%{fullData.name}</b><br>Dataset: %{text}<br>Accuracy score: %{y:.6f}<extra></extra>",
        };
      }
      return {
        type: "bar",
        name: labels.get(id) || id,
        x: [labels.get(id) || id],
        y: [ranking.mean_score],
        marker: { color: R.PALETTE[index % R.PALETTE.length] },
        hovertemplate: "<b>%{x}</b><br>Canonical mean: %{y:.6f}<extra></extra>",
      };
    });
    R.plot(target, traces, {
      margin: { l: 60, r: 18, t: 12, b: 105 },
      xaxis: { title: { text: "Inference method", standoff: 14 }, tickangle: -25, automargin: true },
      yaxis: { title: { text: `${R.humanize(analysis.metric)} score (higher is better)`, standoff: 12 }, gridcolor: "#e7ebf1", zeroline: false },
      showlegend: false,
      boxmode: "group",
    });
  }

  function renderRanks(analysis) {
    const target = R.query("#rank-chart");
    const rankings = orderedRankings(analysis).slice().reverse();
    const labels = labelMap(analysis);
    if (!rankings.length) {
      R.empty(target, "Ranks unavailable", "The Friedman analysis did not produce ranks for this comparison.");
      return;
    }
    R.plot(target, [{
      type: "bar",
      orientation: "h",
      x: rankings.map((ranking) => ranking.mean_rank),
      y: rankings.map((ranking) => labels.get(String(ranking.configuration_id)) || ranking.configuration_id),
      text: rankings.map((ranking) => R.formatNumber(ranking.mean_rank, { maximumFractionDigits: 3 })),
      textposition: "outside",
      customdata: rankings.map((ranking) => [ranking.mean_score, ranking.median_score, ranking.configuration_id]),
      marker: { color: rankings.map((ranking, index) => R.PALETTE[(rankings.length - index - 1) % R.PALETTE.length]) },
      hovertemplate: "<b>%{customdata[2]}</b><br>Mean rank: %{x:.3f}<br>Mean score: %{customdata[0]:.6f}<br>Median score: %{customdata[1]:.6f}<extra></extra>",
    }], {
      margin: { l: 125, r: 45, t: 12, b: 52 },
      xaxis: { title: { text: "Mean Friedman rank (lower is better)", standoff: 14 }, rangemode: "tozero", gridcolor: "#e7ebf1", zeroline: false },
      yaxis: { title: { text: "Inference method", standoff: 12 }, automargin: true },
      showlegend: false,
    });
  }

  function renderPaired(analysis) {
    const target = R.query("#paired-chart");
    const rankings = orderedRankings(analysis);
    const blocks = asArray(analysis.blocks);
    const labels = labelMap(analysis);
    if (!rankings.length || !blocks.length) {
      R.empty(target, "Paired blocks unavailable", "The artifact does not contain per-dataset scores.");
      return;
    }
    const ids = rankings.map((ranking) => String(ranking.configuration_id));
    const names = ids.map((id) => labels.get(id) || id);
    const traces = blocks.map((block) => ({
      type: "scatter",
      mode: "lines+markers",
      name: block.dataset_id,
      x: names,
      y: ids.map((id) => R.number(block.scores && block.scores[id], null)),
      customdata: ids.map((id) => [block.dataset_id, id, block.simulator_id]),
      line: { color: "rgba(82,96,120,0.22)", width: 1 },
      marker: { color: "rgba(37,99,235,0.52)", size: 6 },
      connectgaps: false,
      hovertemplate: "<b>%{customdata[0]}</b><br>Method: %{customdata[1]}<br>Simulator: %{customdata[2]}<br>Accuracy score: %{y:.6f}<extra></extra>",
    }));
    R.plot(target, traces, {
      margin: { l: 62, r: 20, t: 12, b: 85 },
      xaxis: { title: { text: "Inference method", standoff: 14 }, automargin: true },
      yaxis: { title: { text: `${R.humanize(analysis.metric)} score (higher is better)`, standoff: 12 }, gridcolor: "#e7ebf1", zeroline: false },
      showlegend: false,
    });
  }

  function renderForest(analysis) {
    const target = R.query("#forest-chart");
    const pairs = asArray(analysis.pairwise).filter((pair) => Number.isFinite(R.number(pair.median_difference, NaN)));
    if (!pairs.length) {
      R.empty(target, "Effect sizes unavailable", analysis.reason || "No canonical paired comparisons are available.");
      return;
    }
    const reversed = pairs.slice().reverse();
    const labels = reversed.map((pair) => `${pair.control_tool_id || "Control"} − ${pair.competitor_tool_id || pair.competitor_configuration_id}`);
    const estimate = reversed.map((pair) => pair.median_difference);
    const low = reversed.map((pair) => R.number(pair.bootstrap_ci_low, pair.median_difference));
    const high = reversed.map((pair) => R.number(pair.bootstrap_ci_high, pair.median_difference));
    const markerColors = reversed.map((pair) => {
      const difference = R.number(pair.median_difference, NaN);
      if (pair.reject !== true || !Number.isFinite(difference) || difference === 0) return "#64748b";
      return difference > 0 ? "#0f8c84" : "#c2410c";
    });
    R.plot(target, [{
      type: "scatter",
      mode: "markers",
      x: estimate,
      y: labels,
      customdata: reversed.map((pair) => [pair.p_holm, pair.reject, pair.wins, pair.ties, pair.losses, pair.mean_difference]),
      error_x: {
        type: "data",
        symmetric: false,
        array: high.map((value, index) => value - estimate[index]),
        arrayminus: low.map((value, index) => estimate[index] - value),
        color: "#2563eb",
        thickness: 2,
        width: 7,
      },
      marker: { color: markerColors, size: 10 },
      hovertemplate: "<b>%{y}</b><br>Median difference: %{x:.6f}<br>Mean difference: %{customdata[5]:.6f}<br>Holm p: %{customdata[0]:.4g}<br>Rejects H₀: %{customdata[1]}<br>W/T/L: %{customdata[2]}/%{customdata[3]}/%{customdata[4]}<extra></extra>",
    }], {
      margin: { l: 145, r: 20, t: 12, b: 58 },
      xaxis: { title: { text: `Control − competitor median ${R.humanize(analysis.metric)} difference (bootstrap interval)`, standoff: 14 }, gridcolor: "#e7ebf1", zeroline: true, zerolinecolor: "#94a3b8", zerolinewidth: 1.5 },
      yaxis: { title: { text: "Pairwise comparison", standoff: 12 }, automargin: true },
      showlegend: false,
    });
  }

  function rejectionNode(value, status) {
    if (status && status !== "tested") return R.pill(status);
    if (value === true) return R.pill("success", "Yes");
    if (value === false) return R.pill("warning", "No");
    return R.pill("unavailable", "—");
  }

  function holmColumns() {
    return [
      { label: "Control", value: (row) => row.control_tool_id || row.control_configuration_id },
      { label: "Competitor", value: (row) => row.competitor_tool_id || row.competitor_configuration_id },
      { label: "Unadjusted p", value: (row) => row.p_raw, format: R.formatNumber, numeric: true },
      { label: "Holm p", value: (row) => row.p_holm, format: R.formatNumber, numeric: true },
      { label: "Reject H₀", value: (row) => row.reject, render: (value, row) => rejectionNode(value, row.status) },
      { label: "Median difference", value: (row) => row.median_difference, format: R.formatNumber, numeric: true },
      { label: "Bootstrap CI", value: (row) => Number.isFinite(R.number(row.bootstrap_ci_low, NaN)) ? `[${R.formatNumber(row.bootstrap_ci_low)}, ${R.formatNumber(row.bootstrap_ci_high)}]` : "—" },
      { label: "W / T / L", value: (row) => `${firstDefined(row.wins, "—")} / ${firstDefined(row.ties, "—")} / ${firstDefined(row.losses, "—")}` },
      { label: "Rank-biserial", value: (row) => row.rank_biserial, format: R.formatNumber, numeric: true },
    ];
  }

  function renderHolm(analysis) {
    const rows = asArray(analysis.pairwise);
    const columns = holmColumns();
    R.createTable(R.query("#holm-table"), columns, rows, {
      emptyTitle: "No post-hoc tests",
      emptyDetail: analysis.reason || "No paired tests were run for this analysis.",
    });
    const actions = R.query("#holm-table-actions");
    R.clear(actions);
    if (rows.length) actions.appendChild(R.csvButton("Export CSV", `holm-${analysis.metric}-${analysisField(analysis, "level")}.csv`, columns, rows));
  }

  function conditionRows(summary, analysis) {
    const metricField = `mean_${analysis.metric}`;
    const configurations = new Set(asArray(analysis.rankings).map((ranking) => String(ranking.configuration_id)));
    return asArray(summary && summary.aggregates).filter((row) => {
      const datasetRole = analysis.stratum && analysis.stratum.dataset_role;
      if (row.dataset_role && datasetRole && row.dataset_role !== datasetRole) return false;
      if (row.level && row.level !== analysisField(analysis, "level")) return false;
      if (row.regime && row.regime !== analysisField(analysis, "regime")) return false;
      if (row.configuration_id && configurations.size && !configurations.has(String(row.configuration_id))) return false;
      return Number.isFinite(R.number(row[metricField], NaN));
    }).map((row) => Object.assign({}, row, { displayScore: R.number(row[metricField], NaN) }));
  }

  function renderConditions(data, analysis) {
    const target = R.query("#condition-chart");
    const rows = conditionRows(data.summary || {}, analysis);
    if (!rows.length) {
      R.empty(target, "Breakdown unavailable", "The canonical summary has no aggregates for this metric, network level, and regime.");
      return;
    }
    const ids = R.unique(rows.map((row) => String(row.configuration_id)));
    const labels = labelMap(analysis);
    const traces = ids.map((id, index) => {
      const current = rows.filter((row) => String(row.configuration_id) === id);
      return {
        type: "bar",
        name: labels.get(id) || current[0].tool_id || id,
        x: current.map((row) => row.condition_id),
        y: current.map((row) => row.displayScore),
        customdata: current.map((row) => [row.tool_id, row.n_datasets, row.n_executions, row.dataset_role]),
        marker: { color: R.PALETTE[index % R.PALETTE.length] },
        hovertemplate: "<b>%{customdata[0]}</b><br>Condition: %{x}<br>Canonical mean: %{y:.6f}<br>Datasets: %{customdata[1]}<br>Executions: %{customdata[2]}<br>Partition: %{customdata[3]}<extra></extra>",
      };
    });
    R.plot(target, traces, {
      margin: { l: 62, r: 20, t: 12, b: 95 },
      barmode: "group",
      xaxis: { title: { text: "Experimental condition", standoff: 14 }, tickangle: -22, automargin: true },
      yaxis: { title: { text: `Mean ${R.humanize(analysis.metric)} score (higher is better)`, standoff: 12 }, gridcolor: "#e7ebf1", zeroline: false },
      legend: { orientation: "h", y: -0.3 },
    });
  }

  function runtimeRows(data, analysis) {
    const source = firstDefined(data.runtimes, data.runtime, data.performance, []);
    const rows = asArray(source.rows || source.observations || source);
    const configurations = new Set(asArray(analysis.rankings).map((ranking) => String(ranking.configuration_id)));
    return rows.filter((row) => {
      const datasetRole = analysis.stratum && analysis.stratum.dataset_role;
      if (row.dataset_role && datasetRole && row.dataset_role !== datasetRole) return false;
      if (row.configuration_id && configurations.size && !configurations.has(String(row.configuration_id))) return false;
      if (row.regime && row.regime !== analysisField(analysis, "regime")) return false;
      return Number.isFinite(R.number(firstDefined(row.wall_time_seconds, row.runtime_seconds, row.median_wall_time_seconds), NaN));
    });
  }

  function renderRuntime(data, analysis) {
    const target = R.query("#runtime-chart");
    const rows = runtimeRows(data, analysis);
    if (!rows.length) {
      R.empty(target, "Runtime unavailable", "This workflow did not publish canonical runtime observations for the selected configurations.");
      return;
    }
    const labels = labelMap(analysis);
    const ids = R.unique(rows.map((row) => String(firstDefined(row.configuration_id, row.tool_id))));
    const traces = ids.map((id, index) => {
      const current = rows.filter((row) => String(firstDefined(row.configuration_id, row.tool_id)) === id);
      return {
        type: "box",
        name: labels.get(id) || current[0].tool_id || id,
        y: current.map((row) => R.number(firstDefined(row.wall_time_seconds, row.runtime_seconds, row.median_wall_time_seconds), NaN)),
        text: current.map((row) => firstDefined(row.dataset_id, row.run_id, "—")),
        boxpoints: "all",
        jitter: 0.22,
        marker: { color: R.PALETTE[index % R.PALETTE.length], size: 6, opacity: 0.7 },
        line: { color: R.PALETTE[index % R.PALETTE.length] },
        hovertemplate: "<b>%{fullData.name}</b><br>Execution: %{text}<br>Runtime: %{y:.1f} s<extra></extra>",
      };
    });
    R.plot(target, traces, {
      margin: { l: 62, r: 20, t: 12, b: 95 },
      xaxis: { title: { text: "Inference method", standoff: 14 }, tickangle: -22, automargin: true },
      yaxis: { title: { text: "Runtime (seconds; logarithmic scale, lower is better)", standoff: 12 }, type: "log", gridcolor: "#e7ebf1", zeroline: false },
      showlegend: false,
    });
  }

  function exportHref(path) {
    if (!path) return null;
    const raw = String(path);
    if (!R.safeLocalHref(raw)) return null;
    return raw;
  }

  function normalizeBundles(exportsData) {
    const datasets = exportsData && Array.isArray(exportsData.datasets)
      ? exportsData.datasets
      : [];
    return datasets.flatMap((dataset) => {
      const regimes = asArray(dataset.regimes);
      if (!regimes.length) {
        return [{
          datasetId: String(dataset.dataset_id),
          datasetZip: dataset.truth_zip,
          regime: null,
          inferenceZip: null,
          evaluationZip: null,
        }];
      }
      return regimes.map((regime) => ({
        datasetId: String(dataset.dataset_id),
        datasetZip: dataset.truth_zip,
        regime: regime.regime,
        inferenceZip: regime.inference_zip,
        evaluationZip: regime.evaluation_zip,
      }));
    }).sort((left, right) => left.datasetId.localeCompare(right.datasetId) || String(left.regime).localeCompare(String(right.regime)));
  }

  function bundleColumns() {
    return [
      { label: "Dataset", value: (row) => row.datasetId },
      { label: "Regime", value: (row) => R.humanize(row.regime) },
      { label: "Ground truth", value: (row) => row.datasetZip, render: (value) => R.fileLink("Dataset ZIP", exportHref(value), true) },
      { label: "Joint inference", value: (row) => row.inferenceZip, render: (value) => R.fileLink("Inference ZIP", exportHref(value), true) },
      { label: "Joint evaluation", value: (row) => row.evaluationZip, render: (value) => R.fileLink("Evaluation ZIP", exportHref(value), true) },
    ];
  }

  function renderBundles(exportsData, regime) {
    const allRows = normalizeBundles(exportsData || {});
    const rows = allRows.filter((row) => !regime || !row.regime || row.regime === regime);
    const complete = rows.filter((row) => row.datasetZip && row.inferenceZip && row.evaluationZip).length;
    const status = R.query("#bundle-status");
    R.clear(status);
    status.appendChild(R.badge(complete === rows.length && rows.length ? "success" : "warning", `${complete}/${rows.length} complete trios`));
    R.createTable(R.query("#bundle-table"), bundleColumns(), rows, {
      emptyTitle: "No joint ZIP bundles",
      emptyDetail: "The export manifest contains no bundles for this regime.",
    });
  }

  function normalizeLinks(value) {
    if (Array.isArray(value)) return value;
    if (!value || typeof value !== "object") return [];
    return Object.entries(value).flatMap(([label, href]) => {
      if (Array.isArray(href)) return href.map((item, index) => ({ label: `${R.humanize(label)} ${index + 1}`, href: item }));
      if (href && typeof href === "object") return Object.entries(href).map(([child, childHref]) => ({ label: `${R.humanize(label)} · ${R.humanize(child)}`, href: childHref }));
      return [{ label: R.humanize(label), href }];
    });
  }

  function renderProvenance(data) {
    const workflow = data.workflow || {};
    const identity = workflow.identity || data.provenance || {};
    const statistics = data.statistics || {};
    R.renderKeyValues(R.query("#provenance"), "Reproducible identity", [
      { label: "Profile ID", value: workflow.profile_id || statistics.profile_id },
      { label: "Purpose", value: workflow.purpose || statistics.purpose },
      { label: "SPATHI commit", value: identity.spathi_commit || workflow.spathi_commit },
      { label: "ANDREA commit", value: identity.andrea_commit || workflow.andrea_commit },
      { label: "Benchmark commit", value: identity.benchmark_commit || (data.summary && data.summary.benchmark_commit) },
      { label: "Fingerprint", value: identity.fingerprint || workflow.fingerprint },
      { label: "Analysis mode", value: statistics.contract && statistics.contract.status },
      { label: "Schema", value: statistics.schema_version },
    ]);
    R.renderLinks(R.query("#report-links"), "Artifacts", normalizeLinks(data.links || data.artifacts));
  }

  function renderAnalysis(data, analysis) {
    renderAnalysisState(analysis);
    renderAnalysisBadges(analysis);
    renderScores(analysis);
    renderRanks(analysis);
    renderPaired(analysis);
    renderForest(analysis);
    renderHolm(analysis);
    renderConditions(data, analysis);
    renderRuntime(data, analysis);
    renderBundles(data.exports || {}, analysisField(analysis, "regime"));
  }

  R.boot((data) => {
    const statistics = data.statistics || {};
    const analyses = asArray(statistics.analyses || data.analyses);
    const workflow = data.workflow || {
      status: statistics.status,
      profile_id: statistics.profile_id,
      purpose: statistics.purpose,
    };
    R.renderHeader(workflow, "comparison", data.integrity || {});
    renderStatus(data, analyses);
    R.renderDatasetOverview(R.query("#dataset-overview"), data.datasets || {});
    renderOverview(data, analyses);
    buildFilters(analyses, (analysis) => renderAnalysis(data, analysis));
    if (!analyses.length) {
      ["#score-chart", "#rank-chart", "#paired-chart", "#forest-chart", "#holm-table", "#condition-chart", "#runtime-chart"].forEach((selector) => R.empty(R.query(selector), "No analyses", "There are no data to visualize."));
      renderBundles(data.exports || {}, null);
    }
    renderProvenance(data);
  });
}());
