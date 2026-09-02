(function () {
  "use strict";

  const R = window.SpathiReport;
  const STAGE_ORDER = ["convergence", "structure", "estimator", "validation"];

  function asArray(value) {
    if (Array.isArray(value)) return value;
    if (value && typeof value === "object") return Object.entries(value).map(([id, item]) => Object.assign({ id }, item));
    return [];
  }

  function firstDefined() {
    for (let index = 0; index < arguments.length; index += 1) {
      const value = arguments[index];
      if (value !== null && value !== undefined) return value;
    }
    return null;
  }

  function normalizeCandidate(row, rank, winnerId) {
    const candidate = row.candidate || row.configuration || row;
    const id = firstDefined(
      candidate.configuration_id,
      candidate.candidate_id,
      candidate.id,
      row.configuration_id,
      row.candidate_id,
      row.id,
      `candidate-${rank + 1}`,
    );
    return {
      id: String(id),
      rank: firstDefined(row.rank, rank + 1),
      score: R.number(firstDefined(row.hierarchical_aupr, row.hierarchical_score, row.score, row.primary_score, candidate.score), NaN),
      stability: R.number(firstDefined(row.stability, candidate.stability), NaN),
      time: R.number(firstDefined(row.median_wall_time_seconds, row.wall_time_seconds, row.runtime_seconds, candidate.median_wall_time_seconds), NaN),
      memory: R.number(firstDefined(row.median_peak_memory_mb, row.peak_memory_mb, candidate.median_peak_memory_mb), NaN),
      simplicity: firstDefined(row.simplicity, candidate.simplicity),
      datasetCount: firstDefined(row.dataset_count, row.n_datasets, candidate.dataset_count),
      seedCount: firstDefined(row.algorithm_seed_count, row.n_seeds, candidate.algorithm_seed_count),
      role: firstDefined(candidate.selection_role, row.selection_role, candidate.variant_id),
      parameters: candidate.parameters || row.parameters || {},
      selected: Boolean(row.selected) || String(id) === String(winnerId),
      sha256: firstDefined(candidate.configuration_sha256, row.configuration_sha256),
    };
  }

  function normalizeStage(stage, index) {
    const document = stage.ranking_document || stage.ranking_data || stage.document || stage;
    const rawRanking = Array.isArray(document.ranking)
      ? document.ranking
      : Array.isArray(stage.candidates)
        ? stage.candidates
        : Array.isArray(stage.ranking)
          ? stage.ranking
          : [];
    const rawWinner = document.winner || stage.winner || {};
    const winnerCandidate = rawWinner.candidate || rawWinner;
    const winnerId = firstDefined(
      winnerCandidate.configuration_id,
      winnerCandidate.candidate_id,
      winnerCandidate.id,
      document.winner_candidate_id,
      stage.winner_id,
    );
    const candidates = rawRanking.map((row, rank) => normalizeCandidate(row, rank, winnerId));
    const summary = stage.summary || document.summary || {};
    const observations = asArray(firstDefined(
      stage.observations,
      stage.calibration_rows,
      summary.calibration_rows,
      document.observations,
      [],
    ));
    const normalized = {
      id: String(firstDefined(stage.id, stage.stage_id, document.stage_id, `stage-${index + 1}`)),
      profileId: firstDefined(stage.profile_id, document.profile_id),
      status: firstDefined(stage.status, document.status, candidates.length ? "ranked" : "incomplete"),
      datasetRole: firstDefined(document.dataset_role, stage.dataset_role),
      primaryMetric: firstDefined(document.primary_metric, stage.primary_metric, "aupr_directed"),
      testAccessed: Boolean(firstDefined(document.test_data_accessed, stage.test_data_accessed, false)),
      candidates,
      observations,
      winnerId: winnerId || (candidates[0] && candidates[0].id),
      purpose: firstDefined(stage.purpose, document.purpose),
      strategy: firstDefined(stage.strategy, document.strategy),
      varyingParameters: firstDefined(stage.varying_parameters, document.varying_parameters, []),
      selectedIds: firstDefined(stage.selected_ids, document.selected_ids, []),
      memoryAvailable: Boolean(firstDefined(stage.memory_available, document.memory_available, false)),
      tieBreakers: document.tie_breakers || stage.tie_breakers || [],
      selectedCandidates: document.selected_candidates || stage.selected_candidates || [],
      source: document,
    };
    const competition = parameterCompetition(normalized);
    normalized.parameterCompetition = competition;
    const rawLabels = normalized.candidates.map((candidate) => candidateParameterLabel(candidate, competition));
    const occurrences = new Map();
    rawLabels.forEach((label) => occurrences.set(label, (occurrences.get(label) || 0) + 1));
    normalized.candidates.forEach((candidate, candidateIndex) => {
      const rawLabel = rawLabels[candidateIndex];
      candidate.displayLabel = occurrences.get(rawLabel) > 1
        ? `${rawLabel} · candidate ${candidate.rank}`
        : rawLabel;
    });
    return normalized;
  }

  function scalarKey(value) {
    if (value === null) return "null";
    if (typeof value === "object") return JSON.stringify(value);
    return String(value);
  }

  function displayParameterValue(value) {
    if (value === null) return "null";
    if (typeof value === "boolean") return value ? "true" : "false";
    if (typeof value === "number") return R.formatNumber(value, { maximumFractionDigits: 6 });
    if (typeof value === "object") return JSON.stringify(value);
    return R.humanize(String(value));
  }

  function parameterCompetition(stage) {
    const candidates = stage.candidates || [];
    const configured = Array.isArray(stage.varyingParameters) ? stage.varyingParameters : [];
    const allKeys = R.unique(candidates.flatMap((candidate) => Object.keys(candidate.parameters || {})));
    const changing = allKeys.filter((key) => (
      R.unique(candidates.map((candidate) => scalarKey(candidate.parameters[key]))).length > 1
    ));
    const keys = configured.length
      ? configured.filter((key) => changing.includes(key))
      : changing;
    return keys.map((key) => ({
      key,
      label: R.humanize(key),
      values: R.unique(candidates.map((candidate) => scalarKey(candidate.parameters[key])))
        .map((value) => displayParameterValue(value === "null" ? null : value)),
    }));
  }

  function candidateParameterLabel(candidate, competition) {
    const parameters = candidate.parameters || {};
    let dimensions = competition;
    if (!dimensions.length) {
      const fallbackKeys = [
        "n_estimators",
        "distance_metric",
        "weight_mode",
        "group_size_correction",
        "tree_method",
        "max_features",
        "min_samples_leaf",
      ].filter((key) => Object.prototype.hasOwnProperty.call(parameters, key));
      dimensions = fallbackKeys.slice(0, 2).map((key) => ({ key, label: R.humanize(key) }));
    }
    const label = dimensions.map((dimension) => (
      `${dimension.label}: ${displayParameterValue(parameters[dimension.key])}`
    )).join(" · ");
    return label || `Candidate ${candidate.rank}`;
  }

  function competitionSummary(stage) {
    if (!stage.parameterCompetition.length) return "No parameter varies in this stage";
    return stage.parameterCompetition
      .map((item) => `${item.label}: ${item.values.join(" / ")}`)
      .join(" · ");
  }

  function stageSort(left, right) {
    const leftIndex = STAGE_ORDER.indexOf(left.id);
    const rightIndex = STAGE_ORDER.indexOf(right.id);
    if (leftIndex === -1 && rightIndex === -1) return left.id.localeCompare(right.id);
    if (leftIndex === -1) return 1;
    if (rightIndex === -1) return -1;
    return leftIndex - rightIndex;
  }

  function selectionParts(selectedConfig) {
    const selected = selectedConfig || {};
    const candidate = selected.selected_candidate || selected.candidate || selected.selected_config || selected;
    const evidence = selected.selection_evidence || selected.evidence || {};
    return { selected, candidate, evidence };
  }

  function renderStatus(data, stages) {
    const target = R.query("#status-banner");
    const workflow = data.workflow || {};
    const integrity = data.integrity || {};
    const selected = selectionParts(data.selected || data.selected_config).selected;
    const incomplete = stages.filter((stage) => !["ranked", "complete", "completed", "selected", "tested"].includes(String(stage.status).toLowerCase()));
    const expectedTrue = ["all_stages_ranked", "single_spathi_commit", "selection_frozen"];
    const integrityFailed = expectedTrue.some((key) => Object.prototype.hasOwnProperty.call(integrity, key) && integrity[key] !== true)
      || (Object.prototype.hasOwnProperty.call(integrity, "test_data_accessed_before_freeze") && integrity.test_data_accessed_before_freeze !== false);
    const testAccessed = Boolean(selected.test_accessed_before_freeze) || Boolean(integrity.test_data_accessed_before_freeze) || stages.some((stage) => stage.testAccessed);
    if (testAccessed) {
      R.notice(target, "error", "Possible information leakage", "The artifact records access to test data before the configuration was frozen. Do not interpret this selection as valid.");
    } else if (integrityFailed || (integrity.status && ["failed", "invalid", "error"].includes(String(integrity.status).toLowerCase()))) {
      R.notice(target, "error", "Integrity not validated", integrity.message || "At least one report integrity check failed.");
    } else if (incomplete.length || !["complete", "completed"].includes(String(workflow.status || "").toLowerCase())) {
      R.notice(target, "warning", "Incomplete calibration", `${incomplete.length || "One or more"} stages are not closed. Candidates are shown, but they do not constitute a final selection.`);
    } else {
      R.notice(target, "success", "Calibration complete", "The selection is frozen, and development remains separate from validation throughout the report.");
    }
  }

  function renderOverview(data, stages) {
    const selectedParts = selectionParts(data.selected || data.selected_config);
    const evidence = selectedParts.evidence;
    const totalCandidates = stages.reduce((sum, stage) => sum + stage.candidates.length, 0);
    const totalObservations = stages.reduce((sum, stage) => sum + stage.observations.length, 0);
    const completeStages = stages.filter((stage) => ["ranked", "complete", "completed", "selected", "tested"].includes(String(stage.status).toLowerCase())).length;
    R.addStatCards(R.query("#overview-cards"), [
      { label: "Completed stages", value: `${completeStages}/${stages.length || 0}`, detail: stages.map((stage) => R.humanize(stage.id)).join(" · ") || "No stages" },
      { label: "Evaluated candidates", value: R.formatInteger(totalCandidates), detail: "Sum of each stage's search space" },
      { label: "Observations", value: R.formatInteger(totalObservations), detail: "Recorded datasets and algorithm seeds" },
      { label: "Primary metric", value: R.humanize(firstDefined(evidence.primary_metric, stages[0] && stages[0].primaryMetric)), detail: "Canonical hierarchical selection criterion" },
      { label: "Frozen score", value: R.formatNumber(evidence.hierarchical_score), detail: (data.selected || data.selected_config) && (data.selected || data.selected_config).status ? R.statusLabel((data.selected || data.selected_config).status) : "—" },
    ]);
  }

  function parameterGrid(parameters) {
    const list = R.element("dl", { className: "parameter-grid" });
    const entries = Object.entries(parameters || {});
    if (!entries.length) {
      const note = R.element("p", { text: "The artifact does not include this configuration's parameters." });
      return note;
    }
    entries.forEach(([key, value]) => {
      const item = R.element("div", { className: "parameter-item" });
      item.appendChild(R.element("dt", { text: R.humanize(key) }));
      item.appendChild(R.element("dd", { text: value === null ? "null" : typeof value === "object" ? JSON.stringify(value) : String(value) }));
      list.appendChild(item);
    });
    return list;
  }

  function renderSelection(data, stages) {
    const target = R.query("#selection-card");
    R.clear(target);
    const parts = selectionParts(data.selected || data.selected_config);
    const candidate = parts.candidate || {};
    const evidence = parts.evidence || {};
    const id = firstDefined(candidate.candidate_id, candidate.configuration_id, candidate.id, evidence.winner_candidate_id);
    if (!id) {
      R.empty(target, "No selected configuration yet", "The selection will appear after validation has completed and been frozen.");
      return;
    }

    const hero = R.element("article", { className: "selection-hero" });
    hero.appendChild(R.badge(parts.selected.status || parts.selected.phase || "selected"));
    const validationStage = stages.find((stage) => stage.id === "validation") || stages[stages.length - 1];
    const normalizedWinner = validationStage && validationStage.candidates.find((item) => (
      String(item.id) === String(id)
      || (candidate.configuration_sha256 && item.sha256 === candidate.configuration_sha256)
    ));
    hero.appendChild(R.element("h3", {
      text: normalizedWinner
        ? normalizedWinner.displayLabel
        : candidateParameterLabel(
          { rank: 1, parameters: candidate.parameters || {} },
          validationStage ? validationStage.parameterCompetition : [],
        ),
    }));
    hero.appendChild(R.element("p", { text: "Winner of the frozen validation stage" }));
    const score = R.element("div", { className: "selection-score" });
    score.appendChild(R.element("strong", { text: R.formatNumber(evidence.hierarchical_score, { maximumFractionDigits: 6 }) }));
    score.appendChild(R.element("span", { text: R.humanize(evidence.primary_metric || "score") }));
    hero.appendChild(score);

    const parameters = R.element("article", { className: "card" });
    const heading = R.element("div", { className: "card-heading" });
    const headingCopy = R.element("div");
    headingCopy.appendChild(R.element("p", { className: "card-kicker", text: "Parameters" }));
    headingCopy.appendChild(R.element("h3", { text: "Complete configuration" }));
    heading.appendChild(headingCopy);
    parameters.appendChild(heading);
    parameters.appendChild(parameterGrid(candidate.parameters || {}));
    target.appendChild(hero);
    target.appendChild(parameters);
  }

  function renderStageIntro(stage) {
    const target = R.query("#stage-intro");
    R.clear(target);
    const copy = R.element("div", { className: "stage-intro__copy" });
    copy.appendChild(R.element("h3", { text: R.humanize(stage.id) }));
    const details = [
      stage.profileId,
      `${stage.candidates.length} candidates`,
      `${stage.observations.length} observations`,
      stage.datasetRole ? `${R.humanize(stage.datasetRole)} partition` : null,
    ].filter(Boolean).join(" · ");
    copy.appendChild(R.element("p", { text: details }));
    if (stage.purpose) copy.appendChild(R.element("p", { text: stage.purpose }));
    if (stage.strategy) copy.appendChild(R.element("p", { text: `Strategy: ${R.humanize(stage.strategy)}` }));
    const competition = R.element("div", { className: "parameter-competition" });
    competition.appendChild(R.element("span", {
      className: "parameter-competition__heading",
      text: "Parameters and values under comparison",
    }));
    const competitionList = R.element("div", { className: "parameter-competition__list" });
    if (stage.parameterCompetition.length) {
      stage.parameterCompetition.forEach((item) => {
        competitionList.appendChild(R.element("span", {
          className: "parameter-competition__item",
          text: `${item.label}: ${item.values.join(" / ")}`,
        }));
      });
    } else {
      competitionList.appendChild(R.element("span", {
        className: "parameter-competition__item",
        text: "No parameter varies in this stage",
      }));
    }
    competition.appendChild(competitionList);
    copy.appendChild(competition);
    const badges = R.element("div", { className: "badge-row" });
    badges.appendChild(R.badge(stage.status));
    if (stage.datasetRole) badges.appendChild(R.badge(stage.datasetRole === "test" ? "warning" : "neutral", R.humanize(stage.datasetRole)));
    if (stage.selectedIds.length) badges.appendChild(R.badge("selected", `${stage.selectedIds.length} advance`));
    if (!stage.memoryAvailable) badges.appendChild(R.badge("warning", "Memory not measured"));
    R.append(target, copy, badges);
  }

  function candidateLabel(candidate) {
    const label = candidate.displayLabel || `Candidate ${candidate.rank}`;
    return label.length > 52 ? `${label.slice(0, 49)}…` : label;
  }

  function renderRanking(stage) {
    const target = R.query("#ranking-chart");
    const candidates = stage.candidates.filter((candidate) => Number.isFinite(candidate.score)).slice(0, 25).reverse();
    if (!candidates.length) {
      R.empty(target, "Ranking unavailable", "This stage does not contain canonical accuracy scores.");
      return;
    }
    R.plot(target, [{
      type: "bar",
      orientation: "h",
      x: candidates.map((candidate) => candidate.score),
      y: candidates.map(candidateLabel),
      customdata: candidates.map((candidate) => [candidate.displayLabel, candidate.rank, candidate.stability, candidate.time]),
      marker: { color: candidates.map((candidate) => candidate.selected ? "#0f8c84" : "#8ea4bf") },
      hovertemplate: "<b>%{customdata[0]}</b><br>Accuracy score: %{x:.6f}<br>Rank: %{customdata[1]}<br>Stability: %{customdata[2]:.5f}<br>Runtime: %{customdata[3]:.1f} s<extra></extra>",
    }], {
      margin: { l: 235, r: 18, t: 12, b: 76 },
      xaxis: { title: { text: `${R.humanize(stage.primaryMetric)} accuracy score (higher is better)`, standoff: 14 }, gridcolor: "#e7ebf1", zeroline: false },
      yaxis: { title: { text: "Candidate parameterization", standoff: 12 }, automargin: true },
      showlegend: false,
    });
  }

  function paretoFront(candidates) {
    const sorted = candidates.slice().sort((left, right) => left.time - right.time || right.score - left.score);
    let bestScore = -Infinity;
    return sorted.filter((candidate) => {
      if (candidate.score > bestScore) {
        bestScore = candidate.score;
        return true;
      }
      return false;
    });
  }

  function renderPareto(stage) {
    const target = R.query("#pareto-chart");
    const title = R.query("#pareto-chart-title");
    const note = title.closest(".card").querySelector(".legend-note");
    const convergence = stage.id === "convergence"
      ? stage.candidates.filter((candidate) => Number.isFinite(R.number(candidate.parameters.n_estimators, NaN)) && Number.isFinite(candidate.score))
      : [];
    if (convergence.length) {
      title.textContent = "Tree-count convergence and cost";
      if (note) note.textContent = "Tree counts are equally spaced so accuracy and runtime are easy to compare";
      const ordered = convergence.slice().sort((left, right) => Number(left.parameters.n_estimators) - Number(right.parameters.n_estimators));
      const treeCategories = ordered.map((candidate) => String(candidate.parameters.n_estimators));
      const traces = [{
        type: "scatter",
        mode: "lines+markers",
        name: `Accuracy — ${R.humanize(stage.primaryMetric)}`,
        x: treeCategories,
        y: ordered.map((candidate) => candidate.score),
        text: ordered.map((candidate) => candidate.displayLabel),
        customdata: ordered.map((candidate) => [candidate.time, candidate.stability, candidate.selected ? "yes" : "no"]),
        line: { color: "#1d4ed8", width: 3 },
        marker: { color: "#1d4ed8", size: ordered.map((candidate) => candidate.selected ? 13 : 9) },
        hovertemplate: "<b>%{text}</b><br>Trees (n_estimators): %{x}<br>Accuracy score: %{y:.6f}<br>Median runtime: %{customdata[0]:.1f} s<br>Stability: %{customdata[1]:.5f}<br>Advances: %{customdata[2]}<extra></extra>",
      }];
      const timed = ordered.filter((candidate) => Number.isFinite(candidate.time));
      if (timed.length) {
        traces.push({
          type: "scatter",
          mode: "lines+markers",
          name: "Cost — median runtime",
          x: timed.map((candidate) => String(candidate.parameters.n_estimators)),
          y: timed.map((candidate) => candidate.time),
          text: timed.map((candidate) => candidate.displayLabel),
          yaxis: "y2",
          line: { color: "#db6b22", width: 2, dash: "dot" },
          marker: { color: "#db6b22", size: 7 },
          hovertemplate: "<b>%{text}</b><br>Trees (n_estimators): %{x}<br>Median runtime: %{y:.1f} s<extra></extra>",
        });
      }
      R.plot(target, traces, {
        xaxis: {
          title: { text: "Number of trees (n_estimators; categorical spacing)", standoff: 14 },
          type: "category",
          categoryorder: "array",
          categoryarray: treeCategories,
          gridcolor: "#e7ebf1",
          zeroline: false,
        },
        yaxis: { title: { text: `${R.humanize(stage.primaryMetric)} score<br>(higher is better)`, standoff: 12 }, gridcolor: "#e7ebf1", zeroline: false },
        yaxis2: { title: { text: "Median runtime (seconds)<br>(lower is better)", standoff: 12 }, overlaying: "y", side: "right", showgrid: false, zeroline: false },
        margin: { l: 88, r: 104, t: 58, b: 78 },
        legend: { orientation: "h", x: 0, y: 1.17 },
      });
      return;
    }
    title.textContent = "Accuracy versus runtime";
    if (note) note.textContent = "Pareto frontier highlighted";
    const candidates = stage.candidates.filter((candidate) => Number.isFinite(candidate.score) && Number.isFinite(candidate.time));
    if (!candidates.length) {
      R.empty(target, "Cost unavailable", "Accuracy score and runtime were not recorded together for this stage.");
      return;
    }
    const frontier = paretoFront(candidates);
    const traces = [{
      type: "scatter",
      mode: "markers",
      name: "Candidates",
      x: candidates.map((candidate) => candidate.time),
      y: candidates.map((candidate) => candidate.score),
      text: candidates.map((candidate) => candidate.displayLabel),
      customdata: candidates.map((candidate) => [candidate.stability, candidate.memory, candidate.selected ? "yes" : "no"]),
      marker: {
        color: candidates.map((candidate) => candidate.selected ? "#0f8c84" : "#2563eb"),
        size: candidates.map((candidate) => candidate.selected ? 13 : 9),
        line: { color: "#ffffff", width: 1.5 },
        opacity: 0.84,
      },
      hovertemplate: "<b>%{text}</b><br>Median runtime: %{x:.1f} s<br>Accuracy score: %{y:.6f}<br>Stability: %{customdata[0]:.5f}<br>Memory: %{customdata[1]:.1f} MiB<br>Selected: %{customdata[2]}<extra></extra>",
    }];
    if (frontier.length > 1) {
      traces.push({
        type: "scatter",
        mode: "lines+markers",
        name: "Accuracy/runtime Pareto frontier",
        x: frontier.map((candidate) => candidate.time),
        y: frontier.map((candidate) => candidate.score),
        text: frontier.map((candidate) => candidate.displayLabel),
        line: { color: "#db6b22", width: 2, dash: "dot" },
        marker: { color: "#db6b22", size: 6 },
        hovertemplate: "<b>%{text}</b><br>Median runtime: %{x:.1f} s<br>Accuracy score: %{y:.6f}<extra></extra>",
      });
    }
    R.plot(target, traces, {
      xaxis: { title: { text: "Median runtime (seconds; lower is better)", standoff: 14 }, gridcolor: "#e7ebf1", zeroline: false },
      yaxis: { title: { text: `${R.humanize(stage.primaryMetric)} accuracy score (higher is better)`, standoff: 12 }, gridcolor: "#e7ebf1", zeroline: false },
      legend: { orientation: "h", y: -0.2 },
    });
  }

  function observationFields(row) {
    return {
      configurationId: firstDefined(row.configuration_id, row.candidate_id, row.id),
      datasetId: firstDefined(row.dataset_id, row.block_id, row.condition_id, "unknown dataset"),
      seed: firstDefined(row.algorithm_seed, row.seed, row.replicate),
      score: R.number(firstDefined(row.aupr_directed, row.hierarchical_aupr, row.score, row.primary_score), NaN),
      time: R.number(firstDefined(row.wall_time_seconds, row.runtime_seconds), NaN),
    };
  }

  function renderObservations(stage) {
    const target = R.query("#observations-chart");
    const observations = stage.observations.map(observationFields).filter((row) => row.configurationId && Number.isFinite(row.score));
    if (!observations.length) {
      R.empty(target, "Observations unavailable", "The artifact only preserves the aggregate ranking for this stage.");
      return;
    }
    const labels = new Map(stage.candidates.map((candidate) => [candidate.id, candidate.displayLabel]));
    const ids = R.unique(observations.map((row) => row.configurationId));
    const traces = ids.map((id, index) => {
      const rows = observations.filter((row) => row.configurationId === id);
      return {
        type: "box",
        name: candidateLabel({ id, rank: index + 1, displayLabel: labels.get(id) }),
        y: rows.map((row) => row.score),
        text: rows.map((row) => row.datasetId),
        customdata: rows.map((row) => [row.seed, row.time, labels.get(id) || `Candidate ${index + 1}`]),
        boxpoints: "all",
        jitter: 0.26,
        pointpos: 0,
        marker: { color: R.PALETTE[index % R.PALETTE.length], size: 6, opacity: 0.7 },
        line: { color: R.PALETTE[index % R.PALETTE.length] },
        hovertemplate: "<b>%{customdata[2]}</b><br>Dataset: %{text}<br>Algorithm seed: %{customdata[0]}<br>Accuracy score: %{y:.6f}<br>Runtime: %{customdata[1]:.1f} s<extra></extra>",
      };
    });
    R.plot(target, traces, {
      margin: { l: 62, r: 18, t: 12, b: 155 },
      xaxis: { title: { text: "Candidate parameterization", standoff: 14 }, tickangle: -30, automargin: true },
      yaxis: { title: { text: `${R.humanize(stage.primaryMetric)} accuracy score (higher is better)`, standoff: 12 }, gridcolor: "#e7ebf1", zeroline: false },
      showlegend: false,
    });
  }

  function normalizedParameterColumn(key, candidates) {
    const raw = candidates.map((candidate) => candidate.parameters[key]);
    const numeric = raw.every((value) => value !== null && value !== undefined && Number.isFinite(Number(value)));
    if (numeric) {
      const numbers = raw.map(Number);
      const minimum = Math.min.apply(null, numbers);
      const maximum = Math.max.apply(null, numbers);
      return numbers.map((value) => maximum === minimum ? 0.5 : (value - minimum) / (maximum - minimum));
    }
    const labels = R.unique(raw.map(scalarKey));
    return raw.map((value) => {
      const index = labels.indexOf(scalarKey(value));
      return labels.length < 2 ? 0.5 : index / (labels.length - 1);
    });
  }

  function renderParameterHeatmap(target, stage, candidates, keys) {
    const candidateLabels = candidates.map((candidate) => candidateLabel(candidate));
    const columns = keys.map((key) => normalizedParameterColumn(key, candidates));
    return R.plot(target, [{
      type: "heatmap",
      x: keys.map(R.humanize),
      y: candidateLabels,
      z: candidates.map((_candidate, rowIndex) => columns.map((column) => column[rowIndex])),
      text: candidates.map((candidate) => keys.map((key) => scalarKey(candidate.parameters[key]))),
      texttemplate: "%{text}",
      colorscale: [[0, "#dbeafe"], [0.5, "#60a5fa"], [1, "#1e3a8a"]],
      showscale: false,
      hovertemplate: "<b>%{y}</b><br>%{x}: %{text}<extra></extra>",
    }], {
      margin: { l: 190, r: 24, t: 28, b: 90 },
      xaxis: { title: { text: "Varying parameter", standoff: 14 }, tickangle: -20, automargin: true },
      yaxis: { title: { text: "Candidate parameterization", standoff: 12 }, automargin: true },
    });
  }

  function svgNode(name, attributes) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", name);
    Object.entries(attributes || {}).forEach(([key, value]) => {
      if (value !== null && value !== undefined) node.setAttribute(key, String(value));
    });
    return node;
  }

  function wrappedSvgText(parent, text, x, y, options) {
    const settings = options || {};
    const maxCharacters = settings.maxCharacters || 22;
    const words = String(text).split(/\s+/);
    const lines = [];
    let current = "";
    words.forEach((word) => {
      const next = current ? `${current} ${word}` : word;
      if (current && next.length > maxCharacters) {
        lines.push(current);
        current = word;
      } else {
        current = next;
      }
    });
    if (current) lines.push(current);
    const label = svgNode("text", {
      x,
      y,
      class: settings.className || "parallel-axis__text",
      "text-anchor": settings.anchor || "start",
    });
    lines.forEach((line, index) => {
      const span = svgNode("tspan", {
        x,
        dy: index === 0 ? 0 : (settings.lineHeight || 12),
      });
      span.textContent = line;
      label.appendChild(span);
    });
    parent.appendChild(label);
    return label;
  }

  function numericTicks(values) {
    const unique = R.unique(values).slice().sort((left, right) => left - right);
    if (unique.length <= 6) return unique;
    return [unique[0], unique[Math.floor((unique.length - 1) / 2)], unique[unique.length - 1]];
  }

  function parallelAxis(label, rawValues, formatValue, compactTicks) {
    const numeric = rawValues.every((value) => typeof value === "number" && Number.isFinite(value));
    if (numeric) {
      const minimum = Math.min.apply(null, rawValues);
      const maximum = Math.max.apply(null, rawValues);
      const normalized = rawValues.map((value) => (
        maximum === minimum ? 0.5 : (value - minimum) / (maximum - minimum)
      ));
      const tickValues = compactTicks && minimum !== maximum
        ? [minimum, maximum]
        : numericTicks(rawValues);
      return {
        label,
        normalized,
        ticks: tickValues.map((value) => ({
          normalized: maximum === minimum ? 0.5 : (value - minimum) / (maximum - minimum),
          label: formatValue(value),
        })),
      };
    }
    const categories = R.unique(rawValues.map(scalarKey));
    return {
      label,
      normalized: rawValues.map((value) => {
        const index = categories.indexOf(scalarKey(value));
        return categories.length < 2 ? 0.5 : index / (categories.length - 1);
      }),
      ticks: categories.map((value, index) => ({
        normalized: categories.length < 2 ? 0.5 : index / (categories.length - 1),
        label: displayParameterValue(value === "null" ? null : value),
      })),
    };
  }

  function mixChannel(low, high, amount) {
    return Math.round(low + ((high - low) * amount));
  }

  function scoreBlue(score, minimum, maximum) {
    const amount = maximum === minimum ? 0.75 : Math.max(0, Math.min(1, (score - minimum) / (maximum - minimum)));
    const low = [147, 197, 253];
    const high = [30, 58, 138];
    return `rgb(${mixChannel(low[0], high[0], amount)} ${mixChannel(low[1], high[1], amount)} ${mixChannel(low[2], high[2], amount)})`;
  }

  function renderSvgParallelCoordinates(target, stage, candidates, keys) {
    if (typeof document.createElementNS !== "function") return false;
    R.clear(target);
    const scoreValues = candidates.map((candidate) => candidate.score);
    const minimumScore = Math.min.apply(null, scoreValues);
    const maximumScore = Math.max.apply(null, scoreValues);
    const axes = keys.map((key) => parallelAxis(
      R.humanize(key),
      candidates.map((candidate) => candidate.parameters[key]),
      (value) => displayParameterValue(value),
    ));
    axes.push(parallelAxis(
      `${R.humanize(stage.primaryMetric)} score`,
      scoreValues,
      (value) => R.formatNumber(value, { maximumFractionDigits: 5 }),
      true,
    ));

    const chart = R.element("div", { className: "parallel-coordinates" });
    const scroll = R.element("div", { className: "parallel-coordinates__scroll" });
    const gap = 230;
    const horizontalPadding = 78;
    const width = Math.max(900, (horizontalPadding * 2) + (Math.max(axes.length - 1, 1) * gap));
    const height = 560;
    const top = 82;
    const bottom = 485;
    const svg = svgNode("svg", {
      class: "parallel-coordinates__svg",
      viewBox: `0 0 ${width} ${height}`,
      width,
      height,
      role: "img",
      "aria-label": `Parallel coordinates for calibration parameters and ${R.humanize(stage.primaryMetric)} score`,
    });
    const lineLayer = svgNode("g", { class: "parallel-lines" });
    const hitLayer = svgNode("g", { class: "parallel-line-hits" });
    const axisLayer = svgNode("g", { class: "parallel-axes" });
    const visibleLines = [];
    const hitLines = [];

    function xPosition(index) {
      return axes.length === 1
        ? width / 2
        : horizontalPadding + (index * ((width - (horizontalPadding * 2)) / (axes.length - 1)));
    }

    function yPosition(normalized) {
      return bottom - (normalized * (bottom - top));
    }

    candidates.forEach((candidate, candidateIndex) => {
      const points = axes.map((axis, axisIndex) => (
        `${xPosition(axisIndex)},${yPosition(axis.normalized[candidateIndex])}`
      ));
      const pathData = points.map((point, index) => `${index ? "L" : "M"}${point}`).join(" ");
      const color = scoreBlue(candidate.score, minimumScore, maximumScore);
      const visible = svgNode("path", {
        d: pathData,
        class: `parallel-line${candidate.selected ? " parallel-line--selected" : ""}`,
        stroke: color,
      });
      const title = svgNode("title");
      title.textContent = `${candidate.displayLabel}; ${R.humanize(stage.primaryMetric)} ${R.formatNumber(candidate.score, { maximumFractionDigits: 6 })}`;
      visible.appendChild(title);
      lineLayer.appendChild(visible);
      visibleLines.push(visible);

      const hit = svgNode("path", {
        d: pathData,
        class: "parallel-line-hit",
        tabindex: "0",
        role: "button",
        "aria-label": title.textContent,
      });
      hitLayer.appendChild(hit);
      hitLines.push(hit);
    });

    axes.forEach((axis, axisIndex) => {
      const x = xPosition(axisIndex);
      const group = svgNode("g", { class: "parallel-axis" });
      group.appendChild(svgNode("line", { x1: x, y1: top, x2: x, y2: bottom }));
      wrappedSvgText(group, axis.label, x, 26, {
        anchor: "middle",
        className: "parallel-axis__label",
        maxCharacters: 24,
        lineHeight: 14,
      });
      axis.ticks.forEach((tick) => {
        const y = yPosition(tick.normalized);
        group.appendChild(svgNode("line", {
          x1: x - 5,
          y1: y,
          x2: x + 5,
          y2: y,
          class: "parallel-axis__tick",
        }));
        const lastAxis = axisIndex === axes.length - 1;
        wrappedSvgText(group, tick.label, lastAxis ? x - 9 : x + 9, y + 3, {
          anchor: lastAxis ? "end" : "start",
          className: "parallel-axis__tick-label",
          maxCharacters: 23,
          lineHeight: 11,
        });
      });
      axisLayer.appendChild(group);
    });

    const tooltip = R.element("div", {
      className: "parallel-coordinates__tooltip",
      attrs: { role: "status", "aria-live": "polite" },
    });

    function fillTooltip(candidate) {
      R.clear(tooltip);
      tooltip.appendChild(R.element("strong", { text: candidate.displayLabel }));
      tooltip.appendChild(R.element("span", {
        text: `${R.humanize(stage.primaryMetric)}: ${R.formatNumber(candidate.score, { maximumFractionDigits: 6 })}`,
      }));
      tooltip.appendChild(R.element("span", {
        text: keys.map((key) => `${R.humanize(key)}: ${displayParameterValue(candidate.parameters[key])}`).join(" · "),
      }));
    }

    function activate(candidateIndex, event) {
      visibleLines.forEach((line, index) => {
        line.classList.toggle("is-active", index === candidateIndex);
        line.classList.toggle("is-muted", index !== candidateIndex);
      });
      hitLines.forEach((line, index) => line.classList.toggle("is-active", index === candidateIndex));
      fillTooltip(candidates[candidateIndex]);
      tooltip.classList.add("is-visible");
      if (event && Number.isFinite(event.clientX)) {
        const bounds = chart.getBoundingClientRect();
        tooltip.style.left = `${Math.max(12, event.clientX - bounds.left + 14)}px`;
        tooltip.style.top = `${Math.max(12, event.clientY - bounds.top + 14)}px`;
      } else {
        tooltip.style.left = "18px";
        tooltip.style.top = "18px";
      }
    }

    function deactivate() {
      visibleLines.forEach((line) => line.classList.remove("is-active", "is-muted"));
      hitLines.forEach((line) => line.classList.remove("is-active"));
      tooltip.classList.remove("is-visible");
    }

    hitLines.forEach((line, candidateIndex) => {
      line.addEventListener("mouseenter", (event) => activate(candidateIndex, event));
      line.addEventListener("mousemove", (event) => activate(candidateIndex, event));
      line.addEventListener("mouseleave", deactivate);
      line.addEventListener("focus", () => activate(candidateIndex));
      line.addEventListener("blur", deactivate);
    });

    svg.appendChild(lineLayer);
    svg.appendChild(axisLayer);
    svg.appendChild(hitLayer);
    scroll.appendChild(svg);
    chart.appendChild(scroll);
    chart.appendChild(tooltip);
    const scale = R.element("div", { className: "parallel-coordinates__scale" });
    scale.appendChild(R.element("span", {
      text: `Lower ${R.humanize(stage.primaryMetric)} · ${R.formatNumber(minimumScore, { maximumFractionDigits: 5 })}`,
    }));
    scale.appendChild(R.element("span", { className: "parallel-coordinates__gradient" }));
    scale.appendChild(R.element("span", {
      text: `Higher ${R.humanize(stage.primaryMetric)} · ${R.formatNumber(maximumScore, { maximumFractionDigits: 5 })}`,
    }));
    chart.appendChild(scale);
    chart.appendChild(R.element("p", {
      className: "parallel-coordinates__hint",
      text: "Lines are 3 px wide and colored on a monochromatic blue accuracy scale. Hover or focus a line to highlight one candidate and inspect its exact values.",
    }));
    target.appendChild(chart);
    return true;
  }

  function renderParameters(stage) {
    const target = R.query("#parameters-chart");
    const candidates = stage.candidates.filter((candidate) => Number.isFinite(candidate.score));
    if (candidates.length < 2) {
      R.empty(target, "Search space unavailable", "At least two scored candidates are required to compare parameters.");
      return;
    }
    const configuredKeys = Array.isArray(stage.varyingParameters) && stage.varyingParameters.length
      ? stage.varyingParameters
      : R.unique(candidates.flatMap((candidate) => Object.keys(candidate.parameters || {})));
    const keys = configuredKeys
      .filter((key) => R.unique(candidates.map((candidate) => scalarKey(candidate.parameters[key]))).length > 1);
    if (!keys.length) {
      R.empty(target, "No varying parameters", "Candidates in this stage share the same recorded parameterization.");
      return;
    }
    try {
      if (renderSvgParallelCoordinates(target, stage, candidates, keys)) return;
    } catch (error) {
      if (window.console) window.console.warn("SVG parallel coordinates failed; using heatmap fallback.", error);
    }
    renderParameterHeatmap(target, stage, candidates, keys);
  }

  function candidateColumns() {
    return [
      { label: "Rank", value: (row) => row.rank, numeric: true },
      {
        label: "Candidate parameterization",
        value: (row) => row.displayLabel,
        render: (value, row) => {
          const wrapper = R.element("div");
          wrapper.appendChild(R.element("span", { text: value }));
          if (row.selected) wrapper.appendChild(R.pill("selected", "Advances"));
          return wrapper;
        },
      },
      { label: "Accuracy score", value: (row) => row.score, format: R.formatNumber, numeric: true },
      { label: "Stability", value: (row) => row.stability, format: R.formatNumber, numeric: true },
      { label: "Runtime", value: (row) => row.time, format: R.formatSeconds, numeric: true },
      { label: "Memory", value: (row) => row.memory, format: R.formatBytesMb, numeric: true },
      { label: "Datasets", value: (row) => row.datasetCount, format: R.formatInteger, numeric: true },
      { label: "Algorithm seeds", value: (row) => row.seedCount, format: R.formatInteger, numeric: true },
    ];
  }

  function renderCandidateTable(stage) {
    const columns = candidateColumns();
    R.createTable(R.query("#candidate-table"), columns, stage.candidates, {
      emptyTitle: "No candidates",
      emptyDetail: "This stage does not yet include a canonical ranking.",
    });
    const actions = R.query("#candidate-table-actions");
    R.clear(actions);
    if (stage.candidates.length) actions.appendChild(R.csvButton("Export CSV", `calibration-${stage.id}-candidates.csv`, columns, stage.candidates));
  }

  function renderStage(stage) {
    renderStageIntro(stage);
    renderRanking(stage);
    renderPareto(stage);
    renderObservations(stage);
    renderParameters(stage);
    renderCandidateTable(stage);
  }

  function renderTabs(stages) {
    const target = R.query("#stage-tabs");
    R.clear(target);
    if (!stages.length) {
      R.empty(R.query("#stage-panel"), "No stages", "The calibration workflow records no stage results.");
      return;
    }
    stages.forEach((stage, index) => {
      const button = R.element("button", {
        className: "tab-button",
        attrs: {
          type: "button",
          role: "tab",
          "aria-selected": index === 0 ? "true" : "false",
          "aria-controls": "stage-panel",
          tabindex: index === 0 ? "0" : "-1",
        },
      });
      button.appendChild(R.element("span", {
        className: "tab-button__title",
        text: R.humanize(stage.id),
      }));
      button.appendChild(R.element("span", {
        className: "tab-button__parameters",
        text: competitionSummary(stage),
      }));
      button.addEventListener("click", () => {
        R.queryAll(".tab-button", target).forEach((other) => {
          other.setAttribute("aria-selected", other === button ? "true" : "false");
          other.setAttribute("tabindex", other === button ? "0" : "-1");
        });
        renderStage(stage);
      });
      button.addEventListener("keydown", (event) => {
        if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
        event.preventDefault();
        const buttons = R.queryAll(".tab-button", target);
        const current = buttons.indexOf(button);
        const next = event.key === "Home"
          ? 0
          : event.key === "End"
            ? buttons.length - 1
            : (current + (event.key === "ArrowRight" ? 1 : -1) + buttons.length) % buttons.length;
        buttons[next].focus();
        buttons[next].click();
      });
      target.appendChild(button);
    });
    renderStage(stages[0]);
  }

  function normalizeLinks(value) {
    if (Array.isArray(value)) return value;
    if (!value || typeof value !== "object") return [];
    return Object.entries(value).flatMap(([label, href]) => {
      if (Array.isArray(href)) return href.map((item, index) => ({ label: `${R.humanize(label)} ${index + 1}`, href: item }));
      if (href && typeof href === "object") return Object.entries(href).map(([childLabel, childHref]) => ({ label: `${R.humanize(label)} · ${R.humanize(childLabel)}`, href: childHref }));
      return [{ label: R.humanize(label), href }];
    });
  }

  function renderProvenance(data) {
    const workflow = data.workflow || {};
    const identity = workflow.identity || data.provenance || {};
    const selected = selectionParts(data.selected || data.selected_config);
    R.renderKeyValues(R.query("#provenance"), "Reproducible identity", [
      { label: "Flow ID", value: workflow.flow_id },
      { label: "Evidence budget", value: workflow.calibration_budget },
      { label: "SPATHI commit", value: identity.spathi_commit || selected.candidate.spathi_commit },
      { label: "ANDREA commit", value: identity.andrea_commit },
      { label: "Benchmark commit", value: identity.benchmark_commit },
      { label: "Fingerprint", value: identity.fingerprint },
      { label: "Search space SHA-256", value: identity.search_space_sha256 || selected.evidence.search_space_sha256 },
      { label: "Ranking SHA-256", value: selected.evidence.validation_ranking_sha256 },
      { label: "Frozen at", value: selected.evidence.frozen_at },
    ]);
    R.renderLinks(R.query("#report-links"), "Artifacts", normalizeLinks(data.links || data.artifacts));
  }

  R.boot((data) => {
    const workflow = data.workflow || {};
    const rawStages = data.stages || workflow.stages || [];
    const stages = asArray(rawStages).map(normalizeStage).sort(stageSort);
    R.renderHeader(workflow, "calibration", data.integrity || {});
    renderStatus(data, stages);
    R.renderDatasetOverview(R.query("#dataset-overview"), data.datasets || {});
    renderOverview(data, stages);
    renderSelection(data, stages);
    renderTabs(stages);
    renderProvenance(data);
  });
}());
