(function () {
  "use strict";

  const R = window.SpathiReport;
  const PHASE_ORDER = [
    "geometry__scout", "geometry__confirmation",
    "estimator__scout", "estimator__confirmation",
    "interaction__scout", "interaction__confirmation",
    "estimator_scale_transfer__scout", "estimator_scale_transfer__confirmation",
    "convergence__final", "validation__final",
  ];
  const REPORT_KEYS = [
    "schema_version", "report_type", "workflow", "integrity", "objective",
    "dataset_partition", "datasets", "selected", "stages", "links", "report",
  ];
  const REPORT_METADATA_KEYS = [
    "schema_version", "fingerprint", "fingerprint_scope", "renderer",
    "self_contained", "statistics_recomputed",
  ];
  const WORKFLOW_KEYS = ["flow_id", "status", "calibration_budget", "identity"];
  const IDENTITY_KEYS = [
    "benchmark_commit", "benchmark_origin", "spathi_commit", "andrea_commit",
    "search_space_sha256", "calibration_budget", "fingerprint",
  ];
  const INTEGRITY_KEYS = [
    "phase_order", "all_stages_ranked", "test_data_accessed_before_freeze",
    "single_spathi_commit", "selection_frozen",
  ];
  const STAGE_KEYS = [
    "id", "stage_id", "fidelity", "order", "purpose", "dataset_role",
    "primary_metric", "strategy", "advance_limit", "candidate_source",
    "dataset_selector", "algorithm_seed_indices", "target_scope",
    "expected_physical_fits", "expected_logical_results",
    "candidates", "observations", "varying_parameters",
    "compared_parameters", "selected_ids", "winner", "test_data_accessed",
    "memory_available",
  ];
  const CANDIDATE_KEYS = [
    "rank", "configuration_id", "configuration_sha256", "variant_id",
    "parent_configuration_id", "parameters", "score", "stability",
    "median_wall_time_seconds", "median_peak_memory_mb", "simplicity",
    "dataset_count", "algorithm_seed_count", "selected", "rationale",
  ];
  const OBSERVATION_KEYS = [
    "configuration_id", "configuration_sha256", "dataset_role", "dataset_id",
    "algorithm_seed", "aupr_directed", "wall_time_seconds", "peak_memory_mb",
    "n_group_contexts", "group_contexts", "spathi_commit",
    "spathi_build_fingerprint", "spathi_image_id", "run_id", "execution_sha256",
  ];
  const SELECTED_KEYS = [
    "schema_version", "id", "search_space", "calibration_budget", "status",
    "phase", "selected_policy", "selection_evidence", "test_accessed_before_freeze",
    "notes",
  ];
  const POLICY_KEYS = [
    "policy_id", "source_candidate_id", "spathi_commit",
    "source_configuration_sha256", "policy_sha256", "group_policy",
  ];
  const EVIDENCE_KEYS = [
    "calibration_budget", "validation_ranking", "validation_ranking_sha256",
    "search_space_sha256", "validation_manifest_sha256", "benchmark_summary_sha256",
    "spathi_build_fingerprint", "spathi_image_id", "primary_metric",
    "hierarchical_score", "winner_candidate_id", "winner_configuration_sha256",
    "winner_policy_sha256", "tie_breakers_applied", "frozen_at",
  ];
  const DATASET_SUMMARY_KEYS = [
    "dataset_count", "total_cells", "simulators", "roles", "genes_min",
    "genes_max", "cells_min", "cells_max", "groups_min", "groups_max",
    "transcription_factors_min", "transcription_factors_max",
  ];
  const DATASET_ROW_KEYS = [
    "dataset_id", "simulator_id", "condition_id", "dataset_role", "split_id",
    "seed", "genes", "cells", "column_kind", "groups", "group_labels",
    "transcription_factors",
  ];

  function contractError(path, expectation) {
    throw new Error(`Invalid calibration report payload: ${path} ${expectation}.`);
  }

  function requireObject(value, path) {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      contractError(path, "must be an object");
    }
    return value;
  }

  function requireArray(value, path) {
    if (!Array.isArray(value)) contractError(path, "must be an array");
    return value;
  }

  function requireExactKeys(value, path, expectedKeys) {
    const object = requireObject(value, path);
    const observed = Object.keys(object);
    const missing = expectedKeys.filter((key) => !Object.prototype.hasOwnProperty.call(object, key));
    const unexpected = observed.filter((key) => !expectedKeys.includes(key));
    if (missing.length) contractError(path, `is missing ${missing.join(", ")}`);
    if (unexpected.length) contractError(path, `contains unexpected fields ${unexpected.join(", ")}`);
    return object;
  }

  function requireString(value, path) {
    if (typeof value !== "string" || !value) contractError(path, "must be a non-empty string");
    return value;
  }

  function requireBoolean(value, path) {
    if (typeof value !== "boolean") contractError(path, "must be a boolean");
    return value;
  }

  function requireNumber(value, path, nullable) {
    if (nullable && value === null) return value;
    if (typeof value !== "number" || !Number.isFinite(value)) {
      contractError(path, nullable ? "must be a finite number or null" : "must be a finite number");
    }
    return value;
  }

  function requireInteger(value, path) {
    if (!Number.isInteger(value)) contractError(path, "must be an integer");
    return value;
  }

  function requireLiteral(value, expected, path) {
    if (value !== expected) contractError(path, `must equal ${JSON.stringify(expected)}`);
  }

  function validateCandidate(candidate, path) {
    requireExactKeys(candidate, path, CANDIDATE_KEYS);
    requireInteger(candidate.rank, `${path}.rank`);
    requireString(candidate.configuration_id, `${path}.configuration_id`);
    requireString(candidate.configuration_sha256, `${path}.configuration_sha256`);
    requireString(candidate.variant_id, `${path}.variant_id`);
    if (candidate.parent_configuration_id !== null) {
      requireString(candidate.parent_configuration_id, `${path}.parent_configuration_id`);
    }
    const parameters = requireObject(candidate.parameters, `${path}.parameters`);
    if (!Object.keys(parameters).length) contractError(`${path}.parameters`, "must not be empty");
    requireNumber(candidate.score, `${path}.score`);
    requireNumber(candidate.stability, `${path}.stability`);
    requireNumber(candidate.median_wall_time_seconds, `${path}.median_wall_time_seconds`);
    requireNumber(candidate.median_peak_memory_mb, `${path}.median_peak_memory_mb`, true);
    requireInteger(candidate.simplicity, `${path}.simplicity`);
    requireInteger(candidate.dataset_count, `${path}.dataset_count`);
    requireInteger(candidate.algorithm_seed_count, `${path}.algorithm_seed_count`);
    requireBoolean(candidate.selected, `${path}.selected`);
    requireString(candidate.rationale, `${path}.rationale`);
  }

  function validateObservation(row, path) {
    requireExactKeys(row, path, OBSERVATION_KEYS);
    requireString(row.configuration_id, `${path}.configuration_id`);
    requireString(row.configuration_sha256, `${path}.configuration_sha256`);
    requireString(row.dataset_role, `${path}.dataset_role`);
    requireString(row.dataset_id, `${path}.dataset_id`);
    requireInteger(row.algorithm_seed, `${path}.algorithm_seed`);
    requireNumber(row.aupr_directed, `${path}.aupr_directed`);
    requireNumber(row.wall_time_seconds, `${path}.wall_time_seconds`);
    requireNumber(row.peak_memory_mb, `${path}.peak_memory_mb`, true);
    requireInteger(row.n_group_contexts, `${path}.n_group_contexts`);
    requireArray(row.group_contexts, `${path}.group_contexts`).forEach((group, index) => {
      requireString(group, `${path}.group_contexts[${index}]`);
    });
    ["spathi_commit", "spathi_build_fingerprint", "spathi_image_id", "run_id", "execution_sha256"]
      .forEach((key) => requireString(row[key], `${path}.${key}`));
  }

  function validateSelected(selected) {
    requireExactKeys(selected, "selected", SELECTED_KEYS);
    requireLiteral(selected.schema_version, "1.0", "selected.schema_version");
    requireLiteral(selected.id, "spathi_selected_v1", "selected.id");
    requireLiteral(selected.search_space, "experiments/v1/spathi/search-space.json", "selected.search_space");
    requireLiteral(selected.status, "selected", "selected.status");
    requireLiteral(selected.phase, "frozen", "selected.phase");
    requireLiteral(selected.test_accessed_before_freeze, false, "selected.test_accessed_before_freeze");
    requireArray(selected.notes, "selected.notes").forEach((note, index) => {
      requireString(note, `selected.notes[${index}]`);
    });

    const policy = requireExactKeys(selected.selected_policy, "selected.selected_policy", POLICY_KEYS);
    ["policy_id", "source_candidate_id", "spathi_commit", "source_configuration_sha256", "policy_sha256"]
      .forEach((key) => requireString(policy[key], `selected.selected_policy.${key}`));
    requireLiteral(policy.policy_id, "spathi_conditional_v1", "selected.selected_policy.policy_id");
    const groupPolicy = requireExactKeys(
      policy.group_policy,
      "selected.selected_policy.group_policy",
      ["routing_parameter", "shared_parameters", "branches"],
    );
    requireLiteral(groupPolicy.routing_parameter, "group_count", "selected.selected_policy.group_policy.routing_parameter");
    requireArray(groupPolicy.shared_parameters, "selected.selected_policy.group_policy.shared_parameters")
      .forEach((parameter, index) => {
        requireString(parameter, `selected.selected_policy.group_policy.shared_parameters[${index}]`);
      });
    const branches = requireExactKeys(
      groupPolicy.branches,
      "selected.selected_policy.group_policy.branches",
      ["single_group", "multi_group"],
    );
    ["single_group", "multi_group"].forEach((branchName) => {
      const branchPath = `selected.selected_policy.group_policy.branches.${branchName}`;
      const branch = requireExactKeys(branches[branchName], branchPath, ["condition", "parameters"]);
      const conditionKeys = branchName === "single_group"
        ? ["minimum_group_count", "maximum_group_count"]
        : ["minimum_group_count"];
      const condition = requireExactKeys(branch.condition, `${branchPath}.condition`, conditionKeys);
      requireLiteral(condition.minimum_group_count, branchName === "single_group" ? 1 : 2, `${branchPath}.condition.minimum_group_count`);
      if (branchName === "single_group") {
        requireLiteral(condition.maximum_group_count, 1, `${branchPath}.condition.maximum_group_count`);
      }
      const parameters = requireObject(branch.parameters, `${branchPath}.parameters`);
      if (!Object.keys(parameters).length) contractError(`${branchPath}.parameters`, "must not be empty");
    });

    const evidence = requireExactKeys(selected.selection_evidence, "selected.selection_evidence", EVIDENCE_KEYS);
    [
      "calibration_budget", "validation_ranking", "validation_ranking_sha256",
      "search_space_sha256", "validation_manifest_sha256", "benchmark_summary_sha256",
      "spathi_build_fingerprint", "spathi_image_id", "primary_metric",
      "winner_candidate_id", "winner_configuration_sha256", "winner_policy_sha256",
      "frozen_at",
    ].forEach((key) => requireString(evidence[key], `selected.selection_evidence.${key}`));
    requireNumber(evidence.hierarchical_score, "selected.selection_evidence.hierarchical_score");
    const tieBreakers = requireArray(evidence.tie_breakers_applied, "selected.selection_evidence.tie_breakers_applied");
    if (JSON.stringify(tieBreakers) !== JSON.stringify(["stability", "simplicity", "canonical_configuration_id"])) {
      contractError("selected.selection_evidence.tie_breakers_applied", "must equal stability, simplicity, canonical configuration ID");
    }
  }

  function validateDatasets(datasets) {
    requireExactKeys(datasets, "datasets", ["summary", "rows"]);
    const summary = requireExactKeys(datasets.summary, "datasets.summary", DATASET_SUMMARY_KEYS);
    [
      "dataset_count", "total_cells", "genes_min", "genes_max", "cells_min",
      "cells_max", "groups_min", "groups_max", "transcription_factors_min",
      "transcription_factors_max",
    ].forEach((key) => requireInteger(summary[key], `datasets.summary.${key}`));
    ["simulators", "roles"].forEach((collection) => {
      requireArray(summary[collection], `datasets.summary.${collection}`).forEach((entry, index) => {
        const path = `datasets.summary.${collection}[${index}]`;
        requireExactKeys(entry, path, ["id", "count"]);
        requireString(entry.id, `${path}.id`);
        requireInteger(entry.count, `${path}.count`);
      });
    });
    const rows = requireArray(datasets.rows, "datasets.rows");
    if (!rows.length) contractError("datasets.rows", "must not be empty");
    rows.forEach((row, index) => {
      const path = `datasets.rows[${index}]`;
      requireExactKeys(row, path, DATASET_ROW_KEYS);
      ["dataset_id", "simulator_id", "condition_id", "dataset_role", "split_id", "column_kind"]
        .forEach((key) => requireString(row[key], `${path}.${key}`));
      ["seed", "genes", "cells", "groups", "transcription_factors"]
        .forEach((key) => requireInteger(row[key], `${path}.${key}`));
      requireArray(row.group_labels, `${path}.group_labels`).forEach((group, groupIndex) => {
        requireString(group, `${path}.group_labels[${groupIndex}]`);
      });
    });
    if (summary.dataset_count !== rows.length) {
      contractError("datasets.summary.dataset_count", "must equal datasets.rows.length");
    }
  }

  function validateStage(stage, index) {
    const path = `stages[${index}]`;
    requireExactKeys(stage, path, STAGE_KEYS);
    requireLiteral(stage.id, PHASE_ORDER[index], `${path}.id`);
    requireString(stage.stage_id, `${path}.stage_id`);
    if (!stage.id.startsWith(`${stage.stage_id}__`)) {
      contractError(`${path}.stage_id`, "must identify its phase prefix");
    }
    if (!["scout", "confirmation", "final"].includes(stage.fidelity)) {
      contractError(`${path}.fidelity`, "must be scout, confirmation or final");
    }
    requireLiteral(stage.order, index + 1, `${path}.order`);
    requireString(stage.purpose, `${path}.purpose`);
    requireLiteral(stage.dataset_role, stage.stage_id === "validation" ? "validation" : "development", `${path}.dataset_role`);
    requireLiteral(stage.primary_metric, "aupr_directed", `${path}.primary_metric`);
    requireString(stage.strategy, `${path}.strategy`);
    requireInteger(stage.advance_limit, `${path}.advance_limit`);
    requireString(stage.candidate_source, `${path}.candidate_source`);
    if (stage.dataset_selector !== null) {
      const selector = requireExactKeys(
        stage.dataset_selector,
        `${path}.dataset_selector`,
        ["condition_ids", "replicate_indices"],
      );
      requireArray(selector.condition_ids, `${path}.dataset_selector.condition_ids`)
        .forEach((value, valueIndex) => requireString(value, `${path}.dataset_selector.condition_ids[${valueIndex}]`));
      requireArray(selector.replicate_indices, `${path}.dataset_selector.replicate_indices`)
        .forEach((value, valueIndex) => requireInteger(value, `${path}.dataset_selector.replicate_indices[${valueIndex}]`));
    }
    requireArray(stage.algorithm_seed_indices, `${path}.algorithm_seed_indices`)
      .forEach((value, valueIndex) => requireInteger(value, `${path}.algorithm_seed_indices[${valueIndex}]`));
    requireObject(stage.target_scope, `${path}.target_scope`);
    requireInteger(stage.expected_physical_fits, `${path}.expected_physical_fits`);
    requireInteger(stage.expected_logical_results, `${path}.expected_logical_results`);
    const candidates = requireArray(stage.candidates, `${path}.candidates`);
    if (!candidates.length) contractError(`${path}.candidates`, "must not be empty");
    candidates.forEach((candidate, candidateIndex) => {
      validateCandidate(candidate, `${path}.candidates[${candidateIndex}]`);
      requireLiteral(candidate.rank, candidateIndex + 1, `${path}.candidates[${candidateIndex}].rank`);
    });
    const observations = requireArray(stage.observations, `${path}.observations`);
    if (!observations.length) contractError(`${path}.observations`, "must not be empty");
    observations.forEach((row, rowIndex) => validateObservation(row, `${path}.observations[${rowIndex}]`));
    const varying = requireArray(stage.varying_parameters, `${path}.varying_parameters`);
    varying.forEach((key, keyIndex) => requireString(key, `${path}.varying_parameters[${keyIndex}]`));
    const compared = requireArray(stage.compared_parameters, `${path}.compared_parameters`);
    compared.forEach((entry, entryIndex) => {
      const entryPath = `${path}.compared_parameters[${entryIndex}]`;
      requireExactKeys(entry, entryPath, ["name", "values"]);
      requireString(entry.name, `${entryPath}.name`);
      requireArray(entry.values, `${entryPath}.values`).forEach((value, valueIndex) => {
        requireString(value, `${entryPath}.values[${valueIndex}]`);
      });
    });
    if (JSON.stringify(varying) !== JSON.stringify(compared.map((entry) => entry.name))) {
      contractError(path, "must reconcile varying_parameters with compared_parameters");
    }
    requireArray(stage.selected_ids, `${path}.selected_ids`).forEach((id, idIndex) => {
      requireString(id, `${path}.selected_ids[${idIndex}]`);
    });
    const candidateIds = new Set(candidates.map((candidate) => candidate.configuration_id));
    const selectedFromCandidates = candidates
      .filter((candidate) => candidate.selected)
      .map((candidate) => candidate.configuration_id)
      .sort();
    if (JSON.stringify(selectedFromCandidates) !== JSON.stringify(stage.selected_ids)) {
      contractError(path, "must reconcile selected_ids with candidate selection flags");
    }
    observations.forEach((row) => {
      if (!candidateIds.has(row.configuration_id)) {
        contractError(path, `contains an observation for unknown configuration ${row.configuration_id}`);
      }
    });
    if (stage.stage_id === "validation") {
      const winner = requireExactKeys(stage.winner, `${path}.winner`, ["candidate_id", "configuration_sha256", "hierarchical_score"]);
      requireString(winner.candidate_id, `${path}.winner.candidate_id`);
      requireString(winner.configuration_sha256, `${path}.winner.configuration_sha256`);
      requireNumber(winner.hierarchical_score, `${path}.winner.hierarchical_score`);
    } else if (stage.winner !== null) {
      contractError(`${path}.winner`, "must be null outside validation");
    }
    requireLiteral(stage.test_data_accessed, false, `${path}.test_data_accessed`);
    requireBoolean(stage.memory_available, `${path}.memory_available`);
  }

  function validateCalibrationPayload(data) {
    requireExactKeys(data, "report", REPORT_KEYS);
    requireLiteral(data.schema_version, "1.0", "schema_version");
    requireLiteral(data.report_type, "calibration", "report_type");
    const report = requireExactKeys(data.report, "report", REPORT_METADATA_KEYS);
    requireLiteral(report.schema_version, "1.0", "report.schema_version");
    requireString(report.fingerprint, "report.fingerprint");
    requireLiteral(report.fingerprint_scope, "scientific_inputs", "report.fingerprint_scope");
    const renderer = requireExactKeys(
      report.renderer,
      "report.renderer",
      ["schema_version", "report_type", "dependencies", "resources", "fingerprint"],
    );
    requireLiteral(renderer.schema_version, "1.0", "report.renderer.schema_version");
    requireLiteral(renderer.report_type, "calibration", "report.renderer.report_type");
    requireString(renderer.fingerprint, "report.renderer.fingerprint");
    const dependencies = requireExactKeys(
      renderer.dependencies,
      "report.renderer.dependencies",
      ["jinja2", "plotly"],
    );
    requireString(dependencies.jinja2, "report.renderer.dependencies.jinja2");
    requireString(dependencies.plotly, "report.renderer.dependencies.plotly");
    const resourceNames = [
      "common.py", "calibration.py", "templates/base.html.j2",
      "templates/calibration.html.j2", "assets/report.css", "assets/common.js",
      "assets/calibration.js", "vendor/plotly.js",
    ];
    const reportResources = requireExactKeys(renderer.resources, "report.renderer.resources", resourceNames);
    resourceNames.forEach((name) => {
      const path = `report.renderer.resources.${name}`;
      const resource = requireExactKeys(reportResources[name], path, ["sha256", "size_bytes"]);
      requireString(resource.sha256, `${path}.sha256`);
      requireInteger(resource.size_bytes, `${path}.size_bytes`);
    });
    requireLiteral(report.self_contained, true, "report.self_contained");
    requireLiteral(report.statistics_recomputed, false, "report.statistics_recomputed");
    const workflow = requireExactKeys(data.workflow, "workflow", WORKFLOW_KEYS);
    requireString(workflow.flow_id, "workflow.flow_id");
    requireLiteral(workflow.status, "complete", "workflow.status");
    if (!["reduced", "full"].includes(workflow.calibration_budget)) {
      contractError("workflow.calibration_budget", "must be reduced or full");
    }
    const identity = requireExactKeys(workflow.identity, "workflow.identity", IDENTITY_KEYS);
    IDENTITY_KEYS.forEach((key) => requireString(identity[key], `workflow.identity.${key}`));
    requireLiteral(identity.calibration_budget, workflow.calibration_budget, "workflow.identity.calibration_budget");

    const integrity = requireExactKeys(data.integrity, "integrity", INTEGRITY_KEYS);
    const phaseOrder = requireArray(integrity.phase_order, "integrity.phase_order");
    if (JSON.stringify(phaseOrder) !== JSON.stringify(PHASE_ORDER)) {
      contractError("integrity.phase_order", `must equal ${PHASE_ORDER.join(", ")}`);
    }
    requireLiteral(integrity.all_stages_ranked, true, "integrity.all_stages_ranked");
    requireLiteral(integrity.test_data_accessed_before_freeze, false, "integrity.test_data_accessed_before_freeze");
    requireLiteral(integrity.single_spathi_commit, true, "integrity.single_spathi_commit");
    requireLiteral(integrity.selection_frozen, true, "integrity.selection_frozen");
    requireExactKeys(data.objective, "objective", [
      "primary_metric", "edge_semantics", "direction", "aggregation",
      "tie_breakers", "forbidden_shortcuts",
    ]);
    requireExactKeys(data.dataset_partition, "dataset_partition", [
      "assignment_unit", "stratify_by", "roles", "constraints",
    ]);
    validateDatasets(data.datasets);
    validateSelected(data.selected);
    requireArray(data.links, "links").forEach((link, index) => {
      const path = `links[${index}]`;
      requireExactKeys(link, path, ["label", "href"]);
      requireString(link.label, `${path}.label`);
      requireString(link.href, `${path}.href`);
    });
    const stages = requireArray(data.stages, "stages");
    if (stages.length !== PHASE_ORDER.length) {
      contractError("stages", `must contain exactly ${PHASE_ORDER.length} phases`);
    }
    stages.forEach(validateStage);

    const validation = stages[PHASE_ORDER.length - 1];
    const policy = data.selected.selected_policy;
    const evidence = data.selected.selection_evidence;
    const validationSelected = validation.candidates.filter((candidate) => candidate.selected);
    if (
      data.selected.calibration_budget !== workflow.calibration_budget
      || evidence.calibration_budget !== workflow.calibration_budget
      || policy.spathi_commit !== identity.spathi_commit
      || evidence.search_space_sha256 !== identity.search_space_sha256
      || validation.winner.candidate_id !== policy.source_candidate_id
      || validation.winner.configuration_sha256 !== policy.source_configuration_sha256
      || evidence.winner_candidate_id !== policy.source_candidate_id
      || evidence.winner_configuration_sha256 !== policy.source_configuration_sha256
      || evidence.winner_policy_sha256 !== policy.policy_sha256
      || validationSelected.length !== 1
      || validationSelected[0].configuration_id !== policy.source_candidate_id
    ) {
      contractError("selected", "must reconcile with the validation winner");
    }
    return data;
  }

  function buildCandidate(candidate) {
    return {
      id: candidate.configuration_id,
      rank: candidate.rank,
      score: candidate.score,
      stability: candidate.stability,
      time: candidate.median_wall_time_seconds,
      memory: candidate.median_peak_memory_mb,
      simplicity: candidate.simplicity,
      datasetCount: candidate.dataset_count,
      seedCount: candidate.algorithm_seed_count,
      parameters: candidate.parameters,
      selected: candidate.selected,
    };
  }

  function buildStage(stage) {
    const model = {
      id: stage.id,
      stageId: stage.stage_id,
      fidelity: stage.fidelity,
      status: "ranked",
      datasetRole: stage.dataset_role,
      primaryMetric: stage.primary_metric,
      candidates: stage.candidates.map(buildCandidate),
      observations: stage.observations,
      purpose: stage.purpose,
      strategy: stage.strategy,
      varyingParameters: stage.varying_parameters,
      selectedIds: stage.selected_ids,
      memoryAvailable: stage.memory_available,
      candidateSource: stage.candidate_source,
      datasetSelector: stage.dataset_selector,
      seedIndices: stage.algorithm_seed_indices,
      targetScope: stage.target_scope,
      physicalFits: stage.expected_physical_fits,
      logicalResults: stage.expected_logical_results,
      parameterCompetition: stage.compared_parameters.map((entry) => ({
        key: entry.name,
        label: R.humanize(entry.name),
        values: entry.values.map((value) => displayParameterValue(value === "null" ? null : value)),
      })),
    };
    const rawLabels = model.candidates.map((candidate) => candidateParameterLabel(candidate, model.parameterCompetition));
    const occurrences = new Map();
    rawLabels.forEach((label) => occurrences.set(label, (occurrences.get(label) || 0) + 1));
    model.candidates.forEach((candidate, candidateIndex) => {
      const rawLabel = rawLabels[candidateIndex];
      candidate.displayLabel = occurrences.get(rawLabel) > 1
        ? `${rawLabel} · configuration ${candidate.rank}`
        : rawLabel;
    });
    return model;
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

  function candidateParameterLabel(candidate, competition) {
    const label = competition.map((dimension) => (
      `${dimension.label}: ${displayParameterValue(candidate.parameters[dimension.key])}`
    )).join(" · ");
    return label || "Fixed parameterization";
  }

  function competitionSummary(stage) {
    if (!stage.parameterCompetition.length) return "No parameter varies in this stage";
    return stage.parameterCompetition
      .map((item) => `${item.label}: ${item.values.join(" / ")}`)
      .join(" · ");
  }

  function selectionParts(selected) {
    const policy = selected.selected_policy;
    return {
      selected,
      evidence: selected.selection_evidence,
      candidate: {
        id: policy.source_candidate_id,
        parameters: policy.group_policy.branches.multi_group.parameters,
        policyId: policy.policy_id,
      },
    };
  }

  function renderStatus() {
    R.notice(
      R.query("#status-banner"),
      "success",
      "Calibration complete",
      "The selection is frozen, and development remains separate from validation throughout the report.",
    );
  }

  function renderOverview(data, stages) {
    const selectedParts = selectionParts(data.selected);
    const evidence = selectedParts.evidence;
    const totalCandidates = stages.reduce((sum, stage) => sum + stage.candidates.length, 0);
    const totalObservations = stages.reduce((sum, stage) => sum + stage.observations.length, 0);
    R.addStatCards(R.query("#overview-cards"), [
      { label: "Completed phases", value: `${stages.length}/${PHASE_ORDER.length}`, detail: stages.map((stage) => R.humanize(stage.id)).join(" · ") },
      { label: "Evaluated candidates", value: R.formatInteger(totalCandidates), detail: "Sum of each stage's search space" },
      { label: "Observations", value: R.formatInteger(totalObservations), detail: "Recorded datasets and algorithm seeds" },
      { label: "Primary metric", value: R.humanize(evidence.primary_metric), detail: "Canonical hierarchical selection criterion" },
      { label: "Frozen score", value: R.formatNumber(evidence.hierarchical_score), detail: R.statusLabel(data.selected.status) },
    ]);
  }

  function parameterGrid(parameters) {
    const list = R.element("dl", { className: "parameter-grid" });
    const entries = Object.entries(parameters);
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
    const parts = selectionParts(data.selected);
    const candidate = parts.candidate;
    const evidence = parts.evidence;

    const hero = R.element("article", { className: "selection-hero" });
    hero.appendChild(R.badge(parts.selected.status));
    const validationStage = stages[PHASE_ORDER.length - 1];
    const selectedCandidate = validationStage.candidates.find((item) => item.id === candidate.id);
    hero.appendChild(R.element("h3", {
      text: selectedCandidate.displayLabel,
    }));
    hero.appendChild(R.element("p", {
      text: `Frozen conditional policy ${candidate.policyId}: single-group datasets use the dedicated cell-distance branch; the parameters below are the calibrated multi-group branch.`,
    }));
    const score = R.element("div", { className: "selection-score" });
    score.appendChild(R.element("strong", { text: R.formatNumber(evidence.hierarchical_score, { maximumFractionDigits: 6 }) }));
    score.appendChild(R.element("span", { text: R.humanize(evidence.primary_metric) }));
    hero.appendChild(score);

    const parameters = R.element("article", { className: "card" });
    const heading = R.element("div", { className: "card-heading" });
    const headingCopy = R.element("div");
    headingCopy.appendChild(R.element("p", { className: "card-kicker", text: "Parameters" }));
    headingCopy.appendChild(R.element("h3", { text: "Complete configuration" }));
    heading.appendChild(headingCopy);
    parameters.appendChild(heading);
    parameters.appendChild(parameterGrid(candidate.parameters));
    target.appendChild(hero);
    target.appendChild(parameters);
  }

  function renderStageIntro(stage) {
    const target = R.query("#stage-intro");
    R.clear(target);
    const copy = R.element("div", { className: "stage-intro__copy" });
    copy.appendChild(R.element("h3", { text: R.humanize(stage.id) }));
    const details = [
      `${stage.candidates.length} candidates`,
      `${stage.observations.length} observations`,
      `${R.humanize(stage.datasetRole)} partition`,
      `${stage.physicalFits} physical fits`,
      `${stage.logicalResults} logical results`,
    ].join(" · ");
    copy.appendChild(R.element("p", { text: details }));
    copy.appendChild(R.element("p", { text: stage.purpose }));
    copy.appendChild(R.element("p", { text: `Strategy: ${R.humanize(stage.strategy)}` }));
    copy.appendChild(R.element("p", {
      text: `Candidate source: ${R.humanize(stage.candidateSource)} · Algorithm seed indices: ${stage.seedIndices.join(", ")} · Target scope: ${stage.targetScope.method === "all" ? "all targets" : `${stage.targetScope.size} targets selected by ${R.humanize(stage.targetScope.method)}`}`,
    }));
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
    badges.appendChild(R.badge("neutral", R.humanize(stage.datasetRole)));
    if (stage.selectedIds.length) badges.appendChild(R.badge("selected", `${stage.selectedIds.length} advance`));
    if (!stage.memoryAvailable) badges.appendChild(R.badge("warning", "Memory not measured"));
    R.append(target, copy, badges);
  }

  function candidateLabel(candidate) {
    const label = candidate.displayLabel;
    return label.length > 52 ? `${label.slice(0, 49)}…` : label;
  }

  function renderRanking(stage) {
    const target = R.query("#ranking-chart");
    const candidates = stage.candidates.filter((candidate) => Number.isFinite(candidate.score)).slice(0, 25).reverse();
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
    const convergence = stage.stageId === "convergence"
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
      configurationId: row.configuration_id,
      datasetId: row.dataset_id,
      seed: row.algorithm_seed,
      score: row.aupr_directed,
      time: row.wall_time_seconds,
    };
  }

  function renderObservations(stage) {
    const target = R.query("#observations-chart");
    const observations = stage.observations.map(observationFields).filter((row) => row.configurationId && Number.isFinite(row.score));
    const labels = new Map(stage.candidates.map((candidate) => [candidate.id, candidate.displayLabel]));
    const ids = R.unique(observations.map((row) => row.configurationId));
    const traces = ids.map((id, index) => {
      const rows = observations.filter((row) => row.configurationId === id);
      return {
        type: "box",
        name: candidateLabel({ id, rank: index + 1, displayLabel: labels.get(id) }),
        y: rows.map((row) => row.score),
        text: rows.map((row) => row.datasetId),
        customdata: rows.map((row) => [row.seed, row.time, labels.get(id)]),
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
    const keys = stage.varyingParameters
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
    actions.appendChild(R.csvButton("Export CSV", `calibration-${stage.id}-candidates.csv`, columns, stage.candidates));
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

  function renderProvenance(data) {
    const workflow = data.workflow;
    const identity = workflow.identity;
    const selected = selectionParts(data.selected);
    R.renderKeyValues(R.query("#provenance"), "Reproducible identity", [
      { label: "Flow ID", value: workflow.flow_id },
      { label: "Evidence budget", value: workflow.calibration_budget },
      { label: "SPATHI commit", value: identity.spathi_commit },
      { label: "ANDREA commit", value: identity.andrea_commit },
      { label: "Benchmark commit", value: identity.benchmark_commit },
      { label: "Fingerprint", value: identity.fingerprint },
      { label: "Search space SHA-256", value: identity.search_space_sha256 },
      { label: "Ranking SHA-256", value: selected.evidence.validation_ranking_sha256 },
      { label: "Frozen at", value: selected.evidence.frozen_at },
    ]);
    R.renderLinks(R.query("#report-links"), "Artifacts", data.links);
  }

  R.boot((data) => {
    validateCalibrationPayload(data);
    const workflow = data.workflow;
    const stages = data.stages.map(buildStage);
    R.renderHeader(workflow, "calibration", data.integrity);
    renderStatus();
    R.renderDatasetOverview(R.query("#dataset-overview"), data.datasets);
    renderOverview(data, stages);
    renderSelection(data, stages);
    renderTabs(stages);
    renderProvenance(data);
  });
}());
