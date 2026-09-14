(function () {
  "use strict";

  const PALETTE = [
    "#0f8c84",
    "#2563eb",
    "#7c3aed",
    "#db6b22",
    "#d9467a",
    "#64748b",
    "#0891b2",
    "#65a30d",
  ];

  const STATUS_LABELS = {
    complete: "Complete",
    completed: "Complete",
    selected: "Selected",
    ranked: "Ranked",
    tested: "Tested",
    passed: "Valid",
    valid: "Valid",
    ok: "OK",
    success: "Successful",
    descriptive_only: "Descriptive only",
    descriptive: "Descriptive",
    exploratory: "Exploratory",
    confirmatory: "Confirmatory",
    incomplete: "Incomplete",
    pending: "Pending",
    warning: "Warning",
    failed: "Failed",
    error: "Error",
    invalid: "Invalid",
    not_testable: "Not testable",
    unavailable: "Unavailable",
    unknown: "Unknown",
  };

  function query(selector, root) {
    return (root || document).querySelector(selector);
  }

  function queryAll(selector, root) {
    return Array.from((root || document).querySelectorAll(selector));
  }

  function clear(node) {
    if (!node) return;
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function element(tagName, options) {
    const node = document.createElement(tagName);
    const opts = options || {};
    if (opts.className) node.className = opts.className;
    if (opts.text !== undefined && opts.text !== null) node.textContent = String(opts.text);
    if (opts.title) node.title = String(opts.title);
    if (opts.id) node.id = String(opts.id);
    if (opts.attrs) {
      Object.entries(opts.attrs).forEach(([name, value]) => {
        if (value !== undefined && value !== null) node.setAttribute(name, String(value));
      });
    }
    return node;
  }

  function append(parent) {
    const children = Array.prototype.slice.call(arguments, 1);
    children.flat(Infinity).forEach((child) => {
      if (child === null || child === undefined) return;
      parent.appendChild(child instanceof Node ? child : document.createTextNode(String(child)));
    });
    return parent;
  }

  function parseData() {
    const source = query("#report-data");
    if (!source) throw new Error("The embedded report data block was not found.");
    try {
      return JSON.parse(source.textContent || "{}");
    } catch (error) {
      throw new Error(`The embedded JSON is invalid: ${error.message}`);
    }
  }

  function isFiniteNumber(value) {
    return typeof value === "number" && Number.isFinite(value);
  }

  function number(value, fallback) {
    if (value === null || value === undefined || value === "") return fallback;
    if (isFiniteNumber(value)) return value;
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  function formatNumber(value, options) {
    if (!isFiniteNumber(number(value, NaN))) return "—";
    const numeric = Number(value);
    const opts = options || {};
    const absolute = Math.abs(numeric);
    if (absolute !== 0 && (absolute < 0.0001 || absolute >= 1000000)) {
      return numeric.toExponential(opts.exponentialDigits === undefined ? 2 : opts.exponentialDigits);
    }
    return numeric.toLocaleString("en-US", {
      maximumFractionDigits: opts.maximumFractionDigits === undefined ? 4 : opts.maximumFractionDigits,
      minimumFractionDigits: opts.minimumFractionDigits || 0,
    });
  }

  function formatInteger(value) {
    const numeric = number(value, NaN);
    return Number.isFinite(numeric) ? Math.round(numeric).toLocaleString("en-US") : "—";
  }

  function formatSeconds(value) {
    const seconds = number(value, NaN);
    if (!Number.isFinite(seconds)) return "—";
    if (seconds < 60) return `${formatNumber(seconds, { maximumFractionDigits: 1 })} s`;
    if (seconds < 3600) return `${formatNumber(seconds / 60, { maximumFractionDigits: 1 })} min`;
    return `${formatNumber(seconds / 3600, { maximumFractionDigits: 1 })} h`;
  }

  function formatBytesMb(value) {
    const megabytes = number(value, NaN);
    if (!Number.isFinite(megabytes)) return "—";
    if (megabytes >= 1024) return `${formatNumber(megabytes / 1024, { maximumFractionDigits: 2 })} GiB`;
    return `${formatNumber(megabytes, { maximumFractionDigits: 1 })} MiB`;
  }

  function humanize(value) {
    if (value === null || value === undefined || value === "") return "—";
    const known = {
      aupr: "AUPR",
      aupr_directed: "Directed AUPR",
      auroc: "AUROC",
      f1_at_truth_count: "F1 @ truth edge count",
      epr_at_truth_count: "EPR @ truth edge count",
      group_only: "Group only",
      order_aware: "Order aware",
      prior_assisted: "Prior assisted",
      directed: "Directed",
      topology: "Topology",
      development: "Development",
      validation: "Validation",
      test: "Test",
      exploratory: "Exploratory",
      confirmatory: "Confirmatory",
      control_wins: "Control wins",
      control_mixed_or_inconsistent_evidence: "Mixed evidence",
      no_detected_difference: "No detected difference",
      incomplete_panel: "Incomplete panel",
      partially_tested: "Partially tested",
      confirmatory_incomplete: "Confirmatory incomplete",
      dyngen: "dyngen",
      scmultisim: "scMultiSim",
      n_estimators: "Number of trees (n_estimators)",
      max_features: "Features per split (max_features)",
      min_samples_leaf: "Minimum samples per leaf",
      max_depth: "Maximum tree depth",
      tree_method: "Tree ensemble",
      distance_metric: "Distance metric",
      distance_space: "Distance space",
      distance_standardization: "Distance standardization",
      group_size_correction: "Group-size correction",
      weight_mode: "Weighting mode",
      n_components: "PCA components",
      pca_svd_solver: "PCA solver",
      sqrt: "sqrt",
    };
    const raw = String(value);
    if (Object.prototype.hasOwnProperty.call(known, raw)) return known[raw];
    return raw
      .replace(/[_-]+/g, " ")
      .replace(/\b\w/g, (letter) => letter.toUpperCase());
  }

  function shortId(value, length) {
    if (value === null || value === undefined) return "—";
    const raw = String(value);
    const max = length || 12;
    return raw.length > max ? `${raw.slice(0, max)}…` : raw;
  }

  function unique(values) {
    return Array.from(new Set(values.filter((item) => item !== null && item !== undefined)));
  }

  function statusTone(status) {
    const normalized = String(status || "unknown").toLowerCase();
    if (["complete", "completed", "selected", "ranked", "tested", "passed", "valid", "ok", "success", "confirmatory"].includes(normalized)) return "success";
    if (["exploratory", "descriptive", "descriptive_only", "pending", "warning", "incomplete", "incomplete_panel", "partially_tested", "confirmatory_incomplete", "not_testable", "unavailable"].includes(normalized)) return "warning";
    if (["failed", "error", "invalid"].includes(normalized)) return "error";
    return "neutral";
  }

  function statusLabel(status) {
    const normalized = String(status || "unknown").toLowerCase();
    return STATUS_LABELS[normalized] || humanize(normalized);
  }

  function badge(status, label) {
    return element("span", {
      className: `badge badge--${statusTone(status)}`,
      text: label || statusLabel(status),
    });
  }

  function pill(status, label) {
    return element("span", {
      className: `pill pill--${statusTone(status)}`,
      text: label || statusLabel(status),
    });
  }

  function notice(container, tone, title, message) {
    clear(container);
    const panel = element("div", { className: `notice notice--${tone || "info"}` });
    if (title) panel.appendChild(element("strong", { text: title }));
    if (message) panel.appendChild(element("span", { text: message }));
    container.appendChild(panel);
    return panel;
  }

  function empty(container, title, detail) {
    clear(container);
    const wrapper = element("div", { className: "empty-state" });
    const copy = element("div");
    copy.appendChild(element("strong", { text: title || "No data" }));
    if (detail) copy.appendChild(element("span", { text: detail }));
    wrapper.appendChild(copy);
    container.appendChild(wrapper);
  }

  function statCard(label, value, detail) {
    const card = element("article", { className: "stat-card" });
    card.appendChild(element("p", { className: "stat-card__label", text: label }));
    card.appendChild(element("p", { className: "stat-card__value", text: value }));
    if (detail) card.appendChild(element("p", { className: "stat-card__detail", text: detail }));
    return card;
  }

  function addStatCards(container, cards) {
    clear(container);
    cards.forEach((card) => container.appendChild(statCard(card.label, card.value, card.detail)));
  }

  function safeLocalHref(value) {
    if (value === null || value === undefined) return null;
    const href = String(value).trim();
    if (!href || /[\u0000-\u001f]/.test(href)) return null;
    if (/^(?:[a-z][a-z0-9+.-]*:|\/\/)/i.test(href)) return null;
    return href;
  }

  function fileLink(label, href, download) {
    const safeHref = safeLocalHref(href);
    if (!safeHref) return pill("unavailable", label || "Unavailable");
    const link = element("a", {
      className: "file-link",
      text: label || "Open",
      attrs: { href: safeHref },
    });
    if (download) link.setAttribute("download", "");
    return link;
  }

  function renderKeyValues(container, title, entries) {
    clear(container);
    if (title) container.appendChild(element("h3", { text: title }));
    const pairs = entries.filter((entry) => entry.value !== undefined && entry.value !== null && entry.value !== "");
    if (!pairs.length) {
      empty(container, "No provenance data", "The artifact does not include these metadata.");
      return;
    }
    const list = element("dl", { className: "detail-list" });
    pairs.forEach((entry) => {
      list.appendChild(element("dt", { text: entry.label }));
      list.appendChild(element("dd", { text: entry.format ? entry.format(entry.value) : entry.value }));
    });
    container.appendChild(list);
  }

  function renderLinks(container, title, links) {
    clear(container);
    if (title) container.appendChild(element("h3", { text: title }));
    const valid = (links || []).filter((item) => safeLocalHref(item.href));
    if (!valid.length) {
      empty(container, "No portable links", "No relative artifact links were recorded for this report.");
      return;
    }
    const list = element("ul", { className: "link-list" });
    valid.forEach((item) => {
      const row = element("li");
      const link = element("a", { text: item.label || item.href, attrs: { href: safeLocalHref(item.href) } });
      if (item.download) link.setAttribute("download", "");
      row.appendChild(link);
      list.appendChild(row);
    });
    container.appendChild(list);
  }

  function cellValue(column, row) {
    const raw = typeof column.value === "function" ? column.value(row) : row[column.key];
    if (column.render) return column.render(raw, row);
    if (column.format) return column.format(raw, row);
    return raw === null || raw === undefined || raw === "" ? "—" : String(raw);
  }

  function createTable(container, columns, rows, options) {
    clear(container);
    if (!rows || !rows.length) {
      empty(container, (options && options.emptyTitle) || "No rows", (options && options.emptyDetail) || "No canonical data are available for this table.");
      return;
    }
    const table = element("table", { className: "data-table" });
    const head = element("thead");
    const headRow = element("tr");
    columns.forEach((column) => headRow.appendChild(element("th", { text: column.label, attrs: { scope: "col" } })));
    head.appendChild(headRow);
    table.appendChild(head);
    const body = element("tbody");
    rows.forEach((row) => {
      const tr = element("tr");
      columns.forEach((column) => {
        const td = element("td", { className: column.numeric ? "numeric" : "" });
        const value = cellValue(column, row);
        td.appendChild(value instanceof Node ? value : document.createTextNode(value));
        tr.appendChild(td);
      });
      body.appendChild(tr);
    });
    table.appendChild(body);
    container.appendChild(table);
  }

  function csvEscape(value) {
    const text = value === null || value === undefined ? "" : String(value);
    return `"${text.replace(/"/g, '""')}"`;
  }

  function csvButton(label, filename, columns, rows) {
    const button = element("button", { className: "action-button", text: label || "Export CSV", attrs: { type: "button" } });
    button.addEventListener("click", () => {
      const header = columns.map((column) => csvEscape(column.label)).join(",");
      const records = rows.map((row) => columns.map((column) => {
        const value = typeof column.csvValue === "function"
          ? column.csvValue(row)
          : typeof column.value === "function"
            ? column.value(row)
            : row[column.key];
        return csvEscape(value);
      }).join(","));
      const blob = new Blob([[header].concat(records).join("\n")], { type: "text/csv;charset=utf-8" });
      const href = URL.createObjectURL(blob);
      const anchor = element("a", { attrs: { href, download: filename || "table.csv" } });
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(href);
    });
    return button;
  }

  function makeSelect(container, id, label, values, selected, onChange) {
    const wrapper = element("label", { attrs: { for: id } });
    wrapper.appendChild(element("span", { className: "field-label", text: label }));
    const select = element("select", { className: "field-select", id });
    values.forEach((value) => {
      const option = element("option", { text: humanize(value), attrs: { value } });
      if (String(value) === String(selected)) option.selected = true;
      select.appendChild(option);
    });
    select.addEventListener("change", onChange);
    wrapper.appendChild(select);
    container.appendChild(wrapper);
    return select;
  }

  function replaceSelectOptions(select, values, selected) {
    clear(select);
    values.forEach((value) => {
      const option = element("option", { text: humanize(value), attrs: { value } });
      if (String(value) === String(selected)) option.selected = true;
      select.appendChild(option);
    });
  }

  function baseLayout(overrides) {
    return Object.assign({
      autosize: true,
      margin: { l: 58, r: 24, t: 18, b: 58 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: "Inter, system-ui, sans-serif", color: "#526078", size: 12 },
      hoverlabel: { bgcolor: "#17243a", bordercolor: "#17243a", font: { color: "#ffffff" } },
      xaxis: { gridcolor: "#e7ebf1", zerolinecolor: "#cbd5e1", automargin: true },
      yaxis: { gridcolor: "#e7ebf1", zerolinecolor: "#cbd5e1", automargin: true },
      legend: { orientation: "h", y: -0.18 },
    }, overrides || {});
  }

  function plot(container, traces, layout, config) {
    if (!container) return Promise.resolve();
    clear(container);
    if (!window.Plotly || typeof window.Plotly.newPlot !== "function") {
      empty(container, "Visualization unavailable", "Plotly was not embedded in the report.");
      return Promise.resolve();
    }
    return window.Plotly.newPlot(container, traces, baseLayout(layout), Object.assign({
      responsive: true,
      displaylogo: false,
      scrollZoom: false,
      modeBarButtonsToRemove: ["lasso2d", "select2d"],
      toImageButtonOptions: { format: "svg", filename: "spathi-benchmark-chart" },
    }, config || {}));
  }

  function countValue(value) {
    if (Array.isArray(value)) return value.length;
    return number(value, NaN);
  }

  function rangeText(minimum, maximum) {
    const low = countValue(minimum);
    const high = countValue(maximum);
    if (!Number.isFinite(low) && !Number.isFinite(high)) return "—";
    if (!Number.isFinite(low)) return formatInteger(high);
    if (!Number.isFinite(high) || low === high) return formatInteger(low);
    return `${formatInteger(low)}–${formatInteger(high)}`;
  }

  function countBreakdown(items) {
    return (Array.isArray(items) ? items : [])
      .map((item) => {
        if (!item || typeof item !== "object") return null;
        const id = item.id === null || item.id === undefined ? null : humanize(item.id);
        if (!id) return null;
        const count = countValue(item.count);
        return Number.isFinite(count) ? `${id} (${formatInteger(count)})` : id;
      })
      .filter(Boolean)
      .join(" · ");
  }

  function renderDatasetOverview(container, datasets) {
    if (!container) return;
    clear(container);
    const source = datasets && typeof datasets === "object" ? datasets : {};
    const summary = source.summary && typeof source.summary === "object" ? source.summary : {};
    const rows = Array.isArray(source.rows) ? source.rows : [];
    if (!rows.length && !Object.keys(summary).length) {
      empty(
        container,
        "Dataset overview unavailable",
        "This report does not include the canonical dataset inventory.",
      );
      return;
    }

    const simulatorDetail = countBreakdown(summary.simulators);
    const roleDetail = countBreakdown(summary.roles);
    const cards = element("div", { className: "stat-grid dataset-overview-cards" });
    addStatCards(cards, [
      {
        label: "Datasets",
        value: formatInteger(summary.dataset_count === undefined ? rows.length : summary.dataset_count),
        detail: simulatorDetail || "Independent dataset instances",
      },
      {
        label: "Total cells",
        value: formatInteger(summary.total_cells),
        detail: `Per dataset: ${rangeText(summary.cells_min, summary.cells_max)}`,
      },
      {
        label: "Genes",
        value: rangeText(summary.genes_min, summary.genes_max),
        detail: "Observed genes per dataset",
      },
      {
        label: "Groups",
        value: rangeText(summary.groups_min, summary.groups_max),
        detail: "Cell groups per dataset",
      },
      {
        label: "Transcription factors",
        value: rangeText(
          summary.transcription_factors_min,
          summary.transcription_factors_max,
        ),
        detail: "Eligible regulators per dataset",
      },
      {
        label: "Dataset roles",
        value: Array.isArray(summary.roles) ? formatInteger(summary.roles.length) : "—",
        detail: roleDetail || "No role breakdown recorded",
      },
    ]);
    container.appendChild(cards);

    const tableCard = element("div", { className: "card table-wrap dataset-overview-table" });
    createTable(tableCard, [
      { label: "Dataset", value: (row) => row.dataset_id },
      { label: "Simulator", value: (row) => humanize(row.simulator_id) },
      { label: "Condition", value: (row) => humanize(row.condition_id) },
      { label: "Role", value: (row) => humanize(row.dataset_role) },
      { label: "Split", value: (row) => humanize(row.split_id) },
      { label: "Seed", value: (row) => row.seed, format: formatInteger, numeric: true },
      { label: "Genes", value: (row) => countValue(row.genes), format: formatInteger, numeric: true },
      { label: "Cells", value: (row) => countValue(row.cells), format: formatInteger, numeric: true },
      { label: "Columns", value: (row) => humanize(row.column_kind) },
      {
        label: "Groups",
        value: (row) => countValue(row.groups),
        render: (value, row) => element("span", {
          text: formatInteger(value),
          title: Array.isArray(row.group_labels) ? row.group_labels.join(", ") : null,
        }),
        numeric: true,
      },
      {
        label: "TFs",
        value: (row) => countValue(row.transcription_factors),
        format: formatInteger,
        numeric: true,
      },
    ], rows, {
      emptyTitle: "No dataset rows",
      emptyDetail: "The summary is available, but no per-dataset inventory was published.",
    });
    container.appendChild(tableCard);
  }

  function renderHeader(workflow, reportType, integrity) {
    const badges = query("#header-badges");
    clear(badges);
    badges.appendChild(badge((workflow && workflow.status) || "unknown"));
    if (workflow && (workflow.calibration_budget || workflow.profile || workflow.profile_id)) {
      badges.appendChild(badge(
        "neutral",
        humanize(workflow.calibration_budget || workflow.profile || workflow.profile_id),
      ));
    }
    if (integrity && integrity.status) badges.appendChild(badge(integrity.status, `Integrity: ${statusLabel(integrity.status)}`));

    const meta = query("#workflow-meta");
    clear(meta);
    const identity = (workflow && workflow.identity) || {};
    const runtimeCapacity = workflow && workflow.runtime_plan && workflow.runtime_plan.capacity;
    const inferenceSchedule = workflow && workflow.scheduling && workflow.scheduling.inference;
    const inferenceUtilization = inferenceSchedule && inferenceSchedule.utilization;
    const values = [
      [reportType === "calibration" ? "Workflow" : "Profile", workflow && (workflow.flow_id || workflow.profile_id)],
      ["CPU budget", runtimeCapacity && `${formatInteger(runtimeCapacity.cpu_threads)} threads`],
      ["RAM budget", runtimeCapacity && bytes(runtimeCapacity.memory_bytes)],
      ["Inference CPU use", inferenceUtilization && `${formatNumber(100 * inferenceUtilization.cpu_capacity_utilization, { maximumFractionDigits: 1 })}%`],
      ["SPATHI", identity.spathi_commit || (workflow && workflow.spathi_commit)],
      ["ANDREA", identity.andrea_commit || (workflow && workflow.andrea_commit)],
      ["Benchmark", identity.benchmark_commit || (workflow && workflow.benchmark_commit)],
      ["Fingerprint", identity.fingerprint || (workflow && workflow.fingerprint)],
    ];
    values.filter((item) => item[1]).forEach(([label, value]) => {
      const wrapper = element("div", { className: "workflow-meta__item", title: value });
      wrapper.appendChild(element("strong", { text: label }));
      wrapper.appendChild(document.createTextNode(shortId(value, label === "Workflow" || label === "Profile" ? 36 : 14)));
      meta.appendChild(wrapper);
    });
  }

  function renderFatal(error) {
    const target = query("#fatal-error");
    if (!target) return;
    target.classList.remove("is-hidden");
    clear(target);
    target.appendChild(element("strong", { text: "The report could not be opened" }));
    target.appendChild(element("span", { text: error && error.message ? error.message : String(error) }));
  }

  function boot(callback) {
    try {
      callback(parseData());
    } catch (error) {
      renderFatal(error);
      if (window.console) window.console.error(error);
    }
  }

  window.SpathiReport = Object.freeze({
    PALETTE,
    addStatCards,
    append,
    badge,
    baseLayout,
    boot,
    clear,
    createTable,
    csvButton,
    element,
    empty,
    fileLink,
    formatBytesMb,
    formatInteger,
    formatNumber,
    formatSeconds,
    humanize,
    isFiniteNumber,
    makeSelect,
    notice,
    number,
    pill,
    plot,
    query,
    queryAll,
    renderHeader,
    renderDatasetOverview,
    renderKeyValues,
    renderLinks,
    replaceSelectOptions,
    safeLocalHref,
    shortId,
    statCard,
    statusLabel,
    statusTone,
    unique,
  });
}());
