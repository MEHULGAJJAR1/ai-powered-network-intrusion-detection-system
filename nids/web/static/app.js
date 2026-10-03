"use strict";

const CLASS_ORDER = ["Normal", "DoS", "Probe", "R2L", "U2R"];
const CLASS_COLORS = { Normal: "#4ecf9a", DoS: "#ff657d", Probe: "#e7ae54", R2L: "#9e83f5", U2R: "#e183c4" };
const RISK_COLORS = { critical: "#ff657d", high: "#f29b5d", medium: "#e1bd62", low: "#4ecfbd" };
const LABELS = { overview: "Overview", monitor: "Live simulation", predict: "Analyze traffic", batch: "CSV batch scan", history: "Prediction history", alerts: "Security alerts", performance: "Model performance", exploration: "Data exploration", explainability: "Explainability", architecture: "System architecture" };
const state = {
  page: "overview", schema: null, metrics: null, modelInfo: null, importance: null, stats: null,
  latestPrediction: null, historyPage: 1, historyPages: 1, alertsPage: 1, alertsPages: 1,
  historySort: ["timestamp", "desc"], monitorTimer: null, monitorBusy: false, monitorCount: 0,
  monitorAttackCount: 0, batchFile: null, exploration: null
};

function $(id) { return document.getElementById(id); }
function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
}
function setText(id, value) { const node = $(id); if (node) node.textContent = value == null ? "—" : String(value); }
function numberFormat(value) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 });
}
function percent(value, digits = 1) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(digits)}%`;
}
function localTime(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit" });
}
function classCss(value) {
  return ({ Normal: "normal", DoS: "dos", Probe: "probe", R2L: "r2l", U2R: "u2r" })[value] || "low";
}
function badge(value, kind = "class") {
  const safe = escapeHtml(value || "unknown");
  return `<span class="badge badge-${kind === "risk" ? escapeHtml(String(value || "low").toLowerCase()) : classCss(value)}">${safe}</span>`;
}
function toast(message, type = "") {
  const region = $("toast-region");
  const item = document.createElement("div");
  item.className = `toast ${type}`;
  item.textContent = message;
  region.appendChild(item);
  window.setTimeout(() => item.remove(), 4200);
}
async function apiRequest(url, options = {}) {
  let response;
  try { response = await fetch(url, options); }
  catch (_error) { throw new Error("Could not reach the NIDS service. Check that the backend is running."); }
  const contentType = response.headers.get("content-type") || "";
  const data = contentType.includes("application/json") ? await response.json() : null;
  if (!response.ok) {
    const message = data?.error || `Request failed (${response.status}).`;
    const detail = Array.isArray(data?.details) ? data.details.map((x) => x.message).join("; ") : "";
    throw new Error(detail ? `${message} ${detail}` : message);
  }
  return data;
}
function jsonOptions(body) { return { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }; }
function emptyChart(container, message = "No records available") {
  if (container) container.innerHTML = `<div class="empty-state compact"><span class="empty-icon">◌</span><strong>${escapeHtml(message)}</strong></div>`;
}

function renderBars(container, entries, colorMap = {}, options = {}) {
  if (!container) return;
  const values = entries.filter((entry) => Number.isFinite(Number(entry.value)));
  if (!values.length || values.every((entry) => Number(entry.value) === 0)) {
    emptyChart(container, options.empty || "No measured data yet");
    return;
  }
  const max = Math.max(...values.map((entry) => Number(entry.value)), 0.000001);
  const top = options.limit ? values.slice(0, options.limit) : values;
  container.innerHTML = top.map((entry) => {
    const value = Number(entry.value);
    const width = Math.max(1, (value / max) * 100);
    const color = colorMap[entry.name] || options.color || "#4ddbd0";
    const display = options.percent ? `${(value * 100).toFixed(1)}%` : numberFormat(value);
    return `<div class="bar-row"><span class="bar-name" title="${escapeHtml(entry.name)}">${escapeHtml(entry.name)}</span><div class="bar-track" role="img" aria-label="${escapeHtml(entry.name)} ${escapeHtml(display)}"><div class="bar-fill" style="width:${width}%;background:${escapeHtml(color)}"></div></div><span class="bar-value">${escapeHtml(display)}</span></div>`;
  }).join("");
}
function renderDonut(chart, legend, distribution) {
  const entries = CLASS_ORDER.map((name) => ({ name, value: Number(distribution?.[name] || 0) }));
  const total = entries.reduce((sum, entry) => sum + entry.value, 0);
  if (!total) {
    emptyChart(chart, "No prediction history yet");
    if (legend) legend.innerHTML = `<div class="empty-state compact"><strong>No classifications recorded</strong><p>Run a prediction to populate this chart.</p></div>`;
    return;
  }
  const radius = 54;
  const circumference = 2 * Math.PI * radius;
  let offset = 0;
  const circles = entries.filter((entry) => entry.value > 0).map((entry) => {
    const length = (entry.value / total) * circumference;
    const markup = `<circle cx="80" cy="80" r="${radius}" fill="none" stroke="${CLASS_COLORS[entry.name]}" stroke-width="15" stroke-dasharray="${length} ${circumference - length}" stroke-dashoffset="-${offset}" transform="rotate(-90 80 80)"><title>${escapeHtml(entry.name)}: ${numberFormat(entry.value)}</title></circle>`;
    offset += length;
    return markup;
  }).join("");
  chart.innerHTML = `<svg viewBox="0 0 160 160" role="img" aria-label="Prediction class distribution"><circle cx="80" cy="80" r="54" fill="none" stroke="rgba(130,160,182,.12)" stroke-width="15"/>${circles}<text class="donut-total" x="80" y="77">${numberFormat(total)}</text><text class="donut-caption" x="80" y="94">PREDICTIONS</text></svg>`;
  legend.innerHTML = entries.map((entry) => `<div class="legend-row"><span class="legend-name"><span class="legend-dot" style="background:${CLASS_COLORS[entry.name]}"></span>${escapeHtml(entry.name)}</span><span class="legend-value">${numberFormat(entry.value)} <span class="legend-share">· ${((entry.value / total) * 100).toFixed(1)}%</span></span></div>`).join("");
}
function renderHeatmap(container, matrix, labels, normalized = false) {
  if (!container || !matrix || !labels || !matrix.length) {
    if (container) emptyChart(container, "No measured matrix available");
    return;
  }
  const flat = matrix.flat().map(Number).filter(Number.isFinite);
  const max = Math.max(...flat, 0.00001);
  const head = `<thead><tr><th></th>${labels.map((label) => `<th>${escapeHtml(label)}</th>`).join("")}</tr></thead>`;
  const body = matrix.map((row, rowIndex) => `<tr><th class="heat-label">${escapeHtml(labels[rowIndex])}</th>${row.map((value) => {
    const number = Number(value) || 0;
    const ratio = normalized ? Math.min(1, Math.abs(number)) : Math.min(1, Math.abs(number) / max);
    const level = Math.min(5, Math.floor(ratio * 5));
    const display = normalized ? number.toFixed(2) : numberFormat(number);
    return `<td class="heat-cell heat-${level}" title="${escapeHtml(labels[rowIndex])}: ${escapeHtml(display)}">${escapeHtml(display)}</td>`;
  }).join("")}</tr>`).join("");
  container.innerHTML = `<table class="heatmap" aria-label="Data matrix">${head}<tbody>${body}</tbody></table>`;
}

function setStatus(available, text) {
  const dot = $("top-status-dot");
  const sideDot = $("sidebar-model-dot");
  [dot, sideDot].forEach((node) => {
    if (!node) return;
    node.classList.remove("status-dot-muted", "status-dot-warn", "status-dot-red");
    if (!available) node.classList.add("status-dot-muted");
  });
  setText("top-status-text", text);
  setText("sidebar-model-status", available ? "Production model ready" : "Model not loaded");
  if (available && sideDot) sideDot.classList.remove("status-dot-muted");
}
async function loadHealth() {
  try {
    const result = await apiRequest("/api/health");
    const available = Boolean(result.model_available);
    state.modelInfo = { ...(state.modelInfo || {}), available, model_name: result.model_name, version: result.model_version };
    setStatus(available, available ? "Model online" : "Model setup required");
    setText("sidebar-model-name", available ? `${result.model_name} · ${result.model_version}` : "Train with KDD Cup 1999");
    const notice = $("model-notice");
    if (notice) notice.hidden = available;
    const startButton = $("monitor-toggle");
    if (startButton) startButton.disabled = !available;
    const predictButton = $("predict-submit");
    if (predictButton) predictButton.disabled = !available;
    const batchButton = $("run-batch");
    if (batchButton) batchButton.disabled = !available || !state.batchFile;
    if (!available) setText("monitor-state", "Model unavailable");
    return result;
  } catch (error) {
    setStatus(false, "Service unavailable");
    toast(error.message, "error");
    return null;
  }
}
async function loadSchema() {
  try {
    state.schema = await apiRequest("/api/schema");
    buildPredictionForm(state.schema.features || []);
  } catch (error) {
    toast(error.message, "error");
  }
}
function buildPredictionForm(features) {
  const host = $("feature-form");
  if (!host) return;
  const serviceValues = ["http", "private", "domain_u", "smtp", "ftp_data", "eco_i", "other", "telnet", "ssh", "ftp", "imap", "ecr_i", "ntp_u", "pop_3", "auth", "domain", "finger", "https", "remote_job", "time", "urp_i"];
  host.innerHTML = features.map((feature) => {
    const name = escapeHtml(feature.name);
    const description = escapeHtml(feature.description || feature.name);
    const id = `feature-${name}`;
    if (feature.type === "categorical") {
      const options = feature.name === "service" ? serviceValues : (feature.options || []);
      const listId = `suggest-${name}`;
      const list = `<datalist id="${listId}">${options.map((value) => `<option value="${escapeHtml(value)}"></option>`).join("")}</datalist>`;
      return `<div class="feature-field"><label for="${id}">${name}</label><input id="${id}" name="${name}" type="text" maxlength="64" list="${listId}" placeholder="${feature.name === "service" ? "e.g. http" : ""}" autocomplete="off">${list}<small title="${description}">${description}</small></div>`;
    }
    const rate = feature.name.endsWith("_rate");
    const binary = ["land", "logged_in", "root_shell", "is_host_login", "is_guest_login"].includes(feature.name);
    const max = rate ? "max=\"1\"" : (binary ? "max=\"1\"" : "");
    const step = binary || feature.name === "su_attempted" ? "1" : "any";
    const hint = rate ? "Rate · 0–1" : binary ? "Binary · 0 or 1" : description;
    return `<div class="feature-field"><label for="${id}">${name}</label><input id="${id}" name="${name}" type="number" min="0" ${max} step="${step}" inputmode="decimal"><small title="${escapeHtml(hint)}">${escapeHtml(hint)}</small></div>`;
  }).join("");
}
function readPredictionForm() {
  const record = {};
  for (const feature of state.schema?.features || []) {
    const input = document.querySelector(`[name="${CSS.escape(feature.name)}"]`);
    if (!input) continue;
    const raw = input.value.trim();
    if (!raw) { record[feature.name] = null; continue; }
    record[feature.name] = feature.type === "numeric" ? Number(raw) : raw;
  }
  return record;
}
function renderPrediction(result, target = "prediction-result") {
  const host = $(target);
  if (!host) return;
  const isAttack = result.prediction !== "Normal";
  const probs = Object.entries(result.probabilities || {}).sort((a, b) => b[1] - a[1]);
  const explanation = result.explanation || {};
  const topFeatures = explanation.top_features || [];
  host.innerHTML = `<div class="panel-heading"><div><h2>Assessment</h2><p>${escapeHtml(localTime(result.timestamp))}</p></div><span class="panel-tag">${escapeHtml(result.model_version || "model")}</span></div><div class="result-class ${isAttack ? "attack" : "normal"}">${escapeHtml(result.prediction)}</div><div class="result-meta"><span class="badge badge-${classCss(result.prediction)}">${isAttack ? "Attack prediction" : "Normal prediction"}</span>${badge(result.risk_level, "risk")}<span class="panel-tag">${percent(result.confidence, 2)} confidence</span></div><div class="explanation-summary">${escapeHtml(explanation.summary || "No explanation was returned.")}</div>${topFeatures.length ? `<div class="local-feature-list">${topFeatures.map((item) => `<div class="local-feature-row"><span>${escapeHtml(item.feature)}</span><span class="${item.direction === "supports" ? "direction-supports" : item.direction === "opposes" ? "direction-opposes" : ""}">${item.contribution == null ? escapeHtml(item.direction || "context") : `${item.contribution > 0 ? "+" : ""}${numberFormat(item.contribution)} · ${escapeHtml(item.direction)}`}</span></div>`).join("")}</div>` : ""}<div class="result-probs">${probs.map(([label, value]) => `<div class="result-prob-row"><span>${escapeHtml(label)}</span><div class="bar-track"><div class="bar-fill" style="width:${Math.max(0, Math.min(100, value * 100))}%;background:${CLASS_COLORS[label] || "#4ddbd0"}"></div></div><strong>${percent(value, 1)}</strong></div>`).join("")}</div><p class="panel-footnote">Risk level is a configured confidence heuristic. Model output is not a confirmed incident.</p>`;
}
async function submitPrediction(event) {
  event.preventDefault();
  if (!state.schema) return toast("Feature schema is not ready.", "error");
  const button = $("predict-submit");
  button.disabled = true;
  button.innerHTML = `<span class="spinner"></span> Analyzing`;
  try {
    const result = await apiRequest("/api/predict", jsonOptions({ record: readPredictionForm() }));
    state.latestPrediction = result;
    renderPrediction(result);
    renderLatestExplanation();
    toast(`${result.prediction} prediction · ${percent(result.confidence, 1)} model score`, result.prediction === "Normal" ? "success" : "");
    await refreshDashboard();
  } catch (error) {
    toast(error.message, "error");
  } finally {
    button.disabled = !state.modelInfo?.available;
    button.innerHTML = `Analyze record <span aria-hidden="true">→</span>`;
  }
}
async function fillSyntheticForm() {
  try {
    const response = await apiRequest("/api/demo-record");
    for (const [name, value] of Object.entries(response.record || {})) {
      const input = document.querySelector(`[name="${CSS.escape(name)}"]`);
      if (input) input.value = value == null ? "" : String(value);
    }
    toast("Synthetic demo features loaded. No ground-truth class was assigned.");
  } catch (error) { toast(error.message, "error"); }
}

async function loadModelInfo() {
  try {
    state.modelInfo = await apiRequest("/api/model-info");
    if (state.modelInfo.available) {
      setText("sidebar-model-name", `${state.modelInfo.model_name} · ${state.modelInfo.version}`);
    }
    $("monitor-toggle").disabled = !state.modelInfo.available;
    $("predict-submit").disabled = !state.modelInfo.available;
    $("run-batch").disabled = !state.modelInfo.available || !state.batchFile;
  } catch (_error) {
    state.modelInfo = { available: false };
    $("monitor-toggle").disabled = true;
    $("predict-submit").disabled = true;
    $("run-batch").disabled = true;
  }
}
async function loadMetrics() {
  try {
    state.metrics = await apiRequest("/api/metrics");
    renderPerformance();
    renderDashboardModel();
  } catch (error) {
    state.metrics = { available: false };
    renderPerformance();
    toast(error.message, "error");
  }
}
function renderDashboardModel() {
  const summary = $("model-quality-summary");
  const chart = $("model-comparison-chart");
  if (!state.metrics?.available) {
    summary.innerHTML = `<div class="empty-state compact"><span class="empty-icon">▥</span><strong>Evaluation unavailable</strong><p>Run training to generate actual test metrics.</p></div>`;
    emptyChart(chart, "No measured model scores");
    return;
  }
  const selectedName = state.metrics.selected_model;
  const selected = state.metrics.models?.[selectedName];
  const test = selected?.test;
  summary.innerHTML = `<div class="quality-line"><span>Selected by validation macro-F1</span><strong>${escapeHtml(selectedName || "—")}</strong></div><div class="quality-line"><span>Held-out test macro-F1</span><strong>${percent(test?.overall?.f1_macro, 2)}</strong></div><div class="quality-line"><span>Test weighted F1</span><strong>${percent(test?.overall?.f1_weighted, 2)}</strong></div>`;
  const entries = Object.entries(state.metrics.models || {}).map(([name, item]) => ({ name: name.replaceAll("_", " "), value: item.test?.overall?.f1_macro })).filter((x) => x.value != null);
  renderBars(chart, entries, {}, { percent: true, color: "#4ddbd0" });
}
function renderPerformance() {
  const metrics = state.metrics;
  if (!metrics?.available) {
    ["perf-selected-model", "perf-macro-f1", "perf-weighted-f1", "perf-roc-auc", "perf-test-size", "perf-model-count"].forEach((id) => setText(id, "—"));
    $("model-metrics-table").innerHTML = `<tr><td colspan="7" class="table-empty">Measured results appear after training on the real KDD dataset.</td></tr>`;
    $("per-class-table").innerHTML = `<tr><td colspan="6" class="table-empty">No measured class metrics.</td></tr>`;
    emptyChart($("confusion-matrix"), "Train and evaluate a model first");
    setText("perf-selection-note", "No trained model");
    return;
  }
  const selectedName = metrics.selected_model;
  const selectedEntry = metrics.models?.[selectedName];
  const selectedTest = selectedEntry?.test;
  const overall = selectedTest?.overall || {};
  setText("perf-selected-model", selectedName || "—");
  setText("perf-macro-f1", percent(overall.f1_macro, 2));
  setText("perf-weighted-f1", percent(overall.f1_weighted, 2));
  setText("perf-roc-auc", percent(overall.roc_auc_ovr_macro, 2));
  setText("perf-test-size", numberFormat(overall.samples));
  setText("perf-selection-note", `Selected on validation macro-F1 ${percent(metrics.selected_validation_macro_f1, 2)}`);
  const modelEntries = Object.entries(metrics.models || {});
  setText("perf-model-count", `${modelEntries.length} measured candidates`);
  $("model-metrics-table").innerHTML = modelEntries.map(([name, entry]) => {
    const score = entry.test?.overall || {};
    return `<tr><td><strong>${escapeHtml(name.replaceAll("_", " "))}</strong>${name === selectedName ? ` <span class="badge badge-low">selected</span>` : ""}</td><td>${percent(score.accuracy, 2)}</td><td>${percent(score.precision_macro, 2)}</td><td>${percent(score.recall_macro, 2)}</td><td>${percent(score.f1_macro, 2)}</td><td>${percent(score.f1_weighted, 2)}</td><td>${percent(score.roc_auc_ovr_macro, 2)}</td></tr>`;
  }).join("") || `<tr><td colspan="7" class="table-empty">No model metrics.</td></tr>`;
  const classes = selectedTest?.per_class || {};
  $("per-class-table").innerHTML = CLASS_ORDER.map((label) => {
    const item = classes[label] || {};
    return `<tr><td>${badge(label)}</td><td>${numberFormat(item.support)}</td><td>${percent(item.precision, 2)}</td><td>${percent(item.recall, 2)}</td><td>${percent(item.f1, 2)}</td><td>${percent(item.false_positive_rate, 3)}</td></tr>`;
  }).join("");
  const matrix = selectedTest?.confusion_matrix;
  renderHeatmap($("confusion-matrix"), matrix?.values, matrix?.labels, false);
  const split = metrics.split || {};
  setText("split-note", `Train for search: ${numberFormat(split.train_rows_for_search)} · Validation: ${numberFormat(split.validation_rows)} · Final development refit: ${numberFormat(split.development_rows_for_final_refit)} · Held-out test: ${numberFormat(split.test_rows)} · Feature-group leakage protection: ${split.group_disjoint_by_feature_fingerprint ? "enabled" : "not recorded"} · Seed: ${split.seed ?? "—"}.`);
}

async function refreshDashboard() {
  try {
    const [stats, history] = await Promise.all([
      apiRequest("/api/stats"), apiRequest("/api/history?page=1&per_page=6&sort_by=timestamp&sort_order=desc")
    ]);
    state.stats = stats;
    setText("kpi-total", numberFormat(stats.total_traffic_analyzed));
    setText("kpi-normal", numberFormat(stats.normal_traffic));
    setText("kpi-attacks", numberFormat(stats.detected_attacks));
    setText("kpi-attack-rate", `${numberFormat(stats.attack_percentage)}%`);
    setText("kpi-high-risk", numberFormat(stats.high_risk_events));
    setText("distribution-count", `${numberFormat(stats.total_traffic_analyzed)} records`);
    setText("nav-alert-count", numberFormat(stats.detected_attacks));
    setText("retention-note", `Counts reflect ${numberFormat(stats.total_traffic_analyzed)} retained inference events (retention cap: ${numberFormat(stats.history_retention_limit)}). Synthetic demo predictions are included and marked in the source field.`);
    renderDonut($("attack-distribution-chart"), $("attack-distribution-legend"), stats.attack_distribution);
    renderBars($("risk-distribution-chart"), Object.entries(stats.risk_distribution || {}).map(([name, value]) => ({ name, value })), RISK_COLORS);
    renderBars($("confidence-distribution-chart"), Object.entries(stats.confidence_distribution_recent_10000 || {}).map(([name, value]) => ({ name, value })), { "0–20%": "#73849a", "20–40%": "#739baa", "40–60%": "#91b9ae", "60–80%": "#58c8b9", "80–100%": "#e1b356" });
    renderRecent(history.items || []);
    setText("dashboard-updated", `Updated ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}`);
  } catch (error) { toast(error.message, "error"); }
}
function renderRecent(items) {
  const body = $("recent-table");
  if (!items.length) {
    body.innerHTML = `<tr><td colspan="4" class="table-empty">No predictions recorded yet.</td></tr>`;
    return;
  }
  body.innerHTML = items.slice(0, 6).map((item) => `<tr><td>${badge(item.prediction)}${item.source === "synthetic_demo" ? ` <span class="feed-source">SYNTHETIC</span>` : ""}</td><td><span class="confidence-cell">${percent(item.confidence, 1)}<span class="confidence-meter"><span style="width:${Math.max(0, Math.min(100, item.confidence * 100))}%"></span></span></span></td><td>${badge(item.risk_level, "risk")}</td><td>${escapeHtml(localTime(item.timestamp))}</td></tr>`).join("");
}

function renderHistoryRows(target, items, isAlerts = false) {
  if (!items.length) {
    target.innerHTML = `<tr><td colspan="6" class="table-empty">${isAlerts ? "No matching alerts recorded." : "No matching predictions recorded."}</td></tr>`;
    return;
  }
  target.innerHTML = items.map((item) => `<tr><td>${escapeHtml(localTime(item.timestamp))}</td><td>${badge(item.prediction)}</td><td><span class="confidence-cell">${percent(item.confidence, 1)}<span class="confidence-meter"><span style="width:${Math.max(0, Math.min(100, Number(item.confidence) * 100))}%"></span></span></span></td><td>${badge(item.risk_level, "risk")}</td><td>${item.source === "synthetic_demo" ? `<span class="feed-source">SYNTHETIC</span>` : escapeHtml(item.source || "api")}</td><td title="${escapeHtml(item.model_version || "")}">${escapeHtml(item.model_version || "—")}</td></tr>`).join("");
}
function dateParam(value) {
  if (!value) return "";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "" : date.toISOString();
}
function buildHistoryParams(kind, page) {
  const params = new URLSearchParams({ page: String(page), per_page: "25" });
  if (kind === "history") {
    const search = $("history-search").value.trim();
    const type = $("history-type").value;
    const severity = $("history-severity").value;
    const confidence = $("history-confidence").value;
    const since = dateParam($("history-since").value);
    const until = dateParam($("history-until").value);
    const [sort, order] = state.historySort;
    if (search) params.set("search", search);
    if (type) params.set("attack_type", type);
    if (severity) params.set("severity", severity);
    if (confidence !== "") params.set("min_confidence", String(Number(confidence) / 100));
    if (since) params.set("since", since);
    if (until) params.set("until", until);
    params.set("sort_by", sort); params.set("sort_order", order);
  } else {
    const type = $("alert-type").value;
    const severity = $("alert-severity").value;
    const confidence = $("alert-confidence").value;
    const since = dateParam($("alert-since").value);
    const until = dateParam($("alert-until").value);
    if (type) params.set("attack_type", type);
    if (severity) params.set("severity", severity);
    if (confidence !== "") params.set("min_confidence", String(Number(confidence) / 100));
    if (since) params.set("since", since);
    if (until) params.set("until", until);
  }
  return params;
}
async function loadHistory(kind = "history", page = 1) {
  const isAlerts = kind === "alerts";
  const endpoint = isAlerts ? "/api/alerts" : "/api/history";
  const params = buildHistoryParams(kind, page);
  try {
    const data = await apiRequest(`${endpoint}?${params.toString()}`);
    const items = data.items || [];
    if (isAlerts) {
      state.alertsPage = data.page; state.alertsPages = data.pages || 1;
      setText("alerts-total", `${numberFormat(data.total)} alerts`);
      setText("alerts-page-info", `Page ${data.page} of ${Math.max(1, data.pages || 1)}`);
      $("alerts-prev").disabled = data.page <= 1;
      $("alerts-next").disabled = data.page >= data.pages;
      renderHistoryRows($("alerts-table"), items, true);
      const counts = { critical: 0, high: 0, medium: 0, low: 0 };
      items.forEach((item) => { counts[item.risk_level] = (counts[item.risk_level] || 0) + 1; });
      $("alert-counts").innerHTML = Object.entries(counts).filter(([, value]) => value > 0).map(([key, value]) => `${badge(key, "risk")} <span>${value}</span>`).join(" ");
    } else {
      state.historyPage = data.page; state.historyPages = data.pages || 1;
      setText("history-total", `${numberFormat(data.total)} records`);
      setText("history-page-info", `Page ${data.page} of ${Math.max(1, data.pages || 1)}`);
      $("history-prev").disabled = data.page <= 1;
      $("history-next").disabled = data.page >= data.pages;
      renderHistoryRows($("history-table"), items, false);
    }
  } catch (error) { toast(error.message, "error"); }
}
async function downloadFile(url, filename) {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.error || `Download failed (${response.status}).`);
    }
    const blob = await response.blob();
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob); link.download = filename;
    document.body.appendChild(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  } catch (error) { toast(error.message, "error"); }
}

async function loadFeatureImportance() {
  try {
    state.importance = await apiRequest("/api/feature-importance?limit=20");
    renderImportance();
  } catch (_error) {
    state.importance = { available: false };
    renderImportance();
  }
}
function renderImportance() {
  const data = state.importance;
  const chart = $("global-importance-chart");
  if (!data?.available) {
    setText("importance-method", "Waiting for model");
    emptyChart(chart, data?.message || "Model importance unavailable");
    return;
  }
  setText("importance-method", data.method || "Model importance");
  renderBars(chart, (data.features || []).map((item) => ({ name: item.feature, value: item.importance })), {}, { color: "#63d5cd" });
  setText("importance-note", data.note || "Global importance is model-specific and not causal.");
}
function renderLatestExplanation() {
  const host = $("latest-explanation");
  const result = state.latestPrediction;
  if (!result) {
    host.innerHTML = `<div class="empty-state compact"><span class="empty-icon">✳</span><strong>No recent prediction</strong><p>Run a single-record prediction to inspect local detail.</p></div>`;
    setText("explanation-method", "No result"); setText("explanation-context", "Run a single-record prediction to inspect local detail");
    return;
  }
  const explanation = result.explanation || {};
  setText("explanation-method", explanation.method || "Explanation");
  setText("explanation-context", `${result.prediction} · ${percent(result.confidence, 1)} · ${localTime(result.timestamp)}`);
  host.innerHTML = `<div class="explanation-summary">${escapeHtml(explanation.summary || "No explanation provided.")}</div><div class="local-feature-list">${(explanation.top_features || []).map((item) => `<div class="local-feature-row"><span>${escapeHtml(item.feature)}</span><span class="${item.direction === "supports" ? "direction-supports" : item.direction === "opposes" ? "direction-opposes" : ""}">${item.contribution == null ? escapeHtml(item.direction || "context") : `${item.contribution > 0 ? "+" : ""}${numberFormat(item.contribution)} · ${escapeHtml(item.direction)}`}</span></div>`).join("") || `<span class="toolbar-muted">No feature-level detail returned.</span>`}</div><p class="panel-footnote">${escapeHtml(explanation.caveat || "Explanations are not causal.")}</p>`;
}

async function loadExploration(force = false) {
  if (state.exploration && !force) { renderExploration(state.exploration); return; }
  try {
    const data = await apiRequest("/api/data-summary");
    state.exploration = data;
    renderExploration(data);
  } catch (error) {
    toast(error.message, "error");
    renderExploration({ available: false, message: error.message });
  }
}
function renderExploration(data) {
  if (!data?.available) {
    ["data-rows", "data-features", "data-duplicates", "data-sample-size"].forEach((id) => setText(id, "—"));
    $("numeric-stats-table").innerHTML = `<tr><td colspan="9" class="table-empty">${escapeHtml(data?.message || "KDD dataset is not present.")}</td></tr>`;
    $("categorical-distributions").innerHTML = `<div class="empty-state compact"><strong>KDD dataset not available</strong><p>${escapeHtml(data?.message || "Place the authentic dataset in data/raw and refresh.")}</p></div>`;
    emptyChart($("data-class-chart"), "Authentic dataset not loaded");
    emptyChart($("correlation-matrix"), "Dataset correlations unavailable");
    setText("numeric-stat-note", "No source data");
    return;
  }
  setText("data-rows", numberFormat(data.dimensions?.rows));
  setText("data-features", numberFormat(data.dimensions?.features));
  setText("data-duplicates", numberFormat(data.duplicate_rows_detected));
  setText("data-sample-size", numberFormat(data.sampled_rows_for_statistics));
  setText("numeric-stat-note", `${numberFormat(data.sampled_rows_for_statistics)} rows summarized`);
  renderBars($("data-class-chart"), Object.entries(data.class_distribution || {}).map(([name, value]) => ({ name, value })), CLASS_COLORS);
  const categorical = data.categorical_distributions || {};
  $("categorical-distributions").innerHTML = Object.entries(categorical).map(([name, values]) => `<div class="category-card"><h3>${escapeHtml(name)}</h3>${values.map((item) => `<div class="category-item"><span title="${escapeHtml(item.value)}">${escapeHtml(item.value)}</span><strong>${numberFormat(item.count)}</strong></div>`).join("")}</div>`).join("");
  const stats = data.numeric_statistics || {};
  $("numeric-stats-table").innerHTML = Object.entries(stats).map(([name, item]) => `<tr><td>${escapeHtml(name)}</td><td>${numberFormat(item.count)}</td><td>${numberFormat(item.mean)}</td><td>${numberFormat(item.std)}</td><td>${numberFormat(item.min)}</td><td>${numberFormat(item["25%"])} </td><td>${numberFormat(item["50%"])} </td><td>${numberFormat(item["75%"])} </td><td>${numberFormat(item.max)}</td></tr>`).join("") || `<tr><td colspan="9" class="table-empty">No numeric statistics.</td></tr>`;
  renderHeatmap($("correlation-matrix"), data.correlation?.values, data.correlation?.features, true);
}

function renderMonitorEvent(result) {
  const feed = $("monitor-feed");
  const empty = feed.querySelector(".empty-state");
  if (empty) empty.remove();
  const node = document.createElement("div");
  node.className = "feed-event";
  node.innerHTML = `<span class="feed-time">${escapeHtml(localTime(result.timestamp))}</span><span class="feed-title">${badge(result.prediction)} · ${escapeHtml(result.prediction === "Normal" ? "Traffic classified normal" : "Attack prediction")}</span><span class="feed-confidence">${percent(result.confidence, 1)} confidence</span><span>${badge(result.risk_level, "risk")}</span><span class="feed-source">SYNTHETIC</span>`;
  feed.prepend(node);
  while (feed.children.length > 40) feed.lastElementChild.remove();
}
async function simulationTick() {
  if (state.monitorBusy || !state.monitorTimer) return;
  state.monitorBusy = true;
  try {
    const result = await apiRequest("/api/demo/predict", { method: "POST" });
    state.monitorCount += 1;
    if (result.prediction !== "Normal") state.monitorAttackCount += 1;
    setText("monitor-events", numberFormat(state.monitorCount));
    setText("monitor-attacks", numberFormat(state.monitorAttackCount));
    setText("monitor-last", new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }));
    renderMonitorEvent(result);
    state.latestPrediction = result;
    renderLatestExplanation();
    await refreshDashboard();
  } catch (error) {
    toast(error.message, "error");
    stopSimulation();
  } finally { state.monitorBusy = false; }
}
function startSimulation() {
  if (state.monitorTimer || !state.modelInfo?.available) return;
  state.monitorTimer = true;
  $("monitor-toggle").innerHTML = "■ Stop simulation";
  $("monitor-toggle").classList.remove("button-primary");
  $("monitor-toggle").classList.add("button-secondary");
  $("monitor-dot").classList.remove("status-dot-muted");
  $("monitor-state").textContent = "Running · synthetic";
  $("monitor-dot").style.background = "var(--green)";
  simulationTick();
  state.monitorTimer = window.setInterval(simulationTick, 2200);
}
function stopSimulation() {
  if (typeof state.monitorTimer === "number") window.clearInterval(state.monitorTimer);
  state.monitorTimer = null;
  const button = $("monitor-toggle");
  if (button) {
    button.innerHTML = "▶ Start simulation";
    button.classList.add("button-primary");
    button.classList.remove("button-secondary");
  }
  const dot = $("monitor-dot");
  if (dot) { dot.classList.add("status-dot-muted"); dot.style.background = ""; }
  setText("monitor-state", state.modelInfo?.available ? "Stopped" : "Model unavailable");
}

function acceptBatchFile(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".csv")) {
    state.batchFile = null; $("run-batch").disabled = true; setText("selected-file", "Only .csv files are accepted.");
    return toast("Choose a file with a .csv extension.", "error");
  }
  state.batchFile = file;
  setText("selected-file", `${file.name} · ${(file.size / 1024).toFixed(1)} KB`);
  $("run-batch").disabled = !state.modelInfo?.available;
}
async function runBatch() {
  if (!state.batchFile) return;
  const button = $("run-batch"); button.disabled = true; button.innerHTML = `<span class="spinner"></span> Predicting`;
  try {
    const form = new FormData(); form.append("file", state.batchFile);
    const response = await fetch("/api/upload-csv", { method: "POST", body: form });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.error || `Batch request failed (${response.status}).`);
    }
    const summaryText = response.headers.get("X-NIDS-Summary");
    const summary = summaryText ? JSON.parse(summaryText) : {};
    const blob = await response.blob();
    const link = document.createElement("a"); link.href = URL.createObjectURL(blob); link.download = `nids_predictions_${new Date().toISOString().replace(/[:.]/g, "-")}.csv`;
    document.body.appendChild(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(link.href), 1000);
    setText("batch-result-count", `${numberFormat(summary.count)} predictions · download started`);
    $("batch-summary").innerHTML = `<div class="quality-line"><span>Rows classified</span><strong>${numberFormat(summary.count)}</strong></div><div class="quality-line"><span>Non-Normal predictions</span><strong>${numberFormat(summary.attack_count)}</strong></div><div class="panel-footnote">The downloaded CSV contains model probabilities, a policy risk level, and global feature context (not per-row SHAP). Results are from the trained model.</div>`;
    renderBars($("batch-distribution"), Object.entries(summary.distribution || {}).map(([name, value]) => ({ name, value })), CLASS_COLORS);
    toast("Batch classified. Prediction CSV download started.", "success");
    await refreshDashboard();
  } catch (error) { toast(error.message, "error"); }
  finally { button.disabled = !state.batchFile || !state.modelInfo?.available; button.innerHTML = `Validate &amp; predict <span aria-hidden="true">→</span>`; }
}

function navigate(page) {
  if (!LABELS[page]) return;
  state.page = page;
  document.querySelectorAll(".page").forEach((section) => {
    const active = section.id === `page-${page}`;
    section.classList.toggle("active", active);
    section.setAttribute("aria-hidden", String(!active));
  });
  document.querySelectorAll(".nav-link").forEach((button) => button.classList.toggle("active", button.dataset.page === page));
  setText("breadcrumb-current", LABELS[page]);
  closeMobileMenu();
  if (page === "history") loadHistory("history", state.historyPage || 1);
  if (page === "alerts") loadHistory("alerts", state.alertsPage || 1);
  if (page === "performance") loadMetrics();
  if (page === "exploration") loadExploration();
  if (page === "explainability") { loadFeatureImportance(); renderLatestExplanation(); }
  if (page === "overview") refreshDashboard();
}
function closeMobileMenu() {
  $("sidebar").classList.remove("open"); $("sidebar-scrim").classList.remove("active");
}
function openMobileMenu() {
  $("sidebar").classList.add("open"); $("sidebar-scrim").classList.add("active");
}
function setupTheme() {
  let light = false;
  try { light = localStorage.getItem("nids-theme") === "light"; } catch (_error) { /* storage may be blocked */ }
  document.body.classList.toggle("theme-light", light);
  $("theme-toggle").addEventListener("click", () => {
    const next = !document.body.classList.contains("theme-light");
    document.body.classList.toggle("theme-light", next);
    try { localStorage.setItem("nids-theme", next ? "light" : "dark"); } catch (_error) { /* no-op */ }
  });
}
function setupHandlers() {
  document.querySelectorAll(".nav-link").forEach((button) => button.addEventListener("click", () => navigate(button.dataset.page)));
  document.querySelectorAll("[data-go]").forEach((button) => button.addEventListener("click", () => navigate(button.dataset.go)));
  $("menu-toggle").addEventListener("click", openMobileMenu);
  $("sidebar-close").addEventListener("click", closeMobileMenu);
  $("sidebar-scrim").addEventListener("click", closeMobileMenu);
  $("predict-form").addEventListener("submit", submitPrediction);
  $("predict-form").addEventListener("reset", () => setTimeout(() => {
    $("prediction-result").innerHTML = `<div class="panel-heading"><div><h2>Assessment</h2><p>Prediction and model explanation</p></div></div><div class="empty-state result-empty"><span class="empty-icon">✳</span><strong>Ready for analysis</strong><p>Submit a record to see its predicted class, probabilities, risk policy, and available explanation.</p></div>`;
  }, 0));
  $("fill-synthetic").addEventListener("click", fillSyntheticForm);
  $("refresh-dashboard").addEventListener("click", async () => { await Promise.all([loadHealth(), loadModelInfo(), loadMetrics(), refreshDashboard()]); });
  $("refresh-exploration").addEventListener("click", () => loadExploration(true));
  $("history-apply").addEventListener("click", () => { state.historyPage = 1; loadHistory("history", 1); });
  $("alerts-apply").addEventListener("click", () => { state.alertsPage = 1; loadHistory("alerts", 1); });
  $("history-reset").addEventListener("click", () => { ["history-search", "history-type", "history-severity", "history-confidence", "history-since", "history-until"].forEach((id) => $(id).value = ""); state.historyPage = 1; loadHistory("history", 1); });
  $("alerts-reset").addEventListener("click", () => { ["alert-type", "alert-severity", "alert-confidence", "alert-since", "alert-until"].forEach((id) => $(id).value = ""); state.alertsPage = 1; loadHistory("alerts", 1); });
  $("history-prev").addEventListener("click", () => loadHistory("history", Math.max(1, state.historyPage - 1)));
  $("history-next").addEventListener("click", () => loadHistory("history", Math.min(state.historyPages, state.historyPage + 1)));
  $("alerts-prev").addEventListener("click", () => loadHistory("alerts", Math.max(1, state.alertsPage - 1)));
  $("alerts-next").addEventListener("click", () => loadHistory("alerts", Math.min(state.alertsPages, state.alertsPage + 1)));
  $("history-sort").addEventListener("change", (event) => { state.historySort = event.target.value.split(":"); loadHistory("history", 1); });
  $("history-export").addEventListener("click", () => downloadFile("/api/history/export.csv", "nids_prediction_history.csv"));
  $("alerts-export").addEventListener("click", () => downloadFile("/api/history/export.csv?alerts_only=true", "nids_alert_history.csv"));
  $("monitor-toggle").addEventListener("click", () => state.monitorTimer ? stopSimulation() : startSimulation());
  $("clear-monitor").addEventListener("click", () => { $("monitor-feed").innerHTML = `<div class="empty-state"><span class="empty-icon">⌁</span><strong>Feed cleared</strong><p>New synthetic events will appear here while simulation runs.</p></div>`; });
  $("choose-csv").addEventListener("click", () => $("csv-file").click());
  $("csv-file").addEventListener("change", (event) => acceptBatchFile(event.target.files[0]));
  $("run-batch").addEventListener("click", runBatch);
  const drop = $("upload-drop");
  ["dragenter", "dragover"].forEach((type) => drop.addEventListener(type, (event) => { event.preventDefault(); drop.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach((type) => drop.addEventListener(type, (event) => { event.preventDefault(); drop.classList.remove("dragover"); }));
  drop.addEventListener("drop", (event) => acceptBatchFile(event.dataTransfer.files[0]));
  document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeMobileMenu(); });
}

async function initialize() {
  setupTheme(); setupHandlers();
  await Promise.all([loadSchema(), loadHealth(), loadModelInfo(), loadMetrics(), refreshDashboard(), loadFeatureImportance()]);
  if (state.page === "overview") setText("dashboard-updated", "Live data refreshed");
}
document.addEventListener("DOMContentLoaded", initialize);
