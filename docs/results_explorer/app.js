import * as d3 from "https://cdn.jsdelivr.net/npm/d3-dsv@3/+esm";

const MAX_FILE_BYTES = 15 * 1024 * 1024;
const REQUIRED_COLUMNS = [
  "label",
  "score",
  "audio_recording_id",
  "start_offset_seconds",
  "end_offset_seconds",
];

const fileInput = document.querySelector("#csv-file");
const authTokenInput = document.querySelector("#auth-token");
const testAuthTokenButton = document.querySelector("#test-auth-token");
const authStatus = document.querySelector("#auth-status");
const clearButton = document.querySelector("#clear");
const loadStatus = document.querySelector("#load-status");
const explorer = document.querySelector("#explorer");
const tabs = document.querySelector("#tabs");
const labelTitle = document.querySelector("#label-title");
const viewMin = document.querySelector("#view-min");
const viewMax = document.querySelector("#view-max");
const binCount = document.querySelector("#bin-count");
const selectedScore = document.querySelector("#selected-score");
const resetView = document.querySelector("#reset-view");
const histogram = document.querySelector("#histogram");
const histogramEmpty = document.querySelector("#histogram-empty");
const recordDetails = document.querySelector("#record-details");

let recordsByLabel = new Map();
let orderedLabels = [];
let activeLabel = "";
let viewStateByLabel = new Map();
let selectedBinIndex = null;
let selectedRow = null;
let activeAudioObjectUrl = null;
const AUTH_TOKEN_STORAGE_KEY = "ecosounds-api-token";

function setStatus(message, error = false) {
  loadStatus.textContent = message;
  loadStatus.classList.toggle("error", error);
}

function finiteNumber(value) {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function audioUrl(record) {
  const recordingId = encodeURIComponent(record.audio_recording_id);
  const startOffset = encodeURIComponent(record.start_offset_seconds);
  const endOffset = encodeURIComponent(record.end_offset_seconds);
  return (
    `https://api.ecosounds.org/audio_recordings/${recordingId}/media.wav` +
    `?start_offset=${startOffset}&end_offset=${endOffset}`
  );
}

function ecosoundsUrl(record) {
  const recordingId = encodeURIComponent(record.audio_recording_id);
  const startOffset = encodeURIComponent(record.start_offset_seconds);
  const endOffset = encodeURIComponent(record.end_offset_seconds);
  return (
    `https://www.ecosounds.org/listen/${recordingId}/` +
    `?start=${startOffset}&end=${endOffset}`
  );
}

function labelExtent(records) {
  let minimum = Infinity;
  let maximum = -Infinity;
  for (const record of records) {
    minimum = Math.min(minimum, record.score);
    maximum = Math.max(maximum, record.score);
  }
  return { minimum, maximum };
}

function getActiveState() {
  return viewStateByLabel.get(activeLabel);
}

function resetExplorer() {
  recordsByLabel = new Map();
  orderedLabels = [];
  activeLabel = "";
  viewStateByLabel = new Map();
  selectedBinIndex = null;
  selectedRow = null;
  explorer.hidden = true;
  tabs.replaceChildren();
  histogram.replaceChildren();
  selectedScore.value = "";
  recordDetails.innerHTML =
    '<p class="status">Select a histogram bin to display a recording.</p>';
  clearButton.disabled = true;
  fileInput.value = "";
  revokeActiveAudioUrl();
}

function parseRows(csvText) {
  const parsed = d3.csvParse(csvText);
  const missing = REQUIRED_COLUMNS.filter(
    (column) => !parsed.columns.includes(column),
  );
  if (missing.length > 0) {
    throw new Error(`Missing required columns: ${missing.join(", ")}`);
  }

  let invalidScoreCount = 0;
  let invalidAudioCount = 0;
  const valid = [];

  parsed.forEach((row, index) => {
    const score = finiteNumber(row.score);
    const startOffset = finiteNumber(row.start_offset_seconds);
    const endOffset = finiteNumber(row.end_offset_seconds);
    const recordingId = String(row.audio_recording_id ?? "").trim();
    const label = String(row.label ?? "").trim() || "(missing)";

    if (score === null) {
      invalidScoreCount += 1;
      return;
    }
    if (
      recordingId === "" ||
      startOffset === null ||
      endOffset === null
    ) {
      invalidAudioCount += 1;
      return;
    }

    valid.push({
      ...row,
      csvRow: index + 2,
      label,
      score,
      start_offset_seconds: startOffset,
      end_offset_seconds: endOffset,
      audio_recording_id: recordingId,
    });
  });

  return { records: valid, invalidScoreCount, invalidAudioCount };
}

function groupRecords(records) {
  const grouped = new Map();
  for (const record of records) {
    if (!grouped.has(record.label)) {
      grouped.set(record.label, []);
    }
    grouped.get(record.label).push(record);
  }
  for (const recordsForLabel of grouped.values()) {
    recordsForLabel.sort(
      (left, right) =>
        left.score - right.score || left.csvRow - right.csvRow,
    );
  }
  return grouped;
}

function renderTabs() {
  tabs.replaceChildren();
  for (const label of orderedLabels) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "tab";
    button.classList.toggle("active", label === activeLabel);
    button.setAttribute("role", "tab");
    button.setAttribute("aria-selected", String(label === activeLabel));
    button.textContent =
      `${label} (${recordsByLabel.get(label).length.toLocaleString()})`;
    button.addEventListener("click", () => activateLabel(label));
    tabs.append(button);
  }
}

function activateLabel(label) {
  activeLabel = label;
  selectedBinIndex = null;
  selectedRow = null;
  selectedScore.value = "";
  const state = viewStateByLabel.get(label);
  viewMin.value = state.minimum;
  viewMax.value = state.maximum;
  binCount.value = state.binCount;
  labelTitle.textContent = `Score distribution — ${label}`;
  renderTabs();
  renderHistogram();
  showWaitingForSelection();
}

function updateStateFromControls() {
  const state = getActiveState();
  const minimum = finiteNumber(viewMin.value);
  const maximum = finiteNumber(viewMax.value);
  const bins = Number.parseInt(binCount.value, 10);

  if (minimum === null || maximum === null) {
    throw new Error("Minimum and maximum scores must be numbers.");
  }
  if (minimum >= maximum) {
    throw new Error("Maximum shown score must be greater than minimum.");
  }
  if (!Number.isInteger(bins) || bins < 2 || bins > 200) {
    throw new Error("Histogram bins must be an integer from 2 to 200.");
  }

  state.minimum = minimum;
  state.maximum = maximum;
  state.binCount = bins;
}

function buildBins(records, minimum, maximum, binCountValue) {
  const bins = Array.from({ length: binCountValue }, (_, index) => ({
    index,
    minimum: minimum + ((maximum - minimum) * index) / binCountValue,
    maximum:
      minimum + ((maximum - minimum) * (index + 1)) / binCountValue,
    rows: [],
  }));

  for (const record of records) {
    if (record.score < minimum || record.score > maximum) {
      continue;
    }
    const normalized = (record.score - minimum) / (maximum - minimum);
    const index = Math.min(
      binCountValue - 1,
      Math.max(0, Math.floor(normalized * binCountValue)),
    );
    bins[index].rows.push(record);
  }
  return bins;
}

function svgElement(name, attributes = {}) {
  const element = document.createElementNS(
    "http://www.w3.org/2000/svg",
    name,
  );
  for (const [key, value] of Object.entries(attributes)) {
    element.setAttribute(key, String(value));
  }
  return element;
}

function formatScore(value) {
  if (!Number.isFinite(value)) {
    return "";
  }
  return Number(value).toPrecision(4);
}

function renderHistogram() {
  try {
    updateStateFromControls();
  } catch (error) {
    setStatus(error.message, true);
    return;
  }

  const state = getActiveState();
  const records = recordsByLabel.get(activeLabel);
  const bins = buildBins(
    records,
    state.minimum,
    state.maximum,
    state.binCount,
  );
  const width = 1000;
  const height = 420;
  const margin = { top: 24, right: 24, bottom: 52, left: 68 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const plotBottom = margin.top + plotHeight;
  const maximumCount = Math.max(1, ...bins.map((bin) => bin.rows.length));
  const stepX = plotWidth / state.binCount;

  histogram.replaceChildren();
  histogram.setAttribute("viewBox", `0 0 ${width} ${height}`);

  for (let tickIndex = 0; tickIndex <= 5; tickIndex += 1) {
    const fraction = tickIndex / 5;
    const y = plotBottom - fraction * plotHeight;
    const count = Math.round(maximumCount * fraction);
    histogram.append(
      svgElement("line", {
        x1: margin.left,
        y1: y,
        x2: width - margin.right,
        y2: y,
        class: "grid",
      }),
    );
    const label = svgElement("text", {
      x: margin.left - 10,
      y: y + 4,
      "text-anchor": "end",
      class: "tick",
    });
    label.textContent = count.toLocaleString();
    histogram.append(label);
  }

  for (let tickIndex = 0; tickIndex <= 5; tickIndex += 1) {
    const fraction = tickIndex / 5;
    const x = margin.left + fraction * plotWidth;
    const score =
      state.minimum + fraction * (state.maximum - state.minimum);
    const label = svgElement("text", {
      x,
      y: plotBottom + 25,
      "text-anchor": "middle",
      class: "tick",
    });
    label.textContent = formatScore(score);
    histogram.append(label);
  }

  histogram.append(
    svgElement("line", {
      x1: margin.left,
      y1: plotBottom,
      x2: width - margin.right,
      y2: plotBottom,
      class: "axis",
    }),
    svgElement("line", {
      x1: margin.left,
      y1: margin.top,
      x2: margin.left,
      y2: plotBottom,
      class: "axis",
    }),
  );

  bins.forEach((bin) => {
    if (bin.rows.length === 0) {
      return;
    }
    const x = margin.left + bin.index * stepX;
    const barHeight = (bin.rows.length / maximumCount) * plotHeight;
    const rect = svgElement("rect", {
      x: x + Math.max(0.5, stepX * 0.04),
      y: plotBottom - barHeight,
      width: Math.max(1, stepX * 0.92),
      height: barHeight,
      class: `bin${bin.index === selectedBinIndex ? " selected" : ""}`,
      tabindex: "0",
      role: "button",
      "aria-label":
        `${bin.rows.length} rows from ${formatScore(bin.minimum)} ` +
        `to ${formatScore(bin.maximum)}`,
    });
    rect.addEventListener("pointerdown", (event) => {
      selectFromBin(event, bin, barHeight, plotBottom);
    });
    rect.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectFromBin(event, bin, barHeight, plotBottom, 0.5);
      }
    });
    histogram.append(rect);
  });

  const visibleCount = bins.reduce(
    (total, bin) => total + bin.rows.length,
    0,
  );
  histogramEmpty.hidden = visibleCount > 0;
  setStatus(
    `${records.length.toLocaleString()} valid rows for "${activeLabel}"; ` +
      `${visibleCount.toLocaleString()} in the visible score range.`,
  );
}

function selectFromBin(
  event,
  bin,
  barHeight,
  plotBottom,
  forcedFraction = null,
) {
  if (bin.rows.length === 0) {
    return;
  }

  let verticalFraction = forcedFraction;
  if (verticalFraction === null) {
    const bounds = histogram.getBoundingClientRect();
    const svgY = ((event.clientY - bounds.top) / bounds.height) * 420;
    const distanceFromBottom = plotBottom - svgY;
    verticalFraction = Math.max(
      0,
      Math.min(1, distanceFromBottom / barHeight),
    );
  }

  const rowIndex = Math.min(
    bin.rows.length - 1,
    Math.floor(verticalFraction * bin.rows.length),
  );
  selectedBinIndex = bin.index;
  selectedRow = bin.rows[rowIndex];
  selectedScore.value = formatScore(selectedRow.score);
  renderHistogram();
  renderSelectedRecord(selectedRow);
}

function showWaitingForSelection() {
  recordDetails.innerHTML =
    '<p class="status">Select a histogram bin to display a recording.</p>';
}

function renderSelectedRecord(record) {
  const source = audioUrl(record);
  const listenUrl = ecosoundsUrl(record);
  recordDetails.innerHTML = `
    <dl class="detail-grid">
      <dt>Score</dt><dd>${escapeHtml(record.score)}</dd>
      <dt>Label</dt><dd>${escapeHtml(record.label)}</dd>
      <dt>Recording</dt><dd>${escapeHtml(record.audio_recording_id)}</dd>
      <dt>Time</dt>
      <dd>${escapeHtml(record.start_offset_seconds)}–${escapeHtml(
        record.end_offset_seconds,
      )} seconds</dd>
      <dt>CSV row</dt><dd>${escapeHtml(record.csvRow)}</dd>
    </dl>
    <div class="spectrogram-shell">
      <oe-axes>
        <oe-indicator>
        <oe-spectrogram
            id="results-spectrogram"
          src=""
            offset="${escapeHtml(record.start_offset_seconds)}"
            window-size="1024"
            scaling="stretch"
          ></oe-spectrogram>
        </oe-indicator>
      </oe-axes>
    </div>
    <oe-media-controls for="results-spectrogram"></oe-media-controls>
    <p id="audio-status" class="status">Loading authenticated audio…</p>
    <p><a href="${escapeHtml(
      listenUrl,
    )}" target="_blank" rel="noopener">Open EcoSounds</a></p>
    <p class="hint">Audio URL: ${escapeHtml(source)}</p>
  `;
  loadAudio(record, source);
}

async function loadAudio(record, source) {
  const requestRow = record;
  const spectrogram = recordDetails.querySelector("#results-spectrogram");
  const audioStatus = recordDetails.querySelector("#audio-status");

  try {
    const response = await fetch(source, {
      headers: authorizationHeaders(),
      credentials: "omit",
    });
    if (response.status === 401) {
      clearAuthToken("Audio request returned HTTP 401; token cleared.");
    }
    if (!response.ok) {
      const authMessage =
        response.status === 401 || response.status === 403
          ? " Check the API token."
          : "";
      throw new Error(
        `Audio request failed with HTTP ${response.status}.${authMessage}`,
      );
    }

    const audioBlob = await response.blob();
    if (selectedRow !== requestRow) {
      return;
    }

    revokeActiveAudioUrl();
    activeAudioObjectUrl = URL.createObjectURL(audioBlob);
    spectrogram.src = activeAudioObjectUrl;
    audioStatus.textContent = "Audio loaded.";
  } catch (error) {
    if (selectedRow !== requestRow) {
      return;
    }
    audioStatus.textContent = error.message;
    audioStatus.classList.add("error");
  }
}

function revokeActiveAudioUrl() {
  if (activeAudioObjectUrl !== null) {
    URL.revokeObjectURL(activeAudioObjectUrl);
    activeAudioObjectUrl = null;
  }
}

window.addEventListener("beforeunload", revokeActiveAudioUrl);

function authorizationHeaders() {
  const token = authTokenInput.value.trim();
  return token
    ? { Authorization: `Token token="${token}"` }
    : {};
}

function setAuthStatus(message, error = false) {
  authStatus.textContent = message;
  authStatus.classList.toggle("error", error);
}

function clearAuthToken(message = "API token cleared.") {
  authTokenInput.value = "";
  localStorage.removeItem(AUTH_TOKEN_STORAGE_KEY);
  setAuthStatus(message);
}

function saveAuthToken() {
  const token = authTokenInput.value.trim();
  if (token) {
    localStorage.setItem(AUTH_TOKEN_STORAGE_KEY, token);
    setAuthStatus("Token saved in local storage. Not yet tested.");
  } else {
    localStorage.removeItem(AUTH_TOKEN_STORAGE_KEY);
    setAuthStatus("No token saved.");
  }
}

async function testAuthToken() {
  const token = authTokenInput.value.trim();
  if (!token) {
    setAuthStatus("Enter a token before testing it.", true);
    return;
  }

  testAuthTokenButton.disabled = true;
  setAuthStatus("Testing token…");
  try {
    const response = await fetch("https://api.ecosounds.org/status", {
      headers: {
        Accept: "application/json",
        ...authorizationHeaders(),
      },
      credentials: "omit",
    });

    if (response.status === 401) {
      clearAuthToken("Token was rejected with HTTP 401 and has been cleared.");
      return;
    }
    if (!response.ok) {
      throw new Error(`Token test failed with HTTP ${response.status}.`);
    }

    saveAuthToken();
    setAuthStatus("Status check succeeded; token was not rejected with 401.");
  } catch (error) {
    setAuthStatus(error.message, true);
  } finally {
    testAuthTokenButton.disabled = false;
  }
}

fileInput.addEventListener("change", async () => {
  const [file] = fileInput.files;
  if (!file) {
    return;
  }
  if (file.size > MAX_FILE_BYTES) {
    setStatus(
      `This file is ${(file.size / 1024 / 1024).toFixed(
        1,
      )} MB. The explorer limit is 15 MB.`,
      true,
    );
    return;
  }

  try {
    setStatus(`Reading ${file.name}…`);
    const text = await file.text();
    const parsed = parseRows(text);
    if (parsed.records.length === 0) {
      throw new Error("The CSV has no usable rows.");
    }

    recordsByLabel = groupRecords(parsed.records);
    orderedLabels = [...recordsByLabel.keys()].sort((a, b) =>
      a.localeCompare(b),
    );
    viewStateByLabel = new Map();
    for (const label of orderedLabels) {
      const extent = labelExtent(recordsByLabel.get(label));
      viewStateByLabel.set(label, {
        minimum: extent.minimum,
        maximum:
          extent.maximum === extent.minimum
            ? extent.maximum +
              Math.max(Math.abs(extent.maximum) * 0.01, 0.01)
            : extent.maximum,
        binCount: 40,
      });
    }

    explorer.hidden = false;
    clearButton.disabled = false;
    activateLabel(orderedLabels[0]);
    const warnings = [];
    if (parsed.invalidScoreCount) {
      warnings.push(
        `${parsed.invalidScoreCount.toLocaleString()} row(s) with invalid scores`,
      );
    }
    if (parsed.invalidAudioCount) {
      warnings.push(
        `${parsed.invalidAudioCount.toLocaleString()} row(s) with incomplete audio fields`,
      );
    }
    setStatus(
      `Loaded ${parsed.records.length.toLocaleString()} rows across ` +
        `${orderedLabels.length} labels from ${file.name}.` +
        (warnings.length ? ` Skipped ${warnings.join(" and ")}.` : ""),
    );
  } catch (error) {
    resetExplorer();
    setStatus(error.message, true);
  }
});

clearButton.addEventListener("click", () => {
  resetExplorer();
  setStatus("Choose a CSV up to 15 MB.");
});

for (const control of [viewMin, viewMax, binCount]) {
  control.addEventListener("change", () => {
    selectedBinIndex = null;
    selectedRow = null;
    selectedScore.value = "";
    renderHistogram();
    showWaitingForSelection();
  });
}

resetView.addEventListener("click", () => {
  const extent = labelExtent(recordsByLabel.get(activeLabel));
  const state = getActiveState();
  state.minimum = extent.minimum;
  state.maximum =
    extent.maximum === extent.minimum
      ? extent.maximum + Math.max(Math.abs(extent.maximum) * 0.01, 0.01)
      : extent.maximum;
  viewMin.value = state.minimum;
  viewMax.value = state.maximum;
  selectedBinIndex = null;
  selectedScore.value = "";
  renderHistogram();
  showWaitingForSelection();
});

authTokenInput.addEventListener("change", () => {
  saveAuthToken();
  if (selectedRow !== null) {
    renderSelectedRecord(selectedRow);
  }
});

testAuthTokenButton.addEventListener("click", testAuthToken);

const savedAuthToken = localStorage.getItem(AUTH_TOKEN_STORAGE_KEY);
if (savedAuthToken) {
  authTokenInput.value = savedAuthToken;
  setAuthStatus("Token restored from local storage.");
}
