/* Bone-R landing — interactive multi-model overlay + benchmark table.
 * Reads docs/data/predictions.json and docs/data/benchmark.json.
 * Falls back to embedded synthetic demo data so the page renders standalone. */

const TRAITS = {
  yolov8: "fast anchor-free",
  fasterrcnn: "two-stage precision",
  retinanet: "focal-loss recall",
  fcos: "anchor-free 1-stage",
};

// ---- Synthetic fallback so the page works before real data is generated ----
const DEMO_PREDICTIONS = {
  image: "synthetic-demo",
  width: 640, height: 640,
  models: {
    yolov8:     { color: "#00c800", boxes: [
      { xyxy: [250, 300, 360, 372], conf: 0.88, type: "transverse/linear (best guess)", severity: "moderate (best guess)" },
      { xyxy: [398, 150, 452, 210], conf: 0.41, type: "localized/simple (best guess)", severity: "low (best guess)" } ] },
    retinanet:  { color: "#0080ff", boxes: [
      { xyxy: [244, 296, 368, 380], conf: 0.79, type: "transverse/linear (best guess)", severity: "moderate (best guess)" },
      { xyxy: [402, 156, 458, 214], conf: 0.55, type: "localized/simple (best guess)", severity: "low (best guess)" } ] },
    fasterrcnn: { color: "#ff0000", boxes: [
      { xyxy: [255, 305, 355, 368], conf: 0.92, type: "transverse/linear (best guess)", severity: "high (best guess)" } ] },
  },
};
const DEMO_BENCH = {
  yolov8:     { map50: 0.62, map5095: 0.31, precision: 0.71, recall: 0.66 },
  retinanet:  { map50: 0.58, map5095: 0.29, precision: 0.64, recall: 0.72 },
  fasterrcnn: { map50: 0.60, map5095: 0.33, precision: 0.78, recall: 0.59 },
};

let PRED = DEMO_PREDICTIONS;
let isDemo = true;
const enabled = {};
let conf = 0.25;

const canvas = document.getElementById("viewer");
const ctx = canvas.getContext("2d");

async function loadJSON(path) {
  try {
    const r = await fetch(path, { cache: "no-store" });
    if (!r.ok) throw new Error(r.status);
    return await r.json();
  } catch (e) { return null; }
}

let bgImage = null;  // real radiograph drawn behind the boxes when available

function drawBackdrop() {
  const { width: W, height: H } = canvas;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#05070a";
  ctx.fillRect(0, 0, W, H);
  // Real radiograph (FracAtlas, CC BY 4.0) drawn full-canvas; box coords are in
  // this image's pixel space, so canvas is sized to the image (see init()).
  if (bgImage && bgImage.complete && bgImage.naturalWidth) {
    ctx.drawImage(bgImage, 0, 0, W, H);
    return;
  }
  // Synthetic "bone" silhouette so the demo reads as an X-ray, not a real patient.
  if (isDemo) {
    const g = ctx.createLinearGradient(0, 0, 0, H);
    g.addColorStop(0, "#23303f"); g.addColorStop(1, "#0c1118");
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.moveTo(290, 90); ctx.quadraticCurveTo(250, 100, 270, 180);
    ctx.lineTo(300, 320); ctx.quadraticCurveTo(280, 400, 320, 520);
    ctx.lineTo(360, 520); ctx.quadraticCurveTo(345, 400, 350, 320);
    ctx.lineTo(380, 180); ctx.quadraticCurveTo(400, 100, 350, 90);
    ctx.closePath(); ctx.fill();
    ctx.fillStyle = "rgba(255,255,255,0.04)";
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = "#3a4654";
    ctx.font = "12px monospace";
    ctx.fillText("SYNTHETIC DEMO — not a real radiograph", 14, H - 14);
  }
}

function hexToRgba(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}

let lastBoxes = [];
function render() {
  drawBackdrop();
  lastBoxes = [];
  for (const [name, m] of Object.entries(PRED.models)) {
    if (!enabled[name]) continue;
    for (const b of m.boxes) {
      if (b.conf < conf) continue;
      const [x1, y1, x2, y2] = b.xyxy;
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = m.color;
      ctx.fillStyle = hexToRgba(m.color, 0.10 + 0.25 * b.conf);
      ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.fillStyle = m.color;
      ctx.font = "12px monospace";
      ctx.fillText(`${name} ${b.conf.toFixed(2)}`, x1, Math.max(12, y1 - 5));
      lastBoxes.push({ name, ...b });
    }
  }
}

function buildToggles() {
  const wrap = document.getElementById("modelToggles");
  wrap.innerHTML = "";
  for (const [name, m] of Object.entries(PRED.models)) {
    enabled[name] = true;
    const visible = m.boxes.filter(b => b.conf >= conf).length;
    const label = document.createElement("label");
    label.className = "toggle";
    label.innerHTML = `
      <input type="checkbox" checked data-model="${name}" />
      <span class="swatch" style="background:${m.color}"></span>
      <span>${name}</span>
      <span class="count" id="cnt-${name}">${visible}</span>`;
    wrap.appendChild(label);
  }
  wrap.querySelectorAll("input").forEach(cb =>
    cb.addEventListener("change", e => {
      enabled[e.target.dataset.model] = e.target.checked;
      render();
    }));
}

function refreshCounts() {
  for (const [name, m] of Object.entries(PRED.models)) {
    const el = document.getElementById(`cnt-${name}`);
    if (el) el.textContent = m.boxes.filter(b => b.conf >= conf).length;
  }
}

function buildBench(bench) {
  const tbody = document.querySelector("#benchTable tbody");
  tbody.innerHTML = "";
  const rows = Object.entries(bench).filter(([, r]) => !r.error);
  const bestMap = Math.max(...rows.map(([, r]) => r.map50 || 0));
  for (const [name, r] of rows) {
    const color = (PRED.models[name] && PRED.models[name].color) || "#888";
    const cls = r.map50 === bestMap ? "best" : "";
    const fmt = v => (v == null ? "—" : v.toFixed ? v.toFixed(3) : v);
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><span class="dot" style="background:${color}"></span>${name}</td>
      <td class="${cls}">${fmt(r.map50)}</td>
      <td>${fmt(r.map5095)}</td>
      <td>${fmt(r.precision)}</td>
      <td>${fmt(r.recall)}</td>
      <td>${TRAITS[name] || ""}</td>`;
    tbody.appendChild(tr);
  }
}

function wireControls() {
  const slider = document.getElementById("confSlider");
  slider.addEventListener("input", e => {
    conf = parseFloat(e.target.value);
    document.getElementById("confVal").textContent = conf.toFixed(2);
    refreshCounts();
    render();
  });

  canvas.addEventListener("mousemove", e => {
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvas.width / rect.width);
    const y = (e.clientY - rect.top) * (canvas.height / rect.height);
    const hit = lastBoxes.find(b =>
      x >= b.xyxy[0] && x <= b.xyxy[2] && y >= b.xyxy[1] && y <= b.xyxy[3]);
    const info = document.getElementById("boxInfo");
    info.innerHTML = hit
      ? `<strong>${hit.name}</strong> · conf ${hit.conf.toFixed(2)}<br>
         ${hit.type}<br>severity: ${hit.severity}`
      : "Hover a box for details.";
  });
}

async function init() {
  document.getElementById("year").textContent = new Date().getFullYear();
  const realPred = await loadJSON("data/predictions.json");
  if (realPred && realPred.models) {
    PRED = realPred;
    // Keep the synthetic backdrop for the bundled demo film; a real radiograph
    // would be drawn from an actual image instead.
    isDemo = !realPred.image || realPred.image === "synthetic-demo";
    if (!isDemo) {
      document.getElementById("sourceHint").innerHTML =
        `Real model output on a FracAtlas test X-ray (<code>${realPred.image}</code>, CC BY 4.0).`;
      // Size the canvas to the image so box pixel-coords align, then load it.
      if (realPred.width && realPred.height) {
        canvas.width = realPred.width;
        canvas.height = realPred.height;
      }
      bgImage = new Image();
      bgImage.onload = render;          // redraw once the radiograph is decoded
      bgImage.src = `assets/${realPred.image}`;
    }
  }
  const bench = (await loadJSON("data/benchmark.json")) || DEMO_BENCH;
  buildToggles();
  wireControls();
  buildBench(bench);
  render();
}

init();
