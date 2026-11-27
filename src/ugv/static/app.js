const $ = (selector) => document.querySelector(selector);
const connection = $("#connection");
const arucoCanvas = $("#aruco-overlay");
const arucoContext = arucoCanvas.getContext("2d");
const joystick = $("#joystick");
const joystickKnob = $("#joystick-knob");
const speed = $("#speed");
const speedValue = $("#speed-value");
const JOY_DEADZONE = 0.08;

let keys = new Set();
let stick = { forward: 0, strafe: 0 };
let stickPointer = null;
let socket = null;
let heartbeat = null;
let driveRefresh = null;
let reconnectTimer = null;
let state = {};
let needleDeg = 0;

window.__viewFit = "cover";

function applyJoyDeadzone(v) {
  if (Math.abs(v) < JOY_DEADZONE) return 0;
  return (v - Math.sign(v) * JOY_DEADZONE) / (1 - JOY_DEADZONE);
}

const PLANE_KP_NAMES = [
  "nose", "tail", "wingL", "wingR", "hubL", "hubR", "hubRear",
  "vtailL", "vtailR", "gearN", "gearL", "gearR", "tyreL", "tyreR", "tyreT",
];
const PLANE_BONES = [
  [0, 1], [0, 9], [2, 4], [4, 5], [5, 3], [6, 1],
  [1, 7], [1, 8], [10, 11], [9, 10], [9, 11], [12, 14], [14, 13],
];

function drawPlane(next, x0, y0, scale) {
  if (!next.plane_enabled) return;
  const pts = next.plane_points || [];
  const byIdx = new Map();
  for (const pt of pts) {
    if (!Array.isArray(pt) || pt.length < 3) continue;
    byIdx.set(pt[0], [x0 + pt[1] * scale, y0 + pt[2] * scale, pt[3] ?? 1]);
  }

  if (Array.isArray(next.plane_box) && next.plane_box.length === 3) {
    const [bx, by, bsize] = next.plane_box;
    arucoContext.save();
    arucoContext.setLineDash([6, 4]);
    arucoContext.strokeStyle = "rgba(52, 211, 153, 0.75)";
    arucoContext.lineWidth = 1.5;
    arucoContext.strokeRect(x0 + bx * scale, y0 + by * scale, bsize * scale, bsize * scale);
    arucoContext.restore();
  }

  arucoContext.save();
  arucoContext.strokeStyle = "rgba(52, 211, 153, 0.9)";
  arucoContext.lineWidth = 2;
  for (const [a, b] of PLANE_BONES) {
    const pa = byIdx.get(a);
    const pb = byIdx.get(b);
    if (!pa || !pb) continue;
    arucoContext.beginPath();
    arucoContext.moveTo(pa[0], pa[1]);
    arucoContext.lineTo(pb[0], pb[1]);
    arucoContext.stroke();
  }

  arucoContext.font = "600 11px ui-monospace, SFMono-Regular, Menlo, monospace";
  for (const [idx, [px, py]] of byIdx.entries()) {
    const isNear = idx >= 12;
    arucoContext.fillStyle = isNear ? "#fbbf24" : "#34d399";
    arucoContext.beginPath();
    arucoContext.arc(px, py, isNear ? 3.5 : 4.5, 0, Math.PI * 2);
    arucoContext.fill();
    arucoContext.strokeStyle = "#0f172a";
    arucoContext.lineWidth = 1.5;
    arucoContext.stroke();
    const label = PLANE_KP_NAMES[idx] || String(idx);
    arucoContext.fillStyle = "#ecfdf5";
    arucoContext.fillText(label, px + 6, py - 5);
  }

  if (next.plane_pose && Array.isArray(next.plane_box)) {
    const p = next.plane_pose;
    const tx = Math.max(12, x0 + next.plane_box[0] * scale);
    const ty = Math.max(20, y0 + next.plane_box[1] * scale - 8);
    arucoContext.fillStyle = "#34d399";
    arucoContext.font = "700 12px ui-monospace, SFMono-Regular, Menlo, monospace";
    arucoContext.fillText(
      `PLANE (${next.plane_model || ""}) x=${p.x_m.toFixed(2)}m y=${p.y_m >= 0 ? "+" : ""}${p.y_m.toFixed(2)}m yaw=${p.yaw_deg >= 0 ? "+" : ""}${p.yaw_deg.toFixed(0)}° · ${pts.length}pts · ${next.plane_ms ?? "—"}ms`,
      tx,
      ty
    );
  }
  arucoContext.restore();
}

function drawAruco(next) {
  const ratio = devicePixelRatio || 1;
  const width = arucoCanvas.clientWidth;
  const height = arucoCanvas.clientHeight;
  if (arucoCanvas.width !== Math.round(width * ratio) || arucoCanvas.height !== Math.round(height * ratio)) {
    arucoCanvas.width = Math.round(width * ratio);
    arucoCanvas.height = Math.round(height * ratio);
  }
  arucoContext.setTransform(ratio, 0, 0, ratio, 0, 0);
  arucoContext.clearRect(0, 0, width, height);
  if (!next.aruco_visible && !next.plane_enabled) return;

  const image = $("#camera-feed");
  const naturalWidth = image.naturalWidth || 1280;
  const naturalHeight = image.naturalHeight || 720;
  const fit = image.style.objectFit || window.__viewFit || "cover";
  const scale = fit === "cover"
    ? Math.max(width / naturalWidth, height / naturalHeight)
    : Math.min(width / naturalWidth, height / naturalHeight);
  const x0 = (width - naturalWidth * scale) / 2;
  const y0 = (height - naturalHeight * scale) / 2;

  drawPlane(next, x0, y0, scale);
  if (!next.aruco_visible) return;
  const markers = next.aruco_markers?.length
    ? next.aruco_markers
    : [{ id: next.aruco_id, distance_m: next.aruco_distance_m, corners: next.aruco_corners, target: true }];

  markers.forEach((marker) => {
    if (!marker.corners || marker.corners.length !== 4) return;
    const points = marker.corners.map(([x, y]) => [x0 + x * scale, y0 + y * scale]);
    const color = marker.target ? "#fb7185" : "#38bdf8";
    arucoContext.setLineDash(marker.tracked ? [6, 4] : []);
    arucoContext.beginPath();
    arucoContext.moveTo(...points[0]);
    points.slice(1).forEach((point) => arucoContext.lineTo(...point));
    arucoContext.closePath();
    arucoContext.lineWidth = marker.target ? 3 : 2;
    arucoContext.strokeStyle = color;
    arucoContext.stroke();
    arucoContext.fillStyle = color;
    arucoContext.font = "700 12px system-ui, sans-serif";
    const source = marker.source === "decode-fallback"
      ? ` · ${marker.confidence === 1 ? "EXACT" : "CORRECTED"}`
      : marker.tracked ? ` · ${String(marker.source || "").toUpperCase()} · ${marker.age_ms}ms` : "";
    const dist = Number(marker.distance_m);
    const distLabel = Number.isFinite(dist) ? `${dist.toFixed(2)} m` : "—";
    arucoContext.fillText(`TAG ${marker.id}${source} · ${distLabel}`, points[0][0], Math.max(14, points[0][1] - 8));
  });
  arucoContext.setLineDash([]);
}

function vector() {
  const forward = Math.max(-1, Math.min(1, Number(keys.has("KeyW")) - Number(keys.has("KeyS")) + stick.forward));
  const strafe = Math.max(-1, Math.min(1, Number(keys.has("KeyD")) - Number(keys.has("KeyA")) + stick.strafe));
  const pivot = Number(keys.has("KeyQ")) - Number(keys.has("KeyE"));
  return { forward, strafe, turn: pivot };
}

function send(message) {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(message));
}

function drive() {
  const next = vector();
  if (!state.armed || !state.you_are_owner) return;
  send({ type: "drive", ...next, speed_limit: Number(speed.value) });
}

function stop(reason = "client-stop") {
  keys.clear();
  stick = { forward: 0, strafe: 0 };
  stickPointer = null;
  joystickKnob.style.transform = "translate(0, 0)";
  send({ type: "stop", reason });
}

function render(next) {
  state = next;
  const cameraEnabled = next.camera_enabled !== false;
  const camBtn = $("#camera-toggle");
  camBtn.setAttribute("aria-pressed", String(cameraEnabled));
  camBtn.classList.toggle("active", cameraEnabled);
  camBtn.disabled = Boolean(next.owner && !next.you_are_owner);

  const feed = $("#camera-feed");
  if (!cameraEnabled && feed.hasAttribute("src")) feed.removeAttribute("src");
  if (cameraEnabled && !feed.hasAttribute("src")) feed.src = "/stream.mjpg";
  feed.style.visibility = cameraEnabled ? "visible" : "hidden";

  $("#aruco-toggle").disabled = !cameraEnabled;
  const planeBtn = $("#plane-toggle");
  if (planeBtn) {
    planeBtn.disabled = !cameraEnabled;
    planeBtn.setAttribute("aria-pressed", String(Boolean(next.plane_enabled)));
    planeBtn.classList.toggle("active", Boolean(next.plane_enabled));
  }
  $("#follow-toggle").disabled = !cameraEnabled;

  const cameraFresh = next.camera_frame_age_ms != null && next.camera_frame_age_ms < 500;
  $("#feed-stats").textContent = `${!cameraEnabled ? "CAM PAUSED" : cameraFresh ? "LIVE" : "STALE"} · ${(next.camera_fps || 0).toFixed(1)} FPS`;

  const autoParkEnabled = Boolean(next.auto_park);
  const autoParkStatus = String(next.auto_park_status || "").trim();
  const autoParkStageRaw = String(next.auto_park_stage || "").trim();
  const autoParkStage = autoParkStageRaw && autoParkStageRaw.toLowerCase() !== "idle" ? autoParkStageRaw : "";
  const autoParkInfo = [autoParkStage, autoParkStatus].filter((v, i, a) => v && a.indexOf(v) === i).join(" · ");
  const autoParkPill = $("#auto-park-pill");
  const showParkPill = autoParkEnabled || Boolean(autoParkStage) || (autoParkStatus && !["idle", "off", "none"].includes(autoParkStatus.toLowerCase()));
  autoParkPill.hidden = !showParkPill;
  if (showParkPill) {
    autoParkPill.textContent = autoParkInfo ? `PARK · ${autoParkInfo.toUpperCase()}` : "AUTO PARK";
    autoParkPill.classList.toggle("stopped", /stop|fail|abort|error/i.test(`${autoParkStage} ${autoParkStatus}`));
  }

  const canControl = Boolean(next.armed && next.you_are_owner);
  const parkBtn = $("#auto-park-toggle");
  parkBtn.disabled = (!state.armed && !state.auto_park) || (!canControl);
  parkBtn.setAttribute("aria-pressed", String(autoParkEnabled));
  parkBtn.classList.toggle("active", autoParkEnabled);
  parkBtn.textContent = autoParkEnabled && autoParkInfo ? `Auto Park · ${autoParkInfo}` : "Auto Park";
  parkBtn.title = (!state.armed && !state.auto_park) ? "Arm robot first" : "";
  const parkDetail = autoParkInfo ? ` (${autoParkInfo})` : "";
  parkBtn.title = canControl
    ? (autoParkEnabled ? `Stop Auto Park${parkDetail}` : `Start Auto Park${parkDetail}`)
    : `Arm controls to Auto Park${parkDetail}`;

  const armBtn = $("#arm");
  armBtn.textContent = canControl ? "Disarm" : "Arm";
  armBtn.classList.toggle("armed", canControl);

  const motion = $("#motion");
  motion.textContent = next.stopped ? "STOPPED" : "MOVING";
  motion.classList.toggle("moving", !next.stopped);

  const planeSummary = next.plane_enabled
    ? (next.plane_pose
      ? `PLANE ${next.plane_pose.x_m.toFixed(2)}m · ${next.plane_pose.yaw_deg.toFixed(0)}°`
      : (next.plane_status || "PLANE"))
    : null;
  const tagSummary = next.aruco_visible
    ? `#${next.aruco_id ?? "?"} · ${Number(next.aruco_distance_m ?? 0).toFixed(2)}m`
    : (planeSummary || (next.heading_calibrated ? `${Math.round(((Number(next.heading_deg) || 0) % 360 + 360) % 360)}°` : (next.aruco_status || "—")));
  $("#pose-tag").textContent = tagSummary;
  $("#imu-status").textContent = next.imu_connected ? `${(next.imu_rate_hz || 0).toFixed(0)} Hz · ${next.imu_age_ms ?? "—"} ms` : (next.imu_status || "offline");

  $("#forward").textContent = Number(next.forward || 0).toFixed(2);
  $("#strafe").textContent = Number(next.strafe || 0).toFixed(2);
  $("#turn").textContent = Number(next.turn || 0).toFixed(2);

  $("#reason").textContent = next.reason || "—";
  $("#turn-feedback").textContent = `${(next.turn_target_dps || 0).toFixed(0)}°/s tgt · ${(next.heading_rate_dps || 0).toFixed(0)}°/s`;
  $("#ownership").textContent = next.you_are_owner ? "controller" : next.owner ? "busy" : "available";
  $("#clients").textContent = String(next.clients ?? 0);
  const watchdog = $("#watchdog");
  if (watchdog && next.watchdog_ms != null) watchdog.textContent = `${next.watchdog_ms} ms`;
  $("#pan").textContent = `${next.pan_us ?? 1500} µs`;
  $("#tilt").textContent = `${next.tilt_us ?? 1500} µs`;
  $("#aruco-readout").textContent = next.aruco_visible
    ? `${next.aruco_markers?.length || 1} tag(s) · #${next.aruco_id} · ${Number(next.aruco_distance_m ?? 0).toFixed(2)} m`
    : (next.aruco_status || "none");
  const planeReadout = $("#plane-readout");
  if (planeReadout) {
    const fpsPart = next.plane_enabled && next.plane_fps ? ` · ${Number(next.plane_fps).toFixed(1)} fps` : "";
    planeReadout.textContent = `${next.plane_status || "disabled"}${fpsPart}`;
  }
  const planeModelSel = $("#plane-model-select");
  if (planeModelSel && Array.isArray(next.plane_available_models)) {
    const models = next.plane_available_models;
    const key = models.join(",");
    if (planeModelSel.dataset.models !== key) {
      planeModelSel.dataset.models = key;
      planeModelSel.innerHTML = models.length
        ? models.map((m) => `<option value="${m}">${m}</option>`).join("")
        : `<option value="">no models</option>`;
    }
    if (next.plane_model && planeModelSel.value !== next.plane_model && models.includes(next.plane_model)) {
      planeModelSel.value = next.plane_model;
    }
    planeModelSel.disabled = !models.length;
  }
  $("#vision-rate").textContent = `${(next.aruco_fps || 0).toFixed(1)} fps`;
  $("#vision-time").textContent = `${(next.aruco_processing_ms || 0).toFixed(0)} ms · ${next.aruco_result_age_ms ?? "—"} ms`;

  const vectorText = (v, d) => v ? ["x", "y", "z"].map((a) => Number(v[a] || 0).toFixed(d)).join(" / ") : "— / — / —";
  $("#imu-accel").textContent = vectorText(next.imu_accel_g, 3);
  $("#imu-gyro").textContent = vectorText(next.imu_gyro_dps, 1);

  const select = $("#controller-select");
  if (select && next.controller && select.value !== next.controller) select.value = next.controller;

  $("#aruco-toggle").setAttribute("aria-pressed", String(Boolean(next.aruco_enabled)));
  $("#aruco-toggle").classList.toggle("active", Boolean(next.aruco_enabled));
  $("#follow-toggle").setAttribute("aria-pressed", String(Boolean(next.aruco_follow)));
  $("#follow-toggle").classList.toggle("active", Boolean(next.aruco_follow));

  const heading = Number(next.heading_deg) || 0;
  // heading_deg wraps at 360; turn the needle by the shortest arc so it never sweeps backwards.
  needleDeg += ((heading - needleDeg) % 360 + 540) % 360 - 180;
  $("#heading-needle").style.transform = `rotate(${needleDeg}deg)`;
  $("#heading-degrees").textContent = next.heading_calibrated ? String(((Math.round(heading) % 360) + 360) % 360).padStart(3, "0") : "—";
  const headingStatus = (next.heading_status || "hold still · zeroing").toUpperCase();
  const headingRate = Number(next.heading_rate_dps) || 0;
  $("#heading-status").textContent = next.heading_calibrated && !next.heading_stationary
    ? `${headingStatus} · ${headingRate.toFixed(0)}°/S`
    : headingStatus;
  $("#heading-calibrate").disabled = !next.imu_connected;
  $("#heading-widget").classList.toggle("uncalibrated", !next.heading_calibrated);
  $("#heading-widget").classList.toggle("moving", !next.heading_stationary && next.heading_calibrated);

  drawAruco(next);
  window.renderParking?.(next.parking, next);
  schedulePerfDraw();
}

function connect() {
  clearTimeout(reconnectTimer);
  socket = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  socket.addEventListener("open", () => {
    connection.className = "connection online";
    connection.lastElementChild.textContent = "Online";
    heartbeat = setInterval(() => send({ type: "heartbeat" }), 200);
    driveRefresh = setInterval(() => {
      if (["KeyW", "KeyS", "KeyA", "KeyD", "KeyQ", "KeyE"].some((k) => keys.has(k)) || stick.forward || stick.strafe) drive();
    }, 150);
  });
  socket.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    if (message.type === "state") render(message.state);
  });
  socket.addEventListener("close", () => {
    keys.clear();
    clearInterval(heartbeat);
    clearInterval(driveRefresh);
    connection.className = "connection offline";
    connection.lastElementChild.textContent = "Offline";
    reconnectTimer = setTimeout(connect, 1000);
  });
}

document.addEventListener("keydown", (event) => {
  if (event.code === "Space" || event.code === "Escape") {
    event.preventDefault();
    stop(event.code === "Space" ? "space-stop" : "escape-stop");
    return;
  }
  if (["INPUT", "TEXTAREA", "SELECT"].includes(event.target?.tagName) && event.target.type !== "range") return;
  if (event.code.startsWith("Arrow")) {
    event.preventDefault();
    if (!state.armed || !state.you_are_owner) return;
    const moves = { ArrowLeft: ["pan", 25], ArrowRight: ["pan", -25], ArrowUp: ["tilt", -25], ArrowDown: ["tilt", 25] };
    const [axis, delta] = moves[event.code];
    send({ type: "camera", axis, delta });
    return;
  }
  if (!["KeyW", "KeyA", "KeyS", "KeyD", "KeyQ", "KeyE"].includes(event.code) || event.repeat) return;
  event.preventDefault();
  keys.add(event.code);
  drive();
});

document.addEventListener("keyup", (event) => {
  if (!["KeyW", "KeyA", "KeyS", "KeyD", "KeyQ", "KeyE"].includes(event.code)) return;
  event.preventDefault();
  keys.delete(event.code);
  drive();
});

$("#stop").addEventListener("click", () => stop("button-stop"));
$("#arm").addEventListener("click", () => (state.armed && state.you_are_owner ? stop("disarm") : send({ type: "arm" })));
$("#auto-park-toggle").addEventListener("click", () => {
  if (!state.armed || !state.you_are_owner) return;
  send({ type: "auto-park", enabled: !state.auto_park });
});
$("#camera-toggle").addEventListener("click", () => send({ type: "camera-capture", enabled: state.camera_enabled === false }));
$("#aruco-toggle").addEventListener("click", () => send({ type: "aruco", enabled: !state.aruco_enabled, follow: state.aruco_follow ? false : null }));
$("#plane-toggle")?.addEventListener("click", () => send({ type: "plane", enabled: !state.plane_enabled }));
$("#plane-model-select")?.addEventListener("change", (event) => {
  if (event.target.value) send({ type: "plane", model: event.target.value });
});
$("#follow-toggle").addEventListener("click", () => send({ type: "aruco", enabled: true, follow: !state.aruco_follow, speed_limit: Number(speed.value) }));
$("#heading-calibrate").addEventListener("click", () => send({ type: "heading-calibrate" }));
$("#controller-select")?.addEventListener("change", (event) => send({ type: "controller", controller: event.target.value }));
$("#view-fit-toggle").addEventListener("click", () => {
  window.__viewFit = window.__viewFit === "cover" ? "contain" : "cover";
  const feed = $("#camera-feed");
  if (feed) feed.style.objectFit = window.__viewFit;
  const el = $("#view-fit-toggle");
  const isCover = window.__viewFit === "cover";
  el.setAttribute("aria-pressed", String(isCover));
  el.classList.toggle("active", isCover);
  el.title = isCover ? "Fit: Cover (click for Contain)" : "Fit: Contain (click for Cover)";
  drawAruco(state);
});

speed.addEventListener("input", () => {
  speedValue.textContent = speed.value;
  if (state.aruco_follow && state.you_are_owner) send({ type: "aruco", enabled: true, speed_limit: Number(speed.value) });
  if (keys.size) drive();
});

function updateStick(event) {
  const rect = joystick.getBoundingClientRect();
  const radius = Math.min(rect.width, rect.height) * 0.38;
  const dx = event.clientX - (rect.left + rect.width / 2);
  const dy = event.clientY - (rect.top + rect.height / 2);
  const magnitude = Math.hypot(dx, dy);
  const scale = magnitude > radius ? radius / magnitude : 1;
  const x = dx * scale;
  const y = dy * scale;
  stick = { strafe: applyJoyDeadzone(x / radius), forward: applyJoyDeadzone(-y / radius) };
  joystickKnob.style.transform = `translate(${x}px, ${y}px)`;
  if (state.armed && state.you_are_owner) drive();
}

joystick.addEventListener("pointerdown", (event) => {
  event.preventDefault();
  stickPointer = event.pointerId;
  joystick.setPointerCapture(event.pointerId);
  updateStick(event);
});
joystick.addEventListener("pointermove", (event) => {
  if (event.pointerId === stickPointer) updateStick(event);
});
function releaseStick(event) {
  if (event.pointerId !== stickPointer) return;
  stickPointer = null;
  stick = { forward: 0, strafe: 0 };
  joystickKnob.style.transform = "translate(0, 0)";
  drive();
}
joystick.addEventListener("pointerup", releaseStick);
joystick.addEventListener("pointercancel", releaseStick);

window.addEventListener("blur", () => stop("window-blur"));
document.addEventListener("visibilitychange", () => { if (document.hidden) stop("hidden"); });
window.addEventListener("pagehide", () => stop("pagehide"));

const perfPanel = $("#perf-panel");
const perfToggle = $("#perf-toggle");
const perfClose = $("#perf-close");
const perfChart = $("#perf-chart");
const perfSummary = $("#perf-summary");
const perfLegend = $("#perf-legend");
const perfDetails = $("#perf-details");

const PERF_LEGEND = [
  ["Decode", "#38bdf8"],
  ["Detect", "#818cf8"],
  ["SubPix", "#c084fc"],
  ["PnP", "#f472b6"],
  ["Plan/Ctrl", "#34d399"],
  ["Drive", "#fbbf24"],
  ["ArUco", "#fb7185"],
  ["Cam/IMU/IO", "#60a5fa"],
];

if (perfLegend) {
  perfLegend.innerHTML = PERF_LEGEND.map(
    ([label, color]) => `<span class="perf-legend-item"><span class="perf-swatch" style="background:${color}"></span>${label}</span>`
  ).join("");
}

let perfRaf = 0;

function readStage(stages, names, fallbackMs = 0) {
  for (const name of names) {
    const raw = stages?.[name];
    if (raw == null) continue;
    if (typeof raw === "number") {
      const v = Number.isFinite(raw) ? raw : 0;
      return { last_ms: v, p50_ms: v, p95_ms: v, hz: null };
    }
    if (typeof raw === "object") {
      const last = Number(raw.last_ms ?? raw.ms ?? raw.last ?? 0);
      return {
        last_ms: Number.isFinite(last) ? last : 0,
        p50_ms: raw.p50_ms != null && Number.isFinite(Number(raw.p50_ms)) ? Number(raw.p50_ms) : null,
        p95_ms: raw.p95_ms != null && Number.isFinite(Number(raw.p95_ms)) ? Number(raw.p95_ms) : null,
        hz: raw.hz != null && Number.isFinite(Number(raw.hz)) ? Number(raw.hz) : null,
      };
    }
  }
  const fb = Number(fallbackMs);
  return { last_ms: Number.isFinite(fb) ? fb : 0, p50_ms: null, p95_ms: null, hz: null };
}

function sumStats(parts, agg) {
  const last = agg?.last_ms > 0 ? agg.last_ms : parts.reduce((s, p) => s + p.last_ms, 0);
  const hasP50 = agg?.p50_ms != null || parts.some((p) => p.p50_ms != null);
  const hasP95 = agg?.p95_ms != null || parts.some((p) => p.p95_ms != null);
  const p50 = agg?.p50_ms != null ? agg.p50_ms : (hasP50 ? parts.reduce((s, p) => s + (p.p50_ms ?? p.last_ms), 0) : null);
  const p95 = agg?.p95_ms != null ? agg.p95_ms : (hasP95 ? parts.reduce((s, p) => s + (p.p95_ms ?? p.last_ms), 0) : null);
  return { last, p50, p95 };
}

function fmtMs(v) {
  return v == null || !Number.isFinite(v) ? "—" : v.toFixed(1);
}

function drawPerfChart(next) {
  if (!perfPanel || perfPanel.hidden || !perfChart) return;
  const ctx = perfChart.getContext("2d");
  if (!ctx) return;

  const dpr = devicePixelRatio || 1;
  const cssW = perfChart.clientWidth || 380;
  const cssH = perfChart.clientHeight || 176;
  const targetW = Math.round(cssW * dpr);
  const targetH = Math.round(cssH * dpr);
  if (perfChart.width !== targetW || perfChart.height !== targetH) {
    perfChart.width = targetW;
    perfChart.height = targetH;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  const perf = next?.perf && typeof next.perf === "object" ? next.perf : {};
  const stages = perf.stages && typeof perf.stages === "object" ? perf.stages : {};
  const sys = perf.system && typeof perf.system === "object" ? perf.system : {};
  const budgetMs = Number(perf.budget_ms) > 0 ? Number(perf.budget_ms) : 33.33;

  const decode = readStage(stages, ["decode_ms", "decode"]);
  const detect = readStage(stages, ["detect_ms", "detect"]);
  const subpix = readStage(stages, ["subpix_ms", "subpix"]);
  const pnp = readStage(stages, ["pnp_ms", "pnp"]);
  const visionAgg = readStage(stages, ["autopark_vision", "vision", "vision_ms"]);
  const plan = readStage(stages, ["plan_ms", "plan"]);
  const driveSt = readStage(stages, ["drive_ms", "drive", "ctrl_ms", "ctrl"]);
  const loopAgg = readStage(stages, ["park_loop", "autopark_loop", "loop_ms"]);
  const aruco = readStage(stages, ["aruco_ms", "aruco"], next?.aruco_processing_ms ?? 0);
  const camAge = readStage(stages, ["camera_frame_age_ms", "cam_age_ms", "camera_age"], next?.camera_frame_age_ms ?? 0);
  const imuAge = readStage(stages, ["imu_age_ms", "imu_age"], next?.imu_age_ms ?? 0);
  const bcast = readStage(stages, ["broadcast_ms", "broadcast"]);

  const visionStat = sumStats([decode, detect, subpix, pnp], visionAgg.last_ms > 0 ? visionAgg : null);
  const visionTotalMs = visionStat.last;
  const loopStat = sumStats(
    [{ last_ms: visionTotalMs, p50_ms: visionStat.p50, p95_ms: visionStat.p95 }, plan, driveSt],
    loopAgg.last_ms > 0 ? loopAgg : null
  );
  const arucoStat = sumStats([aruco], aruco);
  const ioStat = sumStats([camAge, imuAge, bcast], null);

  const rows = [
    {
      label: "AutoPark Vision",
      stat: visionStat,
      segments: (decode.last_ms + detect.last_ms + subpix.last_ms + pnp.last_ms) > 0 || visionAgg.last_ms === 0
        ? [
            { value: decode.last_ms, color: "#38bdf8" },
            { value: detect.last_ms, color: "#818cf8" },
            { value: subpix.last_ms, color: "#c084fc" },
            { value: pnp.last_ms, color: "#f472b6" },
          ]
        : [{ value: visionAgg.last_ms, color: "#38bdf8" }],
    },
    {
      label: "Park Loop",
      stat: loopStat,
      segments: [
        { value: visionTotalMs, color: "#38bdf8" },
        { value: plan.last_ms, color: "#34d399" },
        { value: driveSt.last_ms, color: "#fbbf24" },
      ],
    },
    {
      label: "ArUco Tags",
      stat: arucoStat,
      segments: [{ value: aruco.last_ms, color: "#fb7185" }],
    },
    {
      label: "IO / Age",
      stat: ioStat,
      segments: [
        { value: camAge.last_ms, color: "#60a5fa" },
        { value: imuAge.last_ms, color: "#2dd4bf" },
        { value: bcast.last_ms, color: "#a78bfa" },
      ],
    },
  ];

  const bars = rows.map((r) => Math.max(r.stat.last, r.segments.reduce((s, seg) => s + seg.value, 0)));
  const maxMs = Math.max(60, ...bars);

  const padL = 8;
  const padR = 8;
  const topY = 20;
  const bottomY = cssH - 6;
  const plotW = Math.max(40, cssW - padL - padR);
  const rowH = (bottomY - topY) / rows.length;
  const xOf = (ms) => padL + Math.min(1, Math.max(0, ms / maxMs)) * plotW;

  // Budget & 50ms dashed vertical markers
  const markers = [
    { ms: budgetMs, label: `${budgetMs.toFixed(1)}ms (30Hz)`, color: "rgba(52, 211, 153, 0.72)" },
    { ms: 50, label: "50ms (20Hz)", color: "rgba(251, 191, 36, 0.65)" },
  ];
  ctx.save();
  ctx.font = "600 9.5px ui-monospace, SFMono-Regular, Menlo, monospace";
  ctx.textBaseline = "top";
  for (const m of markers) {
    const x = Math.round(xOf(m.ms)) + 0.5;
    ctx.strokeStyle = m.color;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(x, topY - 2);
    ctx.lineTo(x, bottomY);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = m.color;
    const w = ctx.measureText(m.label).width;
    const tx = Math.min(cssW - padR - w, Math.max(padL, x - w / 2));
    ctx.fillText(m.label, tx, 4);
  }
  ctx.restore();

  rows.forEach((row, idx) => {
    const ry = topY + idx * rowH;
    const barY = ry + 15;
    const barH = Math.max(8, Math.min(13, rowH - 20));

    ctx.font = "600 10.5px system-ui, -apple-system, sans-serif";
    ctx.textBaseline = "top";
    ctx.textAlign = "left";
    ctx.fillStyle = "#e2e8f0";
    ctx.fillText(row.label, padL, ry + 2);

    const statsText = `${fmtMs(row.stat.last)} / ${fmtMs(row.stat.p50)} / ${fmtMs(row.stat.p95)} ms`;
    ctx.font = "500 10px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.textAlign = "right";
    ctx.fillStyle = "#cbd5e1";
    ctx.fillText(statsText, cssW - padR, ry + 2);

    // Track background
    ctx.fillStyle = "rgba(255, 255, 255, 0.08)";
    ctx.beginPath();
    ctx.roundRect(padL, barY, plotW, barH, 3);
    ctx.fill();

    // Stacked segments
    let cursorX = padL;
    for (const seg of row.segments) {
      if (!(seg.value > 0)) continue;
      const segW = Math.max(1.5, (seg.value / maxMs) * plotW);
      const drawW = Math.min(padL + plotW - cursorX, segW);
      if (drawW <= 0) break;
      ctx.fillStyle = seg.color;
      ctx.fillRect(cursorX, barY, drawW, barH);
      cursorX += drawW;
    }
  });

  if (perfSummary) {
    const parts = [];
    if (sys.cpu_pct != null) parts.push(`CPU ${Number(sys.cpu_pct).toFixed(0)}%`);
    if (sys.temp_c != null) parts.push(`${Number(sys.temp_c).toFixed(0)}°C`);
    if (sys.rss_mb != null) parts.push(`${Number(sys.rss_mb).toFixed(0)}MB`);
    if (!parts.length && next?.camera_fps != null) {
      parts.push(`Cam ${Number(next.camera_fps).toFixed(1)}Hz`);
    }
    perfSummary.textContent = parts.length ? parts.join(" · ") : "—";
  }

  if (perfDetails) {
    const sysLine = [
      sys.cpu_pct != null ? `cpu ${Number(sys.cpu_pct).toFixed(0)}%` : null,
      sys.load1 != null ? `load ${Number(sys.load1).toFixed(2)}${sys.cores ? `/${sys.cores}c` : ""}` : null,
      sys.rss_mb != null ? `rss ${Number(sys.rss_mb).toFixed(0)}MB` : null,
      sys.temp_c != null ? `temp ${Number(sys.temp_c).toFixed(1)}°C` : null,
    ].filter(Boolean).join(" · ") || "system: awaiting perf telemetry";

    const camHz = next?.camera_fps != null ? `${Number(next.camera_fps).toFixed(1)}Hz` : "—";
    const imuHz = next?.imu_rate_hz != null ? `${Number(next.imu_rate_hz).toFixed(0)}Hz` : "—";
    const arucoHz = aruco.hz != null ? `${aruco.hz.toFixed(1)}Hz` : (next?.aruco_fps != null ? `${Number(next.aruco_fps).toFixed(1)}Hz` : "—");
    const rateLine = `cam ${camHz} · imu ${imuHz} · aruco ${arucoHz} · budget ${budgetMs.toFixed(1)}ms`;

    const stageKeys = Object.keys(stages);
    const stageLine = stageKeys.length
      ? stageKeys.slice(0, 6).map((k) => {
          const s = readStage(stages, [k]);
          const hz = s.hz != null ? `@${s.hz.toFixed(0)}Hz` : "";
          return `${k.replace(/_ms$/, "")} ${fmtMs(s.p50_ms)}/${fmtMs(s.p95_ms)}${hz}`;
        }).join(" · ")
      : "last / p50 / p95 ms shown per row";

    perfDetails.textContent = `${sysLine}
${rateLine}
${stageLine}`;
  }
}

function schedulePerfDraw() {
  if (!perfPanel || perfPanel.hidden) return;
  if (perfRaf) return;
  perfRaf = requestAnimationFrame(() => {
    perfRaf = 0;
    drawPerfChart(state);
  });
}

function setPerfOpen(open) {
  if (!perfPanel || !perfToggle) return;
  perfPanel.hidden = !open;
  perfToggle.setAttribute("aria-pressed", String(open));
  perfToggle.classList.toggle("active", open);
  if (open) {
    const parkingPanel = $("#parking-panel");
    const mapToggle = $("#map-toggle");
    if (parkingPanel) parkingPanel.hidden = true;
    if (mapToggle) {
      mapToggle.setAttribute("aria-pressed", "false");
      mapToggle.classList.remove("active");
    }
    schedulePerfDraw();
  }
}

perfToggle?.addEventListener("click", () => setPerfOpen(Boolean(perfPanel?.hidden)));
perfClose?.addEventListener("click", () => setPerfOpen(false));
$("#map-toggle")?.addEventListener("click", () => {
  const parkingPanel = $("#parking-panel");
  if (parkingPanel && !parkingPanel.hidden) setPerfOpen(false);
});

connect();
