let needleDeg = 0;
const keys = new Set();
let socket;
let heartbeat;
let driveRefresh;
let reconnectTimer;
let state = {};

const $ = (selector) => document.querySelector(selector);
const connection = $("#connection");
const speed = $("#speed");
const speedValue = $("#speed-value");
const arucoCanvas = $("#aruco-overlay");
const arucoContext = arucoCanvas.getContext("2d");
const joystick = $("#joystick");
const joystickKnob = $("#joystick-knob");
let stick = { forward: 0, strafe: 0 };
let stickPointer = null;
const JOY_DEADZONE = 0.08;
function applyJoyDeadzone(value) {
  const magnitude = Math.abs(value);
  if (magnitude <= JOY_DEADZONE) return 0;
  return Math.sign(value) * ((magnitude - JOY_DEADZONE) / (1 - JOY_DEADZONE));
}

// Telemetry ring buffers (client-side, ~50 Hz loop data at render rate).
const RING_N = 120;
const rings = { loop: [], infer: [], fps: [], age: [], yaw: [], gyro: [] };
function pushRing(name, value) {
  if (!Number.isFinite(value)) return;
  const ring = rings[name];
  ring.push(value);
  if (ring.length > RING_N) ring.shift();
}
function drawSpark(id, series, color, min, max) {
  const el = document.querySelector(id);
  if (!el) return;
  const context = el.getContext("2d");
  const W = el.width, H = el.height;
  context.clearRect(0, 0, W, H);
  if (series.length < 2) return;
  const span = Math.max(1e-6, max - min);
  context.strokeStyle = color;
  context.lineWidth = 1.5;
  context.beginPath();
  series.forEach((v, i) => {
    const x = (i / (RING_N - 1)) * W;
    const y = H - 3 - ((Math.min(max, Math.max(min, v)) - min) / span) * (H - 6);
    if (i === 0) context.moveTo(x, y); else context.lineTo(x, y);
  });
  context.stroke();
}

function drawAruco(next) {
  const ratio = devicePixelRatio || 1;
  const width = arucoCanvas.clientWidth, height = arucoCanvas.clientHeight;
  if (arucoCanvas.width !== Math.round(width * ratio) || arucoCanvas.height !== Math.round(height * ratio)) {
    arucoCanvas.width = Math.round(width * ratio); arucoCanvas.height = Math.round(height * ratio);
  }
  arucoContext.setTransform(ratio, 0, 0, ratio, 0, 0);
  arucoContext.clearRect(0, 0, width, height);
  if (!next.aruco_visible) return;
  const image = $("#camera-feed"), naturalWidth = image.naturalWidth || 640, naturalHeight = image.naturalHeight || 480;
  const scale = Math.min(width / naturalWidth, height / naturalHeight);
  const x0 = (width - naturalWidth * scale) / 2, y0 = (height - naturalHeight * scale) / 2;
  const markers = next.aruco_markers?.length ? next.aruco_markers : [{ id: next.aruco_id, distance_m: next.aruco_distance_m, corners: next.aruco_corners, target: true }];
  markers.forEach(marker => {
    if (marker.corners.length !== 4) return;
    const points = marker.corners.map(([x, y]) => [x0 + x * scale, y0 + y * scale]);
    const color = marker.target ? "#e11d48" : "#2563eb";
    arucoContext.setLineDash(marker.tracked ? [6, 4] : []);
    arucoContext.beginPath(); arucoContext.moveTo(...points[0]); points.slice(1).forEach(point => arucoContext.lineTo(...point)); arucoContext.closePath();
    arucoContext.lineWidth = marker.target ? 3 : 2; arucoContext.strokeStyle = color; arucoContext.stroke();
    arucoContext.fillStyle = color; arucoContext.font = "700 13px system-ui";
    const source = marker.source === "decode-fallback"
      ? ` · ${marker.confidence === 1 ? "EXACT CODE · RESAMPLED" : "CORRECTED CODE · RESAMPLED"}`
      : marker.tracked ? ` · ${marker.source.toUpperCase()} · ${marker.age_ms} ms` : "";
    arucoContext.fillText(`ARUCO ${marker.id}${source} · ${marker.distance_m.toFixed(2)} m`, points[0][0], points[0][1] - 8);
  });
  arucoContext.setLineDash([]);
}

function vector() {
  const forward = Math.max(-1, Math.min(1, Number(keys.has("KeyW")) - Number(keys.has("KeyS")) + stick.forward));
  const strafe = Math.max(-1, Math.min(1, Number(keys.has("KeyD")) - Number(keys.has("KeyA")) + stick.strafe));
  const pivot = Number(keys.has("KeyQ")) - Number(keys.has("KeyE"));
  return {
    forward,
    strafe,
    turn: pivot,
  };
}

function send(message) {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(message));
}

function drive() {
  const next = vector();
  document.querySelectorAll("[data-key]").forEach((button) => button.classList.toggle("active", keys.has(button.dataset.key)));
  if (!state.armed || !state.you_are_owner) return;
  send({ type: "drive", ...next, speed_limit: Number(speed.value) });
}

function stop(reason = "client-stop") {
  keys.clear();
  stick = { forward: 0, strafe: 0 };
  stickPointer = null;
  joystickKnob.style.transform = "translate(0, 0)";
  document.querySelectorAll("[data-key]").forEach((button) => button.classList.remove("active"));
  send({ type: "stop", reason });
}

function render(next) {
  state = next;
  const cameraEnabled = next.camera_enabled !== false;
  $("#camera-toggle").textContent = cameraEnabled ? "CAMERA ON" : "CAMERA OFF";
  $("#camera-toggle").setAttribute("aria-pressed", String(cameraEnabled));
  $("#camera-toggle").disabled = Boolean(next.owner && !next.you_are_owner);
  const feed = $("#camera-feed");
  if (!cameraEnabled && feed.hasAttribute("src")) feed.removeAttribute("src");
  if (cameraEnabled && !feed.hasAttribute("src")) feed.src = "/stream.mjpg";
  feed.style.visibility = cameraEnabled ? "visible" : "hidden";
  $("#aruco-toggle").disabled = !cameraEnabled;
  $("#follow-toggle").disabled = !cameraEnabled;
  const cameraFresh = next.camera_frame_age_ms != null && next.camera_frame_age_ms < 500;
  $("#feed-stats").textContent = `${!cameraEnabled ? "CAMERA PAUSED" : cameraFresh ? "LIVE" : "STALE"} · 1280×720 · ${(next.camera_fps || 0).toFixed(1)} FPS · MONO GS`;
  $("#vision-rate").textContent = `${(next.aruco_fps || 0).toFixed(1)} processed fps`;
  $("#vision-time").textContent = `${(next.aruco_processing_ms || 0).toFixed(0)} ms compute · ${next.aruco_result_age_ms ?? "—"} ms age`;
  $("#motion").textContent = next.stopped ? "STOPPED" : "MOVING";
  $("#motion").classList.toggle("moving", !next.stopped);
  $("#reason").textContent = next.reason;
  $("#turn-feedback").textContent = `${(next.turn_target_dps || 0).toFixed(0)}°/s target · ${(next.heading_rate_dps || 0).toFixed(0)}°/s actual`;
  $("#ownership").textContent = next.you_are_owner ? "controller" : next.owner ? "busy" : "available";
  $("#arm").textContent = next.armed && next.you_are_owner ? "DISARM / STOP" : "ARM CONTROLS";
  $("#arm").classList.toggle("armed", next.armed && next.you_are_owner);
  $("#clients").textContent = next.clients;
  $("#forward").textContent = next.forward.toFixed(2);
  $("#strafe").textContent = next.strafe.toFixed(2);
  $("#turn").textContent = next.turn.toFixed(2);
  $("#watchdog").textContent = `${next.watchdog_ms} ms`;
  $("#pan").textContent = `${next.pan_us} µs`;
  $("#tilt").textContent = `${next.tilt_us} µs`;
  $("#aruco-toggle").textContent = next.aruco_enabled ? "ARUCO ON" : "ARUCO OFF";
  $("#aruco-toggle").setAttribute("aria-pressed", String(next.aruco_enabled));
  $("#follow-toggle").textContent = next.aruco_follow ? "FOLLOW ON" : "FOLLOW OFF";
  $("#follow-toggle").setAttribute("aria-pressed", String(next.aruco_follow));
  $("#aruco-readout").textContent = next.aruco_visible ? `${next.aruco_markers?.length || 1} tag(s) · target ${next.aruco_id} · ${next.aruco_distance_m.toFixed(2)} m` : next.aruco_status;
  $("#imu-status").textContent = next.imu_connected ? `${next.imu_rate_hz.toFixed(0)} Hz · ${next.imu_age_ms} ms` : next.imu_status;
  const vectorText = (vector, digits) => vector ? ["x", "y", "z"].map(axis => vector[axis].toFixed(digits)).join(" / ") : "— / — / —";
  $("#imu-accel").textContent = vectorText(next.imu_accel_g, 3);
  $("#imu-gyro").textContent = vectorText(next.imu_gyro_dps, 1);
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
  Object.entries(next.wheels).forEach(([name, duty]) => document.querySelector(`[data-wheel="${name}"]`).textContent = duty);
  // Mission-Control HUD (additive; every legacy ID above is untouched).
  pushRing("loop", Number(next.tick_ms) || 0);
  pushRing("infer", Number(next.inference_ms) || 0);
  pushRing("fps", Number(next.camera_fps) || 0);
  pushRing("age", Number(next.camera_frame_age_ms) || 0);
  pushRing("yaw", Number(next.heading_deg) || 0);
  pushRing("gyro", Number(next.heading_rate_dps) || 0);
  drawSpark("#spark-loop", rings.loop, "#38bdf8", 0, Math.max(20, ...rings.loop));
  drawSpark("#spark-cam", rings.fps, "#22c55e", 0, Math.max(35, ...rings.fps));
  drawSpark("#spark-imu", rings.gyro, "#f59e0b", Math.min(-45, ...rings.gyro), Math.max(45, ...rings.gyro));
  const bar = { "front-left": "#wheel-fl", "front-right": "#wheel-fr", "rear-left": "#wheel-rl", "rear-right": "#wheel-rr" };
  Object.entries(bar).forEach(([name, id]) => {
    const pwm = Number((next.wheels || {})[name] || 0);
    const el = document.querySelector(id);
    if (el) el.style.width = `${Math.min(100, (Math.abs(pwm) / 4095) * 100).toFixed(1)}%`;
    const num = document.querySelector(`[data-wheel-bar="${name}"]`);
    if (num) num.textContent = String(pwm);
  });
  const fd = next.follow_debug || {};
  const badge = $("#follow-state");
  if (badge) {
    const mode = String(fd.mode || (next.aruco_follow ? "TRACK" : "IDLE")).toUpperCase();
    badge.textContent = mode;
    badge.className = mode === "TRACK" ? "track" : mode === "COAST" ? "coast" : mode === "SEARCH" ? "search" : "";
    badge.id = "follow-state";
  }
  const lead = $("#follow-lead");
  if (lead) lead.textContent = `${Number(fd.err_lead ?? NaN).toFixed?.(3) ?? "—"} / ${Number(fd.z_lead ?? NaN).toFixed?.(3) ?? "—"}`;
  const sigma = $("#follow-sigma");
  if (sigma) sigma.textContent = `${fd.sigma_m ?? "—"} m / ${fd.coast_age_ms ?? 0} ms`;
  const sysid = $("#sysid-readiness");
  if (sysid) sysid.textContent = `${next.sysid_ready ? "READY" : "HOLD"} · rls ${next.rls_scale ?? 1} · bias ${(next.imu_bias_dps?.z ?? 0).toFixed?.(2) ?? "—"} dps`;
  const banner = $("#l7-banner");
  if (banner) {
    const breach = next.controller !== "manual" && !["armed-hold", "policy-hold"].includes(next.controller_stage)
      ? false : (next.reason || "").includes("controller-blocked") || (next.reason || "").includes("watchdog");
    const armedBreach = next.armed && (next.reason || "").match(/(watchdog|blocked|stale|mismatch|expired)/);
    if (armedBreach) {
      banner.textContent = `L7 INTERLOCK · BREACH · ${next.reason}`;
      banner.className = "l7-breach";
    } else {
      banner.textContent = `L7 INTERLOCK · NOMINAL · ${next.controller}/${next.controller_stage} · deadline ${next.watchdog_ms} ms`;
      banner.className = "l7-ok";
    }
    banner.id = "l7-banner";
  }
  const select = $("#controller-select");
  if (select && select.value !== next.controller) select.value = next.controller;
  const fmt = $("#follow-mode-toggle");
  if (fmt) {
    const smooth = (next.follow_debug && next.follow_debug.smooth) || window.__followSmooth;
    fmt.textContent = window.__followSmooth ? "SMOOTH ON" : "SMOOTH OFF";
    fmt.setAttribute("aria-pressed", String(Boolean(window.__followSmooth)));
  }
  window.renderTactical?.(next);
}

function connect() {
  clearTimeout(reconnectTimer);
  socket = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  socket.addEventListener("open", () => {
    connection.className = "connection online";
    connection.lastChild.textContent = " Online";
    heartbeat = setInterval(() => send({ type: "heartbeat" }), 200);
    driveRefresh = setInterval(() => {
      if (["KeyW", "KeyS", "KeyA", "KeyD", "KeyQ", "KeyE"].some(key => keys.has(key)) || stick.forward || stick.strafe) drive();
    }, 150);
  });
  socket.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    if (message.type === "state") render(message.state);
  });
  socket.addEventListener("close", () => {
    keys.clear();
    document.querySelectorAll("[data-key]").forEach(button => button.classList.remove("active"));
    clearInterval(heartbeat);
    clearInterval(driveRefresh);
    connection.className = "connection offline";
    connection.lastChild.textContent = " Offline";
    reconnectTimer = setTimeout(connect, 1000);
  });
}

document.addEventListener("keydown", (event) => {
  if (event.code === "Space") { event.preventDefault(); stop("space-stop"); return; }
  if (event.code.startsWith("Arrow")) {
    event.preventDefault();
    document.querySelector(`[data-key="${event.code}"]`)?.classList.add("active");
    if (!state.armed || !state.you_are_owner) return;
    const moves = { ArrowLeft: ["pan", 25], ArrowRight: ["pan", -25], ArrowUp: ["tilt", -25], ArrowDown: ["tilt", 25] };
    const [axis, delta] = moves[event.code]; send({ type: "camera", axis, delta }); return;
  }
  if (!["KeyW", "KeyA", "KeyS", "KeyD", "KeyQ", "KeyE"].includes(event.code) || event.repeat) return;
  event.preventDefault(); keys.add(event.code); drive();
});
document.addEventListener("keyup", (event) => {
  if (event.code.startsWith("Arrow")) {
    event.preventDefault(); document.querySelector(`[data-key="${event.code}"]`)?.classList.remove("active"); return;
  }
  if (!["KeyW", "KeyA", "KeyS", "KeyD", "KeyQ", "KeyE"].includes(event.code)) return;
  event.preventDefault(); keys.delete(event.code); drive();
});
$("#stop").addEventListener("click", () => stop("button-stop"));
$("#arm").addEventListener("click", () => state.armed && state.you_are_owner ? stop("disarm") : send({ type: "arm" }));
$("#camera-toggle").addEventListener("click", () => send({ type: "camera-capture", enabled: state.camera_enabled === false }));
$("#aruco-toggle").addEventListener("click", () => send({ type: "aruco", enabled: !state.aruco_enabled, follow: state.aruco_follow ? false : null }));
$("#follow-toggle").addEventListener("click", () => send({ type: "aruco", enabled: true, follow: !state.aruco_follow, speed_limit: Number(speed.value) }));
$("#heading-calibrate").addEventListener("click", () => send({ type: "heading-calibrate" }));
$("#controller-select")?.addEventListener("change", (event) => send({ type: "controller", controller: event.target.value }));
window.__followSmooth = false;
window.__viewFit = "contain";
$("#follow-mode-toggle")?.addEventListener("click", () => {
  window.__followSmooth = !window.__followSmooth;
  const el = $("#follow-mode-toggle");
  el.textContent = window.__followSmooth ? "SMOOTH ON" : "SMOOTH OFF";
  el.setAttribute("aria-pressed", String(window.__followSmooth));
  send({ type: "aruco", enabled: true, follow: state.aruco_follow || undefined, follow_mode: window.__followSmooth ? "holonomic" : "legacy" });
});
$("#view-fit-toggle")?.addEventListener("click", () => {
  window.__viewFit = window.__viewFit === "contain" ? "cover" : "contain";
  const feed = $("#camera-feed");
  if (feed) feed.style.objectFit = window.__viewFit;
  const el = $("#view-fit-toggle");
  el.textContent = `FIT: ${window.__viewFit.toUpperCase()}`;
  el.setAttribute("aria-pressed", String(window.__viewFit === "cover"));
});
speed.addEventListener("input", () => {
  speedValue.textContent = speed.value;
  $("#speed-meter").style.width = `${((speed.value - 500) / 1300) * 100}%`;
  if (state.aruco_follow && state.you_are_owner) send({ type: "aruco", enabled: true, speed_limit: Number(speed.value) });
  if (keys.size) drive();
});
function updateStick(event) {
  const rect = joystick.getBoundingClientRect();
  const radius = Math.min(rect.width, rect.height) * .38;
  const dx = event.clientX - (rect.left + rect.width / 2), dy = event.clientY - (rect.top + rect.height / 2);
  const magnitude = Math.hypot(dx, dy), scale = magnitude > radius ? radius / magnitude : 1;
  const x = dx * scale, y = dy * scale;
  stick = { strafe: applyJoyDeadzone(x / radius), forward: applyJoyDeadzone(-y / radius) };
  joystickKnob.style.transform = `translate(${x}px, ${y}px)`;
  if (state.armed && state.you_are_owner) drive();
}
joystick.addEventListener("pointerdown", event => { event.preventDefault(); stickPointer = event.pointerId; joystick.setPointerCapture(event.pointerId); updateStick(event); });
joystick.addEventListener("pointermove", event => { if (event.pointerId === stickPointer) updateStick(event); });
function releaseStick(event) { if (event.pointerId !== stickPointer) return; stickPointer = null; stick = { forward: 0, strafe: 0 }; joystickKnob.style.transform = "translate(0, 0)"; drive(); }
joystick.addEventListener("pointerup", releaseStick);
joystick.addEventListener("pointercancel", releaseStick);
window.addEventListener("blur", () => stop("window-blur"));
document.addEventListener("visibilitychange", () => { if (document.hidden) stop("hidden"); });
window.addEventListener("pagehide", () => stop("pagehide"));

connect();
