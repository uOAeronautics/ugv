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
  if (!next.aruco_visible) return;

  const image = $("#camera-feed");
  const naturalWidth = image.naturalWidth || 640;
  const naturalHeight = image.naturalHeight || 480;
  const fit = image.style.objectFit || window.__viewFit || "cover";
  const scale = fit === "cover"
    ? Math.max(width / naturalWidth, height / naturalHeight)
    : Math.min(width / naturalWidth, height / naturalHeight);
  const x0 = (width - naturalWidth * scale) / 2;
  const y0 = (height - naturalHeight * scale) / 2;
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

  const tagSummary = next.aruco_visible
    ? `#${next.aruco_id ?? "?"} · ${Number(next.aruco_distance_m ?? 0).toFixed(2)}m`
    : (next.heading_calibrated ? `${Math.round(((Number(next.heading_deg) || 0) % 360 + 360) % 360)}°` : (next.aruco_status || "—"));
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

connect();
