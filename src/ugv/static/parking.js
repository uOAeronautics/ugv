(() => {
  const panel = document.querySelector("#parking-panel");
  const canvas = document.querySelector("#parking-map");
  const statusEl = document.querySelector("#parking-status");
  const readoutEl = document.querySelector("#parking-readout");
  const mapToggle = document.querySelector("#map-toggle");
  const mapClose = document.querySelector("#map-close");
  if (!panel || !canvas || !mapToggle) return;
  const ctx = canvas.getContext("2d");

  const trail = [];
  const MAX_TRAIL = 80;

  function setMapOpen(open) {
    panel.hidden = !open;
    mapToggle.setAttribute("aria-pressed", String(open));
    mapToggle.classList.toggle("active", open);
    if (open && typeof send === "function" && typeof state === "object" && !state.plane_enabled && !state.auto_park) {
      send({ type: "plane", enabled: true });
    }
  }

  mapToggle.addEventListener("click", () => setMapOpen(panel.hidden));
  mapClose?.addEventListener("click", () => setMapOpen(false));

  // Invert airframe-in-car pose (x, y, yaw_rad) -> car-in-airframe pose (x, y, yaw_rad).
  function carInAirframe(next) {
    if (Array.isArray(next.auto_park_pose) && next.auto_park_pose.length >= 3) {
      return next.auto_park_pose;
    }
    const p = next.plane_pose;
    if (p && Number.isFinite(p.x_m) && Number.isFinite(p.y_m) && Number.isFinite(p.yaw_deg)) {
      const yaw = (p.yaw_deg * Math.PI) / 180;
      const c = Math.cos(yaw);
      const s = Math.sin(yaw);
      return [-c * p.x_m - s * p.y_m, s * p.x_m - c * p.y_m, -yaw];
    }
    return null;
  }

  window.renderParking = (_legacy, next) => {
    if (!next || panel.hidden) return;
    const pose = carInAirframe(next);
    const stage = next.auto_park && next.auto_park_stage && next.auto_park_stage !== "idle"
      ? next.auto_park_stage.toUpperCase()
      : (pose ? "LIVE" : "NO LOCK");
    if (statusEl) statusEl.textContent = stage;

    if (pose) {
      const last = trail[trail.length - 1];
      if (!last || Math.hypot(pose[0] - last[0], pose[1] - last[1]) > 0.015) {
        trail.push([pose[0], pose[1]]);
        if (trail.length > MAX_TRAIL) trail.shift();
      }
      const deg = (((pose[2] * 180) / Math.PI + 540) % 360) - 180;
      if (readoutEl) {
        readoutEl.textContent = `x ${pose[0] >= 0 ? "+" : ""}${pose[0].toFixed(2)} m · y ${pose[1] >= 0 ? "+" : ""}${pose[1].toFixed(2)} m · ${deg >= 0 ? "+" : ""}${deg.toFixed(0)}°`;
      }
    } else if (readoutEl) {
      readoutEl.textContent = "Waiting for airframe…";
    }

    const dpr = devicePixelRatio || 1;
    const cssW = canvas.clientWidth || 260;
    const cssH = Math.round(cssW * (220 / 260));
    if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {
      canvas.width = Math.round(cssW * dpr);
      canvas.height = Math.round(cssH * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    // Airframe frame: +x nose UP, +y left LEFT. Center slightly up so rear hub/approach fits nicely.
    const dist = pose ? Math.hypot(pose[0], pose[1]) : 0.8;
    const spanM = Math.max(1.15, Math.min(2.4, dist + 0.45));
    const scale = Math.min(cssW, cssH) / (2 * spanM);
    const cx = cssW * 0.5;
    const cy = cssH * 0.44;
    const toPx = (ax, ay) => [cx - ay * scale, cy - ax * scale];

    // Range rings (0.5 m, 1.0 m) + hub_rear orbit ring (0.52 m)
    ctx.save();
    for (const [r, dash, color] of [
      [1.0, [2, 4], "rgba(148, 163, 184, 0.16)"],
      [0.52, [4, 4], "rgba(56, 189, 248, 0.28)"],
    ]) {
      ctx.beginPath();
      ctx.setLineDash(dash);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.arc(cx, cy, r * scale, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    // Target bay box between main gear (-0.165) and nose wheel (+0.137)
    const [bx0, by0] = toPx(0.08, 0.08);
    const [bx1, by1] = toPx(-0.13, -0.08);
    ctx.fillStyle = "rgba(52, 211, 153, 0.16)";
    ctx.strokeStyle = "rgba(52, 211, 153, 0.7)";
    ctx.lineWidth = 1.25;
    ctx.fillRect(bx0, by0, bx1 - bx0, by1 - by0);
    ctx.strokeRect(bx0, by0, bx1 - bx0, by1 - by0);

    // STRL-001 airframe silhouette (metres in model frame)
    ctx.strokeStyle = "rgba(226, 232, 240, 0.72)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    // Fuselage nose (+0.19) to tail (-0.65)
    ctx.moveTo(...toPx(0.19, 0));
    ctx.lineTo(...toPx(-0.65, 0));
    // Wingspan (-0.61 to +0.61 at x=-0.05)
    ctx.moveTo(...toPx(-0.05, 0.61));
    ctx.lineTo(...toPx(-0.05, -0.61));
    // V-tail tips
    ctx.moveTo(...toPx(-0.65, 0));
    ctx.lineTo(...toPx(-0.70, 0.16));
    ctx.moveTo(...toPx(-0.65, 0));
    ctx.lineTo(...toPx(-0.70, -0.16));
    ctx.stroke();

    // Landing gear wheels (nose + main left/right)
    for (const [gx, gy] of [[0.137, 0], [-0.165, 0.131], [-0.164, -0.136]]) {
      const [px, py] = toPx(gx, gy);
      ctx.fillStyle = "#f8fafc";
      ctx.beginPath();
      ctx.arc(px, py, 3, 0, Math.PI * 2);
      ctx.fill();
    }

    // Trail
    if (trail.length > 1) {
      ctx.strokeStyle = "rgba(56, 189, 248, 0.45)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      trail.forEach(([tx, ty], i) => {
        const [px, py] = toPx(tx, ty);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.stroke();
    }

    // Live car footprint (0.203 m L x 0.146 m W)
    if (pose) {
      const [px, py] = toPx(pose[0], pose[1]);
      ctx.save();
      ctx.translate(px, py);
      // In canvas: +x is up (-Y), +y is left (-X), so CCW yaw rotates by -pose[2].
      ctx.rotate(-pose[2]);
      const cw = 0.146 * scale;
      const cl = 0.203 * scale;
      ctx.fillStyle = "rgba(56, 189, 248, 0.30)";
      ctx.strokeStyle = "#38bdf8";
      ctx.lineWidth = 1.8;
      ctx.fillRect(-cw / 2, -cl / 2, cw, cl);
      ctx.strokeRect(-cw / 2, -cl / 2, cw, cl);
      // Front heading triangle
      ctx.fillStyle = "#38bdf8";
      ctx.beginPath();
      ctx.moveTo(0, -cl / 2 - 4);
      ctx.lineTo(-4, -cl / 2 + 2);
      ctx.lineTo(4, -cl / 2 + 2);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }
    ctx.restore();
  };
})();
