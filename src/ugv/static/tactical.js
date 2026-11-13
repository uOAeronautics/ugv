/* Tactical BEV radar: chassis, wheel vectors, gear skids, shield box, covariance, target+lead. */
(() => {
  const canvas = document.querySelector("#tactical-bev");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  let last = null;
  window.renderTactical = (state) => { last = state; draw(state); };
  function draw(next) {
    if (!next) return;
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#060b12";
    ctx.fillRect(0, 0, W, H);
    const extent = 1.5; // metres shown from centre
    const scale = (Math.min(W, H) / 2 - 14) / extent;
    const cx = W / 2, cy = H / 2;
    const toPx = (x, z) => [cx + x * scale, cy + z * scale];
    // grid
    ctx.strokeStyle = "#12283a";
    ctx.lineWidth = 1;
    for (let r = 0.5; r <= extent; r += 0.5) {
      ctx.beginPath();
      ctx.arc(cx, cy, r * scale, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.beginPath(); ctx.moveTo(cx, 6); ctx.lineTo(cx, H - 6);
    ctx.moveTo(6, cy); ctx.lineTo(W - 6, cy); ctx.stroke();
    // stopping-horizon safety shield box (rotated by heading)
    const heading = Number(next.heading_deg || 0) * Math.PI / 180;
    const margin = Number(next.swept_clear_margin_m);
    const shield = Number.isFinite(margin) ? Math.max(0.25, margin + 0.3) : 0.45;
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(-heading);
    ctx.strokeStyle = "#f59e0b";
    ctx.setLineDash([5, 4]);
    ctx.strokeRect(-0.25 * scale, -shield * scale, 0.5 * scale, shield * 2 * scale);
    ctx.setLineDash([]);
    ctx.restore();
    // UGV chassis
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(-heading);
    ctx.fillStyle = "#38bdf8";
    const cw = 0.22 * scale, cl = 0.32 * scale;
    ctx.fillRect(-cw / 2, -cl / 2, cw, cl);
    ctx.fillStyle = "#0c4a6e";
    ctx.fillRect(-cw / 2, -cl / 2 - 6, cw, 6); // front marker
    // 4 wheel vectors from PWM state
    const wheels = next.wheels || {};
    const order = ["front-left", "front-right", "rear-left", "rear-right"];
    const corners = [[-cw / 2, -cl / 2], [cw / 2, -cl / 2], [-cw / 2, cl / 2], [cw / 2, cl / 2]];
    ctx.strokeStyle = "#22c55e";
    ctx.lineWidth = 2;
    order.forEach((name, i) => {
      const pwm = Number(wheels[name] || 0) / 4095;
      const [wx, wy] = corners[i];
      ctx.beginPath();
      ctx.moveTo(wx, wy);
      ctx.lineTo(wx, wy - pwm * 40);
      ctx.stroke();
    });
    ctx.restore();
    // drone landing-gear skids (parking.gear)
    try {
      const gear = next.parking && next.parking.gear;
      if (gear && Number.isFinite(gear.clear_width_m)) {
        const gx = gear.clear_width_m / 2, gl = (gear.bar_length_m || 0.3) / 2;
        ctx.strokeStyle = "#94a3b8";
        ctx.lineWidth = Math.max(2, (gear.bar_radius_m || 0.02) * scale);
        [-gx, gx].forEach((sx) => {
          ctx.beginPath();
          ctx.moveTo(...toPx(sx, -gl));
          ctx.lineTo(...toPx(sx, gl));
          ctx.stroke();
        });
      }
    } catch { /* keep radar live when parking snapshot is partial */ }
    // EKF covariance ellipse
    const sigma = next.parking && next.parking.position_sigma_m;
    if (Number.isFinite(sigma) && sigma > 0) {
      ctx.strokeStyle = "#93c5fd";
      ctx.beginPath();
      ctx.ellipse(cx, cy, sigma * scale, sigma * 0.7 * scale, -heading, 0, Math.PI * 2);
      ctx.stroke();
    }
    // ArUco / follow target vector + lead prediction
    const err = Number(next.aruco_error_x);
    const dist = Number(next.aruco_distance_m);
    const fd = next.follow_debug || {};
    const errLead = Number.isFinite(fd.err_lead) ? fd.err_lead : err;
    const zLead = Number.isFinite(fd.z_lead) ? fd.z_lead : dist;
    if (next.aruco_visible && Number.isFinite(err) && Number.isFinite(dist)) {
      const tx = err * dist; // approx lateral metres
      const [px, py] = toPx(tx, -dist);
      ctx.strokeStyle = "#e11d48";
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(px, py); ctx.stroke();
      ctx.fillStyle = "#e11d48";
      ctx.beginPath(); ctx.arc(px, py, 4, 0, Math.PI * 2); ctx.fill();
      if (Number.isFinite(errLead) && Number.isFinite(zLead)) {
        const [lx, ly] = toPx(errLead * zLead, -zLead);
        ctx.strokeStyle = "#f472b6";
        ctx.setLineDash([4, 3]);
        ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(lx, ly); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = "#f472b6";
        ctx.beginPath(); ctx.arc(lx, ly, 3, 0, Math.PI * 2); ctx.fill();
      }
    }
    ctx.fillStyle = "#7dd3fc";
    ctx.font = "10px ui-monospace, monospace";
    ctx.fillText(`BEVis ${(fd.mode || "IDLE")}`, 8, 14);
  }
  // Re-draw on resize ticks even without new state.
  setInterval(() => { if (last) draw(last); }, 500);
})();
