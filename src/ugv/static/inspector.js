(() => {
  const fsmBar = document.querySelector("#fsm-bar");
  const fsmMeta = document.querySelector("#fsm-meta");
  const panel = document.querySelector("#audit-panel");
  const toggleBtn = document.querySelector("#audit-toggle");
  const closeBtn = document.querySelector("#audit-close");
  const headMeta = document.querySelector("#audit-head-meta");
  const scalarsEl = document.querySelector("#audit-scalars");
  const gatesEl = document.querySelector("#audit-gates");
  const logEl = document.querySelector("#audit-log");
  const countsEl = document.querySelector("#audit-counts");
  const scopeCanvas = document.querySelector("#audit-scope");
  if (!panel || !toggleBtn || !scopeCanvas) return;

  const ctx = scopeCanvas.getContext("2d");
  const ORDER = ["search", "acquire", "orbit", "approach", "enter", "parked"];
  const HISTORY_S = 12.0;
  const samples = []; // { t, w_cmd, gyro, y_cm, duty_mag, boost }
  let userToggled = false;
  let lastRunActive = false;

  function setOpen(open) {
    panel.hidden = !open;
    toggleBtn.setAttribute("aria-pressed", String(open));
    toggleBtn.classList.toggle("active", open);
  }

  toggleBtn.addEventListener("click", () => {
    userToggled = true;
    setOpen(panel.hidden);
  });
  closeBtn?.addEventListener("click", () => {
    userToggled = true;
    setOpen(false);
  });

  function updateFsmBar(next, dbg) {
    if (!fsmBar) return;
    const rawStage = String((dbg && dbg.stage) || next.auto_park_stage || "idle").toLowerCase();
    const active = Boolean(next.auto_park);
    const idx = ORDER.indexOf(rawStage);
    const isWarn = rawStage === "backoff";
    const isErr = rawStage === "abort" || rawStage === "aborted" || rawStage === "error";
    fsmBar.querySelectorAll(".fsm-node").forEach((el) => {
      const s = el.dataset.stage;
      const si = ORDER.indexOf(s);
      el.classList.remove("done", "active", "warn", "err");
      if (isErr && s === "search") {
        el.classList.add("err");
      } else if (isWarn && s === "approach") {
        el.classList.add("warn");
      } else if (idx >= 0) {
        if (si < idx) el.classList.add("done");
        else if (si === idx) el.classList.add("active");
      }
    });
    if (fsmMeta) {
      if (active && dbg) {
        fsmMeta.textContent = `${dbg.stage_age_s.toFixed(1)}s  ${dbg.reason || ""}`;
      } else if (next.auto_park_status) {
        fsmMeta.textContent = `${rawStage !== "idle" ? rawStage + " " : ""}${next.auto_park_status}`;
      } else {
        fsmMeta.textContent = "Idle";
      }
    }
  }

  function renderGates(dbg) {
    if (!gatesEl) return;
    const gates = (dbg && Array.isArray(dbg.gates)) ? dbg.gates : [];
    if (!gates.length) {
      gatesEl.innerHTML = `<div class="muted" style="font-size:11px">Start Auto Park to inspect live trigger gates.</div>`;
      return;
    }
    gatesEl.innerHTML = gates.map((g) => {
      const pct = Math.max(2, Math.min(100, Math.round((Number(g.frac) || 0) * 100)));
      const cls = g.ok ? "" : (pct > 80 ? "fail" : "warn");
      return `<div class="gate-row">
        <span class="gate-lbl">${g.label}</span>
        <div class="gate-track"><div class="gate-fill ${cls}" style="width:${pct}%"></div></div>
        <span class="gate-val mono">${g.val} <span class="gate-lim">${g.limit}</span></span>
      </div>`;
    }).join("");
  }

  function renderLog(dbg) {
    if (!logEl) return;
    const tr = (dbg && Array.isArray(dbg.transitions)) ? dbg.transitions : [];
    if (countsEl) countsEl.textContent = `${tr.length} events`;
    if (!tr.length) {
      logEl.innerHTML = `<div class="muted">No transitions yet</div>`;
      return;
    }
    logEl.innerHTML = tr.slice().reverse().map((e) => (
      `<div class="log-row"><span class="log-t">${Number(e.t).toFixed(2)}s</span><span class="log-edge">${e.from}→${e.to}</span><span class="log-why">${e.why || ""}</span></div>`
    )).join("");
  }

  function drawScope(dbg) {
    const dpr = devicePixelRatio || 1;
    const w = scopeCanvas.clientWidth || 336;
    const h = Math.round(w * (164 / 336));
    if (scopeCanvas.width !== Math.round(w * dpr) || scopeCanvas.height !== Math.round(h * dpr)) {
      scopeCanvas.width = Math.round(w * dpr);
      scopeCanvas.height = Math.round(h * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const mid1 = h * 0.28; // top band: yaw (w_cmd vs gyro, +-1.5 rad/s)
    const mid2 = h * 0.74; // bottom band: lateral y_cm (+-15 cm) + duty
    const amp1 = h * 0.22;
    const amp2 = h * 0.20;

    // Band divider & zero baselines
    ctx.strokeStyle = "#2b303c";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h * 0.52);
    ctx.lineTo(w, h * 0.52);
    ctx.stroke();

    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = "#3a4050";
    ctx.beginPath();
    ctx.moveTo(0, mid1); ctx.lineTo(w, mid1);
    ctx.moveTo(0, mid2); ctx.lineTo(w, mid2);
    ctx.stroke();
    ctx.setLineDash([]);

    // Shaded tolerance corridor for lateral y (+-lat_tol_m)
    const latTolCm = ((dbg && dbg.thresholds && dbg.thresholds.lat_tol_m) || 0.014) * 100;
    const tolH = (latTolCm / 15) * amp2;
    ctx.fillStyle = "rgba(47, 214, 123, 0.14)";
    ctx.fillRect(0, mid2 - tolH, w, tolH * 2);

    // Axis labels
    ctx.fillStyle = "#94a3b8";
    ctx.font = '500 9px "JetBrains Mono", monospace';
    ctx.fillText("YAW rad/s (±1.5)", 6, 11);
    ctx.fillText(`LAT cm (±15) · tol ±${latTolCm.toFixed(1)}`, 6, h * 0.52 + 11);

    if (samples.length < 2) return;
    const tMax = samples[samples.length - 1].t;
    const tMin = tMax - HISTORY_S;
    const xOf = (t) => ((t - tMin) / HISTORY_S) * w;

    function trace(fn, color, dashed, width) {
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      if (dashed) ctx.setLineDash([4, 3]);
      ctx.beginPath();
      let started = false;
      for (const s of samples) {
        const y = fn(s);
        if (y == null || !Number.isFinite(y)) { started = false; continue; }
        const x = xOf(s.t);
        if (!started) { ctx.moveTo(x, y); started = true; }
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.restore();
    }

    // Top band: w_cmd (dashed white) vs gyro (solid cyan)
    trace((s) => mid1 - Math.max(-1.5, Math.min(1.5, s.w_cmd)) / 1.5 * amp1, "#e2e8f0", true, 1.3);
    trace((s) => mid1 - Math.max(-1.5, Math.min(1.5, s.gyro)) / 1.5 * amp1, "#38bdf8", false, 1.8);

    // Bottom band: lateral y_cm (amber) and duty magnitude (green)
    trace((s) => s.y_cm == null ? null : mid2 - Math.max(-15, Math.min(15, s.y_cm)) / 15 * amp2, "#f5a623", false, 1.8);
    trace((s) => mid2 - Math.max(-1, Math.min(1, s.duty_mag)) * amp2 * 0.85, "#2fd67b", false, 1.2);
  }

  window.renderAudit = (next) => {
    if (!next) return;
    const dbg = next.auto_park_debug || null;
    updateFsmBar(next, dbg);

    // Auto-open the inspector when Auto Park starts unless the user explicitly closed it
    const active = Boolean(next.auto_park);
    if (active && !lastRunActive && !userToggled) setOpen(true);
    lastRunActive = active;

    if (dbg && active) {
      const now = performance.now() / 1000;
      const wCmd = Array.isArray(dbg.cmd_vel) ? Number(dbg.cmd_vel[2]) || 0 : 0;
      const gyro = Number(dbg.gyro_ccw) || 0;
      const yCm = Array.isArray(dbg.pose) ? Number(dbg.pose[1]) * 100 : null;
      const dutyMag = Array.isArray(dbg.duty)
        ? Math.hypot(Number(dbg.duty[0]) || 0, Number(dbg.duty[1]) || 0) / 0.44
        : 0;
      samples.push({ t: now, w_cmd: wCmd, gyro, y_cm: yCm, duty_mag: dutyMag });
      while (samples.length > 0 && now - samples[0].t > HISTORY_S + 1) samples.shift();
    }

    if (panel.hidden) return;
    if (headMeta) {
      headMeta.textContent = dbg
        ? `${dbg.stage} · ${dbg.stage_age_s.toFixed(1)}s`
        : "Idle";
    }
    if (scalarsEl) {
      scalarsEl.textContent = dbg
        ? `boost ${Number(dbg.trans_boost || 1).toFixed(2)}× · innov ${Number(dbg.est_innov || 0).toFixed(2)}m · shields ${dbg.shield_hits || 0}`
        : "boost 1.00×";
    }
    renderGates(dbg);
    renderLog(dbg);
    drawScope(dbg);
  };
})();
