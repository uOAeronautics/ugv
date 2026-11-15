// Curve rendering: per-panel canvas drawing, the shared hover cursor and the floating tooltip.
// Reads CHART_CARDS, CV and the series helpers defined in curves.js.

// ---- drawing ----
const PAD = { l: 50, r: 10, t: 8, b: 18 };

function drawCurves() {
  const [x0, x1] = iterDomain();
  CHART_CARDS.forEach((card) => {
    const cardEl = document.querySelector(`[data-card="${card.id}"]`);
    if (!cardEl) return;
    const omitted = CV.selected.filter((rid) => CV.series[rid] && CV.series[rid].format === 'log' && card.id === 'rates');
    const note = cardEl.querySelector('[data-note]');
    note.hidden = !omitted.length;
    note.textContent = omitted.length
      ? `${omitted.length} .log run${omitted.length > 1 ? 's' : ''} omitted: their success/collision are episode counts, not rates.`
      : '';
    cardEl.querySelectorAll('canvas.plot').forEach((canvas) => {
      const pi = Number(canvas.dataset.panel);
      drawPanel(canvas, card, card.panels[pi], [x0, x1]);
    });
  });
  showCrosshairTip();
}

function drawPanel(canvas, card, panel, [x0, x1]) {
  const pal = palette();
  const { ctx, w, h } = fitCanvas(canvas);
  ctx.fillStyle = pal.plot;
  ctx.fillRect(0, 0, w, h);
  const plotW = w - PAD.l - PAD.r;
  const plotH = h - PAD.t - PAD.b;

  const lines = [];
  panel.series.forEach((s, si) => {
    CV.selected.forEach((rid) => {
      const pts = pointsFor(rid, s.keys, s.rate);
      if (pts && pts.length) lines.push({ rid, si, s, pts, color: runColor(rid) });
    });
  });
  const allY = lines.flatMap((l) => l.pts.map((p) => p.y));
  const [yLo, yHi] = yDomain(panel, allY);
  const X = (x) => PAD.l + ((x - x0) / (x1 - x0)) * plotW;
  const Y = (y) => PAD.t + plotH - ((y - yLo) / (yHi - yLo)) * plotH;

  ctx.font = MONO_SM;
  ctx.textBaseline = 'middle';
  for (let i = 0; i <= 3; i++) {
    const v = yLo + ((yHi - yLo) * i) / 3;
    const y = Y(v);
    ctx.strokeStyle = pal.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(PAD.l, Math.round(y) + 0.5);
    ctx.lineTo(w - PAD.r, Math.round(y) + 0.5);
    ctx.stroke();
    ctx.fillStyle = pal.muted;
    ctx.textAlign = 'right';
    ctx.fillText(panel.pct ? `${Math.round(v * 100)}%` : fmt.axis(v), PAD.l - 6, y);
  }
  ctx.textBaseline = 'alphabetic';
  ctx.textAlign = 'center';
  ctx.fillStyle = pal.muted;
  for (let i = 0; i <= 4; i++) {
    const v = x0 + ((x1 - x0) * i) / 4;
    ctx.fillText(String(Math.round(v)), X(v), h - 4);
  }

  if (!lines.length) {
    ctx.fillStyle = pal.muted;
    ctx.textAlign = 'center';
    const missing = panel.series.map((s) => s.label).join(' / ');
    const msg = CV.selected.length ? `none of the selected runs logged ${missing}` : 'select runs to plot';
    ctx.fillText(msg, w / 2, h / 2);
    return;
  }

  ctx.save();
  ctx.beginPath();
  ctx.rect(PAD.l, PAD.t, plotW, plotH);
  ctx.clip();
  lines.forEach((l) => {
    ctx.strokeStyle = l.color;
    ctx.lineWidth = 1.6;
    ctx.lineJoin = 'round';
    ctx.setLineDash(l.s.dash || []);
    ctx.beginPath();
    l.pts.forEach((p, i) => (i === 0 ? ctx.moveTo(X(p.x), Y(p.y)) : ctx.lineTo(X(p.x), Y(p.y))));
    ctx.stroke();
  });
  ctx.setLineDash([]);
  ctx.restore();

  if (CV.hover !== null && CV.hover >= x0 && CV.hover <= x1) {
    const hx = Math.round(X(CV.hover)) + 0.5;
    ctx.strokeStyle = pal.ink2;
    ctx.globalAlpha = 0.55;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(hx, PAD.t);
    ctx.lineTo(hx, PAD.t + plotH);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
    lines.forEach((l) => {
      const p = nearest(l.pts, CV.hover);
      if (!p) return;
      ctx.fillStyle = l.color;
      ctx.beginPath();
      ctx.arc(X(p.x), Y(p.y), 3, 0, Math.PI * 2);
      ctx.fill();
    });
  }
}

function nearest(pts, x) {
  let best = null;
  let bd = Infinity;
  for (const p of pts) {
    const d = Math.abs(p.x - x);
    if (d < bd) {
      bd = d;
      best = p;
    }
  }
  return best;
}

// ---- hover: one shared iteration cursor across all cards ----
function onPlotMove(e, canvas) {
  const [x0, x1] = iterDomain();
  const rect = canvas.getBoundingClientRect();
  const plotW = rect.width - PAD.l - PAD.r;
  const frac = Math.max(0, Math.min(1, (e.clientX - rect.left - PAD.l) / Math.max(1, plotW)));
  CV.hover = x0 + frac * (x1 - x0);
  CV.tipAnchor = { clientX: e.clientX, clientY: e.clientY };
  scheduleRedraw();
}

function onPlotLeave() {
  CV.hover = null;
  CV.tipAnchor = null;
  scheduleRedraw();
}

let redrawQueued = false;
function scheduleRedraw() {
  if (redrawQueued) return;
  redrawQueued = true;
  requestAnimationFrame(() => {
    redrawQueued = false;
    drawCurves();
  });
}

function showCrosshairTip() {
  const tip = $('#chart-tip');
  if (CV.hover === null || !CV.tipAnchor || !CV.selected.length) {
    tip.hidden = true;
    return;
  }
  const iter = Math.round(CV.hover);
  const rows = CHART_CARDS.map((card) => {
    const lines = [];
    card.panels.forEach((panel) =>
      panel.series.forEach((s) =>
        CV.selected.forEach((rid) => {
          const pts = pointsFor(rid, s.keys, s.rate);
          if (!pts || !pts.length) return;
          const p = nearest(pts, iter);
          if (!p) return;
          const val = panel.pct ? fmt.pct(p.y, 1) : fmt.prec(p.y);
          lines.push(`<div class="tip-row"><span>${esc(rid)} · ${esc(s.label)}</span><span>${val}</span></div>`);
        }),
      ),
    );
    if (!lines.length) return '';
    return `<div class="tip-card"><div class="tip-card-title">${esc(card.title)}</div>${lines.join('')}</div>`;
  }).join('');
  tip.innerHTML = `<h4>update ${iter}</h4>${rows || '<span class="muted">no values at this update</span>'}`;
  tip.hidden = false;
  const host = tip.parentElement.getBoundingClientRect();
  const x = CV.tipAnchor.clientX - host.left + 16;
  const y = CV.tipAnchor.clientY - host.top + 16;
  const flip = x + 340 > host.width;
  tip.style.left = `${flip ? x - 356 : x}px`;
  tip.style.top = `${Math.max(0, y)}px`;
}
