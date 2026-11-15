// Tab 2: training curves. Six cards in a 2x3 grid; each card holds one panel per
// scale so SPS (~1.7e5) never flattens std (~0.01). A shared crosshair links every panel.

const CV = {
  runs: [],          // from /api/curves/runs, newest first
  selected: [],      // run ids, colour index = position here
  series: {},        // run id -> { format, points }
  search: '',
  hover: null,       // iteration under the cursor, or null
  tipAnchor: null,   // pointer position for the floating tooltip
};

// Each panel: series keys are read in order (first present wins), `rate` marks
// fraction metrics that only jsonl runs can supply (log counts are not rates).
const CHART_CARDS = [
  {
    id: 'rates',
    title: 'Success & collision rate',
    desc: 'Fraction of episodes per update · 0–100%',
    panels: [
      {
        domain: [0, 1],
        pct: true,
        series: [
          { keys: ['succ_rate', 'succ'], label: 'success', dash: null, rate: true },
          { keys: ['coll_rate', 'coll'], label: 'collision', dash: [5, 4], rate: true },
        ],
      },
    ],
  },
  {
    id: 'dr',
    title: 'Competence-gated ADR scale',
    desc: 'dr_scale over updates · 0.0 – 1.0+',
    panels: [{ domain: 'zero', series: [{ keys: ['dr_scale'], label: 'dr_scale' }] }],
  },
  {
    id: 'ppo',
    title: 'Policy & value loss',
    desc: 'Each loss on its own scale',
    panels: [
      { title: 'pol_loss', domain: 'auto', series: [{ keys: ['pol_loss', 'pol'], label: 'pol_loss' }] },
      { title: 'val_loss', domain: 'auto', series: [{ keys: ['val_loss', 'val'], label: 'val_loss' }] },
    ],
  },
  {
    id: 'aux',
    title: 'Auxiliary pose & extrinsics loss',
    desc: 'aux_loss and dyn_loss differ by ~100×, so each has its own scale',
    panels: [
      { title: 'aux_loss', domain: 'auto', series: [{ keys: ['aux_loss', 'aux'], label: 'aux_loss' }] },
      { title: 'dyn_loss', domain: 'auto', series: [{ keys: ['dyn_loss', 'dyn'], label: 'dyn_loss' }] },
    ],
  },
  {
    id: 'std',
    title: 'Exploration action std',
    desc: 'Policy action standard deviation',
    panels: [{ domain: 'zero', series: [{ keys: ['std'], label: 'std' }] }],
  },
  {
    id: 'ret',
    title: 'Mean episode return & throughput',
    desc: 'return and sps are plotted on separate scales',
    panels: [
      { title: 'mean_return', domain: 'auto', series: [{ keys: ['return', 'mean_return'], label: 'return' }] },
      { title: 'sps (steps / s)', domain: 'zero', series: [{ keys: ['sps'], label: 'sps' }] },
    ],
  },
];

function runColor(rid) {
  const i = CV.selected.indexOf(rid);
  return palette().series[(i < 0 ? 0 : i) % 8];
}

async function loadCurveRuns() {
  const d = await fetchJSON('/api/curves/runs');
  CV.runs = d?.runs || [];
  if (!CV.selected.length && CV.runs.length) CV.selected = CV.runs.slice(0, 3).map((r) => r.run_id);
  renderRunList();
  return loadCurveSeries();
}

async function loadCurveSeries() {
  if (!CV.selected.length) {
    CV.series = {};
    renderLegend();
    drawCurves();
    return;
  }
  const d = await fetchJSON(`/api/curves?runs=${encodeURIComponent(CV.selected.join(','))}`);
  CV.series = d?.series || {};
  renderLegend();
  drawCurves();
}

function renderRunList() {
  const box = $('#run-list');
  const q = CV.search;
  const rows = CV.runs.filter((r) => !q || r.run_id.toLowerCase().includes(q));
  if (!rows.length) {
    box.innerHTML = '<p class="empty">No training runs found.</p>';
    return;
  }
  box.innerHTML = rows
    .map((r) => {
      const on = CV.selected.includes(r.run_id);
      const color = on ? runColor(r.run_id) : 'transparent';
      const meta = `${r.n_iters ?? 0} updates${r.active ? ' · active' : ''}${r.has_log && r.format === 'jsonl' ? ' · + .log' : ''}`;
      return `<label class="run-row">
        <input type="checkbox" value="${esc(r.run_id)}" ${on ? 'checked' : ''} aria-label="Plot ${esc(r.run_id)}">
        <span class="run-swatch" style="background:${color};${on ? '' : 'border:1px solid var(--border-strong)'}"></span>
        <span class="run-main"><span class="run-name" title="${esc(r.run_id)}">${esc(r.run_id)}</span><span class="run-meta">${esc(meta)}</span></span>
        <span class="run-badges"><span class="badge ${r.format === 'jsonl' ? 'jsonl' : 'log'}">${r.format === 'jsonl' ? 'jsonl' : 'log'}</span></span>
      </label>`;
    })
    .join('');
  box.querySelectorAll('input[type=checkbox]').forEach((cb) =>
    cb.addEventListener('change', () => {
      CV.selected = cb.checked
        ? [...CV.selected, cb.value]
        : CV.selected.filter((id) => id !== cb.value);
      renderRunList();
      loadCurveSeries();
    }),
  );
}

function renderLegend() {
  const box = $('#run-legend');
  if (!CV.selected.length) {
    box.innerHTML = '<span class="legend-title">Runs</span><span class="muted">Select runs in the sidebar to plot them.</span>';
    return;
  }
  box.innerHTML =
    '<span class="legend-title">Runs</span>' +
    CV.selected
      .map((rid) => {
        const fmtTag = CV.series[rid]?.format || '';
        return `<span class="chart-legend" style="gap:6px"><i class="run-swatch" style="display:inline-block;background:${runColor(rid)}"></i><span class="mono">${esc(rid)}</span>${fmtTag ? `<span class="badge ${fmtTag === 'jsonl' ? 'jsonl' : 'log'}">${fmtTag}</span>` : ''}</span>`;
      })
      .join('');
}

// ---- card DOM ----
function buildChartGrid() {
  const grid = $('#chart-grid');
  grid.innerHTML = CHART_CARDS.map(
    (card) => `
    <article class="card chart-card" data-card="${card.id}">
      <div class="chart-head">
        <span class="chart-title">${esc(card.title)}</span>
        <span class="chart-desc">${esc(card.desc)}</span>
        <div class="chart-legend">${card.panels
          .flatMap((p) => p.series)
          .map((s) => `<span><i class="dash ${s.dash ? 'dashed' : ''}"></i>${esc(s.label)}</span>`)
          .join('')}</div>
      </div>
      ${card.panels
        .map(
          (p, pi) => `${p.title ? `<span class="panel-label">${esc(p.title)}</span>` : ''}
          <canvas class="plot" data-panel="${pi}" aria-label="${esc(card.title)}"></canvas>`,
        )
        .join('')}
      <span class="chart-note" data-note="${card.id}" hidden></span>
    </article>`,
  ).join('');
  grid.querySelectorAll('canvas.plot').forEach((c) => {
    c.addEventListener('mousemove', (e) => onPlotMove(e, c));
    c.addEventListener('mouseleave', onPlotLeave);
  });
}

// ---- data shaping ----
function pointsFor(rid, keys, rate) {
  const s = CV.series[rid];
  if (!s) return null;
  if (rate && s.format !== 'jsonl') return null; // log-format success/collision are counts
  return (s.points || [])
    .map((p) => {
      let v = null;
      for (const k of keys) {
        if (isNum(p[k])) {
          v = Number(p[k]);
          break;
        }
      }
      return v === null ? null : { x: Number(p.iter), y: v };
    })
    .filter((p) => p && Number.isFinite(p.x));
}

function iterDomain() {
  let lo = Infinity;
  let hi = -Infinity;
  for (const rid of CV.selected) {
    for (const p of CV.series[rid]?.points || []) {
      const x = Number(p.iter);
      if (Number.isFinite(x)) {
        if (x < lo) lo = x;
        if (x > hi) hi = x;
      }
    }
  }
  if (!Number.isFinite(lo)) return [0, 1];
  return hi > lo ? [lo, hi] : [lo, lo + 1];
}

function yDomain(panel, values) {
  if (panel.domain && Array.isArray(panel.domain)) return panel.domain;
  let lo = Infinity;
  let hi = -Infinity;
  values.forEach((v) => {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  });
  if (!Number.isFinite(lo)) return [0, 1];
  if (panel.domain === 'zero') {
    lo = Math.min(0, lo);
    hi = Math.max(hi * 1.08, 1e-9);
    return [lo, hi];
  }
  if (hi - lo < 1e-12) return [lo - 0.5, hi + 0.5];
  const pad = (hi - lo) * 0.08;
  return [lo - pad, hi + pad];
}

function buildCurvesControls() {
  $('#run-search').addEventListener('input', (e) => {
    CV.search = e.target.value.trim().toLowerCase();
    renderRunList();
  });
  $('#sel-top3').addEventListener('click', () => {
    CV.selected = CV.runs.slice(0, 3).map((r) => r.run_id);
    renderRunList();
    loadCurveSeries();
  });
  $('#sel-none').addEventListener('click', () => {
    CV.selected = [];
    renderRunList();
    loadCurveSeries();
  });
  buildChartGrid();
  renderLegend();
}
