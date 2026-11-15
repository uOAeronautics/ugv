// Tab 1: leaderboard, stage pipeline cards, track/source filters, diff panel, real-Pi table.

const LB = {
  entries: [],
  l7: [],
  stage: 'ALL',
  track: 'ALL',
  pareto: false,
  search: '',
  sort: { col: null, dir: 1 },
  diff: [],
};

const isGz = (e) => e.track === 'gazebo_l6' || e.source === 'gazebo_l6';
const isL7 = (e) => e.level === 'L7' || e.track === 'l7' || e.source === 'l7';
const isSimLevel = (e, id) => e.level === id && !isGz(e) && !isL7(e);

// Stage matchers. Gazebo and real-Pi rows are never counted in the sim-level stages.
const STAGE_MATCH = {
  L0: (e) => isSimLevel(e, 'L0'),
  L1: (e) => isSimLevel(e, 'L1'),
  L2: (e) => isSimLevel(e, 'L2'),
  L3: (e) => isSimLevel(e, 'L3'),
  L4: (e) => isSimLevel(e, 'L4'),
  L5: (e) => isSimLevel(e, 'L5'),
  L_real: (e) => isSimLevel(e, 'L_real'),
  L_chaos: (e) => isSimLevel(e, 'L_chaos'),
  L6: (e) => e.level === 'L6' || isGz(e),
  L7: (e) => isL7(e),
};

const PIPELINE = [
  { id: 'L0', title: 'Analytic Nominal' },
  { id: 'L1', title: '80 ms Latency + Noise' },
  { id: 'L2', title: '60-Particle Dynamics' },
  { id: 'L3', title: 'Dropouts + 3 s Blackout' },
  { id: 'L_real', title: '0.40 m Blind Ring' },
  { id: 'L_chaos', title: '1.25× Stress + Faults' },
  { id: 'L6', title: 'Gazebo Harmonic 3D Physics' },
  { id: 'L7', title: 'Real Raspberry Pi 4B' },
];

const STAGE_FILTERS = ['ALL', 'L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L_real', 'L_chaos', 'L6', 'L7'];
const STAGE_LABEL = { L6: 'L6 (Gazebo)', L7: 'L7 (Real Pi)' };
const TRACK_FILTERS = ['ALL', 'student', 'teacher', 'baseline', 'gazebo_l6', 'l7', 'archive'];
const SOURCE_TEXT = { sim: 'VecSim', gazebo_l6: 'Gazebo L6', l7: 'Real Pi L7' };

function stageOf(e) {
  if (isL7(e)) return 'L7';
  if (isGz(e) || e.level === 'L6') return 'L6';
  return e.level || '—';
}

function passesFilters(e) {
  if (LB.stage !== 'ALL' && !(STAGE_MATCH[LB.stage] || (() => false))(e)) return false;
  if (LB.track !== 'ALL' && !(e.track === LB.track || e.source === LB.track)) return false;
  if (LB.pareto && !e.pareto) return false;
  if (LB.search && !String(e.run_id || '').toLowerCase().includes(LB.search)) return false;
  return true;
}

function buildLeaderboardControls() {
  const mk = (box, items, key, label) => {
    box.innerHTML = '';
    items.forEach((v) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.textContent = STAGE_LABEL[v] || v;
      b.dataset[key] = v;
      b.setAttribute('aria-pressed', String(LB[key] === v));
      b.addEventListener('click', () => {
        LB[key] = LB[key] === v && key === 'stage' ? 'ALL' : v;
        syncFilterButtons();
        renderLeaderboard();
      });
      box.appendChild(b);
    });
    box.setAttribute('aria-label', label);
  };
  mk($('#stage-seg'), STAGE_FILTERS, 'stage', 'Stage or level');
  mk($('#track-seg'), TRACK_FILTERS, 'track', 'Track or source');
  $('#pareto-only').addEventListener('change', (e) => {
    LB.pareto = e.target.checked;
    renderLeaderboard();
  });
  $('#search-box').addEventListener('input', (e) => {
    LB.search = e.target.value.trim().toLowerCase();
    renderLeaderboard();
  });
  $$('#leaderboard-table thead th[data-col]').forEach((th) =>
    th.addEventListener('click', () => {
      const col = th.dataset.col;
      LB.sort = LB.sort.col === col ? { col, dir: -LB.sort.dir } : { col, dir: 1 };
      renderLeaderboard();
    }),
  );
}

function syncFilterButtons() {
  $$('#stage-seg button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.stage === LB.stage)));
  $$('#track-seg button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.track === LB.track)));
}

async function loadLeaderboard() {
  const d = await fetchJSON('/api/leaderboard');
  LB.entries = d?.entries || [];
  LB.l7 = d?.l7_runs || [];
  renderLeaderboard();
  renderL7Table();
}

function renderLeaderboard() {
  syncFilterButtons();
  renderPipeline();
  renderTable();
  renderDiff();
}

// ---- pipeline cards ----
function renderPipeline() {
  const box = $('#pipeline');
  box.innerHTML = '';
  PIPELINE.forEach((st) => {
    const subset = LB.entries.filter((e) => STAGE_MATCH[st.id](e));
    // rank by the Wilson lower bound, so a 1/1 run cannot outrank 97/100
    const rated = subset.filter((e) => e.has_rate);
    const lb = (e) => (isNum(e.wilson_lo) ? Number(e.wilson_lo) : Number(e.success_rate ?? 0));
    const best = rated.reduce((a, b) => (!a || lb(b) > lb(a) ? b : a), null);
    const isL7Stage = st.id === 'L7';
    let value = '—';
    let sub = 'no runs yet';
    let bestLine = '';
    if (isL7Stage && subset.length) {
      const succ = subset.reduce((s, e) => s + (e.success || 0), 0);
      const eps = subset.reduce((s, e) => s + (e.episodes || 0), 0);
      const arrived = subset.filter((e) => e.outcome === 'arrived').length;
      value = eps ? fmt.pct(succ / eps, 0) : '—';
      sub = `${succ}/${eps} confirmed park · ${arrived} 'arrived' unconfirmed`;
      bestLine = subset[subset.length - 1].run_id;
    } else if (best) {
      const succ = best.success ?? 0;
      const eps = best.episodes ?? 0;
      value = fmt.pct(best.success_rate, 1);
      sub = `best ${succ}/${eps} · ${subset.length} run${subset.length === 1 ? '' : 's'}`;
      bestLine = best.run_id;
    } else if (subset.length) {
      sub = `${subset.length} run${subset.length === 1 ? '' : 's'} (no rate)`;
    }
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `stage${LB.stage === st.id ? ' active' : ''}${subset.length ? '' : ' empty'}`;
    btn.setAttribute('aria-pressed', String(LB.stage === st.id));
    btn.title = `Filter the leaderboard to ${st.id}`;
    btn.innerHTML = `
      <span class="stage-id">${esc(st.id)}</span>
      <span class="stage-title">${esc(st.title)}</span>
      <span class="stage-value">${value}</span>
      <span class="stage-sub">${esc(sub)}</span>
      <span class="stage-best">${esc(bestLine)}</span>`;
    btn.addEventListener('click', () => {
      LB.stage = LB.stage === st.id ? 'ALL' : st.id;
      renderLeaderboard();
      $('#leaderboard-table').scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    });
    box.appendChild(btn);
  });
}

// ---- table ----
function sortVal(e, col) {
  switch (col) {
    case 'stage':
      return stageOf(e);
    case 'source':
      return sourceOf(e);
    case 'success_rate':
    case 'wilson_lo':
    case 'collisions':
    case 'timeouts':
    case 'time_median':
    case 'pos_p95':
    case 'yaw_p95_deg':
      return e[col] ?? null;
    default:
      return e[col] ?? '';
  }
}

function sortRows(rows) {
  const col = LB.sort.col;
  if (!col) return rows;
  const dir = LB.sort.dir;
  return [...rows].sort((a, b) => {
    const va = sortVal(a, col);
    const vb = sortVal(b, col);
    // nulls always sink to the bottom regardless of direction
    if (va === null && vb === null) return 0;
    if (va === null) return 1;
    if (vb === null) return -1;
    if (typeof va === 'number' && typeof vb === 'number') return (va - vb) * dir;
    return String(va).localeCompare(String(vb)) * dir;
  });
}

function ciCell(e) {
  if (!isNum(e.wilson_lo) || !isNum(e.wilson_hi)) return '<span class="muted">—</span>';
  const lo = Number(e.wilson_lo);
  const hi = Number(e.wilson_hi);
  return `<span class="ci" title="Wilson 95% interval">
    <span class="ci-track" aria-hidden="true"><span class="ci-fill" style="left:${(lo * 100).toFixed(1)}%;width:${Math.max(0.5, (hi - lo) * 100).toFixed(1)}%"></span></span>
    <span class="ci-text">[${lo.toFixed(2)}, ${hi.toFixed(2)}]</span></span>`;
}

function renderTable() {
  const tbody = $('#leaderboard-table tbody');
  const rows = sortRows(LB.entries.filter(passesFilters));
  $('#leaderboard-empty').hidden = rows.length > 0;
  tbody.innerHTML = rows
    .map((e) => {
      const src = sourceOf(e);
      const rate = e.has_rate
        ? `<span class="mono">${e.success ?? 0}/${e.episodes ?? 0}</span> <span class="muted">${fmt.pct(e.success_rate, 1)}</span>`
        : '<span class="muted">—</span>';
      const pos = isNum(e.pos_p95) ? fmt.num(Number(e.pos_p95) * 1000, 1) : '—';
      const pressed = LB.diff.includes(e.run_id);
      return `<tr>
        <td class="mono" title="${esc(e.run_id)}">${esc(e.run_id)}</td>
        <td>${esc(stageOf(e))}</td>
        <td><span class="badge ${src}">${esc(SOURCE_TEXT[src])}</span></td>
        <td>${rate}</td>
        <td>${ciCell(e)}</td>
        <td class="num">${e.collisions ?? 0}</td>
        <td class="num">${e.timeouts ?? 0}</td>
        <td class="num">${fmt.num(e.time_median, 2)}</td>
        <td class="num">${pos}</td>
        <td class="num">${fmt.num(e.yaw_p95_deg, 2)}</td>
        <td>${e.pareto ? '<span class="badge pareto">Pareto</span> ' : ''}<span class="badge">${esc(e.track || '—')}</span></td>
        <td class="mono muted">${esc(e.parent_id || '—')}</td>
        <td><button type="button" class="btn btn-diff" data-run="${esc(e.run_id)}" aria-pressed="${pressed}">${pressed ? 'Picked' : 'Diff'}</button></td>
      </tr>`;
    })
    .join('');
  tbody.querySelectorAll('.btn-diff').forEach((b) => b.addEventListener('click', () => toggleDiff(b.dataset.run)));
  $$('#leaderboard-table thead th[data-col]').forEach((th) => {
    th.setAttribute('aria-sort', LB.sort.col === th.dataset.col ? (LB.sort.dir > 0 ? 'ascending' : 'descending') : 'none');
  });
}

// ---- diff of two runs ----
function toggleDiff(runId) {
  const i = LB.diff.indexOf(runId);
  if (i >= 0) LB.diff.splice(i, 1);
  else {
    LB.diff.push(runId);
    if (LB.diff.length > 2) LB.diff.shift();
  }
  renderLeaderboard();
}

function renderDiff() {
  const panel = $('#diff-panel');
  if (LB.diff.length < 2) {
    panel.hidden = true;
    panel.innerHTML = '';
    return;
  }
  const [a, b] = LB.diff.map((id) => LB.entries.find((e) => e.run_id === id));
  if (!a || !b) {
    panel.hidden = true;
    return;
  }
  const rows = [
    ['Stage', stageOf(a), stageOf(b)],
    ['Source', SOURCE_TEXT[sourceOf(a)], SOURCE_TEXT[sourceOf(b)]],
    ['Success rate', fmt.pct(a.success_rate), fmt.pct(b.success_rate)],
    ['Wilson 95% CI', ciText(a), ciText(b)],
    ['Episodes', a.episodes ?? '—', b.episodes ?? '—'],
    ['Collisions', a.collisions ?? '—', b.collisions ?? '—'],
    ['Timeouts', a.timeouts ?? '—', b.timeouts ?? '—'],
    ['Median time (s)', fmt.num(a.time_median), fmt.num(b.time_median)],
    ['Pos p95 (mm)', isNum(a.pos_p95) ? fmt.num(a.pos_p95 * 1000, 1) : '—', isNum(b.pos_p95) ? fmt.num(b.pos_p95 * 1000, 1) : '—'],
    ['Yaw p95 (°)', fmt.num(a.yaw_p95_deg), fmt.num(b.yaw_p95_deg)],
    ['Pareto score', fmt.num(a.pareto_score, 3), fmt.num(b.pareto_score, 3)],
  ];
  panel.hidden = false;
  panel.innerHTML = `
    <div class="diff-head">
      <strong>Diff</strong>
      <span class="mono">${esc(a.run_id)}</span><span class="muted">vs</span><span class="mono">${esc(b.run_id)}</span>
      <button type="button" class="btn btn-ghost btn-sm" id="diff-close" style="margin-left:auto">Close</button>
    </div>
    <div class="table-scroll" style="max-height:none">
      <table class="diff-table"><thead><tr><th>Metric</th><th>${esc(a.run_id)}</th><th>${esc(b.run_id)}</th></tr></thead>
      <tbody>${rows.map((r) => `<tr><td>${esc(r[0])}</td><td class="num">${esc(r[1])}</td><td class="num">${esc(r[2])}</td></tr>`).join('')}</tbody></table>
    </div>`;
  $('#diff-close').addEventListener('click', () => {
    LB.diff = [];
    renderLeaderboard();
  });
}

function ciText(e) {
  return isNum(e.wilson_lo) && isNum(e.wilson_hi) ? `[${Number(e.wilson_lo).toFixed(2)}, ${Number(e.wilson_hi).toFixed(2)}]` : '—';
}

// ---- real Pi L7 field runs ----
function renderL7Table() {
  $('#l7-count').textContent = LB.l7.length ? `(${LB.l7.length} runs)` : '';
  const tbody = $('#l7-table tbody');
  tbody.innerHTML = LB.l7
    .map(
      (r) => `<tr>
        <td class="mono">${esc(r.run_id)}</td>
        <td>${esc(r.mode || '—')}</td>
        <td><span class="badge ${outcomeClass(r.outcome)}">${esc(r.outcome || '—')}</span></td>
        <td class="num">${fmt.num(r.seconds, 1)}</td>
        <td class="num">${fmt.num(r.hz, 1)}</td>
        <td class="num">${fmt.num(r.vision_ms_p50, 1)}</td>
        <td class="num">${fmt.num(r.loop_ms_p50, 2)}</td>
        <td class="num">${fmt.num(r.loop_ms_p95, 2)}</td>
        <td><button type="button" class="btn btn-sm" data-open="l7:${esc(r.run_id)}">Open in Arena</button></td>
      </tr>`,
    )
    .join('');
  tbody.querySelectorAll('[data-open]').forEach((b) =>
    b.addEventListener('click', () => openInArena(b.dataset.open)),
  );
}
