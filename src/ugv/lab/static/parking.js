import { api, esc, outcomeColor, outcomeLabel, pct, STATE_COLOR } from './app.js';
import { barChart, hideTip, histogram, scatter, showTip } from './charts.js';
import { drawWorld, makeView } from './world2d.js';

let raf = 0;
export function dispose() { cancelAnimationFrame(raf); }

export async function render(main) {
  const [evals, world] = await Promise.all([api('/api/park/evals'), api('/api/park/world')]);
  if (!evals.length) {
    main.innerHTML = '<h1>Parking</h1><div class="empty">No evaluations in runs/park_eval yet. Run <span class="mono">uv run python tools/park/eval.py --name default --sweep --traces 16</span>.</div>';
    return;
  }
  main.innerHTML = `
    <div class="row"><h1>Parking in simulation</h1><span class="spacer"></span>
      <label class="muted">evaluation <select id="pk-pick">${evals.map((e, i) => `<option value="${i}">${esc(e.name)} · ${pct(e.main.rate)}</option>`).join('')}</select></label></div>
    <p class="lede">Each episode draws a new airframe pose, start pose, car (gains, deadzones, lag,
    roller slip, gyro noise) and camera calibration error. Perception is real PnP on the 3D model's
    keypoints; success is judged on the true pose: parked inside ±2 cm, ±2 cm, ±5°, no contact.</p>
    <div id="pk-body"></div>`;
  const pick = main.querySelector('#pk-pick');
  const show = () => body(main.querySelector('#pk-body'), evals[+pick.value], world);
  pick.addEventListener('change', show);
  show();
}

function body(root, ev, world) {
  cancelAnimationFrame(raf);
  const m = ev.main;
  const col = m.columns;
  const ok = col.outcome.map((o) => o === 'parked');
  root.innerHTML = `
    <div class="grid g4">
      ${tile('Parked', pct(m.rate), `${m.success}/${m.n} · CI ${pct(m.wilson95[0], 0)}–${pct(m.wilson95[1], 0)}`)}
      ${tile('Collisions', m.collisions, `${pct(m.collisions / m.n)} of episodes`)}
      ${tile('Median time', m.median_time_s ? `${m.median_time_s.toFixed(0)}<small> s</small>` : '–', 'successful episodes, start to parked')}
      ${tile('Retries', m.mean_retries.toFixed(2), 'mean BACKOFFs per episode')}
    </div>
    <div class="grid g2" style="margin-top:16px">
      <section class="card"><h2>Outcomes</h2><div id="pk-out"></div></section>
      <section class="card"><h2>Robustness sweep <span class="muted">${Object.keys(ev.sweep || {}).length ? '' : 'run with --sweep'}</span></h2><div id="pk-sweep"></div></section>
      <section class="card"><h2>Final error <span class="muted">x vs y, parked episodes; box = tolerance</span></h2><div id="pk-err"></div></section>
      <section class="card"><h2>Time to park <span class="muted">successful episodes</span></h2><div id="pk-time"></div></section>
      <section class="card span2"><h2>Where the time goes <span class="muted">mean seconds per state</span></h2><div id="pk-states"></div></section>
      <section class="card span2" id="pk-replay-card"><h2>Replay <span class="muted">${ev.has_traces ? 'true pose (solid) and the car\'s estimate (dashed)' : 'run eval with --traces N'}</span></h2><div id="pk-replay"></div></section>
    </div>`;
  barChart(root.querySelector('#pk-out'), Object.entries(m.outcomes).map(([k, v]) => ({ label: outcomeLabel(k), value: v, color: outcomeColor(k) })), { labelWidth: 190, fmt: (v) => `${v} (${pct(v / m.n, 0)})` });
  const sw = Object.entries(ev.sweep || {});
  barChart(root.querySelector('#pk-sweep'), [['full', m], ...sw].map(([k, r]) => ({ label: k, value: r.rate, tip: `<b>${k}</b>: ${r.success}/${r.n} · ${r.collisions} collisions<br>${esc(JSON.stringify(r.outcomes))}` })), { max: 1, fmt: (v) => pct(v, 0) });
  const pts = col.final_error.map((e, i) => ({ e, o: col.outcome[i] })).filter((p) => p.o.startsWith('parked'));
  scatter(root.querySelector('#pk-err'), pts.map(({ e, o }) => ({ x: e[0] * 100, y: e[1] * 100, color: outcomeColor(o), tip: `${o}<br>x ${(e[0] * 100).toFixed(1)} cm, y ${(e[1] * 100).toFixed(1)} cm, yaw ${(e[2] * 57.3).toFixed(1)}°` })), { xLabel: 'x error (cm)', yLabel: 'y (cm)', box: [-2, 2, -2, 2] });
  histogram(root.querySelector('#pk-time'), col.time_s.filter((_, i) => ok[i]), { xLabel: 'seconds', unit: 's', color: 'var(--s1)' });
  barChart(root.querySelector('#pk-states'), Object.entries(col.states).map(([k, v]) => ({ label: k, value: v, color: STATE_COLOR[k] || 'var(--s1)' })), { fmt: (v) => `${v.toFixed(1)} s` });
  if (ev.has_traces) replay(root.querySelector('#pk-replay'), ev, world);
}

const tile = (k, v, d) => `<section class="card stat"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></section>`;

async function replay(root, ev, world) {
  const traces = await api(`/api/park/traces/${encodeURIComponent(ev.file.replace(/\.json$/, ''))}`);
  root.innerHTML = `
    <div class="row" style="margin-bottom:10px">
      <select id="rp-ep">${traces.map((t, i) => `<option value="${i}">episode ${t.episode} · ${esc(t.outcome)}</option>`).join('')}</select>
      <button id="rp-play" class="primary" type="button">Play</button>
      <input id="rp-t" type="range" min="0" max="1000" value="1000" style="flex:1">
      <span id="rp-label" class="mono muted"></span>
    </div>
    <div class="grid g2">
      <canvas id="rp-map" class="map"></canvas>
      <div><div class="timeline" id="rp-line"></div><div class="legend" id="rp-legend"></div>
      <p class="muted" id="rp-note" style="margin-top:12px"></p></div>
    </div>`;
  const canvas = root.querySelector('#rp-map');
  const sel = root.querySelector('#rp-ep'), slider = root.querySelector('#rp-t');
  let tr = traces[0];
  const view = makeView(canvas, 2.4);
  const draw = () => {
    const k = Math.round((slider.value / 1000) * (tr.trace.length - 1));
    const rows = tr.trace.slice(0, k + 1);
    drawWorld(view, world.bay, {
      path: rows.map((r) => r.car), est: rows.filter((r) => r.est).map((r) => r.est),
      car: rows[k]?.car, carEst: rows[k]?.est, state: rows[k]?.state,
    });
    root.querySelector('#rp-label').textContent = `t=${rows[k]?.t ?? 0}s ${rows[k]?.state ?? ''}`;
  };
  const load = () => {
    tr = traces[+sel.value];
    const line = root.querySelector('#rp-line');
    const segs = [];
    for (const r of tr.trace) { if (!segs.length || segs.at(-1).s !== r.state) segs.push({ s: r.state, n: 0 }); segs.at(-1).n++; }
    line.innerHTML = segs.map((g) => `<span data-s="${g.s}" style="flex:${g.n};background:${STATE_COLOR[g.s] || 'var(--ink-3)'}"></span>`).join('');
    line.querySelectorAll('span').forEach((s) => {
      s.addEventListener('mousemove', (e) => showTip(e, s.dataset.s));
      s.addEventListener('mouseleave', hideTip);
    });
    root.querySelector('#rp-legend').innerHTML = [...new Set(segs.map((g) => g.s))].map((s) => `<span><i style="background:${STATE_COLOR[s]}"></i>${s}</span>`).join('');
    const fe = tr.final_error;
    root.querySelector('#rp-note').innerHTML = `Outcome <b>${esc(tr.outcome)}</b>; final error x ${(fe[0] * 100).toFixed(1)} cm, y ${(fe[1] * 100).toFixed(1)} cm, yaw ${(fe[2] * 57.3).toFixed(1)}°. The view is the airframe frame: nose to the right, target box in green, obstacles (gear, struts, low props) in red.`;
    slider.value = 1000;
    draw();
  };
  sel.addEventListener('change', load);
  slider.addEventListener('input', draw);
  root.querySelector('#rp-play').addEventListener('click', () => {
    cancelAnimationFrame(raf);
    slider.value = 0;
    const step = () => {
      slider.value = Math.min(1000, +slider.value + 4);
      draw();
      if (+slider.value < 1000) raf = requestAnimationFrame(step);
    };
    step();
  });
  load();
}
