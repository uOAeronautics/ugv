// Episode player for the Park page: every traced run at once, or one run with a scrubber, its
// state timeline, the true path and the car's own estimate.
import { api, esc, outcomeColor, outcomeLabel, STATE_COLOR } from './app.js';
import { hideTip, showTip } from './charts.js';
import { drawWorld, makeView } from './world2d.js';

export function player(root, ev, world, plan) {
  root.innerHTML = `
    <div style="position:relative"><canvas class="plan" id="rp-map" style="height:min(60vh,600px)"></canvas>
      <div class="hud tl" id="rp-hud"></div>
      <div class="keys hud bl" style="margin:0"><span style="--c:var(--bad)">car hits</span><span style="--c:var(--warn)">camera stand hits</span><span style="--c:var(--ink-3)">overhead</span><span style="--c:var(--good)">bay</span></div></div>
    <div class="player">
      <div class="row"><select id="rp-ep" aria-label="Run"></select><button id="rp-play" class="go" type="button">Play</button>
        <input id="rp-t" type="range" min="0" max="1000" value="1000" aria-label="Time"><span id="rp-label" class="mono" style="min-width:140px;text-align:right"></span></div>
      <div class="ticks" id="rp-line" style="height:10px;margin-top:12px"></div><div class="keys" id="rp-legend"></div>
    </div>`;
  const sel = root.querySelector('#rp-ep'), slider = root.querySelector('#rp-t'), play = root.querySelector('#rp-play');
  const canvas = root.querySelector('#rp-map'), hud = root.querySelector('#rp-hud');
  let traces = [], tr = null, view = null, playing = false, raf = 0;
  const fit = () => { view = makeView(canvas, 1.3, [-0.3, 0]); };

  const draw = () => {
    if (!view) fit();
    if (!tr) {
      drawWorld(view, world.bay, plan, { paths: traces.map((t) => ({ pts: t.trace.map((r) => r.car), color: outcomeColor(t.outcome), alpha: 0.8, width: 1.25 })) });
      hud.innerHTML = `<b>${traces.length} traced runs</b><span>true path · nose →</span>`;
      root.querySelector('#rp-label').textContent = '';
      return;
    }
    const k = Math.round((slider.value / 1000) * (tr.trace.length - 1));
    const rows = tr.trace.slice(0, k + 1), r = rows[k];
    drawWorld(view, world.bay, plan, {
      paths: [{ pts: rows.filter((q) => q.est).map((q) => q.est), color: 'var(--ink-2)', dashed: true, width: 1 }, { pts: rows.map((q) => q.car), color: 'var(--signal)', width: 1.75 }],
      car: r?.car, carEst: r?.est,
    });
    root.querySelector('#rp-label').textContent = `${(r?.t ?? 0).toFixed(1)} s ${r?.state ?? ''}`;
    const fe = tr.final_error;
    hud.innerHTML = `<b>Run ${tr.episode} · ${esc(outcomeLabel(tr.outcome))}</b><span>solid true · dashed estimate</span><span>final ${(fe[0] * 100).toFixed(1)} / ${(fe[1] * 100).toFixed(1)} cm · ${(fe[2] * 57.3).toFixed(1)}°</span>`;
  };
  const load = () => {
    stop();
    tr = sel.value === 'all' ? null : traces[+sel.value];
    play.disabled = !tr;
    const line = root.querySelector('#rp-line'), legend = root.querySelector('#rp-legend');
    if (!tr) { line.innerHTML = ''; legend.innerHTML = ''; slider.disabled = true; draw(); return; }
    slider.disabled = false;
    const segs = [];
    for (const r of tr.trace) { if (!segs.length || segs.at(-1).s !== r.state) segs.push({ s: r.state, n: 0 }); segs.at(-1).n++; }
    line.innerHTML = segs.map((g) => `<i style="flex:${g.n};background:${STATE_COLOR[g.s] || 'var(--ink-3)'};margin-right:1px" data-s="${g.s}"></i>`).join('');
    line.querySelectorAll('i').forEach((s) => { s.addEventListener('mousemove', (e) => showTip(e, s.dataset.s)); s.addEventListener('mouseleave', hideTip); });
    legend.innerHTML = [...new Set(segs.map((g) => g.s))].map((s) => `<span style="--c:${STATE_COLOR[s]}">${s}</span>`).join('');
    slider.value = 1000;
    draw();
  };
  const stop = () => { playing = false; cancelAnimationFrame(raf); play.textContent = 'Play'; };
  play.addEventListener('click', () => {
    if (playing || !tr) return stop();
    playing = true; play.textContent = 'Pause';
    if (+slider.value >= 1000) slider.value = 0;
    let last = performance.now();
    const step = (now) => {
      slider.value = Math.min(1000, +slider.value + (1000 * (now - last)) / 1000 / (tr.trace.at(-1).t / 4)); // 4x real time
      last = now;
      draw();
      if (+slider.value < 1000) raf = requestAnimationFrame(step); else stop();
    };
    raf = requestAnimationFrame(step);
  });
  sel.addEventListener('change', load);
  slider.addEventListener('input', () => { stop(); draw(); });

  if (!ev.has_traces) {
    hud.innerHTML = '<b>No traces in this evaluation</b><span>run eval with --traces N</span>';
    fit(); drawWorld(view, world.bay, plan, {});
    return { redraw: () => { fit(); drawWorld(view, world.bay, plan, {}); }, stop };
  }
  api(`/api/park/traces/${encodeURIComponent(ev.file.replace(/\.json$/, ''))}`).then((t) => {
    traces = t;
    sel.innerHTML = `<option value="all">All ${t.length} traced runs</option>` + t.map((x, i) => `<option value="${i}">Run ${x.episode} · ${esc(outcomeLabel(x.outcome))}</option>`).join('');
    load();
  });
  return { redraw: () => { fit(); draw(); }, stop };
}
