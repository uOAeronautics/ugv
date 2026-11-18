// Park: is the controller good enough, what fails, and what has not been tested yet.
import { api, esc, outcomeColor, outcomeLabel, pct, STATE_COLOR, strip } from './app.js';
import { histogram, meters, outcomeTicks, scatter, tips } from './charts.js';
import { player } from './replay.js';
import { tuning } from './tuning.js';
import { loadPlan } from './world2d.js';

let pl = null;
export function dispose() { pl?.stop(); pl = null; removeEventListener('resize', onResize); }
export const retheme = () => pl?.redraw();
const onResize = () => pl?.redraw();

export async function render(main) {
  const [evals, world, plan, searches, ledger] = await Promise.all([
    api('/api/park/evals'), api('/api/park/world'), loadPlan(),
    api('/api/park/searches').catch(() => []), api('/api/ledger').catch(() => ({ records: [] })),
  ]);
  if (!evals.length) {
    main.innerHTML = `${strip('Park')}<div class="empty">No evaluation in runs/park_eval yet: <code>uv run python tools/park/eval.py --name default --sweep --traces 16</code></div>`;
    return;
  }
  const order = evals.slice().sort((a, b) => b.main.rate - a.main.rate);
  main.innerHTML = `${strip('Park', [], `<select id="pk-pick" aria-label="Evaluation">${order.map((e, i) => `<option value="${i}">${esc(e.name)} · ${pct(e.main.rate)}</option>`).join('')}</select>`)}
    <div id="pk-body"></div><div id="pk-tune"></div>`;
  const task = (id) => ledger.records.find((r) => r.kind === 'task' && r.id.endsWith(id));
  const pick = main.querySelector('#pk-pick');
  const show = () => body(main, order[+pick.value], world, plan, task);
  pick.addEventListener('change', show);
  show();
  tuning(main.querySelector('#pk-tune'), searches);
  tips(main);
  addEventListener('resize', onResize);
}

function body(main, ev, world, plan, task) {
  pl?.stop();
  const m = ev.main, col = m.columns;
  const real = task('park-on-car'), eval_ = task('real-pose-eval-set');
  const root = main.querySelector('#pk-body');
  root.innerHTML = `
    <section class="ruled c12 gates">
      ${gate('good', 'Simulation', `${m.success} / ${m.n} parked · ${m.collisions} collisions`)}
      ${gate(real?.state === 'done' ? 'good' : 'warn', 'Real car', real?.state === 'done' ? 'parked under supervision' : `not run · task <a href="#ledger/${real?.id ?? ''}">park-on-car</a> ${esc(real?.state ?? '')}`)}
      ${gate(eval_?.state === 'done' ? 'good' : 'warn', 'Detector on real frames', eval_?.state === 'done' ? 'evaluated' : `0 frames · task <a href="#ledger/${eval_?.id ?? ''}">real-pose-eval-set</a> ${esc(eval_?.state ?? '')}`)}
    </section>
    <section class="ruled c12">
      <div class="sp4">
        <div class="lbl">Parked <small>within ±2 cm, ±2 cm, ±5°</small></div>
        <div class="hero">${(100 * m.rate).toFixed(1)}<small>%</small></div>
        <div class="prov" style="margin:12px 0 22px">${m.success} / ${m.n} · 95% CI ${pct(m.wilson95[0], 1)}–${pct(m.wilson95[1], 1)}</div>
        <div id="pk-out"></div>
        <dl class="kv">
          <div><dt>Collisions</dt><dd>${m.collisions}<small> / ${m.n}</small></dd></div>
          <div><dt>Median time</dt><dd>${m.median_time_s ? m.median_time_s.toFixed(1) : '–'}<small> s</small></dd></div>
          <div><dt>Retries</dt><dd>${m.mean_retries.toFixed(2)}<small> per run</small></dd></div>
          <div><dt>Off target</dt><dd>${m.outcomes['parked-off-target'] ?? 0}<small> / ${m.n}</small></dd></div>
        </dl>
        <div class="lbl" style="margin-top:28px">Robustness <small>parked rate per condition · | = full noise</small></div><div id="pk-sweep"></div>
      </div>
      <div class="sp8" style="padding:0" id="pk-player"></div>
    </section>
    <section class="ruled c12">
      <div class="sp4"><div class="lbl">Final error <small>cm, parked runs · brackets = tolerance</small></div><div id="pk-err"></div></div>
      <div class="sp4"><div class="lbl">Time to park <small>seconds, parked runs</small></div><div id="pk-time"></div></div>
      <div class="sp4"><div class="lbl">Time per state <small>mean seconds per run</small></div><div id="pk-states"></div></div>
    </section>`;
  outcomeTicks(root.querySelector('#pk-out'), col.outcome, outcomeColor, outcomeLabel);
  const pts = col.final_error.map((e, i) => ({ e, o: col.outcome[i] })).filter((p) => p.o.startsWith('parked'));
  scatter(root.querySelector('#pk-err'), pts.map(({ e, o }) => ({ x: e[0] * 100, y: e[1] * 100, r: 2.2, color: outcomeColor(o), tip: `${o}<br>x ${(e[0] * 100).toFixed(1)} · y ${(e[1] * 100).toFixed(1)} cm · ${(e[2] * 57.3).toFixed(1)}°` })), { box: [-2, 2, -2, 2], height: 210, xLabel: 'x' });
  histogram(root.querySelector('#pk-time'), col.time_s.filter((_, i) => col.outcome[i] === 'parked'), { unit: 's', color: 'var(--ink-2)', height: 210, xLabel: 's' });
  const st = Object.entries(col.states);
  meters(root.querySelector('#pk-states'), st.map(([k, v]) => ({ label: k, value: v, color: STATE_COLOR[k], text: v.toFixed(1) })), { max: Math.max(...st.map(([, v]) => v)) });
  meters(root.querySelector('#pk-sweep'), [['full noise', m], ...Object.entries(ev.sweep || {})].map(([k, r]) => ({
    label: k, value: r.rate, text: pct(r.rate, 0), color: r.rate >= m.rate - 0.02 ? 'var(--ink)' : 'var(--warn)', tip: `<b>${esc(k)}</b>: ${r.success}/${r.n} parked · ${r.collisions} collisions`,
  })), { ref: m.rate });
  pl = player(root.querySelector('#pk-player'), ev, world, plan);
}

const gate = (cls, k, v) => `<div class="sp4 gate"><span class="st ${cls}"></span><b>${k}</b><span>${v}</span></div>`;
