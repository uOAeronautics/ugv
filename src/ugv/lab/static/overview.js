import { api, esc, outcomeColor, outcomeLabel, pct } from './app.js';
import { barChart } from './charts.js';

export async function render(main) {
  const [evals, searches, pose, ledger] = await Promise.all([
    api('/api/park/evals').catch(() => []),
    api('/api/park/searches').catch(() => []),
    api('/api/pose').catch(() => ({ smoke: [], training: [] })),
    api('/api/ledger').catch(() => ({ records: [], status: [], roadmap: [] })),
  ]);
  const best = evals.slice().sort((a, b) => b.main.rate - a.main.rate)[0];
  const tasks = ledger.records.filter((r) => r.kind === 'task');
  const live = tasks.filter((t) => ['open', 'claimed', 'blocked'].includes(t.state));
  const attempts = ledger.records.filter((r) => r.kind === 'attempt');
  const smoke = pose.smoke.find((s) => s.views === 'car') || pose.smoke[0];
  const trained = pose.training[pose.training.length - 1];

  main.innerHTML = `
    <h1>Overview</h1>
    <p class="lede">The car should find the STRL-001 airframe from anywhere in the room and park under
    its belly, using the airframe's own 3D model instead of tags. Everything below is read from
    <span class="mono">runs/</span> and <span class="mono">research/</span>; simulation numbers are
    not hardware numbers.</p>
    <div class="grid g4">
      ${stat('Sim parking', best ? pct(best.main.rate) : '–', best ? `${best.main.success}/${best.main.n} · 95% CI ${pct(best.main.wilson95[0], 0)}–${pct(best.main.wilson95[1], 0)} · <span class="mono">${esc(best.name)}</span>` : 'run tools/park/eval.py')}
      ${stat('Collisions', best ? String(best.main.collisions) : '–', best ? `in ${best.main.n} episodes, full noise` : '')}
      ${stat('Pose (synthetic)', smoke ? pct((smoke.keypoint_pnp_sigma1?.within_5deg_10cm ?? 0) / (smoke.keypoint_pnp_sigma1?.n || 1), 0) : '–', 'PnP within 5° / 10 cm at 1 px keypoint noise')}
      ${stat('Open tasks', String(live.length), `${attempts.length} attempts recorded · ${searches.reduce((a, s) => a + s.nodes.length, 0)} search nodes`)}
    </div>
    <div class="grid g2" style="margin-top:16px">
      <section class="card"><h2>Parking outcomes <span class="muted">${best ? esc(best.name) : ''}</span></h2><div id="ov-out"></div></section>
      <section class="card"><h2>Robustness <span class="muted">success by condition</span></h2><div id="ov-sweep"></div></section>
      <section class="card"><h2>Roadmap <span class="muted">ledger.py roadmap</span></h2><pre class="road">${esc(ledger.roadmap.join('\n'))}</pre></section>
      <section class="card"><h2>Live tasks <span class="muted">ledger.py status</span></h2><pre class="road">${esc(ledger.status.join('\n') || 'none')}</pre></section>
    </div>
    ${trained ? `<section class="card" style="margin-top:16px"><h2>Keypoint detector <span class="muted">${esc(trained.name)}</span></h2><p class="muted">${esc(JSON.stringify(trained.summary || {}))}</p></section>` : ''}
  `;
  if (best) {
    barChart(main.querySelector('#ov-out'), Object.entries(best.main.outcomes).map(([k, v]) => ({ label: outcomeLabel(k), value: v, color: outcomeColor(k) })), { labelWidth: 170, fmt: (v) => `${v} (${pct(v / best.main.n, 0)})` });
    const sweep = Object.entries(best.sweep || {});
    barChart(main.querySelector('#ov-sweep'), [['full', best.main], ...sweep].map(([k, r]) => ({ label: k, value: r.rate, color: 'var(--s1)', tip: `<b>${k}</b>: ${r.success}/${r.n} parked · ${r.collisions} collisions` })), { max: 1, fmt: (v) => pct(v, 0) });
  } else {
    main.querySelector('#ov-out').innerHTML = main.querySelector('#ov-sweep').innerHTML = '<div class="empty">no evaluation yet: <span class="mono">uv run python tools/park/eval.py --name default --sweep</span></div>';
  }
}

function stat(k, v, d) {
  return `<section class="card stat"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></section>`;
}
