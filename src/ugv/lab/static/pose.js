import { api, esc, pct } from './app.js';
import { lineChart } from './charts.js';

export async function render(main) {
  const pose = await api('/api/pose', { fresh: true });
  main.innerHTML = `
    <h1>Pose of the airframe</h1>
    <p class="lede">6-DoF pose of STRL-001 from one car-camera image, through its rigid 12-point
    keypoint skeleton (nose, tail, wingtips, rotor hubs, V-tail tips, nose wheel, main wheels) and
    PnP. Synthetic views come from the mesh with random backgrounds, lighting and camera poses.
    Synthetic numbers debug the pipeline; only real frames are evidence.</p>
    <div id="ps-train"></div>
    <div class="grid g2" id="ps-smoke" style="margin-top:16px"></div>`;
  const train = main.querySelector('#ps-train');
  if (pose.training.length) {
    for (const t of pose.training) {
      const sec = document.createElement('section');
      sec.className = 'card';
      sec.style.marginBottom = '16px';
      const m = t.metrics;
      const keys = Object.keys(m[0] || {}).filter((k) => k !== 'step' && k !== 'epoch' && typeof m[0][k] === 'number');
      sec.innerHTML = `<h2>Keypoint detector · ${esc(t.name)} <span class="muted">${m.length} logged steps</span></h2>
        <div class="grid g2"><div class="c1"></div><div class="c2"></div></div>
        ${t.summary ? `<pre class="road" style="margin-top:12px">${esc(JSON.stringify(t.summary, null, 1))}</pre>` : ''}`;
      train.appendChild(sec);
      const x = (r, i) => r.step ?? r.epoch ?? i;
      const losses = keys.filter((k) => k.includes('loss'));
      const errs = keys.filter((k) => !k.includes('loss'));
      lineChart(sec.querySelector('.c1'), losses.map((k) => ({ name: k, points: m.filter((r) => r[k] != null).map((r, i) => [x(r, i), r[k]]) })), { xLabel: 'step', yLabel: 'loss' });
      lineChart(sec.querySelector('.c2'), errs.map((k) => ({ name: k, points: m.filter((r) => r[k] != null).map((r, i) => [x(r, i), r[k]]) })), { xLabel: 'step', yLabel: 'metric' });
    }
  } else {
    train.innerHTML = '<div class="empty">No detector training run yet (runs/pose_train/&lt;name&gt;/metrics.jsonl).</div>';
  }
  const smoke = main.querySelector('#ps-smoke');
  for (const s of pose.smoke) {
    const rows = Object.entries(s).filter(([k, v]) => v && typeof v === 'object' && 'n' in v);
    const sec = document.createElement('section');
    sec.className = 'card';
    sec.innerHTML = `<h2>${esc(s.name)} <span class="muted">${esc(s.views || 'generic')} views</span></h2>
      <table><thead><tr><th>method</th><th class="num">solved</th><th class="num">≤5° & ≤10 cm</th><th class="num">median rot</th><th class="num">median pos</th></tr></thead><tbody>
      ${rows.map(([k, v]) => `<tr><td class="mono">${esc(k)}</td><td class="num">${v.solved}/${v.n}</td><td class="num">${pct(v.within_5deg_10cm / v.n, 0)}</td>
      <td class="num">${Number.isFinite(v.median_rot_deg) ? v.median_rot_deg.toFixed(2) + '°' : '–'}</td><td class="num">${Number.isFinite(v.median_pos_m) ? (v.median_pos_m * 100).toFixed(1) + ' cm' : '–'}</td></tr>`).join('')}
      </tbody></table>
      ${s.sheet ? `<img class="sheet" style="margin-top:12px" alt="synthetic views with keypoints" src="/runs/${encodeURIComponent(s.name)}/sheet.jpg">` : ''}`;
    smoke.appendChild(sec);
  }
  if (!pose.smoke.length) smoke.innerHTML = '<div class="empty">Run <span class="mono">uv run python tools/pose/smoke.py --views car --out runs/pose_smoke_car</span>.</div>';
}
