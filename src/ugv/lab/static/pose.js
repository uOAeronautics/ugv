import { api, esc, pct } from './app.js';
import { lineChart } from './charts.js';

const tile = (k, v, d) => `<section class="card stat"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></section>`;

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
      const sm = t.summary || {};
      sec.innerHTML = `<h2>Keypoint detector · ${esc(t.name)} <span class="muted">${m.length} epochs · ${esc(sm.data || '')}</span></h2>
        ${t.summary ? `<div class="grid g4" style="margin-bottom:12px">
          ${tile('Keypoint error', `${(sm.val_px_median_at_1280 ?? 0).toFixed(1)}<small> px</small>`, `median at 1280x720 · p90 ${sm.val_px_p90_at_1280 != null ? sm.val_px_p90_at_1280.toFixed(0) : '–'} px`)}
          ${tile('Pose within 5° / 10 cm', pct(sm.val_pnp_within_5deg_10cm ?? 0, 0), `PnP solved on ${pct(sm.val_pnp_solved ?? 0, 0)} of views`)}
          ${tile('Visibility', pct(sm.val_vis_acc ?? 0, 0), 'per keypoint, visible or not')}
          ${tile('CPU latency', `${sm.desktop_cpu_ms_1thread ?? '–'}<small> ms</small>`, `desktop, 1 thread · ${((sm.params || 0) / 1e6).toFixed(2)} M params · input ${esc((sm.input || []).join('x'))}`)}
        </div><p class="muted">${esc(sm.note || '')}</p>` : ''}
        <div class="grid g3"><div><div class="muted">loss</div><div class="c1"></div></div><div><div class="muted">keypoint error (px)</div><div class="c2"></div></div><div><div class="muted">rates</div><div class="c3"></div></div></div>`;
      train.appendChild(sec);
      const x = (r, i) => r.epoch ?? i;
      const series = (keys) => keys.filter((k) => m.some((r) => r[k] != null)).map((k) => ({ name: k.replace('val_', '').replace('train_', ''), points: m.filter((r) => r[k] != null).map((r, i) => [x(r, i), r[k]]) }));
      lineChart(sec.querySelector('.c1'), series(['train_kp_loss', 'train_vis_loss']), { xLabel: 'epoch' });
      const px = m[0]?.val_px_median_at_1280 != null ? ['val_px_median_at_1280', 'val_px_p90_at_1280'] : ['val_px_median', 'val_px_p90'];
      lineChart(sec.querySelector('.c2'), series(px), { xLabel: 'epoch' });
      lineChart(sec.querySelector('.c3'), series(['val_vis_acc', 'val_pnp_solved', 'val_pnp_within_5deg_10cm']), { xLabel: 'epoch', yDomain: [0, 1], yFmt: (v) => pct(v, 0) });
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
