// Pose: what the keypoint detector sees on held-out views, the skeleton it finds, and the pose
// PnP makes of it; then how the detector trained. Synthetic views debug the pipeline only.
import { api, color, esc, pct, strip } from './app.js';
import { lineChart, tips } from './charts.js';

const BONES = [
  ['nose', 'tail'], ['nose', 'gear_nose'], ['wingtip_left', 'hub_left'], ['hub_left', 'hub_right'], ['hub_right', 'wingtip_right'],
  ['hub_rear', 'tail'], ['tail', 'vtail_tip_left'], ['tail', 'vtail_tip_right'], ['gear_main_left', 'gear_main_right'],
  ['gear_nose', 'gear_main_left'], ['gear_nose', 'gear_main_right'],
];
const STAGE = { 'kp-v1': 'stage 1 · whole frame 320×180', 'kp-crop-v1': 'stage 2 · 256×256 crop' };

let state = null;
export function dispose() { removeEventListener('keydown', onKey); removeEventListener('resize', onResize); state = null; }
export const retheme = () => state?.draw();
const onResize = () => state?.draw();
const onKey = (e) => {
  if (!state || /INPUT|SELECT/.test(document.activeElement?.tagName || '')) return;
  if (e.key === 'ArrowRight') state.go(state.i + 1);
  if (e.key === 'ArrowLeft') state.go(state.i - 1);
};

export async function render(main) {
  const pose = await api('/api/pose', { fresh: true });
  const runs = pose.training.slice().sort((a, b) => (a.name === 'kp-crop-v1' ? -1 : b.name === 'kp-crop-v1' ? 1 : 0));
  if (!runs.length) {
    main.innerHTML = `${strip('Pose')}<div class="empty">No detector run yet (runs/pose_train/&lt;name&gt;/metrics.jsonl).</div>`;
    return;
  }
  main.innerHTML = `${strip('Pose', [['Method', '12 keypoints → PnP'], ['Real frames evaluated', '0'], ['Stages chained', 'not yet']],
    `<span class="tag syn" data-tip="Rendered views from the same generator as training. They debug the pipeline; only real frames are evidence.">synthetic</span>
     <div class="seg" id="ps-runs">${runs.map((r, i) => `<button type="button" data-i="${i}">${esc(r.name)}</button>`).join('')}</div>`)}
    <div id="ps-body"></div>`;
  tips(main);
  const pickRun = (i) => {
    main.querySelectorAll('#ps-runs button').forEach((b) => b.classList.toggle('on', +b.dataset.i === i));
    body(main.querySelector('#ps-body'), runs[i], pose.smoke);
  };
  main.querySelectorAll('#ps-runs button').forEach((b) => b.addEventListener('click', () => pickRun(+b.dataset.i)));
  pickRun(0);
  addEventListener('keydown', onKey);
  addEventListener('resize', onResize);
}

async function body(root, run, smoke) {
  const sm = run.summary || {}, pv = run.preview;
  root.innerHTML = `
    <section class="ruled c12">
      <div class="sp7" style="padding:0">
        <div class="vbar"><span id="ps-hud"></span><span id="ps-stage"></span></div>
        <div class="viewer" id="ps-view"><img id="ps-img" alt="held-out synthetic view"><canvas id="ps-ov"></canvas></div>
        <div style="padding:12px 24px;border-top:1px solid var(--rule)" class="row">
          <div class="seg" id="ps-layers"><button type="button" class="on" data-l="mesh">STRL mesh</button><button type="button" class="on" data-l="truth">Truth</button><button type="button" class="on" data-l="det">Detector</button><button type="button" class="on" data-l="pnp">PnP pose</button></div>
          <span class="keys" style="margin:0"><span style="--c:var(--s5)">STRL-001 mesh at the true pose</span><span style="--c:var(--ink-3)">rendered truth</span><span style="--c:var(--signal)">detected, prob &gt; 0.5</span><span style="--c:var(--good)">model reprojected through the PnP pose</span></span>
          <span class="grow"></span><button type="button" id="ps-prev" aria-label="Previous view">←</button><button type="button" id="ps-next" aria-label="Next view">→</button>
        </div>
        <div class="film" id="ps-film"></div>
      </div>
      <div class="sp5" id="ps-read"></div>
    </section>
    <section class="ruled c12">
      ${ro('sp3', 'Keypoint error', fx(sm.val_px_median_at_1280, 1), 'px', `median at 1280×720 · p90 ${fx(sm.val_px_p90_at_1280, 0)} px`)}
      ${ro('sp3', 'Pose within 5° / 10 cm', sm.val_pnp_within_5deg_10cm != null ? (100 * sm.val_pnp_within_5deg_10cm).toFixed(0) : '–', '%', `PnP solved on ${pct(sm.val_pnp_solved ?? 0, 0)} of ${sm.val_n ?? ''} views`)}
      ${ro('sp3', 'Visibility', sm.val_vis_acc != null ? (100 * sm.val_vis_acc).toFixed(0) : '–', '%', 'per keypoint: visible or not, correct')}
      ${ro('sp3', 'CPU latency', sm.desktop_cpu_ms_1thread ?? '–', 'ms', `desktop, 1 thread · ${((sm.params || 0) / 1e6).toFixed(2)} M params · Pi not measured`)}
    </section>
    <section class="ruled c12">
      <div class="sp4"><div class="lbl">Loss <small>${run.metrics.length} epochs</small></div><div id="ps-c1"></div></div>
      <div class="sp4"><div class="lbl">Keypoint error <small>px at 1280×720</small></div><div id="ps-c2"></div></div>
      <div class="sp4"><div class="lbl">Rates</div><div id="ps-c3"></div></div>
    </section>
    <section class="ruled c12">${smoke.map((s) => `<div class="sp6">${smokeTable(s)}</div>`).join('')}</section>`;
  const m = run.metrics;
  const series = (keys, names) => keys.map((k, j) => ({ name: names[j], points: m.filter((r) => r[k] != null).map((r, i) => [r.epoch ?? i, r[k]]) })).filter((s) => s.points.length);
  lineChart(root.querySelector('#ps-c1'), series(['train_kp_loss', 'train_vis_loss'], ['keypoint L1', 'visibility BCE']), { xLabel: 'epoch', height: 170 });
  const px = m[0]?.val_px_median_at_1280 != null ? ['val_px_median_at_1280', 'val_px_p90_at_1280'] : ['val_px_median', 'val_px_p90'];
  lineChart(root.querySelector('#ps-c2'), series(px, ['median', 'p90']), { xLabel: 'epoch', height: 170 });
  lineChart(root.querySelector('#ps-c3'), series(['val_vis_acc', 'val_pnp_solved', 'val_pnp_within_5deg_10cm'], ['visibility', 'PnP solved', 'within 5°/10 cm']), { xLabel: 'epoch', yDomain: [0, 1], yFmt: (v) => pct(v, 0), height: 170 });

  if (!pv) {
    root.querySelector('#ps-view').innerHTML = `<div class="empty" style="padding:40px">No preview for this run: <code>cd train && uv run python preview.py --run ../runs/pose_train/${esc(run.name)}</code></div>`;
    state = null;
    return;
  }
  const names = pv.keypoints, idx = Object.fromEntries(names.map((n, i) => [n, i]));
  const layers = new Set(['mesh', 'truth', 'det', 'pnp']);
  const tris = await api('/api/pose/mesh');
  const img = root.querySelector('#ps-img'), ov = root.querySelector('#ps-ov'), view = root.querySelector('#ps-view');
  const film = root.querySelector('#ps-film');
  const grade = (s) => (!s.pnp ? 'bad' : s.pnp.rot_deg < 5 && s.pnp.pos_cm < 10 ? 'good' : 'warn');
  film.innerHTML = pv.samples.map((s, i) => `<button type="button" data-i="${i}" class="g-${grade(s)}" aria-label="view ${i + 1}"><img loading="lazy" src="/runs/pose_train/${encodeURIComponent(run.name)}/${s.file}" alt=""></button>`).join('');

  const draw = () => {
    const s = pv.samples[st.i];
    const [w, h] = s.size;
    const W = view.clientWidth, H = view.clientHeight;
    const k = Math.min(W / w, H / h), ox = (W - w * k) / 2, oy = (H - h * k) / 2;
    img.style.cssText = `left:${ox}px;top:${oy}px;width:${w * k}px;height:${h * k}px`;
    const dpr = devicePixelRatio || 1;
    ov.width = W * dpr; ov.height = H * dpr;
    const g = ov.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const P = ([u, v]) => [ox + u * k, oy + v * k];
    // sensor frame brackets
    g.strokeStyle = color('var(--ink-2)'); g.lineWidth = 1.5;
    const b = 16, x0 = ox - 6, y0 = oy - 6, x1 = ox + w * k + 6, y1 = oy + h * k + 6;
    for (const [x, y, dx, dy] of [[x0, y0, 1, 1], [x1, y0, -1, 1], [x0, y1, 1, -1], [x1, y1, -1, -1]]) { g.beginPath(); g.moveTo(x + dx * b, y); g.lineTo(x, y); g.lineTo(x, y + dy * b); g.stroke(); }
    const skeleton = (pts, ok, col, dash, mark) => {
      g.strokeStyle = col; g.fillStyle = col; g.lineWidth = 1.5; g.setLineDash(dash);
      for (const [a, c] of BONES) {
        const i = idx[a], j = idx[c];
        if (i == null || j == null || !ok[i] || !ok[j]) continue;
        g.beginPath(); g.moveTo(...P(pts[i])); g.lineTo(...P(pts[j])); g.stroke();
      }
      g.setLineDash([]);
      pts.forEach((p, i) => { if (ok[i]) mark(...P(p)); });
    };
    if (layers.has('mesh')) drawMesh(g, tris, s.K, s.R_true, s.t_true, P, color('var(--s5)'), true);
    if (layers.has('pnp') && s.R_pnp) drawMesh(g, tris, s.K, s.R_pnp, s.t_pnp, P, color('var(--good)'), false);
    if (layers.has('truth')) skeleton(s.true, s.true_vis, color('var(--ink-3)'), [], (x, y) => { g.strokeRect(x - 4, y - 4, 8, 8); });
    if (layers.has('pnp') && s.reproj) skeleton(s.reproj, s.reproj.map(() => true), color('var(--good)'), [5, 4], (x, y) => { g.beginPath(); g.moveTo(x - 5, y); g.lineTo(x + 5, y); g.moveTo(x, y - 5); g.lineTo(x, y + 5); g.stroke(); });
    if (layers.has('det')) skeleton(s.pred, s.pred_prob.map((p) => p > 0.5), color('var(--signal)'), [], (x, y) => { g.fillRect(x - 3.5, y - 3.5, 7, 7); });
    root.querySelector('#ps-hud').innerHTML = `<b>${String(st.i + 1).padStart(2, '0')}</b> / ${pv.samples.length}<em>held-out view ${s.index}</em><em>${s.range_m.toFixed(2)} m away</em>`;
    readout(s);
  };
  const readout = (s) => {
    const err = s.pred.map((p, i) => (s.true_vis[i] ? Math.hypot(p[0] - s.true[i][0], p[1] - s.true[i][1]) * s.px_to_1280 : null));
    root.querySelector('#ps-read').innerHTML = `
      <div class="lbl">PnP pose <small>this view</small></div>
      <div class="row" style="gap:36px;align-items:flex-end">
        <div class="ro"><div class="v" style="color:var(--${s.pnp ? (s.pnp.rot_deg < 5 ? 'ink' : 'warn') : 'bad'})">${s.pnp ? s.pnp.rot_deg.toFixed(1) : '—'}<small>°</small></div><div class="d">rotation error</div></div>
        <div class="ro"><div class="v" style="color:var(--${s.pnp ? (s.pnp.pos_cm < 10 ? 'ink' : 'warn') : 'bad'})">${s.pnp ? s.pnp.pos_cm.toFixed(1) : '—'}<small>cm</small></div><div class="d">position error</div></div>
      </div>
      ${s.pnp ? '' : '<p class="prov" style="color:var(--bad)">not solved: fewer than 6 confident keypoints, or no consistent pose</p>'}
      <table style="margin-top:18px"><thead><tr><th>Keypoint</th><th class="num">Prob</th><th class="num">Error px</th></tr></thead><tbody>
      ${names.map((n, i) => `<tr${s.true_vis[i] ? '' : ' class="dim"'}><td><span class="st ${s.pred_prob[i] > 0.5 ? 'live' : 'open'}">${n.replace(/_/g, ' ')}</span></td>
        <td class="num">${s.pred_prob[i].toFixed(2)}</td><td class="num" style="color:var(--${err[i] == null ? 'ink-3' : err[i] > 15 ? 'bad' : err[i] > 5 ? 'warn' : 'ink'})">${err[i] == null ? '–' : err[i].toFixed(1)}</td></tr>`).join('')}
      </tbody></table>
      <p class="prov">Error in pixels at 1280×720 against the rendered truth. Greyed rows are hidden in this view. A filled square means the detector called the point visible.</p>`;
  };
  const st = { i: 0 };
  state = {
    get i() { return st.i; },
    draw,
    go: (i) => {
      st.i = (i + pv.samples.length) % pv.samples.length;
      img.src = `/runs/pose_train/${encodeURIComponent(run.name)}/${pv.samples[st.i].file}`;
      film.querySelectorAll('button').forEach((b) => b.classList.toggle('on', +b.dataset.i === st.i));
      film.querySelector('button.on')?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
    },
  };
  img.addEventListener('load', draw);
  film.addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) state.go(+b.dataset.i); });
  root.querySelector('#ps-prev').addEventListener('click', () => state.go(st.i - 1));
  root.querySelector('#ps-next').addEventListener('click', () => state.go(st.i + 1));
  root.querySelectorAll('#ps-layers button').forEach((b) => b.addEventListener('click', () => {
    b.classList.toggle('on');
    layers[b.classList.contains('on') ? 'add' : 'delete'](b.dataset.l);
    draw();
  }));
  root.querySelector('#ps-stage').textContent = STAGE[run.name] || run.name;
  state.go(0);
}

/** Project the model-frame mesh (mm) through K[R|t]: a filled silhouette, or faint edges. */
function drawMesh(g, tris, K, R, t, P, col, fill) {
  if (!K || !R) return;
  const pr = (x, y, z) => {
    x /= 1000; y /= 1000; z /= 1000;
    const X = R[0][0] * x + R[0][1] * y + R[0][2] * z + t[0], Y = R[1][0] * x + R[1][1] * y + R[1][2] * z + t[1], Z = R[2][0] * x + R[2][1] * y + R[2][2] * z + t[2];
    return Z > 0.02 ? P([K[0][0] * X / Z + K[0][2], K[1][1] * Y / Z + K[1][2]]) : null;
  };
  g.save();
  g.fillStyle = col; g.strokeStyle = col;
  g.globalAlpha = fill ? 0.42 : 0.2; g.lineWidth = 0.5;
  g.beginPath(); // one path, filled once: an even silhouette however many faces overlap
  for (const q of tris) {
    const a = pr(q[0], q[1], q[2]), b = pr(q[3], q[4], q[5]), c = pr(q[6], q[7], q[8]);
    if (!a || !b || !c) continue;
    g.moveTo(...a); g.lineTo(...b); g.lineTo(...c); g.closePath();
  }
  if (fill) g.fill('nonzero'); else g.stroke();
  g.restore();
}

const fx = (v, d) => (v == null ? '–' : v.toFixed(d));
const ro = (cls, k, v, unit, d) => `<div class="ro ${cls}"><div class="k">${k}</div><div class="v">${v}<small>${unit}</small></div><div class="d">${d}</div></div>`;

function smokeTable(s) {
  const rows = Object.entries(s).filter(([, v]) => v && typeof v === 'object' && 'n' in v);
  return `<div class="lbl">Solver check · ${esc(s.views || 'generic')} views <small>exact keypoints plus noise, no detector</small></div>
    <table><thead><tr><th>Method</th><th class="num">Solved</th><th class="num">≤5° &amp; ≤10 cm</th><th class="num">Median rot</th><th class="num">Median pos</th></tr></thead><tbody>
    ${rows.map(([k, v]) => `<tr><td class="mono">${esc(k.replace('keypoint_pnp_sigma', 'PnP, σ = ').replace(/(σ = \d+)$/, '$1 px').replace('sift_bank', 'SIFT bank'))}</td><td class="num">${v.solved}/${v.n}</td><td class="num">${pct(v.within_5deg_10cm / v.n, 0)}</td>
    <td class="num">${Number.isFinite(v.median_rot_deg) ? v.median_rot_deg.toFixed(2) + '°' : '–'}</td><td class="num">${Number.isFinite(v.median_pos_m) ? (v.median_pos_m * 100).toFixed(1) + ' cm' : '–'}</td></tr>`).join('')}
    </tbody></table>`;
}
