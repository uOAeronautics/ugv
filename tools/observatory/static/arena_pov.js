// Right panel: camera POV (real L7 frames or synthetic pinhole), live telemetry and the three strips.

function lastFrameUrl(frames, idx) {
  for (let i = idx; i >= 0; i--) {
    const u = frames[i]?.frame_url || frames[i]?.camera_frame_url;
    if (u) return u;
  }
  return null;
}

function drawPov() {
  const c = $('#pov-canvas');
  const img = $('#pov-img');
  const badge = $('#pov-mode');
  const meta = $('#pov-meta');
  if (!c || !img) return;
  const { ctx, w, h } = fitCanvas(c);
  ctx.clearRect(0, 0, w, h);
  const ep = AR.episodes[AR.epIdx];
  const frames = ep?.frames || [];
  const f = frames[AR.frame] || {};
  const ef = ep?.end_frame_url || AR.bundle?.end_frame_url || null;
  const endBtn = $('#btn-end-frame');
  endBtn.hidden = !ef;
  endBtn.setAttribute('aria-pressed', String(AR.showEnd));
  let url = f.frame_url || f.camera_frame_url || lastFrameUrl(frames, AR.frame);
  if (AR.showEnd && ef) url = ef;
  if (!url && ef) url = ef;

  if (url) {
    drawRealPov(ctx, w, h, img, url, f, badge, meta, AR.showEnd && ef);
    return;
  }
  img.hidden = true;
  badge.textContent = 'Simulated camera';
  drawSyntheticPov(ctx, w, h, f, meta);
}

function drawRealPov(ctx, w, h, img, url, f, badge, meta, isEnd) {
  const pal = palette();
  img.hidden = false;
  $('#pov3d-canvas').hidden = true;
  img.onload = () => drawPov();
  if (img.getAttribute('src') !== url) img.src = url;
  badge.textContent = isEnd ? 'Real L7 end frame' : 'Real L7 camera';
  const tagsPx = Array.isArray(f.tags_px) ? f.tags_px : [];
  const iw = img.naturalWidth || 1280;
  const ih = img.naturalHeight || 720;
  const scale = Math.min(w / iw, h / ih);
  const ox = (w - iw * scale) / 2;
  const oy = (h - ih * scale) / 2;
  ctx.strokeStyle = 'rgba(96,165,250,0.6)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(w / 2 - 8, h / 2);
  ctx.lineTo(w / 2 + 8, h / 2);
  ctx.moveTo(w / 2, h / 2 - 8);
  ctx.lineTo(w / 2, h / 2 + 8);
  ctx.stroke();
  tagsPx.forEach((t) => {
    const raw = t.c;
    if (!Array.isArray(raw)) return;
    const pts = [];
    if (raw.length === 8 && typeof raw[0] === 'number') {
      for (let i = 0; i < 8; i += 2) pts.push([ox + raw[i] * scale, oy + raw[i + 1] * scale]);
    } else if (raw.length >= 4 && Array.isArray(raw[0])) {
      raw.forEach((p) => pts.push([ox + p[0] * scale, oy + p[1] * scale]));
    }
    if (pts.length < 4) return;
    const col = t.how === 'trk' ? pal.accent : pal.succ;
    ctx.strokeStyle = col;
    ctx.lineWidth = 2;
    ctx.beginPath();
    pts.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
    ctx.closePath();
    ctx.stroke();
    ctx.fillStyle = col;
    ctx.font = MONO_SM;
    ctx.textAlign = 'left';
    ctx.fillText(`#${t.id}${t.how ? ` ${t.how}` : ''}`, pts[0][0], Math.max(12, pts[0][1] - 4));
  });
  const tagStr = tagsPx.length ? tagsPx.map((t) => `#${t.id}`).join(', ') : 'no tags';
  const age = f.tag_age != null ? ` · age ${Number(f.tag_age).toFixed(2)} s` : '';
  meta.textContent = `t ${(f.t ?? 0).toFixed(2)} s · detected ${tagStr}${age}${f.reason ? ` · ${f.reason}` : ''}`;
}

function drawSyntheticPov(ctx, W, H, f, meta) {
  const pose = poseOf(f) || { x: 0, y: 1, yaw: 0 };
  const gl = $('#pov3d-canvas');
  const ok = gl && window.POV3D?.render(gl, pose, AR.target, AR.bundle?.geometry);
  gl.hidden = !ok;
  if (!ok) {
    ctx.fillStyle = '#9aa4b2';
    ctx.font = MONO_SM;
    ctx.textAlign = 'center';
    ctx.fillText('WebGL unavailable', W / 2, H / 2);
  }
  const yawDeg = ((pose.yaw * 180) / Math.PI).toFixed(1);
  const seen = f.valid ? 'tag in view' : 'no tag';
  meta.textContent = `x ${pose.x.toFixed(2)} y ${pose.y.toFixed(2)} yaw ${yawDeg}° · ${seen}${f.reason ? ` · ${f.reason}` : ''}`;
}

// ---- telemetry readout ----
function renderTelemetry() {
  const box = $('#telemetry');
  const ep = AR.episodes[AR.epIdx];
  const frames = ep?.frames || [];
  const f = frames[AR.frame];
  if (!f) {
    box.innerHTML = '<h4>Live step</h4><p class="muted">No step selected.</p>';
    return;
  }
  const p = poseOf(f);
  const t = tagOf(f);
  const a = Array.isArray(f.act) ? f.act : [0, 0, 0];
  const fmtP = (q) => (q ? `${q.x.toFixed(3)}, ${q.y.toFixed(3)}, ${((q.yaw * 180) / Math.PI).toFixed(1)}°` : '—');
  const blocked = !!f.shield_blocked;
  const rows = [
    ['Pose (x, y, yaw)', fmtP(p)],
    ['Tag (x, y, yaw)', f.valid && t ? fmtP(t) : 'no tag'],
    ['Tag age', f.tag_age != null ? `${Number(f.tag_age).toFixed(2)} ${Number.isInteger(Number(f.tag_age)) ? 'steps' : 's'}` : '—'],
    ['Action (vx, vy, w)', a.map((v) => Number(v || 0).toFixed(2)).join(', ')],
    ['Shield', `<span class="badge ${blocked ? 'err' : 'ok'}">${blocked ? 'blocked' : 'clear'}</span>`],
    ['Reason', esc(f.reason || '—')],
    ['IMU w · yaw', `${fmt.num(f.imu_w, 3)} · ${fmt.num(f.imu_yaw, 3)}`],
  ];
  box.innerHTML = `<h4>Live step · t ${fmt.num(f.t, 2)} s</h4><dl>${rows
    .map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`)
    .join('')}</dl>`;
}
