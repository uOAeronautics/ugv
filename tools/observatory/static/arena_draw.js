// Top-down arena canvas: grid, goal, blind ring, UGV, pose ghosts, trail, FOV and target airframe.

// Pose / tag / action accessors. Frames store [x, y, yaw] arrays or {x,y,yaw} objects.
function toPose(p) {
  if (!p) return null;
  if (Array.isArray(p)) return { x: Number(p[0]) || 0, y: Number(p[1]) || 0, yaw: Number(p[2]) || 0 };
  return { x: Number(p.x) || 0, y: Number(p.y) || 0, yaw: Number(p.yaw) || 0 };
}
const poseOf = (f) => toPose(f?.pose);
const tagOf = (f) => toPose(f?.tag);
const auxOf = (f) => toPose(f?.aux_pose) || poseOf(f);

// World bounds that frame the whole episode: every pose, the goal and the airframe.
function episodeBox(strl) {
  const frames = AR.episodes[AR.epIdx]?.frames || [];
  let [x0, x1, y0, y1] = strl ? [-1.3, 1.3, -0.6, 0.5] : [-0.5, 0.5, -0.5, 0.5];
  for (const f of frames) {
    const p = poseOf(f);
    if (!p) continue;
    x0 = Math.min(x0, p.x - 0.15);
    x1 = Math.max(x1, p.x + 0.15);
    y0 = Math.min(y0, p.y - 0.15);
    y1 = Math.max(y1, p.y + 0.15);
  }
  return [x0, x1, y0, y1];
}

function arenaProjector(w, h, strl, view25) {
  const k = view25 ? 0.72 : 1;
  const V = AR.view;
  let vx;
  let vy;
  let ppm;
  if (V && V.user) {
    ({ x: vx, y: vy, ppm } = V);
  } else {
    const [x0, x1, y0, y1] = episodeBox(strl);
    const pad = 56;
    ppm = Math.max(40, Math.min(900, (w - 2 * pad) / (x1 - x0), (h - 2 * pad) / ((y1 - y0) * k)));
    vx = (x0 + x1) / 2;
    vy = (y0 + y1) / 2;
    AR.view = { x: vx, y: vy, ppm, user: false };
  }
  const S = (x, y, z = 0) => [w / 2 + (x - vx) * ppm, h / 2 - (y - vy) * ppm * k - (view25 ? z * ppm * 0.68 : 0)];
  return { ppm, vx, vy, k, w, h, S };
}

function drawArenaGrid(ctx, w, h, P, pal) {
  const step = P.ppm >= 110 ? 0.25 : 0.5;
  const xHalf = w / 2 / P.ppm;
  const yHalf = h / 2 / P.ppm / P.k;
  ctx.font = MONO_SM;
  const isMajor = (g) => Math.abs(Math.round(g / 0.5) * 0.5 - g) < 1e-9;
  for (let gx = Math.floor((P.vx - xHalf) / step) * step; gx <= P.vx + xHalf; gx += step) {
    const [sx] = P.S(gx, 0);
    const major = isMajor(gx);
    ctx.strokeStyle = major ? pal.gridStrong : pal.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(Math.round(sx) + 0.5, 0);
    ctx.lineTo(Math.round(sx) + 0.5, h);
    ctx.stroke();
    if (major && Math.abs(gx) > 1e-9 && sx > 48) {
      ctx.fillStyle = pal.muted;
      ctx.textAlign = 'center';
      ctx.fillText(`${gx.toFixed(1)} m`, sx, h - 6);
    }
  }
  for (let gy = Math.floor((P.vy - yHalf) / step) * step; gy <= P.vy + yHalf; gy += step) {
    const [, sy] = P.S(0, gy, 0);
    const major = isMajor(gy);
    ctx.strokeStyle = major ? pal.gridStrong : pal.grid;
    ctx.beginPath();
    ctx.moveTo(0, Math.round(sy) + 0.5);
    ctx.lineTo(w, Math.round(sy) + 0.5);
    ctx.stroke();
    if (major && Math.abs(gy) > 1e-9 && sy < h - 22) {
      ctx.fillStyle = pal.muted;
      ctx.textAlign = 'left';
      ctx.fillText(`${gy.toFixed(1)} m`, 6, sy - 4);
    }
  }
  const [ax, ay] = P.S(0, 0);
  ctx.strokeStyle = pal.ink2;
  ctx.globalAlpha = 0.45;
  ctx.beginPath();
  ctx.moveTo(0, ay);
  ctx.lineTo(w, ay);
  ctx.moveTo(ax, 0);
  ctx.lineTo(ax, h);
  ctx.stroke();
  ctx.globalAlpha = 1;
}

function drawGoal(ctx, P, pal) {
  const k = P.k;
  const [gx, gy] = P.S(0, 0, 0);
  const gw = 0.06 * P.ppm;
  const gh = 0.1 * P.ppm * k;
  ctx.strokeStyle = pal.succ;
  ctx.lineWidth = 2;
  ctx.strokeRect(gx - gw / 2, gy - gh / 2, gw, gh);
  const tipY = gy + gh;
  ctx.beginPath();
  ctx.moveTo(gx, gy);
  ctx.lineTo(gx, tipY);
  ctx.moveTo(gx, tipY);
  ctx.lineTo(gx - 4, tipY - 8);
  ctx.moveTo(gx, tipY);
  ctx.lineTo(gx + 4, tipY - 8);
  ctx.stroke();
  ctx.fillStyle = pal.succ;
  ctx.font = MONO_SM;
  ctx.textAlign = 'left';
  ctx.fillText('goal', gx + 6, gy - 6);
}

// The ring is named by its layer chip, so no text sits on the wing or the tags.
function drawBlindRing(ctx, P, pal) {
  const [bx, by] = P.S(0, 0, 0);
  const r = 0.4 * P.ppm;
  ctx.strokeStyle = pal.warn;
  ctx.setLineDash([6, 5]);
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  if (P.k !== 1) ctx.ellipse(bx, by, r, r * P.k, 0, 0, Math.PI * 2);
  else ctx.arc(bx, by, r, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);
}

function drawTrail(ctx, frames, upto, P, pal) {
  let started = false;
  ctx.strokeStyle = pal.sim;
  ctx.lineWidth = 1.6;
  ctx.globalAlpha = 0.85;
  ctx.beginPath();
  for (let i = 0; i <= upto && i < frames.length; i++) {
    const p = poseOf(frames[i]);
    if (!p) continue;
    const [sx, sy] = P.S(p.x, p.y, 0);
    if (!started) {
      ctx.moveTo(sx, sy);
      started = true;
    } else ctx.lineTo(sx, sy);
  }
  ctx.stroke();
  ctx.globalAlpha = 1;
}

function drawFov(ctx, P, pose, geo, pal) {
  const half = ((geo.fov_half_deg ?? 33.5) * Math.PI) / 180;
  const range = 1.4;
  const [ux, uy] = P.S(pose.x, pose.y, 0);
  // policy frame: yaw 0 faces -y, positive yaw is counter-clockwise
  const fx = Math.sin(pose.yaw);
  const fy = -Math.cos(pose.yaw);
  const angL = pose.yaw - half;
  const angR = pose.yaw + half;
  const [lx, ly] = P.S(pose.x + range * Math.sin(angL), pose.y - range * Math.cos(angL), 0);
  const [rx, ry] = P.S(pose.x + range * Math.sin(angR), pose.y - range * Math.cos(angR), 0);
  const [mx, my] = P.S(pose.x + range * fx, pose.y + range * fy, 0);
  ctx.fillStyle = pal.accent;
  ctx.globalAlpha = 0.12;
  ctx.beginPath();
  ctx.moveTo(ux, uy);
  ctx.lineTo(lx, ly);
  ctx.quadraticCurveTo(mx, my, rx, ry);
  ctx.closePath();
  ctx.fill();
  ctx.globalAlpha = 0.45;
  ctx.strokeStyle = pal.accent;
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.globalAlpha = 1;
}

// Ghosts are named by their layer chips (colour-coded), so they carry no canvas text.
function drawGhost(ctx, P, pose, color, dash) {
  const [sx, sy] = P.S(pose.x, pose.y, 0);
  const w = 0.16 * P.ppm;
  const hgt = 0.22 * P.ppm;
  ctx.save();
  ctx.translate(sx, sy);
  ctx.rotate(-pose.yaw);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.setLineDash(dash);
  ctx.strokeRect(-w / 2, -hgt / 2, w, hgt);
  ctx.setLineDash([]);
  ctx.restore();
}

function drawUgv(ctx, P, pose, f, geo, pal) {
  const [ux, uy] = P.S(pose.x, pose.y, 0);
  const blocked = !!f.shield_blocked;
  const body = blocked ? pal.err : pal.succ;
  const bw = (geo.car_w ?? 0.16) * P.ppm;
  const bl = (geo.car_l ?? 0.22) * P.ppm;
  ctx.save();
  ctx.translate(ux, uy);
  ctx.rotate(-pose.yaw);
  // wheels
  const ww = 0.03 * P.ppm;
  const wl = 0.06 * P.ppm;
  ctx.fillStyle = pal.ink2;
  [
    [-bw / 2 - ww / 2, -bl / 2 + 2],
    [bw / 2 - ww / 2, -bl / 2 + 2],
    [-bw / 2 - ww / 2, bl / 2 - wl - 2],
    [bw / 2 - ww / 2, bl / 2 - wl - 2],
  ].forEach(([wx, wy]) => ctx.fillRect(wx, wy, ww, wl));
  ctx.globalAlpha = 0.2;
  ctx.fillStyle = body;
  ctx.fillRect(-bw / 2, -bl / 2, bw, bl);
  ctx.globalAlpha = 1;
  ctx.strokeStyle = body;
  ctx.lineWidth = 2;
  ctx.strokeRect(-bw / 2, -bl / 2, bw, bl);
  // heading arrow
  const tipY = bl / 2 + 6;
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.lineTo(0, tipY);
  ctx.moveTo(0, tipY);
  ctx.lineTo(-5, tipY - 8);
  ctx.moveTo(0, tipY);
  ctx.lineTo(5, tipY - 8);
  ctx.stroke();
  ctx.restore();
  ctx.fillStyle = body;
  ctx.font = `600 ${MONO_SM}`;
  ctx.textAlign = 'left';
  ctx.fillText(blocked ? 'UGV · BLOCKED' : 'UGV', ux + bw / 2 + 6, uy + 4);
}

function drawArena() {
  const c = $('#arena-canvas');
  if (!c) return;
  const { ctx, w, h } = fitCanvas(c);
  const pal = palette();
  ctx.fillStyle = pal.plot;
  ctx.fillRect(0, 0, w, h);

  const geo = AR.bundle?.geometry || {};
  const strl = AR.target === 'strl001_belly';
  const P = arenaProjector(w, h, strl, AR.view25);
  drawArenaGrid(ctx, w, h, P, pal);

  const L = AR.layers;
  const ep = AR.episodes[AR.epIdx];
  const frames = ep?.frames || [];
  const f = frames[AR.frame] || {};
  const pose = poseOf(f);

  if (L.blind) drawBlindRing(ctx, P, pal);
  drawGoal(ctx, P, pal);
  if (strl) drawStrlFloor(ctx, P.S, P.ppm, { ...L, shield: L.shield }, geo, pal);
  else drawQuadSkids(ctx, P.S, P.ppm, L, geo, pal);
  if (L.trail && frames.length) drawTrail(ctx, frames, AR.frame, P, pal);
  if (L.fov && pose) drawFov(ctx, P, pose, geo, pal);
  const tag = tagOf(f);
  if (L.tag && tag) drawGhost(ctx, P, tag, pal.warn, [4, 4]);
  const aux = auxOf(f);
  if (L.aux && aux) drawGhost(ctx, P, aux, pal.gz, [2, 3]);
  if (L.true && pose) drawUgv(ctx, P, pose, f, geo, pal);
  if (strl) drawStrlAirframe(ctx, P.S, P.ppm, L, pal);

  ctx.fillStyle = pal.ink2;
  ctx.font = MONO_SM;
  ctx.textAlign = 'left';
  const airLbl = strl ? 'STRL001 plane belly' : 'quad skids';
  const n = Math.max(0, frames.length - 1);
  ctx.fillText(`t ${(f.t ?? 0).toFixed(2)} s · frame ${AR.frame}/${n} · ${airLbl} · ${AR.view25 ? '2.5D' : '2D'}`, 12, 18);
  if (!frames.length) {
    ctx.fillStyle = pal.muted;
    ctx.textAlign = 'center';
    ctx.font = `13px ${cssVar('--font-ui') || 'sans-serif'}`;
    ctx.fillText('This bundle has no step frames', w / 2, h / 2);
  } else if (!pose) {
    ctx.fillStyle = pal.muted;
    ctx.textAlign = 'center';
    ctx.font = `13px ${cssVar('--font-ui') || 'sans-serif'}`;
    ctx.fillText('No pose in this frame', w / 2, h / 2);
  }
}
