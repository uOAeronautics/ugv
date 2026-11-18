// Plan view in the airframe frame (x nose -> right, y left -> up, metres). The airframe is drawn
// from its mesh triangles, coloured by what the car meets: overhead structure it drives under,
// parts only the camera stand hits, and parts the whole car hits.
import { api, color as resolve, css } from './app.js';

export const loadPlan = () => api('/api/park/plan');

export function makeView(canvas, extent = 2.4, centre = [-0.4, 0]) {
  const dpr = devicePixelRatio || 1;
  const w = canvas.clientWidth || 480;
  const h = canvas.clientHeight || w;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const scale = Math.min(w, h) / (2 * extent);
  const toPx = (x, y) => [w / 2 + (x - centre[0]) * scale, h / 2 - (y - centre[1]) * scale];
  return { canvas, ctx, w, h, size: Math.min(w, h), scale, toPx, extent, centre, dpr, layer: null };
}

/** The airframe silhouette, rendered once per view and theme into an offscreen layer. */
function silhouette(view, plan) {
  const key = `${document.documentElement.dataset.mode}:${view.w}x${view.h}:${view.scale}`;
  if (view.layer?.key === key) return view.layer.canvas;
  const off = document.createElement('canvas');
  off.width = view.canvas.width;
  off.height = view.canvas.height;
  const g = off.getContext('2d');
  g.setTransform(view.dpr, 0, 0, view.dpr, 0, 0);
  const over = css('--ink'), tall = css('--warn'), low = css('--bad');
  const { body_top_mm: bodyTop, car_top_mm: carTop } = plan;
  g.lineJoin = 'round';
  for (const t of plan.tris) {
    const lo = t[6];
    const [a, b] = view.toPx(t[0] / 1000, t[1] / 1000), [c, d] = view.toPx(t[2] / 1000, t[3] / 1000), [e, f] = view.toPx(t[4] / 1000, t[5] / 1000);
    g.beginPath(); g.moveTo(a, b); g.lineTo(c, d); g.lineTo(e, f); g.closePath();
    if (lo < bodyTop) { g.globalAlpha = 0.85; g.fillStyle = low; }
    else if (lo < carTop) { g.globalAlpha = 0.75; g.fillStyle = tall; }
    else { g.globalAlpha = 0.07; g.fillStyle = over; }
    g.fill();
    if (lo >= carTop) { g.globalAlpha = 0.05; g.strokeStyle = over; g.lineWidth = 0.6; g.stroke(); }
  }
  g.globalAlpha = 1;
  view.layer = { key, canvas: off };
  return off;
}

function rectPath(ctx, toPx, pose, length, width, x0 = -length / 2, y0 = -width / 2) {
  const [x, y, yaw] = pose;
  const c = Math.cos(yaw), s = Math.sin(yaw);
  const pts = [[x0, y0], [x0 + length, y0], [x0 + length, y0 + width], [x0, y0 + width]].map(([u, v]) => toPx(x + c * u - s * v, y + s * u + c * v));
  ctx.beginPath();
  pts.forEach(([px, py], i) => (i ? ctx.lineTo(px, py) : ctx.moveTo(px, py)));
  ctx.closePath();
  return pts;
}

export function drawCar(view, bay, pose, { color, dashed = false, fill = true } = {}) {
  const { ctx, toPx } = view;
  const car = bay.car;
  color = color || css('--signal');
  rectPath(ctx, toPx, pose, car.length, car.width);
  if (fill && !dashed) { ctx.globalAlpha = 0.22; ctx.fillStyle = color; ctx.fill(); ctx.globalAlpha = 1; }
  ctx.setLineDash(dashed ? [4, 3] : []);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.setLineDash([]);
  if (!dashed) {
    rectPath(ctx, toPx, pose, car.tower_x[1] - car.tower_x[0], car.tower_y[1] - car.tower_y[0], car.tower_x[0], car.tower_y[0]);
    ctx.fillStyle = color;
    ctx.fill();
    // heading: the camera's line of sight
    const [x, y, yaw] = pose;
    const [a, b] = toPx(x + Math.cos(yaw) * car.length * 0.5, y + Math.sin(yaw) * car.length * 0.5);
    const [c, d] = toPx(x + Math.cos(yaw) * car.length * 1.6, y + Math.sin(yaw) * car.length * 1.6);
    ctx.beginPath(); ctx.moveTo(a, b); ctx.lineTo(c, d); ctx.lineWidth = 1; ctx.stroke();
  }
}

/** Brackets at the corners of the parking tolerance box. */
function drawTarget(view, bay) {
  const { ctx, toPx } = view;
  const t = bay.target, tol = bay.tolerance;
  const L = bay.car.length / 2 + tol.x, W = bay.car.width / 2 + tol.y, k = 0.035;
  const c = Math.cos(t[2]), s = Math.sin(t[2]);
  const P = (u, v) => toPx(t[0] + c * u - s * v, t[1] + s * u + c * v);
  ctx.strokeStyle = css('--good');
  ctx.lineWidth = 2;
  for (const [sx, sy] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) {
    const a = P(sx * L, sy * (W - k)), b = P(sx * L, sy * W), d = P(sx * (L - k), sy * W);
    ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.lineTo(...d); ctx.stroke();
  }
}

export function drawWorld(view, bay, plan, { paths = [], car, carEst, label, grid = true, scaleBar = true } = {}) {
  const { ctx, w, h, toPx, scale } = view;
  ctx.clearRect(0, 0, w, h);
  if (grid) {
    ctx.strokeStyle = css('--rule');
    ctx.lineWidth = 1;
    for (let v = -6; v <= 6; v += 0.5) {
      const [a] = toPx(v, 0), [, b] = toPx(0, v);
      ctx.beginPath(); ctx.moveTo(Math.round(a) + 0.5, 0); ctx.lineTo(Math.round(a) + 0.5, h); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, Math.round(b) + 0.5); ctx.lineTo(w, Math.round(b) + 0.5); ctx.stroke();
    }
  }
  if (plan) ctx.drawImage(silhouette(view, plan), 0, 0, w, h);
  drawTarget(view, bay);
  for (const p of paths) {
    if (p.pts.length < 2) continue;
    ctx.beginPath();
    p.pts.forEach((q, i) => { const [x, y] = toPx(q[0], q[1]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.strokeStyle = resolve(p.color); ctx.globalAlpha = p.alpha ?? 1; ctx.lineWidth = p.width ?? 1.5;
    ctx.setLineDash(p.dashed ? [4, 4] : []); ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
  }
  if (carEst) drawCar(view, bay, carEst, { color: css('--ink-2'), dashed: true });
  if (car) drawCar(view, bay, car);
  if (label) {
    ctx.fillStyle = css('--ink');
    ctx.font = '600 13px "Barlow Condensed", sans-serif';
    ctx.fillText(label, 12, 20);
  }
  if (scaleBar) {
    const x0 = w - 16 - 0.5 * scale;
    ctx.strokeStyle = css('--ink-2'); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x0, h - 14); ctx.lineTo(w - 16, h - 14); ctx.moveTo(x0, h - 18); ctx.lineTo(x0, h - 10); ctx.moveTo(w - 16, h - 18); ctx.lineTo(w - 16, h - 10); ctx.stroke();
    ctx.fillStyle = css('--ink-2'); ctx.font = '11px "JetBrains Mono", monospace'; ctx.textAlign = 'right';
    ctx.fillText('0.5 m', w - 16, h - 22); ctx.textAlign = 'left';
  }
}
