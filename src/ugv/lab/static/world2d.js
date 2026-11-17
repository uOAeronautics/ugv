// Top-down drawing in the airframe frame (x nose -> right, y left -> up), metres.

export function makeView(canvas, extent = 2.4, centre = [-0.4, 0]) {
  const dpr = devicePixelRatio || 1;
  const size = canvas.clientWidth || 480;
  canvas.width = size * dpr;
  canvas.height = size * dpr;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const scale = size / (2 * extent);
  const toPx = (x, y) => [size / 2 + (x - centre[0]) * scale, size / 2 - (y - centre[1]) * scale];
  return { canvas, ctx, size, scale, toPx, extent, centre };
}

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function rect(view, pose, length, width, x0 = -length / 2, y0 = -width / 2) {
  const { ctx, toPx } = view;
  const [x, y, yaw] = pose;
  const c = Math.cos(yaw), s = Math.sin(yaw);
  const pts = [[x0, y0], [x0 + length, y0], [x0 + length, y0 + width], [x0, y0 + width]].map(([u, v]) => toPx(x + c * u - s * v, y + s * u + c * v));
  ctx.beginPath();
  pts.forEach(([px, py], i) => (i ? ctx.lineTo(px, py) : ctx.moveTo(px, py)));
  ctx.closePath();
}

export function drawCar(view, bay, pose, { fill, stroke, dashed = false } = {}) {
  const { ctx } = view;
  const car = bay.car;
  rect(view, pose, car.length, car.width);
  ctx.setLineDash(dashed ? [5, 4] : []);
  if (fill) { ctx.fillStyle = fill; ctx.fill(); }
  ctx.strokeStyle = stroke || css('--ink');
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.setLineDash([]);
  if (!dashed) {
    // camera stand and heading tick
    rect(view, pose, car.tower_x[1] - car.tower_x[0], car.tower_y[1] - car.tower_y[0], car.tower_x[0], car.tower_y[0]);
    ctx.fillStyle = stroke || css('--ink');
    ctx.fill();
  }
}

export function drawWorld(view, bay, { path = [], est = [], car, carEst, state, grid = true } = {}) {
  const { ctx, size, toPx, scale } = view;
  ctx.clearRect(0, 0, size, size);
  ctx.fillStyle = css('--surface-2');
  ctx.fillRect(0, 0, size, size);
  if (grid) {
    ctx.strokeStyle = css('--grid');
    ctx.lineWidth = 1;
    for (let v = -6; v <= 6; v += 0.5) {
      const [a] = toPx(v, 0), [, b] = toPx(0, v);
      ctx.beginPath(); ctx.moveTo(a, 0); ctx.lineTo(a, size); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, b); ctx.lineTo(size, b); ctx.stroke();
    }
  }
  // airframe shadow (everything the car can pass under), then obstacles
  ctx.fillStyle = 'color-mix(in srgb, ' + css('--ink-3') + ' 45%, transparent)';
  const r = Math.max(1.2, 0.008 * scale);
  for (const [x, y] of bay.shadow || []) { const [px, py] = toPx(x, y); ctx.fillRect(px - r, py - r, 2 * r, 2 * r); }
  ctx.fillStyle = css('--serious');
  for (const [x, y] of bay.tall) { const [px, py] = toPx(x, y); ctx.fillRect(px - 1.5, py - 1.5, 3, 3); }
  ctx.fillStyle = css('--critical');
  for (const [x, y] of bay.obstacles) { const [px, py] = toPx(x, y); ctx.fillRect(px - 1.5, py - 1.5, 3, 3); }
  // target
  const t = bay.target, tol = bay.tolerance;
  rect(view, t, bay.car.length + 2 * tol.x, bay.car.width + 2 * tol.y);
  ctx.strokeStyle = css('--good');
  ctx.setLineDash([5, 4]);
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.setLineDash([]);
  const line = (pts, color, dash) => {
    if (pts.length < 2) return;
    ctx.beginPath();
    pts.forEach((p, i) => { const [px, py] = toPx(p[0], p[1]); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.setLineDash(dash ? [4, 4] : []); ctx.stroke(); ctx.setLineDash([]);
  };
  line(est, css('--s7'), true);
  line(path, css('--s1'), false);
  if (carEst) drawCar(view, bay, carEst, { stroke: css('--s7'), dashed: true });
  if (car) drawCar(view, bay, car, { fill: 'color-mix(in srgb, ' + css('--s1') + ' 25%, transparent)', stroke: css('--s1') });
  if (state) {
    ctx.fillStyle = css('--ink-2');
    ctx.font = '12px ui-monospace, monospace';
    ctx.fillText(state, 10, 18);
  }
  // scale bar
  const [a, b] = [toPx(view.centre[0] + view.extent - 0.7, 0)[0], toPx(view.centre[0] + view.extent - 0.2, 0)[0]];
  ctx.strokeStyle = css('--ink-3'); ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(a, size - 14); ctx.lineTo(b, size - 14); ctx.stroke();
  ctx.fillStyle = css('--ink-3'); ctx.font = '11px system-ui'; ctx.fillText('0.5 m', a, size - 20);
}
