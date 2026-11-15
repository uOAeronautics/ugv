// Target airframes for the top-down arena: STRL001 fixed-wing plane belly and quadcopter skids.
// Geometry is in the policy frame (metres, +y forward). `S(x, y, z)` maps to canvas pixels.
// Airframe colours are physical (checkerboard, props) and fixed; labels use the theme ink.

function strokeCapsule(ctx, x1, y1, x2, y2, rPx) {
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.lineCap = 'round';
  ctx.lineWidth = rPx * 2;
  ctx.stroke();
  ctx.lineCap = 'butt';
}

function fillPoly(ctx, pts, fill, stroke, lw = 1.5) {
  if (!pts.length) return;
  ctx.beginPath();
  pts.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
  ctx.closePath();
  if (fill) {
    ctx.fillStyle = fill;
    ctx.fill();
  }
  if (stroke) {
    ctx.strokeStyle = stroke;
    ctx.lineWidth = lw;
    ctx.stroke();
  }
}

// Wheels on the floor (z = 0): nose, left main, right main. Fin-tip skids at x = ±0.58.
const STRL_WHEELS = [
  { x: 0.0, y: -0.45, r: 0.075, w: 0.05, label: 'Nose gear' },
  { x: -0.28, y: -0.05, r: 0.085, w: 0.06, label: 'Left main' },
  { x: 0.28, y: -0.05, r: 0.085, w: 0.06, label: 'Right main' },
];

function drawStrlFloor(ctx, S, ppm, L, geo, pal) {
  const inf = geo.inflation ?? 0.015;

  // ground shadow of wing, pod and tail
  const [a, b] = S(-1.2, -0.25, 0);
  const [c, d] = S(1.2, -0.25, 0);
  const [e, f] = S(1.2, 0.15, 0);
  const [g, hh] = S(-1.2, 0.15, 0);
  fillPoly(ctx, [[a, b], [c, d], [e, f], [g, hh]], 'rgba(37,99,235,0.06)', null);

  // gear is labelled in the legend and the caption, not on the canvas: the wing covers it
  STRL_WHEELS.forEach((wh) => {
    const hw = wh.w / 2;
    const hl = wh.r;
    const pts = [S(wh.x - hw, wh.y - hl, 0), S(wh.x + hw, wh.y - hl, 0), S(wh.x + hw, wh.y + hl, 0), S(wh.x - hw, wh.y + hl, 0)];
    fillPoly(ctx, pts, 'rgba(100,116,139,0.35)', pal.accent, 1.6);
  });

  [-0.58, 0.58].forEach((fx) => {
    const [x1, y1] = S(fx, 0.34, 0);
    const [x2, y2] = S(fx, 0.48, 0);
    ctx.strokeStyle = pal.muted;
    strokeCapsule(ctx, x1, y1, x2, y2, 0.022 * ppm);
  });

  if (L.shield) {
    ctx.strokeStyle = pal.warn;
    ctx.setLineDash([5, 4]);
    ctx.lineWidth = 1.4;
    STRL_WHEELS.forEach((wh) => {
      const rPad = Math.max(wh.r, wh.w / 2) + inf;
      const [wx, wy] = S(wh.x, wh.y, 0);
      ctx.beginPath();
      ctx.arc(wx, wy, rPad * ppm, 0, Math.PI * 2);
      ctx.stroke();
    });
    [-0.58, 0.58].forEach((fx) => {
      const [x1, y1] = S(fx, 0.34, 0);
      const [x2, y2] = S(fx, 0.48, 0);
      ctx.globalAlpha = 0.5;
      strokeCapsule(ctx, x1, y1, x2, y2, (0.022 + inf) * ppm);
      ctx.globalAlpha = 1;
    });
    ctx.setLineDash([]);
  }

  ctx.font = `bold ${MONO_SM}`;
  ctx.textAlign = 'center';
  const [pr, prY] = S(0.86, -0.22, 0);
  const [pl, plY] = S(-0.86, -0.22, 0);
  ctx.fillStyle = pal.succ;
  ctx.fillText('PORT (LEFT +X)', pr, prY);
  ctx.fillStyle = pal.warn;
  ctx.fillText('STBD (RIGHT -X)', pl, plY);
}

// Elevated airframe: drawn over the UGV, so the car reads as sitting under the wing.
function drawStrlAirframe(ctx, S, ppm, L, pal) {
  const zPod = 0.4;
  const podPts = [
    S(-0.11, -0.46, zPod), S(0.11, -0.46, zPod), S(0.15, -0.26, zPod), S(0.15, 0.26, zPod),
    S(0.11, 0.46, zPod), S(-0.11, 0.46, zPod), S(-0.15, 0.26, zPod), S(-0.15, -0.26, zPod),
  ];
  fillPoly(ctx, podPts, 'rgba(148,163,184,0.16)', pal.muted, 1.4);
  const [tx, ty] = S(0, -0.33, 0.24);
  ctx.strokeStyle = pal.warn;
  ctx.lineWidth = 1.3;
  ctx.beginPath();
  ctx.arc(tx, ty, 0.055 * ppm, 0, Math.PI * 2);
  ctx.stroke();

  // high wing with the checkerboard band (semi-transparent so the UGV shows through)
  const zW = 0.585;
  const wingPts = [S(-1.2, -0.14, zW), S(1.2, -0.14, zW), S(1.2, 0.06, zW), S(0.95, 0.16, zW), S(-0.95, 0.16, zW), S(-1.2, 0.06, zW)];
  fillPoly(ctx, wingPts, 'rgba(255,255,255,0.10)', pal.ink2, 1.4);
  const cols = 16;
  const x0 = -1.12;
  const dx = 2.24 / cols;
  const yA = -0.13;
  const yB = -0.03;
  const yC = 0.07;
  for (let i = 0; i < cols; i++) {
    const lx0 = x0 + i * dx;
    const lx1 = lx0 + dx;
    const c1 = i % 2 === 0 ? 'rgba(248,250,252,0.32)' : 'rgba(15,23,42,0.5)';
    const c2 = i % 2 === 0 ? 'rgba(234,88,12,0.32)' : 'rgba(248,250,252,0.22)';
    fillPoly(ctx, [S(lx0, yA, zW), S(lx1, yA, zW), S(lx1, yB, zW), S(lx0, yB, zW)], c1, null);
    fillPoly(ctx, [S(lx0, yB, zW), S(lx1, yB, zW), S(lx1, yC, zW), S(lx0, yC, zW)], c2, null);
  }

  // twin tractor-prop nacelles above the wing
  const zNac = 0.655;
  [-0.43, 0.43].forEach((nx) => {
    const nPts = [S(nx - 0.055, -0.27, zNac), S(nx + 0.055, -0.27, zNac), S(nx + 0.055, 0.07, zNac), S(nx - 0.055, 0.07, zNac)];
    fillPoly(ctx, nPts, 'rgba(148,163,184,0.2)', pal.ink2, 1.2);
    const [p1x, p1y] = S(nx - 0.111, -0.32, 0.685);
    const [p2x, p2y] = S(nx + 0.111, -0.32, 0.685);
    const [hx, hy] = S(nx, -0.34, 0.685);
    ctx.strokeStyle = '#ea580c';
    ctx.lineWidth = 2.4;
    ctx.beginPath();
    ctx.moveTo(p1x, p1y);
    ctx.lineTo(p2x, p2y);
    ctx.stroke();
    ctx.fillStyle = pal.muted;
    ctx.beginPath();
    ctx.arc(hx, hy, 3, 0, Math.PI * 2);
    ctx.fill();
  });

  // sloped inverted-V tail fins: root z = 0.46, tips z = 0.075 at x = ±0.60
  [-1, 1].forEach((s) => {
    const finPts = [S(s * 0.12, 0.16, 0.46), S(s * 0.6, 0.28, 0.075), S(s * 0.6, 0.56, 0.075), S(s * 0.12, 0.46, 0.46)];
    fillPoly(ctx, finPts, 'rgba(255,255,255,0.10)', '#ea580c', 1.5);
    for (let k = 0; k < 4; k++) {
      const t0 = k / 4;
      const t1 = (k + 1) / 4;
      const p0 = [s * (0.12 + 0.48 * t0), 0.16 + 0.12 * t0, 0.46 - 0.385 * t0];
      const p1 = [s * (0.12 + 0.48 * t1), 0.16 + 0.12 * t1, 0.46 - 0.385 * t1];
      const [ax, ay] = S(...p0);
      const [bx, by] = S(...p1);
      ctx.strokeStyle = k % 2 === 0 ? '#ea580c' : '#e2e8f0';
      ctx.lineWidth = 2.6;
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();
    }
  });

  if (L.tags) {
    [
      { id: 20, x: -0.435, y: 0.35, z: 0.245, cant: -0.35, side: -1 },
      { id: 21, x: 0.435, y: 0.35, z: 0.245, cant: 0.35, side: 1 },
      { id: 22, x: 0, y: 0.02, z: 0.345, cant: 0, side: 0 },
    ].forEach((t) => {
      const [ax, ay] = S(t.x, t.y, t.z);
      const sz = Math.max(9, 0.05 * ppm);
      ctx.save();
      ctx.translate(ax, ay);
      ctx.rotate(t.cant);
      ctx.fillStyle = pal.sim;
      ctx.fillRect(-sz / 2, -sz / 2, sz, sz);
      ctx.strokeStyle = pal.ink;
      ctx.lineWidth = 1;
      ctx.strokeRect(-sz / 2, -sz / 2, sz, sz);
      ctx.fillStyle = '#ffffff';
      ctx.font = `bold 9px "JetBrains Mono", monospace`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(String(t.id), 0, 0.5);
      ctx.restore();
      ctx.textBaseline = 'alphabetic';
      ctx.fillStyle = pal.sim;
      ctx.font = `bold ${MONO_SM}`;
      ctx.textAlign = t.side < 0 ? 'right' : t.side > 0 ? 'left' : 'center';
      ctx.fillText(`#${t.id}`, ax + t.side * (sz / 2 + 6), ay + (t.id === 22 ? -sz / 2 - 6 : 4));
    });
  }
}

// Quadcopter target: tags #14/#6/#3 on the deck, true-scale skids on the floor.
const QUAD_TAGS = [
  { id: 14, x: -0.028, y: 0, z: 0.14 },
  { id: 6, x: 0.06, y: -0.05, z: 0.15 },
  { id: 3, x: -0.105, y: -0.049, z: 0.16 },
];

function drawQuadSkids(ctx, S, ppm, L, geo, pal) {
  if (L.tags) {
    const tags = geo.tags || QUAD_TAGS;
    tags.forEach((t) => {
      const [tx, ty] = S(t.x, t.y, t.z ?? 0.15);
      const sz = 0.05 * ppm;
      ctx.fillStyle = pal.sim;
      ctx.fillRect(tx - sz / 2, ty - sz / 2, sz, sz);
      ctx.strokeStyle = pal.ink;
      ctx.lineWidth = 1;
      ctx.strokeRect(tx - sz / 2, ty - sz / 2, sz, sz);
      ctx.fillStyle = '#ffffff';
      ctx.font = `bold 9px "JetBrains Mono", monospace`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(String(t.id), tx, ty + 0.5);
      ctx.textBaseline = 'alphabetic';
    });
  }
  const skidX = geo.skid_x ?? 0.157;
  const halfLen = (geo.gear_len ?? 0.3048) / 2;
  const r = geo.gear_r ?? 0.03;
  [-skidX, skidX].forEach((sx) => {
    const [x1, y1] = S(sx, -halfLen, 0);
    const [x2, y2] = S(sx, halfLen, 0);
    ctx.strokeStyle = pal.accent;
    ctx.globalAlpha = 0.85;
    strokeCapsule(ctx, x1, y1, x2, y2, r * ppm);
    ctx.globalAlpha = 1;
  });
  if (L.shield) {
    const rInf = (geo.gear_r ?? 0.03) + (geo.inflation ?? 0.015);
    [-skidX, skidX].forEach((sx) => {
      const [x1, y1] = S(sx, -halfLen, 0);
      const [x2, y2] = S(sx, halfLen, 0);
      ctx.strokeStyle = pal.warn;
      ctx.setLineDash([5, 4]);
      ctx.lineWidth = 1.4;
      strokeCapsule(ctx, x1, y1, x2, y2, rInf * ppm);
      ctx.setLineDash([]);
    });
  }
}
