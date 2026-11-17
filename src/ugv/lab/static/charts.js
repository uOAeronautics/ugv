// Small SVG charts: line, bar, scatter, histogram. One y-axis each, thin marks, recessive grid,
// a hover tooltip on every mark. Colours are CSS variables (--s1..--s8), assigned in fixed order.

const NS = 'http://www.w3.org/2000/svg';
export const SERIES = ['--s1', '--s2', '--s3', '--s4', '--s5', '--s6', '--s7', '--s8'].map((v) => `var(${v})`);

const tip = () => document.getElementById('tip');
export function showTip(evt, html) {
  const t = tip();
  t.innerHTML = html;
  t.hidden = false;
  const pad = 14;
  const w = t.offsetWidth, h = t.offsetHeight;
  let x = evt.clientX + pad, y = evt.clientY + pad;
  if (x + w > innerWidth - 8) x = evt.clientX - w - pad;
  if (y + h > innerHeight - 8) y = evt.clientY - h - pad;
  t.style.left = `${x}px`;
  t.style.top = `${y}px`;
}
export const hideTip = () => { tip().hidden = true; };

function el(name, attrs = {}, parent) {
  const n = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}

function niceTicks(lo, hi, count = 5) {
  if (!(hi > lo)) { hi = lo + 1; }
  const span = hi - lo;
  const step0 = span / count;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= count) || 10 * mag;
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) ticks.push(+v.toFixed(10));
  return ticks;
}

const fmt = (v) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : +v.toFixed(3));

function frame(container, { height = 220, xDomain, yDomain, xLabel, yLabel, xTicks, yFmt = fmt, xFmt = fmt, left = 44 }) {
  container.innerHTML = '';
  const width = Math.max(280, container.clientWidth || 480);
  const m = { l: left, r: 12, t: 10, b: xLabel ? 36 : 24 };
  const svg = el('svg', { class: 'chart', viewBox: `0 0 ${width} ${height}`, height, role: 'img' }, container);
  const iw = width - m.l - m.r, ih = height - m.t - m.b;
  const sx = (v) => m.l + ((v - xDomain[0]) / (xDomain[1] - xDomain[0] || 1)) * iw;
  const sy = (v) => m.t + ih - ((v - yDomain[0]) / (yDomain[1] - yDomain[0] || 1)) * ih;
  const g = el('g', { class: 'axis' }, svg);
  for (const t of niceTicks(yDomain[0], yDomain[1], 4)) {
    el('line', { x1: m.l, x2: m.l + iw, y1: sy(t), y2: sy(t), class: 'gridline' }, g);
    el('text', { x: m.l - 6, y: sy(t) + 4, 'text-anchor': 'end' }, g).textContent = yFmt(t);
  }
  el('line', { x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih }, g);
  if (xTicks !== false) {
    for (const t of xTicks || niceTicks(xDomain[0], xDomain[1], Math.max(3, Math.floor(iw / 90)))) {
      el('text', { x: sx(t), y: m.t + ih + 15, 'text-anchor': 'middle' }, g).textContent = xFmt(t);
    }
  }
  if (xLabel) el('text', { x: m.l + iw / 2, y: height - 4, 'text-anchor': 'middle' }, g).textContent = xLabel;
  if (yLabel) el('text', { x: m.l, y: m.t - 2 + 8, 'text-anchor': 'start', dx: 4 }, g).textContent = yLabel;
  return { svg, sx, sy, m, iw, ih, width, height };
}

export function legend(container, items) {
  if (items.length < 2) return;
  const div = document.createElement('div');
  div.className = 'legend';
  div.innerHTML = items.map(([name, color]) => `<span><i style="background:${color}"></i>${name}</span>`).join('');
  container.appendChild(div);
}

/** series: [{name, points: [[x, y], ...], color?, dashed?}] */
export function lineChart(container, series, opts = {}) {
  const xs = series.flatMap((s) => s.points.map((p) => p[0]));
  const ys = series.flatMap((s) => s.points.map((p) => p[1]));
  if (!xs.length) { container.innerHTML = '<div class="empty">no data yet</div>'; return; }
  const yDomain = opts.yDomain || [Math.min(0, ...ys), Math.max(...ys) * 1.05 || 1];
  const f = frame(container, { ...opts, xDomain: [Math.min(...xs), Math.max(...xs)], yDomain });
  series.forEach((s, i) => {
    const color = s.color || SERIES[i % SERIES.length];
    const d = s.points.map((p, j) => `${j ? 'L' : 'M'}${f.sx(p[0]).toFixed(1)},${f.sy(p[1]).toFixed(1)}`).join('');
    el('path', { d, fill: 'none', stroke: color, 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-dasharray': s.dashed ? '5 4' : '' }, f.svg);
  });
  // crosshair + tooltip
  const cross = el('line', { y1: f.m.t, y2: f.m.t + f.ih, stroke: 'var(--ink-3)', 'stroke-width': 1, visibility: 'hidden' }, f.svg);
  const hit = el('rect', { x: f.m.l, y: f.m.t, width: f.iw, height: f.ih, fill: 'transparent' }, f.svg);
  const allX = [...new Set(xs)].sort((a, b) => a - b);
  hit.addEventListener('mousemove', (evt) => {
    const r = f.svg.getBoundingClientRect();
    const px = ((evt.clientX - r.left) / r.width) * f.width;
    const x = allX.reduce((a, b) => (Math.abs(f.sx(b) - px) < Math.abs(f.sx(a) - px) ? b : a));
    cross.setAttribute('x1', f.sx(x)); cross.setAttribute('x2', f.sx(x)); cross.setAttribute('visibility', 'visible');
    const rows = series.map((s, i) => {
      const p = s.points.find((q) => q[0] === x);
      return p ? `<div><i style="display:inline-block;width:8px;height:8px;border-radius:2px;background:${s.color || SERIES[i]}"></i> ${s.name}: <b>${(opts.yFmt || fmt)(p[1])}</b></div>` : '';
    }).join('');
    showTip(evt, `<div class="muted">${opts.xLabel || 'x'} ${(opts.xFmt || fmt)(x)}</div>${rows}`);
  });
  hit.addEventListener('mouseleave', () => { cross.setAttribute('visibility', 'hidden'); hideTip(); });
  legend(container, series.map((s, i) => [s.name, s.color || SERIES[i]]));
}

/** bars: [{label, value, color?, tip?}] - horizontal bars, labels on the left. */
export function barChart(container, bars, opts = {}) {
  container.innerHTML = '';
  if (!bars.length) { container.innerHTML = '<div class="empty">no data yet</div>'; return; }
  const max = opts.max ?? Math.max(...bars.map((b) => b.value), 1e-9);
  const rowH = 26, labelW = opts.labelWidth || 130;
  const width = Math.max(280, container.clientWidth || 480);
  const height = bars.length * rowH + 6;
  const svg = el('svg', { class: 'chart', viewBox: `0 0 ${width} ${height}`, height }, container);
  const iw = width - labelW - 60;
  bars.forEach((b, i) => {
    const y = i * rowH + 4;
    el('text', { x: labelW - 8, y: y + 13, 'text-anchor': 'end' }, svg).textContent = b.label;
    el('rect', { x: labelW, y: y + 2, width: iw, height: rowH - 10, rx: 4, fill: 'var(--grid)' }, svg);
    const w = Math.max(2, (b.value / max) * iw);
    const r = el('rect', { x: labelW, y: y + 2, width: w, height: rowH - 10, rx: 4, fill: b.color || SERIES[0] }, svg);
    const t = el('text', { x: labelW + w + 6, y: y + 13 }, svg);
    t.textContent = opts.fmt ? opts.fmt(b.value, b) : fmt(b.value);
    t.style.fill = 'var(--ink-2)';
    for (const node of [r]) {
      node.addEventListener('mousemove', (evt) => showTip(evt, b.tip || `<b>${b.label}</b>: ${opts.fmt ? opts.fmt(b.value, b) : fmt(b.value)}`));
      node.addEventListener('mouseleave', hideTip);
    }
  });
}

/** points: [{x, y, color?, r?, tip?}] */
export function scatter(container, points, opts = {}) {
  if (!points.length) { container.innerHTML = '<div class="empty">no data yet</div>'; return; }
  const xs = points.map((p) => p.x), ys = points.map((p) => p.y);
  const pad = (lo, hi) => { const d = (hi - lo) * 0.08 || 1; return [lo - d, hi + d]; };
  const f = frame(container, { ...opts, xDomain: opts.xDomain || pad(Math.min(...xs), Math.max(...xs)), yDomain: opts.yDomain || pad(Math.min(...ys), Math.max(...ys)) });
  if (opts.box) {
    const [x0, x1, y0, y1] = opts.box;
    el('rect', { x: f.sx(x0), y: f.sy(y1), width: f.sx(x1) - f.sx(x0), height: f.sy(y0) - f.sy(y1), fill: 'none', stroke: 'var(--good)', 'stroke-dasharray': '4 3', 'stroke-width': 1.5 }, f.svg);
  }
  for (const e of opts.edges || []) {
    el('line', { x1: f.sx(e[0].x), y1: f.sy(e[0].y), x2: f.sx(e[1].x), y2: f.sy(e[1].y), stroke: 'var(--line)', 'stroke-width': 1.2 }, f.svg);
  }
  for (const p of points) {
    const c = el('circle', { cx: f.sx(p.x), cy: f.sy(p.y), r: p.r || 4, fill: p.color || SERIES[0], stroke: 'var(--surface)', 'stroke-width': 2 }, f.svg);
    c.addEventListener('mousemove', (evt) => showTip(evt, p.tip || `${fmt(p.x)}, ${fmt(p.y)}`));
    c.addEventListener('mouseleave', hideTip);
    if (p.onclick) { c.style.cursor = 'pointer'; c.addEventListener('click', p.onclick); }
  }
}

export function histogram(container, values, opts = {}) {
  if (!values.length) { container.innerHTML = '<div class="empty">no data yet</div>'; return; }
  const bins = opts.bins || 20;
  const lo = opts.lo ?? Math.min(...values), hi = opts.hi ?? Math.max(...values);
  const w = (hi - lo) / bins || 1;
  const counts = new Array(bins).fill(0);
  for (const v of values) counts[Math.min(bins - 1, Math.max(0, Math.floor((v - lo) / w)))]++;
  const f = frame(container, { ...opts, xDomain: [lo, hi], yDomain: [0, Math.max(...counts) * 1.1] });
  counts.forEach((c, i) => {
    const x0 = f.sx(lo + i * w) + 1, x1 = f.sx(lo + (i + 1) * w) - 1;
    const r = el('rect', { x: x0, y: f.sy(c), width: Math.max(1, x1 - x0), height: f.sy(0) - f.sy(c), rx: 2, fill: opts.color || SERIES[0] }, f.svg);
    r.addEventListener('mousemove', (evt) => showTip(evt, `${fmt(lo + i * w)}–${fmt(lo + (i + 1) * w)} ${opts.unit || ''}: <b>${c}</b>`));
    r.addEventListener('mouseleave', hideTip);
  });
}
