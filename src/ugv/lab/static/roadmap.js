// The roadmap as a dependency graph: tasks in columns by how many prerequisites deep they sit,
// one line per `after` link. Same data as `ledger.py roadmap`.
import { esc } from './app.js';
import { hideTip, showTip } from './charts.js';

const NS = 'http://www.w3.org/2000/svg';

export function roadmap(container, tasks) {
  const live = tasks.filter((t) => t.state !== 'dropped');
  const byId = Object.fromEntries(live.map((t) => [t.id, t]));
  const after = (t) => (Array.isArray(t.after) ? t.after : []).filter((a) => byId[a]);
  const depth = {};
  const d = (t, seen = new Set()) => {
    if (depth[t.id] != null) return depth[t.id];
    if (seen.has(t.id)) return 0;
    seen.add(t.id);
    return (depth[t.id] = after(t).length ? 1 + Math.max(...after(t).map((a) => d(byId[a], seen))) : 0);
  };
  live.forEach((t) => d(t));
  const cols = [];
  for (const t of live) (cols[depth[t.id]] ||= []).push(t);
  const rank = (t) => (t.state === 'done' ? 2 : after(t).every((a) => byId[a].state === 'done') ? 0 : 1);
  cols.forEach((c) => c.sort((a, b) => rank(a) - rank(b) || a.id.localeCompare(b.id)));

  if ((container.clientWidth || 800) < 600) {
    const order = cols.flat().filter((t) => t.state !== 'done');
    container.innerHTML = `<div class="road-list">${order.map((t) => {
      const ready = after(t).every((a) => byId[a].state === 'done');
      return `<a href="#ledger/${t.id}"><span class="st ${ready ? 'live' : 'open'}"></span><b>${esc(t.id.slice(11))}</b><span>${ready ? 'ready' : 'after ' + after(t).filter((a) => byId[a].state !== 'done').map((a) => esc(a.slice(11))).join(', ')}</span></a>`;
    }).join('')}</div>`;
    return;
  }
  const W = Math.max(520, container.clientWidth || 800);
  const gap = 34, nh = 40, vgap = 8;
  const nw = Math.min(230, (W - gap * (cols.length - 1)) / cols.length);
  const H = Math.max(...cols.map((c) => c.length)) * (nh + vgap);
  const pos = {};
  cols.forEach((c, i) => c.forEach((t, j) => { pos[t.id] = { x: i * (nw + gap), y: j * (nh + vgap) }; }));

  container.innerHTML = '';
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('class', 'chart road');
  svg.style.height = `${H}px`;
  const el = (n, a, p = svg) => { const e = document.createElementNS(NS, n); for (const [k, v] of Object.entries(a)) e.setAttribute(k, v); p.appendChild(e); return e; };

  for (const t of live) {
    for (const a of after(t)) {
      const p = pos[a], q = pos[t.id];
      const x0 = p.x + nw, y0 = p.y + nh / 2, x1 = q.x, y1 = q.y + nh / 2, xm = x0 + gap / 2;
      el('path', { d: `M${x0},${y0}H${xm}V${y1}H${x1}`, fill: 'none', stroke: byId[a].state === 'done' ? 'var(--rule)' : 'var(--rule-2)', 'stroke-width': 1 });
    }
  }
  for (const t of live) {
    const { x, y } = pos[t.id];
    const ready = t.state !== 'done' && after(t).every((a) => byId[a].state === 'done');
    const kind = t.state === 'done' ? 'done' : t.state === 'claimed' ? 'claimed' : ready ? 'ready' : 'waiting';
    const g = el('g', { class: `node ${kind}`, tabindex: 0, role: 'link' });
    el('rect', { x: x + 0.5, y: y + 0.5, width: nw - 1, height: nh - 1, fill: 'var(--bg)', stroke: kind === 'ready' ? 'var(--ink)' : 'var(--rule-2)', 'stroke-width': 1 }, g);
    const sq = { done: ['var(--good)', 'var(--good)'], claimed: ['var(--warn)', 'var(--warn)'], ready: ['var(--signal)', 'var(--signal)'], waiting: ['var(--ink-3)', 'none'] }[kind];
    el('rect', { x: x + 9, y: y + 10, width: 7, height: 7, fill: sq[1], stroke: sq[0], 'stroke-width': 1.5 }, g);
    const name = el('text', { x: x + 24, y: y + 17, style: `font:600 13px var(--cond);letter-spacing:.02em;fill:${kind === 'done' ? 'var(--ink-3)' : 'var(--ink)'}` }, g);
    name.textContent = fit(t.id.slice(11), nw - 32, 6.2);
    const sub = el('text', { x: x + 24, y: y + 31, style: 'font:400 10.5px var(--mono);fill:var(--ink-3)' }, g);
    sub.textContent = `${kind.toUpperCase()} · ${t.owner}`;
    g.style.cursor = 'pointer';
    g.addEventListener('click', () => { location.hash = `ledger/${t.id}`; });
    g.addEventListener('keydown', (e) => { if (e.key === 'Enter') location.hash = `ledger/${t.id}`; });
    g.addEventListener('mousemove', (e) => showTip(e, `<b>${esc(t.id.slice(11))}</b><br>${esc(t.summary).slice(0, 260)}${t.summary.length > 260 ? '…' : ''}`));
    g.addEventListener('mouseleave', hideTip);
  }
  container.appendChild(svg);
}

const fit = (s, px, cw) => (s.length * cw <= px ? s : s.slice(0, Math.max(4, Math.floor(px / cw) - 1)) + '…');
