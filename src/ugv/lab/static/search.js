import { api, esc, outcomeColor, pct } from './app.js';
import { lineChart, scatter } from './charts.js';

export async function render(main) {
  const searches = await api('/api/park/searches', { fresh: true });
  main.innerHTML = `
    <h1>Search tree</h1>
    <p class="lede">The controller's parameters are tuned by a failure-guided search: each candidate
    changes a few parameters, chosen mostly among those that plausibly cause its parent's failures,
    and is scored on the same episodes as every other candidate. It is kept only if it beats its
    parent by more than noise (PASS). Every node is recorded, failed ones included, so a later
    search starts from what was already tried.</p>
    <div id="sr"></div>`;
  const root = main.querySelector('#sr');
  if (!searches.length) {
    root.innerHTML = '<div class="empty">No search yet: <span class="mono">uv run python tools/park/search.py --out runs/park_search/s1</span></div>';
    return;
  }
  for (const s of searches) {
    const sec = document.createElement('section');
    sec.className = 'card';
    sec.style.marginBottom = '16px';
    const nodes = s.nodes;
    const passes = nodes.filter((n) => n.verdict === 'PASS').length;
    sec.innerHTML = `<h2>${esc(s.name)} <span class="muted">${nodes.length} nodes · ${passes} PASS · best ${esc(s.best?.node || '…')}</span></h2>
      <div class="grid g2"><div><div class="muted">success per candidate (lines join parent and child)</div><div class="t1"></div></div>
      <div><div class="muted">best so far vs every candidate</div><div class="t2"></div></div></div>
      <h2 style="margin-top:14px">Accepted changes</h2><div class="t3"></div>`;
    root.appendChild(sec);
    const lo = Math.max(0, Math.min(...nodes.map((n) => n.success / n.n)) - 0.05);
    const byId = Object.fromEntries(nodes.map((n, i) => [n.id, { ...n, i }]));
    const color = (n) => (n.verdict === 'PASS' ? 'var(--good)' : n.verdict === 'ROOT' ? 'var(--accent)' : 'var(--ink-3)');
    scatter(sec.querySelector('.t1'), nodes.map((n, i) => ({
      x: i, y: n.success / n.n, color: color(n), r: n.verdict === 'FAIL' ? 3.5 : 5,
      tip: `<b>${n.id}</b> ${n.verdict} · gen ${n.gen}<br>${n.success}/${n.n} parked<br>${esc(Object.entries(n.change).map(([k, v]) => `${k}=${v}`).join(', ') || 'root')}<br><span class="muted">${esc(JSON.stringify(n.outcomes))}</span>`,
    })), {
      xLabel: 'candidate', yFmt: (v) => pct(v, 0), yDomain: [lo, 1],
      edges: nodes.filter((n) => n.parent && byId[n.parent]).map((n) => [{ x: byId[n.parent].i, y: byId[n.parent].success / byId[n.parent].n }, { x: byId[n.id].i, y: n.success / n.n }]),
    });
    let best = 0;
    const bestSeries = nodes.map((n, i) => { best = Math.max(best, n.verdict === 'FAIL' ? best : n.success / n.n); return [i, best]; });
    lineChart(sec.querySelector('.t2'), [
      { name: 'best accepted', points: bestSeries },
      { name: 'candidate', points: nodes.map((n, i) => [i, n.success / n.n]), color: 'var(--ink-3)', dashed: true },
    ], { xLabel: 'candidate', yFmt: (v) => pct(v, 0), yDomain: [lo, 1] });
    const acc = nodes.filter((n) => n.verdict !== 'FAIL');
    sec.querySelector('.t3').innerHTML = `<table><thead><tr><th>node</th><th>gen</th><th class="num">parked</th><th>change</th><th>failures left</th></tr></thead><tbody>
      ${acc.map((n) => `<tr><td class="mono">${n.id}</td><td>${n.gen}</td><td class="num">${n.success}/${n.n}</td>
      <td class="mono">${esc(Object.entries(n.change).map(([k, v]) => `${k}=${v}`).join(', ') || 'root')}</td>
      <td>${Object.entries(n.outcomes).filter(([k]) => k !== 'parked').map(([k, v]) => `<span class="pill" style="--dot:${outcomeColor(k)}">${esc(k)} ${v}</span>`).join(' ')}</td></tr>`).join('')}
      </tbody></table>`;
  }
}
