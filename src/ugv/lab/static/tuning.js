// How the controller's parameters were found: the failure-guided search, one run at a time.
import { esc, outcomeLabel, pct } from './app.js';
import { lineChart, scatter } from './charts.js';

export function tuning(root, searches) {
  if (!searches.length) { root.innerHTML = '<div class="empty">No search yet: <code>uv run python tools/park/search.py --out runs/park_search/s1</code></div>'; return; }
  const list = searches.slice().reverse();
  root.innerHTML = `
    <section class="ruled c12">
      <div class="sp12 row" style="padding-block:14px"><span class="lbl" style="margin:0" data-tip="Each candidate changes a few parameters, chosen among those that plausibly caused its parent's failures. It is kept only if it beats its parent beyond noise and collides no more.">Tuning</span>
        <span class="grow"></span><div class="seg" id="tn-pick">${list.map((s, i) => `<button type="button" data-i="${i}">${esc(s.name)}</button>`).join('')}</div></div>
      <div class="sp8"><div class="lbl">Every candidate <small>● kept ○ rejected · lines join parent and child</small></div><div id="tn-c1"></div></div>
      <div class="sp4"><div class="lbl">Best kept so far</div><div id="tn-c2"></div></div>
      <div class="sp12"><div id="tn-t"></div></div>
    </section>`;
  const show = (i) => {
    root.querySelectorAll('#tn-pick button').forEach((b) => b.classList.toggle('on', +b.dataset.i === i));
    const nodes = list[i].nodes, best = list[i].best;
    const rate = (n) => n.success / n.n;
    const lo = Math.max(0, Math.min(...nodes.map(rate)) - 0.03), hi = Math.min(1, Math.max(...nodes.map(rate)) + 0.02);
    const byId = Object.fromEntries(nodes.map((n, j) => [n.id, { ...n, j }]));
    const col = (n) => (n.verdict === 'PASS' ? 'var(--signal)' : n.verdict === 'ROOT' ? 'var(--ink)' : 'var(--ink-3)');
    const change = (n) => Object.entries(n.change).map(([k, v]) => `${k}=${v}`).join(', ') || 'root';
    scatter(root.querySelector('#tn-c1'), nodes.map((n, j) => ({
      x: j, y: rate(n), color: n.verdict === 'FAIL' ? 'var(--bg)' : col(n), stroke: col(n), r: n.verdict === 'FAIL' ? 3 : 4.5,
      tip: `<b>${n.id}</b> ${n.verdict} · gen ${n.gen}<br>${n.success}/${n.n} parked · ${n.outcomes?.collision ?? 0} collisions<br>${esc(change(n))}`,
    })), {
      xLabel: 'candidate', yFmt: (v) => pct(v, 0), yDomain: [lo, hi], height: 260,
      edges: nodes.filter((n) => n.parent && byId[n.parent]).map((n) => [{ x: byId[n.parent].j, y: rate(byId[n.parent]) }, { x: byId[n.id].j, y: rate(n) }]),
    });
    let b = 0;
    lineChart(root.querySelector('#tn-c2'), [
      { name: 'best kept', points: nodes.map((n, j) => { b = Math.max(b, n.verdict === 'FAIL' ? b : rate(n)); return [j, b]; }), color: 'var(--signal)' },
      { name: 'candidate', points: nodes.map((n, j) => [j, rate(n)]), color: 'var(--ink-3)', dashed: true },
    ], { xLabel: 'candidate', yFmt: (v) => pct(v, 0), yDomain: [lo, hi], height: 260 });
    const kept = nodes.filter((n) => n.verdict !== 'FAIL');
    const fails = (n) => Object.entries(n.outcomes).filter(([k]) => k !== 'parked').map(([k, v]) => `${outcomeLabel(k).replace('abort: ', '')} ${v}`).join(', ');
    root.querySelector('#tn-t').innerHTML = `<table><thead><tr><th>Node</th><th class="wide-only">Gen</th><th class="num">Parked</th><th>Change from parent</th><th class="wide-only">Failures left</th></tr></thead><tbody>
      ${kept.map((n) => `<tr${n.id === best?.node ? ' class="best"' : ''}><td class="mono">${n.id}${n.id === best?.node ? ' ◀ best' : ''}</td><td class="mono wide-only">${n.gen}</td><td class="num">${n.success}/${n.n}</td>
      <td class="mono">${esc(change(n))}</td><td class="mono dim wide-only">${esc(fails(n))}</td></tr>`).join('')}</tbody></table>`;
  };
  root.querySelectorAll('#tn-pick button').forEach((btn) => btn.addEventListener('click', () => show(+btn.dataset.i)));
  show(0);
}
