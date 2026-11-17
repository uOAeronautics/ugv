import { api, esc } from './app.js';

const OUT = { kept: 'good', negative: 'bad', invalid: 'bad', retracted: 'bad', abandoned: 'warn', open: 'info' };
const STATE = { open: 'info', claimed: 'warn', blocked: 'bad', done: 'good', dropped: '' };

export async function render(main) {
  const led = await api('/api/ledger', { fresh: true });
  const recs = led.records;
  const attempts = recs.filter((r) => r.kind === 'attempt');
  const tasks = recs.filter((r) => r.kind === 'task');
  const decisions = recs.filter((r) => r.kind === 'decision');
  main.innerHTML = `
    <h1>Research ledger</h1>
    <p class="lede">Decisions, attempts and tasks, one Markdown file each under
    <span class="mono">research/</span>. Attempts form a tree through their parent, like a
    Dream-RSI discovery tree: what was tried, how it ended and when it may be re-run.</p>
    <div class="row" style="margin-bottom:12px"><input id="rs-q" placeholder="filter…" style="flex:1;max-width:340px;padding:6px 10px;border-radius:7px;border:1px solid var(--line);background:var(--surface);color:var(--ink)"></div>
    <div class="grid g2">
      <section class="card"><h2>Tasks <span class="muted">${tasks.length}</span></h2><div id="rs-tasks"></div></section>
      <section class="card"><h2>Decisions <span class="muted">${decisions.length}</span></h2><div id="rs-dec"></div></section>
      <section class="card span2"><h2>Attempt tree <span class="muted">${attempts.length} attempts</span></h2><div id="rs-tree" class="tree"></div></section>
    </div>`;
  const draw = (q) => {
    const hit = (r) => !q || JSON.stringify(r).toLowerCase().includes(q);
    main.querySelector('#rs-tasks').innerHTML = `<table><tbody>${tasks.filter(hit).sort((a, b) => (a.state > b.state ? 1 : -1)).map((t) => `
      <tr><td><span class="pill ${STATE[t.state] || ''}">${esc(t.state)}</span></td><td><div class="mono">${esc(t.id.slice(11))}</div><div class="muted">${esc(t.summary)}</div></td>
      <td class="muted">${esc(t.owner)}${Array.isArray(t.after) && t.after.length ? `<br>after ${t.after.map((a) => esc(a.slice(11))).join(', ')}` : ''}</td></tr>`).join('')}</tbody></table>`;
    main.querySelector('#rs-dec').innerHTML = `<table><tbody>${decisions.filter(hit).map((d) => `
      <tr><td><span class="pill ${d.status === 'active' ? 'good' : ''}">${esc(d.status)}</span></td><td><div class="mono">${esc(d.id.slice(11))}</div><div class="muted">${esc(d.summary).slice(0, 240)}</div></td></tr>`).join('')}</tbody></table>`;
    const kids = {};
    for (const a of attempts) (kids[a.parent && a.parent !== 'none' ? a.parent : 'root'] ||= []).push(a);
    const node = (a) => {
      const ch = (kids[a.id] || []).filter((c) => hit(c) || (kids[c.id] || []).some(hit));
      const head = `<span class="node"><span class="pill ${OUT[a.outcome] || ''}">${esc(a.outcome)}</span><span class="mono">${esc(a.id.slice(11))}</span>${a.failure_mode && a.failure_mode !== 'none' ? `<span class="pill">${esc(a.failure_mode)}</span>` : ''}</span><div class="q">${esc(a.question)}</div><div class="muted">${esc(a.summary)}</div>`;
      return ch.length ? `<details ${q ? 'open' : ''}><summary>${head}</summary>${ch.map(node).join('')}</details>` : `<div style="margin-left:14px;padding:4px 0 4px 10px;border-left:1px solid var(--line)">${head}</div>`;
    };
    main.querySelector('#rs-tree').innerHTML = (kids.root || []).filter((a) => hit(a) || (kids[a.id] || []).some(hit)).map(node).join('') || '<div class="empty">nothing matches</div>';
  };
  main.querySelector('#rs-q').addEventListener('input', (e) => draw(e.target.value.toLowerCase()));
  draw('');
}
