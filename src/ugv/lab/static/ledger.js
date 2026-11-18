import { api, esc, strip } from './app.js';
import { roadmap } from './roadmap.js';

const OUT = { kept: 'good', negative: 'bad', invalid: 'bad', retracted: 'bad', abandoned: 'warn', open: 'open' };
const TASK = { open: 'open', claimed: 'warn', blocked: 'bad', done: 'good', dropped: '' };
const short = (id) => esc(id.slice(11));

export async function render(main) {
  const led = await api('/api/ledger', { fresh: true });
  const recs = led.records;
  const attempts = recs.filter((r) => r.kind === 'attempt');
  const tasks = recs.filter((r) => r.kind === 'task');
  const decisions = recs.filter((r) => r.kind === 'decision');
  const focus = decodeURIComponent(location.hash.split('/')[1] || '');
  main.innerHTML = `
    ${strip('Ledger', [['Tasks', tasks.length], ['Decisions', decisions.length], ['Attempts', attempts.length]],
      `<input id="rs-q" type="search" placeholder="filter: id, tag, text" value="${esc(focus)}" style="width:min(320px,60vw)" aria-label="Filter records">`)}
    <section class="ruled c12"><div class="sp12"><div class="lbl">Roadmap <small>open tasks and what they wait for · click one to open it</small></div><div id="rs-road"></div></div></section>
    <section class="ruled c12">
      <div class="sp4"><div class="lbl">Tasks <small id="rs-nt"></small></div><div id="rs-tasks"></div></div>
      <div class="sp8"><div class="lbl">Attempt tree <small>parent → child · outcome · failure mode</small></div><div id="rs-tree"></div></div>
    </section>
    <section class="ruled c12"><div class="sp12"><div class="lbl">Decisions</div><div id="rs-dec"></div></div></section>`;
  const draw = (q) => {
    const hit = (r) => !q || JSON.stringify(r).toLowerCase().includes(q);
    const ts = tasks.filter(hit).sort((a, b) => ['claimed', 'open', 'blocked', 'done', 'dropped'].indexOf(a.state) - ['claimed', 'open', 'blocked', 'done', 'dropped'].indexOf(b.state));
    main.querySelector('#rs-nt').textContent = `${ts.length} shown`;
    main.querySelector('#rs-tasks').innerHTML = ts.map((t) => `
      <div class="rec${t.id === focus ? ' focus' : ''}"><div class="rec-h"><span class="st ${TASK[t.state] || ''}">${esc(t.state)}</span><b>${short(t.id)}</b></div>
      <p>${esc(t.summary)}</p>
      ${Array.isArray(t.after) && t.after.length ? `<div class="prov">after ${t.after.map(short).join(', ')}</div>` : ''}</div>`).join('') || '<div class="empty">nothing matches</div>';
    main.querySelector('#rs-dec').innerHTML = `<div class="cols">${decisions.filter(hit).map((d) => `
      <div class="rec"><div class="rec-h"><span class="st ${d.status === 'active' ? 'good' : ''}">${esc(d.status)}</span><b>${short(d.id)}</b></div><p>${esc(d.summary)}</p></div>`).join('')}</div>`;
    const kids = {};
    for (const a of attempts) (kids[a.parent && a.parent !== 'none' ? a.parent : 'root'] ||= []).push(a);
    const keep = (a) => hit(a) || (kids[a.id] || []).some(keep);
    const node = (a) => {
      const ch = (kids[a.id] || []).filter(keep);
      return `<li><div class="rec"><div class="rec-h"><span class="st ${OUT[a.outcome] || ''}">${esc(a.outcome)}</span><b>${short(a.id)}</b>${a.failure_mode && a.failure_mode !== 'none' ? `<span class="prov">${esc(a.failure_mode)}</span>` : ''}</div>
        <p class="q">${esc(a.question)}</p><p>${esc(a.summary)}</p></div>${ch.length ? `<ul>${ch.map(node).join('')}</ul>` : ''}</li>`;
    };
    main.querySelector('#rs-tree').innerHTML = `<ul class="tree">${(kids.root || []).filter(keep).map(node).join('')}</ul>` || '<div class="empty">nothing matches</div>';
  };
  roadmap(main.querySelector('#rs-road'), tasks);
  main.addEventListener('click', (e) => { const r = e.target.closest('.rec'); if (r && !e.target.closest('a')) r.classList.toggle('open'); });
  main.querySelector('#rs-q').addEventListener('input', (e) => draw(e.target.value.toLowerCase()));
  draw(focus.toLowerCase());
  main.querySelector('.rec.focus')?.scrollIntoView({ block: 'center' });
}
