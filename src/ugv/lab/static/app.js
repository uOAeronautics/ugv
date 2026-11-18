// Router, theme and shared helpers. Each page module exports `render(main)`, and optionally
// `dispose()` (leaving the page) and `retheme()` (redraw canvases after a theme change).

const pages = {
  park: () => import('./park.js'),
  challenge: () => import('./challenge.js'),
  pose: () => import('./pose.js'),
  ledger: () => import('./ledger.js'),
};
const ALIAS = { overview: 'park', parking: 'park', search: 'park', tuning: 'park', research: 'ledger' }; // old links
const ORDER = Object.keys(pages);

const cache = new Map();
export async function api(path, { fresh = false, body } = {}) {
  if (body) {
    const r = await fetch(path, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
    if (!r.ok) throw new Error(`${r.status} ${path}`);
    return r.json();
  }
  if (!fresh && cache.has(path)) return cache.get(path);
  const p = fetch(path).then((r) => {
    if (!r.ok) throw new Error(`${r.status} ${path}`);
    return r.json();
  });
  cache.set(path, p);
  p.catch(() => cache.delete(path));
  return p;
}

export const pct = (x, d = 1) => `${(100 * x).toFixed(d)}%`;
export const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
export const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
/** A colour usable on a canvas: resolves `var(--x)` to its current value. */
export const color = (c) => (c.startsWith('var(') ? css(c.slice(4, -1)) : c);

export const OUTCOME_COLOR = {
  parked: 'var(--good)',
  'parked-off-target': 'var(--warn)',
  collision: 'var(--bad)',
  geofence: 'var(--bad)',
  timeout: 'var(--ink-3)',
};
export const outcomeLabel = (o) => (o.startsWith('abort:') ? 'abort: ' + o.slice(6).split(';')[0].replace('misaligned in the bay', 'misaligned') : o);
export const outcomeColor = (o) => OUTCOME_COLOR[o] || (o.startsWith('abort') ? 'var(--s5)' : 'var(--ink-3)');

export const STATES = ['SEARCH', 'ACQUIRE', 'ORBIT', 'APPROACH', 'ENTER', 'BACKOFF', 'PARKED', 'ABORT'];
export const STATE_COLOR = {
  SEARCH: 'var(--s8)', ACQUIRE: 'var(--s5)', ORBIT: 'var(--s7)', APPROACH: 'var(--s6)',
  ENTER: 'var(--signal)', BACKOFF: 'var(--warn)', PARKED: 'var(--good)', ABORT: 'var(--bad)',
};

/** Page strip: title, facts as label/value pairs ([[label, value], ...] or HTML), controls. */
export const strip = (title, facts = [], controls = '') =>
  `<div class="strip"><h1>${title}</h1>${Array.isArray(facts) ? `<dl class="facts">${facts.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join('')}</dl>` : `<span class="prov">${facts}</span>`}<span class="grow"></span>${controls}</div>`;

let current = null;
let currentName = '';
async function route() {
  const raw = (location.hash.slice(1) || 'park').split('/')[0];
  const name = ALIAS[raw] || raw;
  const key = pages[name] ? name : 'park';
  document.querySelectorAll('.tabs a').forEach((a) => a.classList.toggle('on', a.dataset.page === key));
  moveIndicator();
  const main = document.getElementById('page');
  if (current?.dispose) current.dispose();
  currentName = key;
  try {
    const mod = await pages[key]();
    if (currentName !== key) return;
    current = mod;
    main.innerHTML = '';
    const page = document.createElement('div');
    page.className = 'page';
    main.appendChild(page);
    await mod.render(page);
  } catch (err) {
    main.innerHTML = `<div class="page"><div class="empty">Could not load this page: ${esc(err.message)}</div></div>`;
  }
}

function moveIndicator() {
  const on = document.querySelector('.tabs a.on');
  const ind = document.querySelector('.rail-ind');
  if (!on || !ind) return;
  ind.style.width = `${on.offsetWidth}px`;
  ind.style.transform = `translateX(${on.offsetLeft}px)`;
}

function initTheme() {
  const root = document.documentElement;
  const btn = document.getElementById('theme');
  btn.addEventListener('click', () => {
    const next = root.dataset.mode === 'night' ? 'day' : 'night';
    const r = btn.getBoundingClientRect();
    root.style.setProperty('--wx', `${r.left + r.width / 2}px`);
    root.style.setProperty('--wy', `${r.top + r.height / 2}px`);
    const apply = () => {
      root.dataset.theme = next;
      root.dataset.mode = next;
      try { localStorage.setItem('ugv-lab-theme', next); } catch { /* ignore */ }
      current?.retheme?.();
    };
    const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (document.startViewTransition && !still) document.startViewTransition(apply);
    else apply();
  });
}

addEventListener('keydown', (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || /INPUT|SELECT|TEXTAREA/.test(document.activeElement?.tagName || '')) return;
  const i = Number(e.code.replace('Digit', '')) - 1;
  if (e.code.startsWith('Digit') && ORDER[i]) location.hash = ORDER[i];
});
addEventListener('resize', moveIndicator);
document.fonts?.ready.then(moveIndicator);
initTheme();
addEventListener('hashchange', route);
route();
