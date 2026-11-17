// Router and shared helpers. Each page module exports `render(main)`.

const pages = {
  overview: () => import('./overview.js'),
  parking: () => import('./parking.js'),
  search: () => import('./search.js'),
  challenge: () => import('./challenge.js'),
  pose: () => import('./pose.js'),
  research: () => import('./research.js'),
};

const cache = new Map();
export async function api(path, { fresh = false } = {}) {
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

export const OUTCOME_COLOR = {
  parked: 'var(--good)',
  'parked-off-target': 'var(--warning)',
  collision: 'var(--critical)',
  geofence: 'var(--serious)',
  timeout: 'var(--ink-3)',
};
export const outcomeLabel = (o) => (o.startsWith('abort:') ? 'abort: ' + o.slice(6).split(';')[0].replace('misaligned in the bay', 'misaligned') : o);
export const outcomeColor = (o) => OUTCOME_COLOR[o] || (o.startsWith('abort') ? 'var(--serious)' : 'var(--ink-3)');

export const STATE_COLOR = {
  SEARCH: 'var(--s4)', ACQUIRE: 'var(--s5)', ORBIT: 'var(--s7)', APPROACH: 'var(--s1)',
  ENTER: 'var(--s3)', BACKOFF: 'var(--s2)', PARKED: 'var(--good)', ABORT: 'var(--critical)',
};

let current = null;
async function route() {
  const name = (location.hash.slice(1) || 'overview').split('/')[0];
  const load = pages[name] || pages.overview;
  document.querySelectorAll('.rail a').forEach((a) => a.classList.toggle('on', a.dataset.page === name));
  const main = document.getElementById('page');
  if (current?.dispose) current.dispose();
  main.innerHTML = '<div class="muted">loading…</div>';
  try {
    const mod = await load();
    current = mod;
    await mod.render(main);
  } catch (err) {
    main.innerHTML = `<div class="empty">Could not load this page: ${esc(err.message)}</div>`;
  }
  main.focus({ preventScroll: true });
}

function initTheme() {
  const root = document.documentElement;
  let saved = null;
  try { saved = localStorage.getItem('ugv-lab-theme'); } catch { /* storage blocked */ }
  if (saved) root.dataset.theme = saved;
  document.getElementById('theme').addEventListener('click', () => {
    const dark = root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
    root.dataset.theme = dark ? 'light' : 'dark';
    try { localStorage.setItem('ugv-lab-theme', root.dataset.theme); } catch { /* ignore */ }
    route();
  });
}

initTheme();
addEventListener('hashchange', route);
route();
