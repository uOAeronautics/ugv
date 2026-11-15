// Shared helpers for the observatory front end.
// Classic scripts (no modules) so each file passes `node --check`; load order is set in index.html.

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

// Escape anything that came from a file name or a JSON field before it reaches innerHTML.
const esc = (v) =>
  String(v ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);

const isNum = (v) => v !== null && v !== undefined && v !== '' && Number.isFinite(Number(v));

async function fetchJSON(url) {
  try {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${r.status} ${url}`);
    return await r.json();
  } catch (e) {
    console.warn('observatory:', e);
    return null;
  }
}

const fmt = {
  pct: (v, d = 1) => (isNum(v) ? `${(Number(v) * 100).toFixed(d)}%` : '—'),
  num: (v, d = 2) => (isNum(v) ? Number(v).toFixed(d) : '—'),
  int: (v) => (isNum(v) ? Math.round(Number(v)).toLocaleString('en-US') : '—'),
  prec: (v) => (isNum(v) ? Number(v).toPrecision(4) : '—'),
  // compact axis label: keeps 3 significant digits and avoids "0.0000"
  axis(v) {
    const a = Math.abs(v);
    if (a === 0) return '0';
    if (a >= 1e4) return `${(v / 1e3).toFixed(0)}k`;
    if (a >= 100) return v.toFixed(0);
    if (a >= 1) return v.toFixed(2);
    if (a >= 0.01) return v.toFixed(3);
    return v.toExponential(1);
  },
};

// ---- theme (daylight default; the choice is a per-viewer convenience) ----
const THEME_KEY = 'ugv-observatory-theme';

function currentTheme() {
  return document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light';
}

function setTheme(theme) {
  document.documentElement.dataset.theme = theme === 'dark' ? 'dark' : 'light';
  try {
    localStorage.setItem(THEME_KEY, currentTheme());
  } catch (e) {
    /* private window or blocked storage: the theme simply does not persist */
  }
  $$('[data-theme-set]').forEach((b) => {
    b.setAttribute('aria-pressed', String(b.dataset.themeSet === currentTheme()));
  });
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// Snapshot of the tokens canvases need. Read once per draw so light/dark switch cleanly.
function palette() {
  return {
    ink: cssVar('--text-primary'),
    ink2: cssVar('--text-secondary'),
    muted: cssVar('--text-muted'),
    grid: cssVar('--grid'),
    gridStrong: cssVar('--border-strong'),
    plot: cssVar('--plot-bg'),
    accent: cssVar('--accent'),
    succ: cssVar('--succ'),
    warn: cssVar('--warn'),
    err: cssVar('--err'),
    sim: cssVar('--sim'),
    gz: cssVar('--gz'),
    l7: cssVar('--l7'),
    border: cssVar('--border'),
    series: Array.from({ length: 8 }, (_, i) => cssVar(`--s${i + 1}`)),
  };
}

const MONO = '11px Inter, system-ui, sans-serif';
const MONO_SM = '10px Inter, system-ui, sans-serif';

// Size a canvas to its CSS box at device pixel ratio. Draw in CSS pixels afterwards.
function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, canvas.clientWidth);
  const h = Math.max(1, canvas.clientHeight);
  const pw = Math.round(w * dpr);
  const ph = Math.round(h * dpr);
  if (canvas.width !== pw || canvas.height !== ph) {
    canvas.width = pw;
    canvas.height = ph;
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

// Shared outcome vocabulary across the Arena and the leaderboard.
function outcomeClass(outcome) {
  const o = String(outcome || '').toLowerCase();
  if (o.includes('success') || o.includes('parked')) return 'ok';
  if (o.includes('arrived')) return 'warn'; // estimate-based, not a confirmed park
  if (o.includes('collision') || o.includes('coll')) return 'collision';
  if (o.includes('timeout')) return 'timeout';
  return '';
}

// Source key used by badges, filters and stage matching.
function sourceOf(entry) {
  if (entry.source === 'gazebo_l6' || entry.track === 'gazebo_l6') return 'gazebo_l6';
  if (entry.source === 'l7' || entry.track === 'l7' || entry.level === 'L7') return 'l7';
  return 'sim';
}
