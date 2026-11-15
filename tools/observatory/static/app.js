// Bootstrap: header KPIs, theme and auto-refresh, tab switching, and the initial data load.
// Runs last (after the feature modules), so every function it calls is already defined.

const TAB_NAMES = ['leaderboard', 'curves', 'arena', 'contract'];

function activateTab(name) {
  if (!TAB_NAMES.includes(name)) name = 'leaderboard';
  $$('.tab').forEach((t) => {
    const on = t.dataset.tab === name;
    t.classList.toggle('active', on);
    t.setAttribute('aria-selected', String(on));
  });
  $$('.pane').forEach((p) => p.classList.toggle('active', p.id === `tab-${name}`));
  try {
    history.replaceState(null, '', `#${name}`);
  } catch (e) {
    /* sandboxed frames refuse history writes; the tab still switches */
  }
  requestAnimationFrame(() => onTabShown(name));
}

function onTabShown(name) {
  if (name === 'curves') drawCurves();
  if (name === 'arena') renderArena();
}

async function refreshOverview() {
  const d = await fetchJSON('/api/overview');
  if (!d) return;
  const k = d.kpi || {};
  const counts = d.counts || {};
  $('#kpi-eval').textContent = isNum(k.eval_runs) ? fmt.int(k.eval_runs) : fmt.int(counts.archive);
  $('#kpi-gz').textContent = fmt.int(k.gazebo_l6_runs ?? counts.gazebo_l6);
  $('#kpi-l7').textContent = fmt.int(k.l7_runs ?? counts.l7_runs);
  $('#kpi-pi').textContent = isNum(k.pi_step_ms_p50) ? `${Number(k.pi_step_ms_p50).toFixed(3)} ms / step` : 'n/a';
}

function setupTheme() {
  setTheme(currentTheme());
  $$('[data-theme-set]').forEach((b) => b.addEventListener('click', () => {
    setTheme(b.dataset.themeSet);
    renderActiveCanvases();
  }));
}

// Canvases read theme tokens at draw time, so a theme switch must repaint them.
function renderActiveCanvases() {
  const active = document.querySelector('.pane.active')?.id;
  if (active === 'tab-curves') drawCurves();
  if (active === 'tab-arena') renderArena();
}

let refreshTimer = null;
function setupAutoRefresh() {
  $('#auto-refresh').addEventListener('change', (e) => {
    clearInterval(refreshTimer);
    refreshTimer = e.target.checked ? setInterval(refreshLive, 5000) : null;
  });
}

async function refreshLive() {
  await Promise.all([refreshOverview(), loadLeaderboard()]);
  if (document.querySelector('#tab-curves.active')) await loadCurveRuns();
}

function setupTabs() {
  $$('.tab').forEach((t) => t.addEventListener('click', () => activateTab(t.dataset.tab)));
  const fromHash = location.hash.replace('#', '');
  activateTab(TAB_NAMES.includes(fromHash) ? fromHash : 'leaderboard');
  let resizeQueued = false;
  window.addEventListener('resize', () => {
    if (resizeQueued) return;
    resizeQueued = true;
    requestAnimationFrame(() => {
      resizeQueued = false;
      renderActiveCanvases();
    });
  });
}

async function init() {
  setupTheme();
  buildLeaderboardControls();
  buildCurvesControls();
  buildArenaControls();
  buildContractControls();
  setupAutoRefresh();
  setupTabs();
  await Promise.all([refreshOverview(), loadLeaderboard(), loadRolloutOptions(), loadContractRuns()]);
  await loadCurveRuns();
  await loadBundleList();
}

init();
