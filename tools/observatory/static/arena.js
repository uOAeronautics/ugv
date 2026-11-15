// Tab 3 controller: bundle + source filter, episodes, transport, layers, and the live rollout trigger.
// Drawing lives in arena_draw.js, arena_airframe.js and arena_pov.js.

const AR = {
  bundles: [],
  source: 'all',
  bundleId: null,
  bundle: null,
  episodes: [],
  epIdx: 0,
  epFilter: 'all',
  frame: 0,
  playing: false,
  speed: 1,
  target: 'strl001_belly',
  view25: false,
  showEnd: false,
  engine: 'sim',
  rollOpts: null,
  poll: null,
  layers: { true: true, tag: true, aux: false, fov: true, blind: true, shield: true, trail: true, tags: true },
};

const LAYER_DEFS = [
  { id: 'true', label: 'True pose' },
  { id: 'tag', label: 'Delayed tag' },
  { id: 'aux', label: 'Aux pose' },
  { id: 'fov', label: 'FOV cone' },
  { id: 'blind', label: '0.40 m blind ring' },
  { id: 'shield', label: 'Safety shield' },
  { id: 'trail', label: 'Trail' },
  { id: 'tags', label: 'ArUco tags' },
];

const SOURCE_PILLS = [
  { id: 'all', label: 'All' },
  { id: 'gazebo_l6', label: 'Gazebo L6' },
  { id: 'sim', label: 'VecSim' },
  { id: 'l7', label: 'Real Pi L7' },
];

const EP_FILTER_TEXT = { all: 'any', success: 'successful', collision: 'collision', timeout: 'timeout' };

function filteredIdx() {
  return AR.episodes
    .map((e, i) => i)
    .filter((i) => {
      if (AR.epFilter === 'all') return true;
      return outcomeClass(AR.episodes[i].outcome) === (AR.epFilter === 'success' ? 'ok' : AR.epFilter);
    });
}

function renderArena() {
  drawArena();
  drawPov();
  drawStrips();
  renderTelemetry();
  updateTransport();
}

function setFrame(i) {
  const n = AR.episodes[AR.epIdx]?.frames?.length || 0;
  AR.frame = Math.max(0, Math.min(Math.max(0, n - 1), Math.round(i)));
  renderArena();
}

function setEpisode(i) {
  AR.epIdx = i;
  AR.view = null;
  AR.frame = 0;
  stopPlay();
  renderEpisodeList();
  renderArena();
  $('#bundle-meta').textContent = bundleMetaText();
}

function stopPlay() {
  AR.playing = false;
  $('#btn-play').textContent = 'Play';
}

function togglePlay() {
  const frames = AR.episodes[AR.epIdx]?.frames || [];
  if (!frames.length) return;
  if (!AR.playing && AR.frame >= frames.length - 1) AR.frame = 0;
  AR.playing = !AR.playing;
  $('#btn-play').textContent = AR.playing ? 'Pause' : 'Play';
}

function updateTransport() {
  const ep = AR.episodes[AR.epIdx];
  const frames = ep?.frames || [];
  const n = Math.max(0, frames.length - 1);
  const sc = $('#scrubber');
  sc.max = String(n);
  sc.value = String(AR.frame);
  const t = frames[AR.frame]?.t ?? 0;
  $('#time-readout').textContent = `t ${Number(t).toFixed(2)} s · step ${AR.frame}/${n}`;
  const badge = $('#outcome-badge');
  badge.textContent = ep ? ep.outcome || 'unknown' : '—';
  badge.className = `badge ${ep ? outcomeClass(ep.outcome) : ''}`;
}

// ---- episodes ----
function syncEpFilterButtons() {
  $$('#ep-filter-seg button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.epFilter === AR.epFilter)));
}

function renderEpisodeList() {
  const ul = $('#episode-list');
  if (!AR.episodes.length) {
    ul.innerHTML = '<li class="empty">This bundle has no episodes.</li>';
    return;
  }
  const idxs = filteredIdx();
  if (!idxs.length) {
    ul.innerHTML = `<li class="empty">No ${EP_FILTER_TEXT[AR.epFilter]} episodes here.</li>`;
    return;
  }
  ul.innerHTML = idxs
    .map((i) => {
      const e = AR.episodes[i];
      const err = e.final_error != null ? `${fmt.num(e.final_error, 1)} mm` : '—';
      return `<li><button type="button" class="ep-row" data-ep="${i}" aria-current="${i === AR.epIdx}">
        <span>#${esc(e.episode_id ?? i)}</span>
        <span class="badge ${outcomeClass(e.outcome)}">${esc(e.outcome || '—')}</span>
        <span class="muted">${fmt.num(e.duration, 1)} s · ${err}</span>
      </button></li>`;
    })
    .join('');
  ul.querySelectorAll('button.ep-row').forEach((b) => b.addEventListener('click', () => setEpisode(Number(b.dataset.ep))));
}

// ---- layers and target ----
function buildLayers() {
  const box = $('#layers');
  box.innerHTML =
    LAYER_DEFS.map(
      (l) => `<label class="chip"><input type="checkbox" data-layer="${l.id}" ${AR.layers[l.id] ? 'checked' : ''}><span>${esc(l.label)}</span></label>`,
    ).join('') +
    `<span class="chip-sep" aria-hidden="true"></span>
     <button type="button" class="btn btn-sm" id="btn-fit" title="Frame the whole episode (double-click the view)">Fit view</button>
     <label class="chip"><input type="checkbox" id="view-25d" ${AR.view25 ? 'checked' : ''}><span>2.5D oblique</span></label>`;
  box.querySelectorAll('input[data-layer]').forEach((cb) =>
    cb.addEventListener('change', () => {
      AR.layers[cb.dataset.layer] = cb.checked;
      renderArena();
    }),
  );
  $('#view-25d').addEventListener('change', (e) => {
    AR.view25 = e.target.checked;
    renderArena();
  });
}

function setTarget(t) {
  AR.target = t;
  $$('#target-seg button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.target === t)));
}

// ---- playback ----
let lastTs = null;
let playAcc = 0;
function playbackLoop(ts) {
  requestAnimationFrame(playbackLoop);
  if (!AR.playing) {
    lastTs = null;
    playAcc = 0;
    return;
  }
  const frames = AR.episodes[AR.epIdx]?.frames || [];
  if (!frames.length) {
    stopPlay();
    return;
  }
  if (lastTs === null) {
    lastTs = ts;
    return;
  }
  playAcc += ((ts - lastTs) / 1000) * AR.speed;
  lastTs = ts;
  let moved = false;
  while (AR.frame < frames.length - 1) {
    const cur = frames[AR.frame]?.t ?? AR.frame * 0.02;
    const nxt = frames[AR.frame + 1]?.t ?? cur + 0.02;
    const step = Math.max(1e-3, nxt - cur);
    if (playAcc < step) break;
    playAcc -= step;
    AR.frame++;
    moved = true;
  }
  if (AR.frame >= frames.length - 1) stopPlay();
  if (moved) renderArena();
}

function buildArenaControls() {
  buildLayers();
  $('#bundle-select').addEventListener('change', (e) => loadBundle(e.target.value));
  $$('#target-seg button').forEach((b) =>
    b.addEventListener('click', () => {
      setTarget(b.dataset.target);
      renderArena();
    }),
  );
  $$('#ep-filter-seg button').forEach((b) =>
    b.addEventListener('click', () => {
      AR.epFilter = b.dataset.epFilter;
      syncEpFilterButtons();
      const idxs = filteredIdx();
      if (idxs.length && !idxs.includes(AR.epIdx)) setEpisode(idxs[0]);
      else renderEpisodeList();
    }),
  );
  $$('#engine-seg button').forEach((b) => b.addEventListener('click', () => setEngine(b.dataset.engine)));
  $('#rollout-form').addEventListener('submit', submitRollout);
  $('#btn-play').addEventListener('click', togglePlay);
  $('#btn-step-back').addEventListener('click', () => {
    stopPlay();
    setFrame(AR.frame - 1);
  });
  $('#btn-step-fwd').addEventListener('click', () => {
    stopPlay();
    setFrame(AR.frame + 1);
  });
  $('#speed-sel').addEventListener('change', (e) => {
    AR.speed = parseFloat(e.target.value) || 1;
  });
  $('#scrubber').addEventListener('input', (e) => setFrame(Number(e.target.value) || 0));
  $('#btn-end-frame').addEventListener('click', () => {
    AR.showEnd = !AR.showEnd;
    drawPov();
  });
  $('#pov-img').addEventListener('load', () => drawPov());
  $$('.strip').forEach((c) => c.addEventListener('click', scrubFromStrip));
  document.addEventListener('keydown', (e) => {
    if (!document.querySelector('#tab-arena.active')) return;
    if (/INPUT|SELECT|TEXTAREA/.test(document.activeElement?.tagName || '')) return;
    if (e.key === ' ') {
      e.preventDefault();
      togglePlay();
    } else if (e.key === 'ArrowRight') {
      stopPlay();
      setFrame(AR.frame + 1);
    } else if (e.key === 'ArrowLeft') {
      stopPlay();
      setFrame(AR.frame - 1);
    }
  });
  bindArenaView();
  renderEpisodeList();
  requestAnimationFrame(playbackLoop);
}

// Wheel zooms about the cursor, drag pans, double-click or Fit re-frames the whole episode.
function bindArenaView() {
  const c = $('#arena-canvas');
  const fit = () => {
    AR.view = null;
    drawArena();
  };
  $('#btn-fit').addEventListener('click', fit);
  c.addEventListener('dblclick', fit);
  c.addEventListener(
    'wheel',
    (e) => {
      e.preventDefault();
      const V = AR.view;
      if (!V) return;
      const r = c.getBoundingClientRect();
      const mx = (e.clientX - r.left - r.width / 2) / V.ppm;
      const my = -(e.clientY - r.top - r.height / 2) / (V.ppm * (AR.view25 ? 0.72 : 1));
      const ppm = Math.max(30, Math.min(1600, V.ppm * Math.exp(-e.deltaY * 0.0015)));
      const s = V.ppm / ppm;
      AR.view = { x: V.x + mx * (1 - s), y: V.y + my * (1 - s), ppm, user: true };
      drawArena();
    },
    { passive: false },
  );
  let drag = null;
  c.addEventListener('pointerdown', (e) => {
    if (!AR.view) return;
    drag = { x: e.clientX, y: e.clientY, v: { ...AR.view } };
    c.setPointerCapture(e.pointerId);
  });
  c.addEventListener('pointermove', (e) => {
    if (!drag) return;
    const k = AR.view25 ? 0.72 : 1;
    AR.view = {
      x: drag.v.x - (e.clientX - drag.x) / drag.v.ppm,
      y: drag.v.y + (e.clientY - drag.y) / (drag.v.ppm * k),
      ppm: drag.v.ppm,
      user: true,
    };
    drawArena();
  });
  const end = () => (drag = null);
  c.addEventListener('pointerup', end);
  c.addEventListener('pointercancel', end);
}
