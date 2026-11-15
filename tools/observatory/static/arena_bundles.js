// Replay bundles: source filter, bundle selector, and loading a bundle into the arena.

// ---- bundles and source filter ----
async function loadBundleList() {
  const d = await fetchJSON('/api/replay/list');
  AR.bundles = d?.bundles || [];
  renderSourcePills();
  renderBundleSelect();
  if (!AR.bundleId && AR.bundles.length) {
    // prefer the largest Gazebo baseline bundle; episode count, not a lucky 1/1, decides
    const bySize = (a, b) => (b.episodes_count ?? 0) - (a.episodes_count ?? 0);
    const gz = AR.bundles.filter((b) => b.source === 'gazebo_l6').sort(bySize);
    const pick = gz.find((b) => /baseline/i.test(b.bundle_id)) || gz[0] || AR.bundles[0];
    selectBundle(pick.bundle_id);
  }
}

function renderSourcePills() {
  const box = $('#source-seg');
  box.innerHTML = '';
  SOURCE_PILLS.forEach((p) => {
    const count = p.id === 'all' ? AR.bundles.length : AR.bundles.filter((b) => b.source === p.id).length;
    const b = document.createElement('button');
    b.type = 'button';
    b.dataset.source = p.id;
    b.textContent = `${p.label} · ${count}`;
    b.setAttribute('aria-pressed', String(AR.source === p.id));
    b.addEventListener('click', () => {
      AR.source = p.id;
      renderSourcePills();
      renderBundleSelect();
      const visible = AR.bundles.filter((x) => AR.source === 'all' || x.source === AR.source);
      if (visible.length && !visible.some((x) => x.bundle_id === AR.bundleId)) selectBundle(visible[0].bundle_id);
    });
    box.appendChild(b);
  });
}

function bundleLabel(b) {
  const counts = b.episodes_count != null ? ` · ${b.success_count ?? 0}/${b.episodes_count}` : '';
  return `[${SOURCE_PILLS.find((p) => p.id === b.source)?.label || b.source}] ${b.label}${counts}`;
}

function renderBundleSelect() {
  const sel = $('#bundle-select');
  const visible = AR.bundles.filter((b) => AR.source === 'all' || b.source === AR.source);
  sel.innerHTML = visible.length
    ? visible.map((b) => `<option value="${esc(b.bundle_id)}">${esc(bundleLabel(b))}</option>`).join('')
    : '<option value="">No bundles for this source</option>';
  if (AR.bundleId && visible.some((b) => b.bundle_id === AR.bundleId)) sel.value = AR.bundleId;
}

function selectBundle(id) {
  if (!id) return;
  const sel = $('#bundle-select');
  if ([...sel.options].some((o) => o.value === id)) sel.value = id;
  loadBundle(id);
}

async function loadBundle(id) {
  const d = await fetchJSON(`/api/replay/bundle/${encodeURIComponent(id)}`);
  if (!d) {
    $('#bundle-meta').textContent = `Could not load ${id}.`;
    AR.episodes = [];
    renderEpisodeList();
    renderArena();
    return;
  }
  AR.bundleId = id;
  AR.bundle = d;
  let eps = d.episodes || [];
  if (!eps.length && Array.isArray(d.poses)) {
    eps = [{ episode_id: 0, outcome: d.summary?.outcome || 'unknown', duration: 0, frames: d.poses, end_frame_url: d.end_frame_url || null }];
  }
  AR.episodes = eps;
  AR.epIdx = 0;
  AR.view = null;
  AR.frame = 0;
  stopPlay();
  if (d.geometry?.target_type === 'strl001_belly') setTarget('strl001_belly');
  // A filter that matches nothing in the new bundle would show a blank arena: reset it.
  if (AR.epFilter !== 'all' && !filteredIdx().length) AR.epFilter = 'all';
  syncEpFilterButtons();
  AR.epIdx = filteredIdx()[0] ?? 0;
  AR.view = null;
  renderBundleSelect();
  $('#bundle-meta').textContent = bundleMetaText();
  renderEpisodeList();
  renderArena();
}

function bundleMetaText() {
  if (!AR.bundle) return '';
  const eps = AR.episodes;
  const succ = eps.filter((e) => outcomeClass(e.outcome) === 'ok').length;
  const src = AR.bundles.find((b) => b.bundle_id === AR.bundleId)?.source || 'sim';
  const srcTxt = SOURCE_PILLS.find((p) => p.id === src)?.label || src;
  return `${eps.length} episode${eps.length === 1 ? '' : 's'} · ${succ} success · ${srcTxt}`;
}

function openInArena(bundleId) {
  activateTab('arena');
  if (AR.bundles.some((b) => b.bundle_id === bundleId)) {
    AR.source = 'all';
    renderSourcePills();
  }
  selectBundle(bundleId);
}
