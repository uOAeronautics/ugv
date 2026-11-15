// Live rollout trigger: start one VecSim or Gazebo job, poll it, then open the new bundle.

// ---- live rollout trigger ----
function setRolloutStatus(text, kind = '') {
  const el = $('#rollout-status');
  el.textContent = text;
  el.style.color = kind === 'err' ? 'var(--err)' : kind === 'ok' ? 'var(--succ)' : '';
}

async function loadRolloutOptions() {
  const d = await fetchJSON('/api/rollout/options');
  if (!d) return;
  AR.rollOpts = d;
  const lvl = $('#rollout-level');
  lvl.innerHTML = (d.levels || []).map((l) => `<option value="${esc(l)}">${esc(l)}</option>`).join('');
  if ((d.levels || []).includes('L1')) lvl.value = 'L1';
  $('#rollout-form input[name=episodes]').max = String(d.episodes_max || 20);
  setEngine(AR.engine);
}

function setEngine(engine) {
  AR.engine = engine;
  $$('#engine-seg button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.engine === engine)));
  const pol = $('#rollout-policy');
  const list = AR.rollOpts?.engines?.[engine]?.policies || [];
  pol.innerHTML = list.map((p) => `<option value="${esc(p)}">${esc(p)}</option>`).join('');
  if (!list.length) pol.innerHTML = '<option value="">no policies found</option>';
}

async function submitRollout(e) {
  e.preventDefault();
  const fd = new FormData(e.target);
  const body = {
    engine: AR.engine,
    policy: fd.get('policy'),
    level: fd.get('level'),
    episodes: parseInt(fd.get('episodes'), 10),
    seed: parseInt(fd.get('seed'), 10),
  };
  setRolloutStatus('Starting…');
  let r;
  try {
    r = await fetch('/api/rollout', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch (err) {
    setRolloutStatus('Could not reach the observatory server.', 'err');
    return;
  }
  const j = await r.json().catch(() => ({}));
  if (!r.ok) {
    setRolloutStatus(typeof j.detail === 'string' ? j.detail : `Rollout refused (HTTP ${r.status}).`, 'err');
    return;
  }
  pollRollout(j.job_id, body);
}

function pollRollout(jobId, body) {
  if (AR.poll) clearInterval(AR.poll);
  $('#rollout-run').disabled = true;
  const label = `${body.engine === 'gazebo' ? 'Gazebo L6' : 'VecSim'} · ${body.policy} · ${body.level} × ${body.episodes}`;
  AR.poll = setInterval(async () => {
    const s = await fetchJSON(`/api/rollout/${encodeURIComponent(jobId)}`);
    if (!s) return;
    if (s.status === 'running') {
      setRolloutStatus(`Running ${label} …`);
      return;
    }
    clearInterval(AR.poll);
    AR.poll = null;
    $('#rollout-run').disabled = false;
    if (s.status === 'done') {
      setRolloutStatus(`Done: ${s.bundle_id}`, 'ok');
      await loadBundleList();
      const item = AR.bundles.find((b) => b.bundle_id === s.bundle_id);
      if (item) AR.source = item.source;
      renderSourcePills();
      selectBundle(s.bundle_id);
      refreshOverview();
    } else {
      setRolloutStatus(`Failed: ${s.error || 'unknown error'}`, 'err');
    }
  }, 1500);
}
