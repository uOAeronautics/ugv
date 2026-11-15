// Tab 4: Sim-vs-Real observation contract (L7 field run vs simulator reference).

const CHECK_TITLE = {
  valid_pose_rate: 'Valid pose rate',
  loop_rate_hz: 'Control loop rate (Hz)',
  loop_ms_p95: 'Loop time p95 (ms)',
  loop_ms_p50: 'Loop time p50 (ms)',
  frame_age_p50_ms: 'Tag frame age p50 (ms)',
  vision_hz: 'Vision update rate (Hz)',
  dropout_rate: 'Vision dropout rate',
  imu_w_std: 'Gyro rate noise std (rad/s)',
  joint_whited_kde_overlap: 'Joint sim↔real overlap (whitened KDE)',
};

const CHECK_GROUPS = [
  { title: 'Vision & pose', names: ['valid_pose_rate', 'vision_hz', 'dropout_rate'] },
  { title: 'Timing', names: ['loop_rate_hz', 'loop_ms_p95', 'loop_ms_p50', 'frame_age_p50_ms'] },
  { title: 'IMU', names: ['imu_w_std'] },
  { title: 'Joint distribution', names: ['joint_whited_kde_overlap'] },
];

const CHANNEL_TITLE = {
  vision_hz: 'Vision update rate (Hz)',
  frame_age_p50_ms: 'Tag frame age p50 (ms)',
  dropout_rate: 'Dropout rate',
  imu_w_std: 'Gyro noise std (rad/s)',
  loop_ms_p50: 'Loop time p50 (ms)',
  loop_ms_p95: 'Loop time p95 (ms)',
};

function verdictKind(status) {
  const s = String(status || '').toUpperCase();
  if (s.includes('PASS')) return 'pass';
  if (s.includes('FAIL')) return 'fail';
  if (s.includes('WARN')) return 'warn';
  return 'info';
}

function formatMeasured(name, v) {
  if (!isNum(v)) return String(v ?? '—');
  const n = Number(v);
  if (/rate$/.test(name) && Math.abs(n) <= 1) return fmt.pct(n, 1);
  return fmt.prec(n);
}

function checkCard(c) {
  const kind = c.pass === true ? 'pass' : c.pass === false ? 'fail' : 'unknown';
  const label = kind === 'pass' ? 'PASS' : kind === 'fail' ? 'FAIL' : 'NO FLAG';
  const badge = kind === 'pass' ? 'pass' : kind === 'fail' ? 'fail' : 'warn';
  const method = c.method
    ? `<div class="check-raw">${esc(c.method)}${c.dropped && c.dropped.length ? ` · dropped: ${esc(c.dropped.join(', '))}` : ''}</div>`
    : '';
  return `<article class="card check-card ${kind}">
    <div class="check-top">
      <span class="check-name">${esc(CHECK_TITLE[c.name] || c.name)}</span>
      <span class="badge ${badge}">${label}</span>
    </div>
    <div class="check-value${String(formatMeasured(c.name, c.measured)).length > 14 ? ' long' : ''}">${esc(formatMeasured(c.name, c.measured))}</div>
    <div class="check-limit">threshold ${esc(String(c.limit ?? '—'))}</div>
    <div class="check-raw">${esc(c.name)}</div>${method}
  </article>`;
}

function renderChecks(checks) {
  const box = $('#check-groups');
  if (!checks.length) {
    box.innerHTML = '<p class="empty card">This run reported no checks.</p>';
    return;
  }
  const used = new Set();
  const groups = CHECK_GROUPS.map((g) => {
    const items = checks.filter((c) => g.names.includes(c.name));
    items.forEach((c) => used.add(c));
    return { title: g.title, items };
  });
  const rest = checks.filter((c) => !used.has(c));
  if (rest.length) groups.push({ title: 'Other', items: rest });
  box.innerHTML = groups
    .filter((g) => g.items.length)
    .map(
      (g) => `<section class="check-group">
        <h3>${esc(g.title)}</h3>
        <div class="check-cards">${g.items.map(checkCard).join('')}</div>
      </section>`,
    )
    .join('');
}

function channelRow(c) {
  const rm = Number(c.real_mean);
  const sm = Number(c.sim_mean);
  const max = Math.max(isNum(rm) ? Math.abs(rm) : 0, isNum(sm) ? Math.abs(sm) : 0, 1e-12);
  const rw = isNum(rm) ? (Math.abs(rm) / max) * 100 : 0;
  const sw = isNum(sm) ? (Math.abs(sm) / max) * 100 : 0;
  const delta = isNum(rm) && isNum(sm) ? rm - sm : null;
  const rel = delta !== null && Math.abs(sm) > 1e-12 ? (delta / Math.abs(sm)) * 100 : null;
  const line = (mean, std) => (isNum(mean) ? `μ ${fmt.prec(mean)} · σ ${fmt.prec(std)}` : '—');
  return `<div class="channel-row">
    <div class="channel-name">${esc(c.name)}<small>${esc(CHANNEL_TITLE[c.name] || '')}</small></div>
    <div class="dual">
      <div class="dual-line"><span>Real Pi L7</span>
        <div class="dual-bar real" aria-hidden="true"><span style="width:${rw.toFixed(1)}%"></span></div>
        <span class="dual-num">${line(c.real_mean, c.real_std)}</span></div>
      <div class="dual-line"><span>Sim ref</span>
        <div class="dual-bar sim" aria-hidden="true"><span style="width:${sw.toFixed(1)}%"></span></div>
        <span class="dual-num">${line(c.sim_mean, c.sim_std)}</span></div>
      ${delta !== null ? `<span class="chart-desc">Δ ${delta >= 0 ? '+' : ''}${fmt.prec(delta)}${rel !== null ? ` (${rel >= 0 ? '+' : ''}${rel.toFixed(0)}%)` : ''}</span>` : ''}
    </div>
  </div>`;
}

async function loadContractRuns() {
  const d = await fetchJSON('/api/contract/runs');
  const runs = d?.runs || [];
  const sel = $('#contract-run');
  sel.innerHTML = runs.length
    ? [...runs].reverse().map((r) => `<option value="${esc(r)}">${esc(r)}</option>`).join('')
    : '<option value="">no L7 runs with ticks.jsonl</option>';
  if (!runs.length) return;
  // the server's default prefers the newest control run over watch snippets
  const def = await fetchJSON('/api/contract');
  if (def?.run_id && runs.includes(def.run_id)) sel.value = def.run_id;
  loadContract(sel.value, def && def.run_id === sel.value ? def : null);
}

async function loadContract(id, pre) {
  const q = id ? `?run_id=${encodeURIComponent(id)}` : '';
  const d = pre || (await fetchJSON(`/api/contract${q}`));
  if (!d) {
    $('#verdict').textContent = 'N/A';
    $('#verdict').className = 'verdict verdict-info';
    $('#check-groups').innerHTML = '<p class="empty card">Could not load this contract report.</p>';
    return;
  }
  const kind = verdictKind(d.status);
  const verdict = $('#verdict');
  verdict.textContent = String(d.status || 'INFO').toUpperCase();
  verdict.className = `verdict verdict-${kind}`;
  const checks = d.checks || [];
  const passed = checks.filter((c) => c.pass === true).length;
  const noFlag = checks.filter((c) => c.pass === null || c.pass === undefined).length;
  const src = d.sim_source === 'nominal_contract'
    ? 'sim reference unavailable: nominal constants used'
    : `sim reference: ${d.sim_source || 'unknown'}`;
  $('#contract-src').textContent = `${src} · ${passed}/${checks.length} checks pass${noFlag ? ` · ${noFlag} without a flag` : ''}`;
  renderChecks(checks);
  const chBox = $('#channel-list');
  const channels = d.channels || [];
  chBox.innerHTML = channels.length
    ? channels.map(channelRow).join('')
    : '<p class="empty">No channel comparison in this report.</p>';
}

function buildContractControls() {
  $('#contract-run').addEventListener('change', (e) => loadContract(e.target.value));
}
