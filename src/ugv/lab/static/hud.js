// The challenge HUD: markup and the per-frame readouts. It shows what the car knows (its own
// estimate, the camera); the truth, and this car's faults, only when asked (T) or at the end.
import { esc, STATE_COLOR, STATES } from './app.js';
import { ENVIRONMENTS } from './env3d.js';

const KEYCAPS = [['KeyQ', 'Q'], ['KeyW', 'W'], ['KeyE', 'E'], ['KeyA', 'A'], ['KeyS', 'S'], ['KeyD', 'D']];
export const QUALITY = { healthy: 'Healthy sensors', noisy: 'Noisy sensors', bad: 'Bad sensors', broken: 'A sensor breaks' };
export const START_NAME = {
  random: 'Random start', tail: 'Behind the tail', 'nose-on': 'Nose-on', 'nose-oblique': 'Nose, oblique',
  'beside-nose-gear': 'Beside the nose wheel', 'under-wingtip': 'Under a wingtip', side: 'Abeam',
};

function envMenu(hdri) {
  const groups = new Map();
  const add = (group, value, label, sub = '') => {
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push(
      `<button type="button" class="envitem" data-env="${esc(value)}" data-name="${esc(`${label} ${sub} ${group}`.toLowerCase())}" role="option" aria-selected="false"><span>${esc(label)}</span>${sub ? `<small>${esc(sub)}</small>` : ''}</button>`,
    );
  };
  for (const [k, label] of Object.entries(ENVIRONMENTS)) add('Simulated sets', k, label);
  for (const a of hdri) {
    const where = `${a.env === 'indoor' ? 'Indoor' : a.env === 'outdoor' ? 'Outdoor' : 'Other'} · ${a.cat}`;
    add(where, `hdri:${a.id}`, a.name);
  }
  return [...groups].map(([g, items]) => `<div class="envsection"><div class="envgroup">${esc(g)}</div>${items.join('')}</div>`).join('');
}

function markup(hdri = []) {
  return `
  <div class="stage" id="ch-stage">
    <canvas id="ch-3d"></canvas><canvas id="ch-ov"></canvas>
    <div class="plate tl">
      <div class="hud-mode" id="ch-mode">Manual</div>
      <div class="fsm" id="ch-fsm">${['MANUAL', ...STATES].map((s) => `<span data-s="${s}" style="--c:${STATE_COLOR[s] || 'var(--ink)'}">${s}</span>`).join('')}</div>
      <div class="hud-flash" id="ch-flash"></div>
    </div>
    <div class="plate tr tele" id="ch-tele"></div>
    <div class="plate bl mini" title="Click to place the car, drag to aim it"><canvas id="ch-map"></canvas><div class="mini-k">What the car believes · click to place</div></div>
    <div class="plate br pip" id="ch-pip-box"><canvas id="ch-pip"></canvas><canvas id="ch-pip-ov"></canvas><div class="mini-k">Car camera · 10 Hz</div></div>
    <div class="dock">
      <div class="envpick">
        <button type="button" id="ch-env" class="envbtn" aria-haspopup="listbox" aria-expanded="false" aria-label="Environment"><span id="ch-env-label">Lab room</span><i></i></button>
        <div class="envmenu" id="ch-env-menu" hidden>
          <div class="envmenu-head"><span>Environment</span><kbd>ESC</kbd></div>
          <input id="ch-env-q" type="search" placeholder="Search ${hdri.length} panoramas" autocomplete="off" aria-label="Search environments">
          <div class="envlist" id="ch-env-list" role="listbox" aria-label="Environments">${envMenu(hdri)}</div>
          <div class="envmenu-foot">Real panoramas: OpenHDRI, ambientCG, Poly Haven · CC0</div>
        </div>
      </div>
      <select id="ch-quality" aria-label="Sensors">${Object.entries(QUALITY).map(([k, v]) => `<option value="${k}">${v}</option>`).join('')}</select>
      <select id="ch-exposure" aria-label="Exposure" title="Applies live to the running round">
        <option value="0">Exposure: auto</option>
        <option value="1">Exposure: fixed +1 stop</option>
        <option value="2">Exposure: fixed +2 stops</option>
        <option value="3">Exposure: fixed +3 stops</option>
        <option value="4">Exposure: fixed +4 stops</option>
      </select>
      <select id="ch-start" aria-label="Start">${Object.entries(START_NAME).map(([k, v]) => `<option value="${k}">${v}</option>`).join('')}<option value="placed" hidden>Placed on the map</option></select>
      <button type="button" id="ch-auto" aria-pressed="false"><span>Autopilot</span> <kbd>P</kbd></button>
      <button type="button" id="ch-new">New round <kbd>R</kbd></button>
      <button type="button" id="ch-truth" aria-pressed="false">Truth <kbd>T</kbd></button>
      <span class="caps">${KEYCAPS.map(([c, k]) => `<span data-c="${c}">${k}</span>`).join('')}</span>
      <span class="wd" title="Keys release themselves if their key-up is lost"><i><b id="ch-wd"></b></i></span>
      <span class="hint">drag to orbit · scroll in to ride the car camera · <kbd>SPACE</kbd> declare parked</span>
    </div>
    <div class="pad" id="ch-pad">${KEYCAPS.map(([c, k]) => `<button type="button" data-key="${c}">${k}</button>`).join('')}<button type="button" data-key="Space" class="wide">Park</button></div>
    <div class="banner" id="ch-banner"></div>
  </div>`;
}

function bind(main) {
  const $ = (s) => main.querySelector(s);
  const tele = $('#ch-tele'), fsm = $('#ch-fsm'), wd = $('#ch-wd'), banner = $('#ch-banner'), flashEl = $('#ch-flash');
  const caps = [...main.querySelectorAll('.caps span')];
  let flashUntil = 0, handlers = {};
  let rounds = [];
  try { rounds = JSON.parse(localStorage.getItem('ugv-lab-rounds') || '[]'); } catch { /* per-browser only */ }

  $('#ch-start').addEventListener('change', (e) => { handlers.start?.(e.target.value); e.target.blur(); });
  $('#ch-quality').addEventListener('change', (e) => { handlers.quality?.(e.target.value); e.target.blur(); });
  $('#ch-exposure').addEventListener('change', (e) => { handlers.exposure?.(parseFloat(e.target.value)); e.target.blur(); });
  const envBtn = $('#ch-env'), envMenu = $('#ch-env-menu'), envQ = $('#ch-env-q'), envList = $('#ch-env-list'), envLabel = $('#ch-env-label');
  const filterEnv = (q) => {
    const s = q.trim().toLowerCase();
    for (const sec of envList.querySelectorAll('.envsection')) {
      let any = false;
      for (const b of sec.querySelectorAll('.envitem')) {
        const on = !s || b.dataset.name.includes(s);
        b.hidden = !on;
        any ||= on;
      }
      sec.hidden = !any;
    }
  };
  const closeEnv = () => { envMenu.hidden = true; envBtn.setAttribute('aria-expanded', 'false'); };
  const setEnvLabel = (v) => {
    const b = envList.querySelector(`.envitem[data-env="${CSS.escape(v)}"]`);
    if (!b) return;
    envLabel.textContent = b.querySelector('span').textContent;
    envList.querySelectorAll('.envitem').forEach((x) => x.setAttribute('aria-selected', String(x === b)));
  };
  const openEnv = () => {
    envMenu.hidden = false;
    envBtn.setAttribute('aria-expanded', 'true');
    envQ.value = '';
    filterEnv('');
    envQ.focus();
  };
  envBtn.addEventListener('click', () => { if (envMenu.hidden) openEnv(); else closeEnv(); });
  envQ.addEventListener('input', () => filterEnv(envQ.value));
  envQ.addEventListener('keydown', (e) => { if (e.key === 'Escape') { closeEnv(); envBtn.focus(); } });
  envList.addEventListener('click', (e) => {
    const b = e.target.closest('.envitem');
    if (!b) return;
    setEnvLabel(b.dataset.env);
    closeEnv();
    handlers.env?.(b.dataset.env);
    envBtn.blur();
  });
  const onDocDown = (e) => { if (!envMenu.hidden && !e.target.closest('.envpick')) closeEnv(); };
  const onDocKey = (e) => { if (e.key === 'Escape' && !envMenu.hidden) closeEnv(); };
  document.addEventListener('pointerdown', onDocDown);
  document.addEventListener('keydown', onDocKey);
  $('#ch-auto').addEventListener('click', (e) => { handlers.auto?.(); e.currentTarget.blur(); });
  $('#ch-new').addEventListener('click', (e) => { handlers.fresh?.(); e.currentTarget.blur(); });
  $('#ch-truth').addEventListener('click', (e) => { handlers.truth?.(); e.currentTarget.blur(); });
  banner.addEventListener('click', (e) => { const b = e.target.closest('button[data-act]'); if (b) handlers[b.dataset.act]?.(); });

  const cm = (v) => (v * 100).toFixed(1);
  const deg = (v) => ((v * 180) / Math.PI).toFixed(1);
  const faultList = (f) => (f.length ? f.map((x) => `<li>${esc(x)}</li>`).join('') : '<li>no faults: a healthy car</li>');
  return {
    on(h) { handlers = h; },
    start(v) { $('#ch-start').value = v; },
    setup(env, quality, exposure = 0) { setEnvLabel(env); $('#ch-quality').value = quality; $('#ch-exposure').value = String(exposure); },
    auto(on) {
      const b = $('#ch-auto');
      b.classList.toggle('go', on);
      b.setAttribute('aria-pressed', on);
      b.querySelector('span').textContent = on ? 'Autopilot on' : 'Autopilot';
    },
    truth(on) { $('#ch-truth').classList.toggle('on', on); $('#ch-truth').setAttribute('aria-pressed', on); },
    mode(t) { $('#ch-mode').textContent = t; },
    flash(t) { flashEl.textContent = t; flashUntil = performance.now() + 2600; },
    log(row) {
      rounds = [row, ...rounds].slice(0, 50);
      try { localStorage.setItem('ugv-lab-rounds', JSON.stringify(rounds)); } catch { /* ignore */ }
    },
    banner(kind, info = {}) {
      banner.hidden = !kind;
      banner.classList.toggle('ready', kind === 'ready');
      if (!kind) return;
      if (kind === 'ready') {
        banner.innerHTML = `<div><div class="b-k">${esc(info.start)} · round ${info.seed}</div><div class="b-v">Park under the belly</div>
          <p>Enter from the tail · 3.5 cm to spare per side · new faults every round</p>
          <div class="row" style="justify-content:center;margin-top:16px"><span class="prov w-hint"><kbd>W</kbd> drive · <kbd>Space</kbd> parked · drag to look</span><button type="button" class="drive-btn" data-act="drive">Drive</button><button type="button" class="go" data-act="auto">Autopilot from here</button></div></div>`;
        return;
      }
      const ok = kind === 'parked';
      banner.innerHTML = `<div><div class="b-k">${esc(info.who)} · ${info.t.toFixed(1)} s</div><div class="b-v" style="color:var(--${ok ? 'good' : 'bad'})">${esc(ok ? 'Parked' : info.why)}</div>
        <p class="mono">true error x ${cm(info.e[0])} cm · y ${cm(info.e[1])} cm · yaw ${deg(info.e[2])}°</p>
        <ul class="faults">${faultList(info.faults)}</ul>
        <div class="row" style="justify-content:center;margin-top:20px"><button type="button" data-act="retry">Try again</button><button type="button" class="go" data-act="fresh">New round</button></div></div>`;
    },
    update({ snap, est, e, eTrue, input, truth, auto, faults }) {
      const st = snap?.state || 'MANUAL';
      fsm.querySelectorAll('span').forEach((s) => s.classList.toggle('on', s.dataset.s === st));
      flashEl.hidden = performance.now() > flashUntil;
      const held = input.held();
      caps.forEach((c) => c.classList.toggle('on', held.has(c.dataset.c)));
      wd.style.width = held.size ? `${100 * (1 - input.watchdog())}%` : '0';
      wd.closest('.wd').classList.toggle('trip', !!input.tripped());
      const v = snap ? Math.hypot(snap.vel[0], snap.vel[1]) : 0;
      const dist = e ? Math.hypot(e[0], e[1]) : null;
      const camLost = !!snap?.cam_fx?.lost;
      tele.innerHTML = `
        <div><span>Time</span><b>${(snap?.t ?? 0).toFixed(1)}<small>s</small></b></div>
        <div><span>Speed</span><b>${v.toFixed(2)}<small>m/s</small></b></div>
        <div><span>To bay</span><b>${dist == null ? '—' : (dist * 100).toFixed(dist < 0.1 ? 1 : 0)}<small>cm</small></b></div>
        <div><span>Heading off</span><b>${e ? deg(e[2]) : '—'}<small>°</small></b></div>
        <div><span>Keypoints</span><b>${snap?.kp?.length ?? 0}<small>/12</small></b></div>
        <div><span>Pose fix</span><b class="${snap?.fix ? 'fix' : 'nofix'}">${snap?.fix ? 'Yes' : est ? 'Old' : 'No'}</b></div>
        <div><span>Camera</span><b class="${camLost ? 'nofix' : 'fix'}">${camLost ? 'Lost' : 'Live'}</b></div>
        <div title="Trained keypoint network (runs/pose_train kp-v1 then kp-crop-v1), server CPU"><span>Network</span><b>${snap?.cam_fx?.ms != null ? Math.round(snap.cam_fx.ms) : '—'}<small>ms</small></b></div>
        ${truth ? `<div class="truth"><span>True error</span><b>${cm(eTrue[0])} / ${cm(eTrue[1])}<small>cm</small> ${deg(eTrue[2])}<small>°</small></b><ul class="faults">${faultList(faults)}${snap?.floor && snap.floor[1] > 0.05 ? `<li>floor grip ${snap.floor[0].toFixed(2)} · skid ${snap.floor[1].toFixed(2)} · patchy ${snap.floor[2].toFixed(2)}</li>` : ''}</ul></div>` : ''}
        <div class="sub"><span>${auto ? 'Tuned controller' : input.fine() ? 'Fine control' : 'Full control'}</span><span>${rounds.filter((x) => x.ok).length}/${rounds.length} parked by you</span></div>`;
    },
    dispose() {
      document.removeEventListener('pointerdown', onDocDown);
      document.removeEventListener('keydown', onDocKey);
    },
  };
}

export const hud = { markup, bind };
