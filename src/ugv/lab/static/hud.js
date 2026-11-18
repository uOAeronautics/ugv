// The challenge HUD: markup and the per-frame readouts. Pure DOM; the loop lives in challenge.js.
import { esc, STATE_COLOR, STATES } from './app.js';

const KEYCAPS = [['KeyQ', 'Q', 'turn'], ['KeyW', 'W', 'fwd'], ['KeyE', 'E', 'turn'], ['KeyA', 'A', 'left'], ['KeyS', 'S', 'back'], ['KeyD', 'D', 'right']];
const VIEW_NAME = { cam: 'Car camera', chase: 'Chase', top: 'Overhead' };

function markup(levels) {
  return `
  <div class="stage" id="ch-stage">
    <canvas id="ch-3d"></canvas><canvas id="ch-ov"></canvas>
    <div class="plate tl">
      <div class="hud-mode" id="ch-mode">MANUAL</div>
      <div class="fsm" id="ch-fsm">${STATES.map((s) => `<span data-s="${s}" style="--c:${STATE_COLOR[s]}">${s}</span>`).join('')}</div>
      <div class="hud-flash" id="ch-flash"></div>
    </div>
    <div class="plate tr tele" id="ch-tele"></div>
    <div class="plate bl mini"><canvas id="ch-map"></canvas><div class="mini-off" id="ch-mini-off">MAP OFF · CAMERA ONLY</div></div>
    <div class="plate br input">
      <div class="caps">${KEYCAPS.map(([c, k, l]) => `<span data-c="${c}"><b>${k}</b><small>${l}</small></span>`).join('')}</div>
      <div class="wd"><span>KEY WATCHDOG</span><i><b id="ch-wd"></b></i></div>
      <div class="hint"><kbd>SHIFT</kbd> fine · <kbd>SPACE</kbd> declare parked · <kbd>V</kbd> view · <kbd>P</kbd> autopilot · <kbd>R</kbd> new round</div>
    </div>
    <div class="dock">
      <div class="seg" id="ch-level">${Object.entries(levels).map(([k, v]) => `<button type="button" data-l="${k}">${v}</button>`).join('')}</div>
      <div class="seg" id="ch-view">${Object.entries(VIEW_NAME).map(([k, v]) => `<button type="button" data-v="${k}">${v}</button>`).join('')}</div>
      <button type="button" class="go" id="ch-auto">Autopilot</button>
      <button type="button" id="ch-new">New round</button>
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
  const load = () => { try { return JSON.parse(localStorage.getItem('ugv-lab-rounds') || '[]'); } catch { return []; } };
  let rounds = load();

  $('#ch-level').addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b) return; level(b.dataset.l); handlers.level?.(b.dataset.l); b.blur(); });
  $('#ch-view').addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) { handlers.view?.(b.dataset.v); b.blur(); } });
  $('#ch-auto').addEventListener('click', (e) => { handlers.auto?.(); e.target.blur(); });
  $('#ch-new').addEventListener('click', (e) => { handlers.fresh?.(); e.target.blur(); });
  banner.addEventListener('click', (e) => { const b = e.target.closest('button[data-act]'); if (b) handlers[b.dataset.act]?.(); });
  const level = (l) => main.querySelectorAll('#ch-level button').forEach((b) => b.classList.toggle('on', b.dataset.l === l));
  level('nominal');

  const cm = (v) => `${(v * 100).toFixed(1)}`;
  const deg = (v) => `${((v * 180) / Math.PI).toFixed(1)}`;
  return {
    on(h) { handlers = h; },
    view(v) { main.querySelectorAll('#ch-view button').forEach((b) => b.classList.toggle('on', b.dataset.v === v)); },
    mode(t) { $('#ch-mode').textContent = t; },
    flash(t) { flashEl.textContent = t; flashUntil = performance.now() + 2600; },
    log(row) {
      rounds = [row, ...rounds].slice(0, 50);
      try { localStorage.setItem('ugv-lab-rounds', JSON.stringify(rounds)); } catch { /* per-browser only */ }
    },
    banner(kind, info = {}) {
      banner.hidden = !kind;
      banner.classList.toggle('ready', kind === 'ready');
      if (!kind) return;
      const btns = '<div class="row" style="justify-content:center;margin-top:22px"><button type="button" data-act="retry">Same start again</button><button type="button" class="go" data-act="fresh">New round</button><button type="button" data-act="auto">Watch autopilot</button></div>';
      if (kind === 'ready') {
        banner.innerHTML = `<div><div class="b-k">${esc(info.level)} · round ${info.seed}</div><div class="b-v">Park under the belly</div>
          <p>Enter from the tail, between the main gear and the nose wheel. 3.5 cm to spare per side.<br>The car has the real deadzones: small inputs do nothing.</p>
          <div class="row" style="justify-content:center;margin-top:22px"><span class="prov w-hint"><kbd>W</kbd> to drive</span><button type="button" class="drive-btn" data-act="drive">Drive</button><button type="button" class="go" data-act="auto">Watch autopilot from here</button></div></div>`;
      } else if (kind === 'parked') {
        banner.innerHTML = `<div><div class="b-k">${esc(info.who)}</div><div class="b-v" style="color:var(--good)">Parked</div>
          <p class="mono">${info.t.toFixed(1)} s · x ${cm(info.e[0])} cm · y ${cm(info.e[1])} cm · yaw ${deg(info.e[2])}°</p>${btns}</div>`;
      } else {
        banner.innerHTML = `<div><div class="b-k">${esc(info.who)}</div><div class="b-v" style="color:var(--bad)">${esc(info.why || 'Not parked')}</div>
          <p class="mono">${info.t.toFixed(1)} s · off by ${cm(info.e[0])} / ${cm(info.e[1])} cm · ${deg(info.e[2])}°</p>${btns}</div>`;
      }
    },
    update({ r, e, state, kp, believed, input, blind, level: lv }) {
      main.querySelector('.mini').classList.toggle('off', blind);
      fsm.classList.toggle('idle', !state);
      fsm.querySelectorAll('span').forEach((s) => s.classList.toggle('on', s.dataset.s === state));
      flashEl.hidden = performance.now() > flashUntil;
      const held = input.held();
      caps.forEach((c) => c.classList.toggle('on', held.has(c.dataset.c)));
      wd.style.width = held.size ? `${100 * (1 - input.watchdog())}%` : '0';
      wd.parentElement.parentElement.classList.toggle('trip', !!input.tripped());
      const v = r ? Math.hypot(r.vel[0], r.vel[1]) : 0;
      const ok = rounds.filter((x) => x.ok && x.level === lv);
      const dist = Math.hypot(e[0], e[1]);
      tele.innerHTML = `
        <div><span>T</span><b>${(r?.t ?? 0).toFixed(1)}<small>s</small></b></div>
        <div><span>SPEED</span><b>${v.toFixed(2)}<small>m/s</small></b></div>
        <div><span>TO BAY</span><b>${blind ? '—' : (dist * 100).toFixed(dist < 0.1 ? 1 : 0)}<small>cm</small></b></div>
        <div><span>YAW ERR</span><b>${blind ? '—' : deg(e[2])}<small>°</small></b></div>
        <div><span>KEYPOINTS</span><b>${kp ? kp.length : 0}<small>/12</small></b></div>
        <div><span>POSE FIX</span><b class="${believed ? 'fix' : 'nofix'}">${believed ? 'YES' : 'NO'}</b></div>
        <div class="sub"><span>${input.fine() ? 'FINE CONTROL' : r?.auto ? 'TUNED CONTROLLER' : 'FULL CONTROL'}</span><span>${ok.length ? `best ${Math.min(...ok.map((x) => x.t)).toFixed(1)} s` : 'no park yet'}</span></div>`;
    },
  };
}

export const hud = { markup, bind };
