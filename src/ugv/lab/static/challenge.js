// Park it yourself: the car's physics (measured gains, deadzones, lag; optionally a random car),
// the airframe's real collision geometry, and the view from the car's own camera.
import * as THREE from '/static/vendor/three.module.min.js';
import { api, esc } from './app.js';
import { loadStrlModel } from './stl.js';
import { drawWorld, makeView } from './world2d.js';

const LEVELS = {
  easy: { label: 'Easy: nominal car, map on', random: 0, map: true },
  real: { label: 'Realistic: a random car, map on', random: 1, map: true },
  camera: { label: 'Camera only: random car, no map', random: 1, map: false },
};
const KEYS = { KeyW: [0, 1], KeyS: [0, -1], KeyA: [1, 1], KeyD: [1, -1], KeyQ: [2, 1], KeyE: [2, -1] };

let stop = () => {};
export function dispose() { stop(); }

export async function render(main) {
  const world = await api('/api/park/world');
  const evals = await api('/api/park/evals').catch(() => []);
  const bay = world.bay, plant = world.plant;
  main.innerHTML = `
    <div class="row"><h1>Parking challenge</h1><span class="spacer"></span>
      <select id="ch-level">${Object.entries(LEVELS).map(([k, v]) => `<option value="${k}">${v.label}</option>`).join('')}</select>
      <button id="ch-view" type="button">Chase view</button>
      <button id="ch-new" class="primary" type="button">New round</button></div>
    <p class="lede">Find the airframe, line up behind the tail and creep in under the belly. The
    bay has ${(bay.clearance_m * 100).toFixed(1)} cm to spare on each side at best. Drive with
    <kbd>W</kbd><kbd>S</kbd> forward/back, <kbd>A</kbd><kbd>D</kbd> strafe, <kbd>Q</kbd><kbd>E</kbd>
    turn, hold <kbd>Shift</kbd> for fine control, <kbd>Space</kbd> to declare "parked". The car
    has the real deadzones: small inputs do nothing. Simulation only; the real car is never touched.</p>
    <div class="game">
      <div><div class="viewport" id="ch-vp"><canvas id="ch-3d"></canvas>
        <div class="hud"><span id="ch-hud-l"></span><span id="ch-hud-r"></span></div>
        <div class="banner" id="ch-banner"><div><b>Ready</b>Press <kbd>W</kbd> to start, or watch the autopilot.</div></div></div>
        <div class="row" style="margin-top:10px"><button id="ch-auto" type="button" ${evals.some((e) => e.has_traces) ? '' : 'disabled'}>Watch the autopilot</button>
        <span class="muted" id="ch-auto-note"></span></div></div>
      <div class="grid">
        <canvas id="ch-map" class="map"></canvas>
        <section class="card"><h2>Your rounds <span class="muted">stored in this browser</span></h2><div id="ch-board"></div></section>
      </div>
    </div>`;

  // --- state
  let level = LEVELS.easy, chase = false, round = null, keys = new Set(), raf = 0, last = 0, auto = null;
  const vp = main.querySelector('#ch-vp');
  const canvas3d = main.querySelector('#ch-3d');
  const banner = main.querySelector('#ch-banner');
  const mapCanvas = main.querySelector('#ch-map');
  const view2d = makeView(mapCanvas, 2.0, [0, 0]);

  // --- 3D scene
  const renderer = new THREE.WebGLRenderer({ canvas: canvas3d, antialias: true });
  renderer.setPixelRatio(devicePixelRatio || 1);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x11151b);
  scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x30302a, 1.2));
  const sun = new THREE.DirectionalLight(0xffffff, 1.6);
  sun.position.set(2, 4, 1);
  scene.add(sun);
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(12, 12), new THREE.MeshStandardMaterial({ color: 0x2a2d31, roughness: 0.95 }));
  floor.rotation.x = -Math.PI / 2;
  scene.add(floor);
  const grid = new THREE.GridHelper(12, 48, 0x3c4148, 0x33373d);
  grid.position.y = 0.001;
  scene.add(grid);
  const P = (x, y, z = 0) => new THREE.Vector3(x, z, -y); // airframe frame -> three
  // target outline on the floor
  {
    const t = bay.target, l = bay.car.length / 2 + bay.tolerance.x, w = bay.car.width / 2 + bay.tolerance.y;
    const c = Math.cos(t[2]), s = Math.sin(t[2]);
    const pts = [[-l, -w], [l, -w], [l, w], [-l, w], [-l, -w]].map(([u, v]) => P(t[0] + c * u - s * v, t[1] + s * u + c * v, 0.002));
    scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: 0x0ca30c })));
  }
  const fovV = (2 * Math.atan(360 / 968) * 180) / Math.PI;
  const cam = new THREE.PerspectiveCamera(fovV, 16 / 9, 0.02, 40);
  const carMesh = new THREE.Mesh(new THREE.BoxGeometry(bay.car.length, 0.044, bay.car.width), new THREE.MeshStandardMaterial({ color: 0x3987e5 }));
  scene.add(carMesh);
  loadStrlModel(() => {}).then(({ root }) => { root.rotation.y = 0; scene.add(root); }).catch(() => {});

  const resize = () => {
    const w = vp.clientWidth, h = vp.clientHeight;
    renderer.setSize(w, h, false);
    cam.aspect = w / h;
    cam.updateProjectionMatrix();
  };
  addEventListener('resize', resize);
  resize();

  // --- physics
  const randCar = (k) => {
    const j = (v, r) => v * (1 + k * (Math.random() * 2 - 1) * r);
    return { gain: plant.gain.map((g) => j(g, 0.25)), dz: plant.deadzone.map((d) => j(d, 0.4)), tau: plant.tau.map((t) => j(t, 0.5)), slip: k * (Math.random() * 1.2 - 0.6) };
  };
  const collides = (q) => {
    const c = Math.cos(q[2]), s = Math.sin(q[2]);
    const inside = (pts, x0, x1, y0, y1) => pts.some(([x, y]) => {
      const dx = x - q[0], dy = y - q[1];
      const u = c * dx + s * dy, v = -s * dx + c * dy;
      return u >= x0 && u <= x1 && v >= y0 && v <= y1;
    });
    const L = bay.car.length / 2, W = bay.car.width / 2;
    return inside(bay.obstacles, -L, L, -W, W) || inside(bay.tall, bay.car.tower_x[0], bay.car.tower_x[1], bay.car.tower_y[0], bay.car.tower_y[1]);
  };
  const newRound = () => {
    let q;
    do {
      const d = 1.4 + Math.random() * 1.1, b = Math.random() * 2 * Math.PI;
      q = [d * Math.cos(b), d * Math.sin(b), Math.random() * 2 * Math.PI - Math.PI];
    } while (collides(q));
    round = { q, v: [0, 0, 0], car: randCar(level.random), t: 0, started: false, done: null, path: [q.slice(0, 2)], penalty: 0 };
    auto = null;
    banner.hidden = false;
    banner.innerHTML = '<div><b>Ready</b>Press <kbd>W</kbd> to start, or watch the autopilot.</div>';
  };
  const step = (dt) => {
    const r = round;
    if (!r || r.done) return;
    const fine = keys.has('ShiftLeft') || keys.has('ShiftRight');
    const duty = [0, 0, 0];
    for (const k of keys) if (KEYS[k]) duty[KEYS[k][0]] += KEYS[k][1] * (fine ? [0.2, 0.25, 0.14][KEYS[k][0]] : [0.32, 0.38, 0.2][KEYS[k][0]]);
    if (!r.started && duty.some((d) => d)) { r.started = true; banner.hidden = true; }
    if (!r.started) return;
    r.t += dt;
    for (let i = 0; i < 3; i++) {
      const d = Math.max(-plant.duty_cap, Math.min(plant.duty_cap, duty[i]));
      const target = Math.sign(d) * Math.max(Math.abs(d) - r.car.dz[i], 0) * r.car.gain[i];
      r.v[i] += (1 - Math.exp(-dt / r.car.tau[i])) * (target - r.v[i]);
    }
    const w = r.v[2] + r.car.slip * r.v[1];
    const c = Math.cos(r.q[2]), s = Math.sin(r.q[2]);
    const next = [r.q[0] + (c * r.v[0] - s * r.v[1]) * dt, r.q[1] + (s * r.v[0] + c * r.v[1]) * dt, r.q[2] + w * dt];
    if (collides(next)) { finish(false, 'Contact with the airframe'); return; }
    if (Math.hypot(next[0], next[1]) > 4) { finish(false, 'Left the room'); return; }
    r.q = next;
    if (Math.hypot(r.q[0] - r.path.at(-1)[0], r.q[1] - r.path.at(-1)[1]) > 0.02) r.path.push(r.q.slice(0, 2));
  };
  const errors = (q) => {
    const t = bay.target;
    let dy = q[2] - t[2];
    dy = Math.atan2(Math.sin(dy), Math.cos(dy));
    return [q[0] - t[0], q[1] - t[1], dy];
  };
  const finish = (ok, why) => {
    const r = round;
    const e = errors(r.q);
    r.done = { ok, why, t: r.t + r.penalty, e };
    banner.hidden = false;
    banner.innerHTML = `<div><b>${ok ? 'Parked' : esc(why)}</b>${ok ? `${(r.t + r.penalty).toFixed(1)} s · off by ${(e[0] * 100).toFixed(1)} / ${(e[1] * 100).toFixed(1)} cm, ${(e[2] * 57.3).toFixed(1)}°` : 'The real car would need a person to come and reset it.'}<br><br>Press <kbd>R</kbd> or "New round".</div>`;
    save({ when: Date.now(), level: main.querySelector('#ch-level').value, ok, why, t: +(r.t + r.penalty).toFixed(1), e: e.map((v) => +v.toFixed(3)) });
  };
  const declare = () => {
    const r = round;
    if (!r || r.done || !r.started) return;
    const e = errors(r.q), tol = bay.tolerance;
    const still = Math.hypot(r.v[0], r.v[1]) < 0.01;
    if (Math.abs(e[0]) <= tol.x && Math.abs(e[1]) <= tol.y && Math.abs(e[2]) <= (tol.yaw_deg * Math.PI) / 180 && still) finish(true);
    else { r.penalty += 5; flash(`Not parked yet: off by ${(e[0] * 100).toFixed(1)} / ${(e[1] * 100).toFixed(1)} cm, ${(e[2] * 57.3).toFixed(1)}° (+5 s)`); }
  };
  let flashUntil = 0, flashText = '';
  const flash = (t) => { flashText = t; flashUntil = performance.now() + 2500; };

  // --- leaderboard (per-browser convenience; works without storage)
  const load = () => { try { return JSON.parse(localStorage.getItem('ugv-lab-rounds') || '[]'); } catch { return []; } };
  const save = (row) => {
    const rows = [row, ...load()].slice(0, 50);
    try { localStorage.setItem('ugv-lab-rounds', JSON.stringify(rows)); } catch { /* ignore */ }
    board(rows);
  };
  const board = (rows = load()) => {
    const el = main.querySelector('#ch-board');
    if (!rows.length) { el.innerHTML = '<div class="muted">No rounds yet.</div>'; return; }
    const ok = rows.filter((r) => r.ok);
    el.innerHTML = `<div class="muted" style="margin-bottom:6px">${ok.length}/${rows.length} parked · best ${ok.length ? Math.min(...ok.map((r) => r.t)).toFixed(1) + ' s' : '–'}</div>
      <table><tbody>${rows.slice(0, 8).map((r) => `<tr><td><span class="pill ${r.ok ? 'good' : 'bad'}">${r.ok ? 'parked' : esc(r.why)}</span></td><td class="muted">${esc(r.level)}</td><td class="num">${r.t}s</td>
      <td class="num mono">${(r.e[0] * 100).toFixed(1)}/${(r.e[1] * 100).toFixed(1)} cm</td></tr>`).join('')}</tbody></table>`;
  };
  board();

  // --- autopilot replay
  const startAuto = async () => {
    const ev = evals.find((e) => e.has_traces);
    const traces = await api(`/api/park/traces/${encodeURIComponent(ev.file.replace(/\.json$/, ''))}`);
    const tr = traces.find((t) => t.outcome === 'parked') || traces[0];
    auto = { tr, k: 0, acc: 0 };
    round = { q: tr.trace[0].car, v: [0, 0, 0], t: 0, started: true, done: null, path: [], penalty: 0 };
    banner.hidden = true;
    main.querySelector('#ch-auto-note').textContent = `${ev.name}, episode ${tr.episode}: ${tr.outcome}`;
  };
  main.querySelector('#ch-auto').addEventListener('click', startAuto);

  // --- loop
  const frame = (now) => {
    const dt = Math.min(0.05, (now - (last || now)) / 1000);
    last = now;
    if (auto) {
      auto.acc += dt;
      while (auto.acc > 0.05 && auto.k < auto.tr.trace.length - 1) { auto.acc -= 0.05; auto.k++; }
      const row = auto.tr.trace[auto.k];
      round.q = row.car; round.t = row.t; round.state = row.state;
      round.path.push(row.car.slice(0, 2));
    } else {
      step(dt);
    }
    const q = round?.q || [2, 0, Math.PI];
    // camera
    const c = Math.cos(q[2]), s = Math.sin(q[2]);
    carMesh.position.copy(P(q[0], q[1], 0.022));
    carMesh.rotation.y = q[2];
    carMesh.visible = chase;
    if (chase) {
      cam.position.copy(P(q[0] - 0.9 * c, q[1] - 0.9 * s, 0.55));
      cam.lookAt(P(q[0] + 0.3 * c, q[1] + 0.3 * s, 0.05));
    } else {
      const cx = bay.car.camera_x;
      cam.position.copy(P(q[0] + cx * c, q[1] + cx * s, bay.car.camera_z));
      cam.lookAt(P(q[0] + (cx + 1) * c, q[1] + (cx + 1) * s, bay.car.camera_z));
    }
    renderer.render(scene, cam);
    // map
    if (level.map || auto || round?.done) drawWorld(view2d, bay, { path: round?.path || [], car: q, state: auto ? round.state : null });
    else { const g = view2d.ctx; g.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--surface-2'); g.fillRect(0, 0, view2d.size, view2d.size); g.fillStyle = '#888'; g.font = '13px system-ui'; g.fillText('map hidden: drive from the camera, like the car', 16, 28); }
    // HUD
    const e = errors(q);
    main.querySelector('#ch-hud-l').textContent = auto ? `AUTOPILOT ${round.state}  t=${round.t.toFixed(1)}s` : `t=${(round?.t || 0).toFixed(1)}s  v=${Math.hypot(round?.v[0] || 0, round?.v[1] || 0).toFixed(2)} m/s${performance.now() < flashUntil ? '  ' + flashText : ''}`;
    main.querySelector('#ch-hud-r').textContent = level.map ? `to bay ${(Math.hypot(e[0], e[1]) * 100).toFixed(0)} cm  yaw ${(e[2] * 57.3).toFixed(0)}°` : 'camera only';
    if (auto && auto.k >= auto.tr.trace.length - 1 && !round.done) {
      round.done = { ok: auto.tr.outcome === 'parked' };
      banner.hidden = false;
      banner.innerHTML = `<div><b>Autopilot: ${esc(auto.tr.outcome)}</b>${round.t.toFixed(1)} s. Press <kbd>R</kbd> to drive yourself.</div>`;
    }
    raf = requestAnimationFrame(frame);
  };

  const down = (e) => {
    if (['KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyQ', 'KeyE', 'Space'].includes(e.code) && document.activeElement?.tagName !== 'SELECT') e.preventDefault();
    if (e.code === 'Space') declare();
    else if (e.code === 'KeyR') newRound();
    else if (!auto) keys.add(e.code);
  };
  const up = (e) => keys.delete(e.code);
  addEventListener('keydown', down);
  addEventListener('keyup', up);
  addEventListener('blur', () => keys.clear());
  main.querySelector('#ch-new').addEventListener('click', newRound);
  main.querySelector('#ch-level').addEventListener('change', (e) => { level = LEVELS[e.target.value]; newRound(); e.target.blur(); });
  main.querySelector('#ch-view').addEventListener('click', (e) => { chase = !chase; e.target.textContent = chase ? 'Car camera' : 'Chase view'; e.target.blur(); });
  newRound();
  raf = requestAnimationFrame(frame);
  if (location.hash.endsWith('/auto') && evals.some((e) => e.has_traces)) { chase = true; startAuto(); }
  stop = () => {
    cancelAnimationFrame(raf);
    removeEventListener('keydown', down);
    removeEventListener('keyup', up);
    removeEventListener('resize', resize);
    renderer.dispose();
  };
}
