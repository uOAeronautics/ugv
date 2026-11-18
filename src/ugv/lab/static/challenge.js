// Park it yourself, or watch the tuned controller do it from the same start in the same car.
// Rounds, the autopilot (ugv.park.episode.run) and the keypoint detector run on the server; the
// hand-driven plant is a port of ugv.park.plant. Simulation only.
import * as THREE from '/static/vendor/three.module.min.js';
import { api, color } from './app.js';
import { makeInput, stepPlant } from './drive.js';
import { hud } from './hud.js';
import { makeWorld, P } from './scene3d.js';
import { drawWorld, loadPlan, makeView } from './world2d.js';

const LEVELS = { nominal: 'Nominal car', random: 'Random car', blind: 'Camera only' };
const VIEWS = ['cam', 'chase', 'top'];
const W0 = 1280, H0 = 720, F0 = 968;

let stop = () => {};
let rethemeFn = () => {};
export function dispose() { stop(); }
export const retheme = () => rethemeFn();

const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));
const compose = (a, b) => [a[0] + Math.cos(a[2]) * b[0] - Math.sin(a[2]) * b[1], a[1] + Math.sin(a[2]) * b[0] + Math.cos(a[2]) * b[1], wrap(a[2] + b[2])];
const inverse = (a) => [-Math.cos(a[2]) * a[0] - Math.sin(a[2]) * a[1], Math.sin(a[2]) * a[0] - Math.cos(a[2]) * a[1], -a[2]];

export async function render(main) {
  const [world, plan] = await Promise.all([api('/api/park/world'), loadPlan()]);
  const bay = world.bay;
  main.classList.add('stage-page');
  main.innerHTML = hud.markup(LEVELS);
  const $ = (s) => main.querySelector(s);
  const stage = $('#ch-stage');
  const W = await makeWorld($('#ch-3d'), bay, plan.skeleton);
  const ov = $('#ch-ov'), og = ov.getContext('2d');
  const mini = makeView($('#ch-map'), 2.0, [-0.45, 0]);
  const H = hud.bind(main);

  let level = 'nominal', view = 'chase', round = null, raf = 0, last = 0, tick = 0, busy = false;
  const input = makeInput({ onAction: (k) => ({ Space: declare, KeyR: () => newRound(), KeyV: cycleView, KeyP: autopilot })[k]?.() });
  input.pad($('#ch-pad'));

  // --- rounds
  const newRound = async (seed = Math.floor(Math.random() * 1e6)) => {
    const r = await api(`/api/park/round?seed=${seed}&level=${level}`, { fresh: true });
    round = { ...r, q: r.start.slice(), vel: [0, 0, 0], t: 0, started: false, done: null, path: [r.start.slice(0, 2)], penalty: 0, seen: null, auto: null };
    input.reset();
    H.banner('ready', { level: LEVELS[level], seed });
    H.mode('MANUAL');
  };
  const collides = (q) => {
    const c = Math.cos(q[2]), s = Math.sin(q[2]);
    const hit = (pts, x0, x1, y0, y1) => pts.some(([x, y]) => {
      const dx = x - q[0], dy = y - q[1], u = c * dx + s * dy, v = -s * dx + c * dy;
      return u >= x0 && u <= x1 && v >= y0 && v <= y1;
    });
    const L = bay.car.length / 2, Wd = bay.car.width / 2;
    return hit(bay.obstacles, -L, L, -Wd, Wd) || hit(bay.tall, bay.car.tower_x[0], bay.car.tower_x[1], bay.car.tower_y[0], bay.car.tower_y[1]);
  };
  const error = (q) => [q[0] - bay.target[0], q[1] - bay.target[1], wrap(q[2] - bay.target[2])];
  const finish = (ok, why) => {
    const r = round, e = error(r.q);
    r.done = { ok, why, t: r.t + r.penalty, e };
    H.banner(ok ? 'parked' : 'fail', { why, t: r.t + r.penalty, e, who: r.auto ? 'Autopilot' : 'You', outcome: r.auto?.res.outcome });
    if (!r.auto) H.log({ when: Date.now(), level, seed: r.seed, ok, why: why || 'parked', t: +(r.t + r.penalty).toFixed(1), e: e.map((v) => +v.toFixed(3)) });
  };
  const declare = () => {
    const r = round;
    if (!r || r.done || !r.started || r.auto) return;
    const e = error(r.q), tol = bay.tolerance;
    const ok = Math.abs(e[0]) <= tol.x && Math.abs(e[1]) <= tol.y && Math.abs(e[2]) <= (tol.yaw_deg * Math.PI) / 180;
    if (ok && Math.hypot(r.vel[0], r.vel[1]) < 0.01) finish(true);
    else { r.penalty += 5; H.flash(`NOT PARKED · ${(e[0] * 100).toFixed(1)} / ${(e[1] * 100).toFixed(1)} cm · ${((e[2] * 180) / Math.PI).toFixed(1)}° · +5 s`); }
  };
  const autopilot = async () => {
    if (!round || busy) return;
    busy = true;
    H.mode('AUTOPILOT · computing');
    try {
      const res = await api('/api/park/autopilot', { body: { seed: round.seed, level, start: round.start } });
      Object.assign(round, { q: round.start.slice(), vel: [0, 0, 0], t: 0, started: true, done: null, path: [], penalty: 0, auto: { res, k: 0 } });
      H.banner(null);
      H.mode('AUTOPILOT');
    } finally { busy = false; }
  };
  const cycleView = () => { view = VIEWS[(VIEWS.indexOf(view) + 1) % VIEWS.length]; H.view(view); };

  // --- detector frames for the hand-driven car (10 Hz, one request in flight)
  let inflight = false;
  const perceive = async () => {
    if (inflight || !round || round.auto || round.done) return;
    inflight = true;
    const r = round, q = r.q.slice();
    try {
      const res = await api('/api/park/perceive', { body: { seed: r.seed, level, car: q, tick: tick++ } });
      if (r === round) r.seen = { kp: res.kp, est: res.pose ? compose(q, res.pose) : null, t: performance.now() };
    } catch { /* keep the last frame */ } finally { inflight = false; }
  };
  const percTimer = setInterval(perceive, 100);

  // --- camera: the true camera, mount errors included (what the image is formed by)
  const camRig = (q) => {
    const cam = W.cam, w = stage.clientWidth, h = stage.clientHeight;
    const ce = round?.camera_error || { focal_scale: 1, yaw: 0, pitch: 0 };
    const k = Math.min(w / W0, h / H0);
    if (view === 'cam') {
      cam.fov = (2 * Math.atan(h / 2 / (F0 * ce.focal_scale * k)) * 180) / Math.PI;
      const c = Math.cos(q[2]), s = Math.sin(q[2]), cx = bay.car.camera_x;
      cam.position.copy(P(q[0] + cx * c, q[1] + cx * s, bay.car.camera_z));
      cam.lookAt(P(q[0] + (cx + 1) * c, q[1] + (cx + 1) * s, bay.car.camera_z));
      const cy = Math.cos(ce.yaw), sy = Math.sin(ce.yaw), cp = Math.cos(ce.pitch), sp = Math.sin(ce.pitch);
      const M = new THREE.Matrix3().set(1, 0, 0, 0, cp, -sp, 0, sp, cp).multiply(new THREE.Matrix3().set(cy, 0, -sy, 0, 1, 0, sy, 0, cy));
      const S = new THREE.Matrix3().set(1, 0, 0, 0, -1, 0, 0, 0, -1);
      const C = S.clone().multiply(M.clone().transpose()).multiply(S);
      cam.quaternion.multiply(new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().setFromMatrix3(C)));
    } else if (view === 'chase') {
      cam.fov = 50;
      const c = Math.cos(q[2]), s = Math.sin(q[2]);
      cam.position.copy(P(q[0] - 0.75 * c, q[1] - 0.75 * s, 0.42));
      cam.lookAt(P(q[0] + 0.35 * c, q[1] + 0.35 * s, 0.03));
    } else {
      cam.fov = 45;
      cam.position.copy(P(-0.3, -1.9, 3.1));
      cam.lookAt(P(-0.2, 0, 0));
    }
    cam.updateProjectionMatrix();
    return k;
  };

  // --- overlay: sensor frame and detections, car-camera view only
  const overlay = (k, kp) => {
    const w = stage.clientWidth, h = stage.clientHeight, dpr = devicePixelRatio || 1;
    if (ov.width !== Math.round(w * dpr)) { ov.width = Math.round(w * dpr); ov.height = Math.round(h * dpr); }
    og.setTransform(dpr, 0, 0, dpr, 0, 0);
    og.clearRect(0, 0, w, h);
    if (view !== 'cam') return;
    const X = (u) => w / 2 + (u - W0 / 2) * k, Y = (v) => h / 2 + (v - H0 / 2) * k;
    og.strokeStyle = 'rgba(236,232,222,.7)'; og.lineWidth = 1.5;
    const b = 22, x0 = X(0) + 10, x1 = X(W0) - 10, y0 = Y(0) + 10, y1 = Y(H0) - 10;
    for (const [x, y, dx, dy] of [[x0, y0, 1, 1], [x1, y0, -1, 1], [x0, y1, 1, -1], [x1, y1, -1, -1]]) { og.beginPath(); og.moveTo(x + dx * b, y); og.lineTo(x, y); og.lineTo(x, y + dy * b); og.stroke(); }
    og.beginPath(); og.moveTo(w / 2 - 8, h / 2); og.lineTo(w / 2 + 8, h / 2); og.moveTo(w / 2, h / 2 - 8); og.lineTo(w / 2, h / 2 + 8); og.stroke();
    og.fillStyle = color('var(--signal)');
    for (const [, u, v] of kp || []) { og.fillRect(X(u) - 4, Y(v) - 4, 8, 8); og.strokeStyle = '#000'; og.lineWidth = 1; og.strokeRect(X(u) - 4.5, Y(v) - 4.5, 9, 9); }
  };

  // --- loop
  const frame = (now) => {
    const dt = Math.min(0.05, (now - (last || now)) / 1000);
    last = now;
    const r = round;
    let kp = null, believed = null, state = null;
    if (r?.auto) {
      const tr = r.auto.res.trace;
      if (!r.done) r.t += dt;
      while (r.auto.k < tr.length - 1 && tr[r.auto.k + 1].t <= r.t) r.auto.k++;
      const a = tr[r.auto.k], b = tr[Math.min(tr.length - 1, r.auto.k + 1)];
      const f = b.t > a.t ? Math.min(1, (r.t - a.t) / (b.t - a.t)) : 0;
      const prev = r.q;
      r.q = [a.car[0] + f * (b.car[0] - a.car[0]), a.car[1] + f * (b.car[1] - a.car[1]), a.car[2] + f * wrap(b.car[2] - a.car[2])];
      const c = Math.cos(r.q[2]), s = Math.sin(r.q[2]), dx = (r.q[0] - prev[0]) / (dt || 1), dy = (r.q[1] - prev[1]) / (dt || 1);
      r.vel = [c * dx + s * dy, -s * dx + c * dy, wrap(r.q[2] - prev[2]) / (dt || 1)];
      if (!r.path.length || Math.hypot(r.q[0] - r.path.at(-1)[0], r.q[1] - r.path.at(-1)[1]) > 0.01) r.path.push(r.q.slice(0, 2));
      kp = a.kp; state = a.state;
      if (a.est) believed = compose(r.q, inverse(a.est));
      r.est = a.est;
      if (!r.done && r.auto.k >= tr.length - 1) { r.t = r.auto.res.time_s; finish(r.auto.res.success, r.auto.res.success ? null : r.auto.res.outcome); }
    } else if (r && !r.done) {
      const duty = input.duty();
      if (!r.started && duty.some((d) => d)) { r.started = true; H.banner(null); }
      if (r.started) {
        r.t += dt;
        const next = stepPlant(r.q, r.vel, duty, r.plant, dt);
        if (collides(next)) finish(false, 'Contact with the airframe');
        else if (Math.hypot(next[0], next[1]) > 3.5) finish(false, 'Left the room');
        else r.q = next;
        if (Math.hypot(r.q[0] - r.path.at(-1)[0], r.q[1] - r.path.at(-1)[1]) > 0.01) r.path.push(r.q.slice(0, 2));
      }
      if (r.seen && now - r.seen.t < 400) { kp = r.seen.kp; believed = r.seen.est; }
    } else if (r?.done && !r.auto) { r.vel = [0, 0, 0]; }
    const q = r?.q || [1.5, 0, Math.PI];
    W.setCar(q, r?.vel, dt);
    W.showCar(view !== 'cam');
    W.xray(view === 'top');
    W.setSkeleton(believed, 0.9);
    const k = camRig(q);
    W.render();
    overlay(k, kp);
    const blind = level === 'blind' && !r?.done;
    if (!blind) drawWorld(mini, bay, plan, { paths: [{ pts: r?.path || [], color: 'var(--signal)', width: 1.25 }], car: q, carEst: r?.auto ? r.est : null, scaleBar: false, grid: true });
    H.update({ r, q, e: error(q), state, kp, believed, input, blind, level });
    raf = requestAnimationFrame(frame);
  };

  const resize = () => { W.resize(stage.clientWidth, stage.clientHeight); };
  const ro = new ResizeObserver(resize);
  ro.observe(stage);
  resize();
  H.on({ level: (l) => { level = l; newRound(); }, view: (v) => { view = v; H.view(v); }, auto: autopilot, fresh: () => newRound(), retry: () => newRound(round.seed), drive: () => H.banner(null) });
  H.view(view);
  rethemeFn = () => { W.theme(); mini.layer = null; };
  await newRound();
  if (location.hash.endsWith('/auto')) { view = 'chase'; H.view(view); autopilot(); }
  raf = requestAnimationFrame(frame);
  document.body.dataset.driving = '1';
  stop = () => {
    cancelAnimationFrame(raf);
    clearInterval(percTimer);
    ro.disconnect();
    input.dispose();
    W.dispose();
    main.classList.remove('stage-page');
    delete document.body.dataset.driving;
    rethemeFn = () => {};
  };
}
