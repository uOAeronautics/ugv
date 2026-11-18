// Park it yourself, hand the car to the tuned controller and take it back at any moment. The car,
// its faults, the camera and the estimator run on the server (ugv.lab.sim) in real time; the page
// drives, draws and interpolates. Simulation only.
import * as THREE from '/static/vendor/three.module.min.js';
import { api, color } from './app.js';
import { makeInput } from './drive.js';
import { hud, START_NAME } from './hud.js';
import { makeOrbit } from './orbit.js';
import { makeWorld, P } from './scene3d.js';
import { drawWorld, loadPlan, makeView } from './world2d.js';

const W0 = 1280, H0 = 720, F0 = 968, TICK = 20;
// The sensor frame: what the camera actually sees. Both the full-screen ride and the inset show
// exactly this 16:9 rectangle, so the two pictures can be compared pixel for pixel.
const sensorRect = (w, h) => {
  const k = Math.min(w / W0, h / H0), rw = W0 * k, rh = H0 * k;
  return { x: (w - rw) / 2, y: (h - rh) / 2, w: rw, h: rh, k };
};
let stop = () => {};
let rethemeFn = () => {};
export function dispose() { stop(); }
export const retheme = () => rethemeFn();

const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));
const compose = (a, b) => [a[0] + Math.cos(a[2]) * b[0] - Math.sin(a[2]) * b[1], a[1] + Math.sin(a[2]) * b[0] + Math.cos(a[2]) * b[1], wrap(a[2] + b[2])];
const inverse = (a) => [-Math.cos(a[2]) * a[0] - Math.sin(a[2]) * a[1], Math.sin(a[2]) * a[0] - Math.cos(a[2]) * a[1], -a[2]];

export async function render(main) {
  const [world, plan, hdris] = await Promise.all([
    api('/api/park/world'),
    loadPlan(),
    api('/static/hdri.json').catch(() => ({ assets: [] })),
  ]);
  const bay = world.bay;
  main.classList.add('stage-page');
  main.innerHTML = hud.markup(hdris.assets || []);
  const $ = (s) => main.querySelector(s);
  const stage = $('#ch-stage');
  const W = await makeWorld($('#ch-3d'), $('#ch-pip'), bay, plan.skeleton);
  const ov = $('#ch-ov'), pipOv = $('#ch-pip-ov');
  const mapCanvas = $('#ch-map');
  let mini = makeView(mapCanvas, 2.0, [-0.3, 0]);
  const H = hud.bind(main);
  const orbit = makeOrbit($('#ch-3d'), (fpv) => { stage.classList.toggle('fpv', fpv); });

  let sess = null, snap = null, prev = null, recvAt = 0, interval = 100, lastPump = 0, inflight = false;
  let auto = false, truth = false, started = false, done = null, startName = 'random', raf = 0;
  let envName = 'lab', quality = 'bad';
  let paths = { est: [], true: [] };
  const input = makeInput({ onAction: (k) => ({ Space: declare, KeyR: () => newRound(), KeyV: () => orbit.toggle(), KeyP: toggleAuto, KeyT: toggleTruth })[k]?.() });
  input.pad($('#ch-pad'));

  // --- rounds
  const newRound = async ({ seed = Math.floor(Math.random() * 1e6), start = null } = {}) => {
    const body = { seed, env: envName, quality };
    if (start) body.start = start;
    else if (startName !== 'random' && startName !== 'placed') body.preset = startName;
    try {
      sess = await api('/api/park/session', { body });
    } catch {
      H.flash('THAT START TOUCHES THE AIRFRAME OR LEAVES THE ROOM');
      return;
    }
    snap = prev = sess;
    auto = false; started = false; done = null; paths = { est: [], true: [] };
    lastPump = performance.now();
    input.reset();
    H.auto(false);
    H.start(startName);
    H.banner('ready', { start: START_NAME[startName] || 'Placed on the map', seed });
  };
  const finish = (ok, why) => {
    done = { ok, why };
    const e = [snap.car[0] - bay.target[0], snap.car[1] - bay.target[1], wrap(snap.car[2] - bay.target[2])];
    H.banner(ok ? 'parked' : 'fail', { who: auto ? 'Autopilot' : 'You', t: snap.t, why, e, faults: sess.faults });
    if (!auto) H.log({ when: Date.now(), seed: sess.seed, ok, why: why || 'parked', t: snap.t });
  };
  const OUTCOME = { parked: 'Parked', 'parked-off-target': 'Parked off target', collision: 'Contact with the airframe', geofence: 'Left the room', timeout: 'Timed out' };
  const declare = () => {
    if (!snap || done || auto || !started) return;
    const e = [snap.car[0] - bay.target[0], snap.car[1] - bay.target[1], wrap(snap.car[2] - bay.target[2])], tol = bay.tolerance;
    const ok = Math.abs(e[0]) <= tol.x && Math.abs(e[1]) <= tol.y && Math.abs(e[2]) <= (tol.yaw_deg * Math.PI) / 180;
    if (ok && Math.hypot(snap.vel[0], snap.vel[1]) < 0.01) finish(true);
    else H.flash('NOT PARKED YET · open Truth (T) to see by how much');
  };
  const toggleAuto = () => {
    if (!sess || done) return;
    auto = !auto;
    started = true;
    H.auto(auto);
    H.banner(null);
    H.flash(auto ? 'CONTROLLER HAS THE CAR · press a drive key to take over' : 'YOU HAVE THE CAR');
  };
  const toggleTruth = () => { truth = !truth; H.truth(truth); };

  // --- the server advances the world in real time; one request in flight
  const pump = async (now) => {
    if (!sess || inflight || done) return;
    const duty = input.duty();
    const driving = duty.some((d) => d);
    if (auto && driving) { auto = false; H.auto(false); H.flash('YOU HAVE THE CAR'); }
    if (!auto && !started) {
      if (!driving) { lastPump = now; return; }
      started = true;
      H.banner(null);
    }
    const ticks = Math.max(1, Math.min(25, Math.round((now - lastPump) / TICK)));
    if (now - lastPump < TICK) return;
    lastPump += ticks * TICK;
    if (now - lastPump > 500) lastPump = now; // a stall (tab hidden): do not try to catch up
    inflight = true;
    const s = sess;
    try {
      const r = await api(`/api/park/session/${s.id}/step`, { body: { ticks, duty: auto ? null : duty, auto } });
      if (s !== sess) return;
      prev = snap; snap = r; recvAt = performance.now(); interval = ticks * TICK;
      if (r.est) {
        const l = paths.est.at(-1);
        if (l && Math.hypot(r.est[0] - l[0], r.est[1] - l[1]) > 0.25) paths.est = []; // the estimate jumped: new track
        paths.est.push(r.est.slice(0, 2));
      }
      paths.true.push(r.car.slice(0, 2));
      if (r.outcome) finish(r.result === 'parked', OUTCOME[r.result] || (r.result.startsWith('abort') ? `Aborted: ${r.reason || r.result.slice(6)}` : r.result));
    } catch {
      H.flash('SESSION EXPIRED · press R');
      done = { ok: false };
    } finally { inflight = false; }
  };

  // --- cameras: the car's own (the true camera, mount errors included) and the orbit
  const carCamera = (q) => {
    const ce0 = sess?.camera_error || { focal_scale: 1, yaw: 0, pitch: 0 }, fx = snap?.cam_fx;
    const cam = W.carCam, ce = { ...ce0, yaw: ce0.yaw + (fx?.yaw || 0), pitch: ce0.pitch + (fx?.pitch || 0) };
    cam.aspect = W0 / H0;
    cam.fov = (2 * Math.atan(H0 / 2 / (F0 * ce.focal_scale)) * 180) / Math.PI;
    const c = Math.cos(q[2]), s = Math.sin(q[2]), cx = bay.car.camera_x;
    cam.position.copy(P(q[0] + cx * c, q[1] + cx * s, bay.car.camera_z));
    cam.lookAt(P(q[0] + (cx + 1) * c, q[1] + (cx + 1) * s, bay.car.camera_z));
    const cy = Math.cos(ce.yaw), sy = Math.sin(ce.yaw), cp = Math.cos(ce.pitch), sp = Math.sin(ce.pitch);
    const M = new THREE.Matrix3().set(1, 0, 0, 0, cp, -sp, 0, sp, cp).multiply(new THREE.Matrix3().set(cy, 0, -sy, 0, 1, 0, sy, 0, cy));
    const S = new THREE.Matrix3().set(1, 0, 0, 0, -1, 0, 0, 0, -1);
    cam.quaternion.multiply(new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().setFromMatrix3(S.clone().multiply(M.clone().transpose()).multiply(S))));
    cam.updateProjectionMatrix();
  };
  // what the camera faults do to the picture: exposure, blur, a blocked region
  const pipImg = $('#ch-pip'), mainImg = $('#ch-3d');
  // The same fault at the same visual size in both views: blur is in sensor pixels, so it is
  // scaled by how many screen pixels one sensor pixel covers.
  const imageFaults = (img, on, rect) => {
    const fx = snap?.cam_fx, lost = !!fx?.lost;
    const expo = fx?.exposure > 0 ? 'brightness(2.6) contrast(.6)' : fx?.exposure < 0 ? 'brightness(.28)' : '';
    const blur = !lost && fx?.blur > 0.5 ? `blur(${Math.min(16, fx.blur * (rect.w / W0)).toFixed(1)}px)` : '';
    const f = on && !lost ? `grayscale(1) contrast(1.08) ${expo} ${blur}`.trim() : on ? 'brightness(0)' : '';
    if (img.style.filter !== f) img.style.filter = f;
  };
  const detections = (canvas, w, h, r, kp, frame, lost) => {
    const dpr = devicePixelRatio || 1;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); }
    const g = canvas.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    const X = (u) => r.x + (u / W0) * r.w, Y = (v) => r.y + (v / H0) * r.h;
    if (lost) {
      g.fillStyle = '#050505';
      g.fillRect(r.x, r.y, r.w, r.h);
      const big = r.w > 600;
      g.fillStyle = 'rgba(236,232,222,.55)';
      g.font = `500 ${big ? 13 : 9}px ${getComputedStyle(document.body).fontFamily}`;
      g.textAlign = 'center';
      g.fillText('NO SIGNAL', r.x + r.w / 2, r.y + r.h / 2 - (big ? 8 : 4));
      g.fillStyle = 'rgba(236,232,222,.3)';
      g.font = `400 ${big ? 10 : 7}px ${getComputedStyle(document.body).fontFamily}`;
      g.fillText('frame dropped', r.x + r.w / 2, r.y + r.h / 2 + (big ? 12 : 7));
      return;
    }
    if (frame) {
      g.strokeStyle = 'rgba(236,232,222,.7)'; g.lineWidth = 1.5;
      const b = 22, x0 = X(0) + 10, x1 = X(W0) - 10, y0 = Y(0) + 10, y1 = Y(H0) - 10;
      for (const [x, y, dx, dy] of [[x0, y0, 1, 1], [x1, y0, -1, 1], [x0, y1, 1, -1], [x1, y1, -1, -1]]) { g.beginPath(); g.moveTo(x + dx * b, y); g.lineTo(x, y); g.lineTo(x, y + dy * b); g.stroke(); }
    }
    const o = snap?.cam_fx?.occluder;
    if (kp && o) {
      g.fillStyle = 'rgba(8,8,8,.92)';
      g.fillRect(X(o[0]), Y(o[1]), X(o[2]) - X(o[0]), Y(o[3]) - Y(o[1]));
    }
    const s = Math.max(3, 4 * r.k * 1.6);
    g.fillStyle = color('var(--signal)');
    g.strokeStyle = '#000'; g.lineWidth = 1;
    for (const [, u, v] of kp || []) { g.fillRect(X(u) - s, Y(v) - s, 2 * s, 2 * s); g.strokeRect(X(u) - s - 0.5, Y(v) - s - 0.5, 2 * s + 1, 2 * s + 1); }
  };

  // --- loop
  let last = 0;
  const frame = (now) => {
    const dt = Math.min(0.05, (now - (last || now)) / 1000);
    last = now;
    pump(now);
    const f = snap && prev ? Math.min(1, (now - recvAt) / interval) : 1;
    const q = snap ? (prev ? [0, 1].map((i) => prev.car[i] + f * (snap.car[i] - prev.car[i])).concat(prev.car[2] + f * wrap(snap.car[2] - prev.car[2])) : snap.car) : [-1.4, 0, 0];
    const est = snap?.est || null;
    W.setCar(q, snap?.vel, done ? 0 : dt);
    W.setSkeleton(est ? compose(q, inverse(est)) : null, 0.9);
    const fpv = orbit.state.fpv;
    const w = stage.clientWidth, h = stage.clientHeight;
    W.xray(!fpv && orbit.state.el > 1.0);
    if (!fpv) orbit.apply(W.cam, q);
    const pw = pipImg.clientWidth, ph = pipImg.clientHeight; // the canvas, not the plate around it
    const rect = sensorRect(w, h), pipRect = sensorRect(pw, ph);
    carCamera(q);
    // The WebGL background is drawn over the whole canvas, so clip the picture to the sensor
    // frame; the stage behind it is black, which gives the monitor bars.
    mainImg.style.clipPath = fpv
      ? `inset(${rect.y}px ${w - rect.x - rect.w}px ${h - rect.y - rect.h}px ${rect.x}px)`
      : '';
    W.resize(w, h, fpv ? 0 : pw, ph);
    W.render(fpv, !fpv, fpv ? rect : null, pipRect);
    const lost = !!snap?.cam_fx?.lost;
    detections(ov, w, h, fpv ? rect : { ...sensorRect(w, h) }, fpv ? snap?.kp : null, fpv, fpv && lost);
    imageFaults(mainImg, fpv, rect);
    if (!fpv) imageFaults(pipImg, true, pipRect);
    if (!fpv) detections(pipOv, pw, ph, pipRect, snap?.kp, false, lost);
    drawWorld(mini, bay, plan, {
      paths: [{ pts: paths.est, color: 'var(--signal)', width: 1.25 }, ...(truth ? [{ pts: paths.true, color: 'var(--ink-2)', width: 1, dashed: true }] : [])],
      car: est || undefined, carEst: truth ? q : undefined, scaleBar: false,
      label: est ? null : 'NO FIX YET',
    });
    const e = est ? [est[0] - bay.target[0], est[1] - bay.target[1], wrap(est[2] - bay.target[2])] : null;
    const eTrue = [q[0] - bay.target[0], q[1] - bay.target[1], wrap(q[2] - bay.target[2])];
    H.update({ snap, est, e, eTrue, input, truth, auto, faults: sess?.faults || [] });
    H.mode(auto ? 'AUTOPILOT' : 'MANUAL');
    raf = requestAnimationFrame(frame);
  };

  // place the car on the map: press for the spot, drag to aim
  let placing = null;
  const mapXY = (e) => { const r = mapCanvas.getBoundingClientRect(); const px = e.clientX - r.left, py = e.clientY - r.top; return [mini.centre[0] + (px - mini.w / 2) / mini.scale, mini.centre[1] - (py - mini.h / 2) / mini.scale]; };
  mapCanvas.addEventListener('pointerdown', (e) => { placing = { p: mapXY(e), yaw: 0 }; mapCanvas.setPointerCapture(e.pointerId); });
  mapCanvas.addEventListener('pointermove', (e) => { if (!placing) return; const [x, y] = mapXY(e); if (Math.hypot(x - placing.p[0], y - placing.p[1]) > 0.05) placing.yaw = Math.atan2(y - placing.p[1], x - placing.p[0]); });
  mapCanvas.addEventListener('pointerup', () => { if (!placing) return; const s = [...placing.p, placing.yaw]; placing = null; startName = 'placed'; newRound({ start: s }); });

  H.on({
    start: (v) => { startName = v; newRound(); },
    env: (v) => {
      envName = v;
      if (v.startsWith('hdri:')) {
        H.flash('LOADING PANORAMA');
        W.setHdri(v.slice(5)).catch(() => H.flash('PANORAMA UNAVAILABLE · keeping the last set'));
      } else W.setEnv(v);
      newRound();
    },
    quality: (v) => { quality = v; newRound(); },
    auto: toggleAuto, truth: toggleTruth, fresh: () => newRound(),
    retry: () => newRound({ seed: sess.seed, start: sess.start }),
    drive: () => { started = true; H.banner(null); },
  });
  rethemeFn = () => { W.theme(); mini.layer = null; };
  W.setEnv(envName);
  H.setup(envName, quality);
  const onResize = () => { mini = makeView(mapCanvas, 2.0, [-0.3, 0]); };
  addEventListener('resize', onResize);
  await newRound();
  if (location.hash.endsWith('/auto')) toggleAuto();
  raf = requestAnimationFrame(frame);
  stop = () => {
    cancelAnimationFrame(raf);
    input.dispose();
    orbit.dispose();
    H.dispose();
    W.dispose();
    removeEventListener('resize', onResize);
    main.classList.remove('stage-page');
    rethemeFn = () => {};
  };
}
