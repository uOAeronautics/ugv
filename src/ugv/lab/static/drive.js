// Driving the simulated car by hand: the plant (a line-for-line port of ugv.park.plant.Plant.step)
// and the input, keyboard or touch, with a watchdog like the car's own.

export const DUTY_CAP = 1800 / 4095;

/** Port of `Plant.step`: duty -> deadzone/gain -> first-order lag -> pose. Gyro is not needed here. */
export function stepPlant(q, vel, duty, p, dt) {
  const target = [0, 1, 2].map((i) => {
    const d = Math.max(-DUTY_CAP, Math.min(DUTY_CAP, duty[i]));
    const mag = Math.max(Math.abs(d) - p.deadzone[i], 0);
    return Math.sign(d) * mag * p.gain[i] * (1 + p.speed_noise * gauss());
  });
  for (let i = 0; i < 3; i++) vel[i] += (1 - Math.exp(-dt / p.tau[i])) * (target[i] - vel[i]);
  const w = vel[2] + p.strafe_yaw * vel[1];
  const c = Math.cos(q[2]), s = Math.sin(q[2]);
  const yaw = q[2] + w * dt;
  return [q[0] + (c * vel[0] - s * vel[1]) * dt, q[1] + (s * vel[0] + c * vel[1]) * dt, Math.atan2(Math.sin(yaw), Math.cos(yaw))];
}

function gauss() {
  const u = 1 - Math.random(), v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// Duty per held key: [axis, sign]; normal and fine (Shift) magnitudes per axis (vx, vy, w).
const MAP = {
  KeyW: [0, 1], ArrowUp: [0, 1], KeyS: [0, -1], ArrowDown: [0, -1],
  KeyA: [1, 1], KeyD: [1, -1], KeyQ: [2, 1], ArrowLeft: [2, 1], KeyE: [2, -1], ArrowRight: [2, -1],
};
const BY_KEY = { w: 'KeyW', s: 'KeyS', a: 'KeyA', d: 'KeyD', q: 'KeyQ', e: 'KeyE' };
const FULL = [0.32, 0.38, 0.2], FINE = [0.2, 0.25, 0.14];
export const WATCHDOG_MS = 1000; // while a key repeats
export const HOLD_MS = 4000; // when no repeat can be expected (after a modifier or another key's release)

/**
 * Held keys -> duty. A key is released by its keyup, by window blur or a hidden tab, and by the
 * watchdog: while a key is held the OS repeats its keydown every ~30 ms after a ≤660 ms delay;
 * if neither a repeat nor any other key event arrives for WATCHDOG_MS, the keyup was lost
 * (a shortcut, a focus change, a modifier chord) and every key is released. Repeats stop after
 * another key goes down or up, so then the limit is HOLD_MS of no key event at all. Shift is read
 * from each event's modifier state, never tracked as a held key, so it cannot stick either.
 */
export function makeInput({ onAction }) {
  const held = new Set();
  const touch = new Set();
  let last = 0, expect = false, fine = false, tripped = 0;
  const now = () => performance.now();
  const clear = () => { held.clear(); expect = false; };
  const down = (e) => {
    if (/INPUT|SELECT|TEXTAREA/.test(document.activeElement?.tagName || '') || e.ctrlKey || e.metaKey || e.altKey) return;
    fine = e.shiftKey;
    last = now();
    if (MAP[e.code]) {
      e.preventDefault();
      held.add(e.code);
      expect = true;
      tripped = 0;
      return;
    }
    if (!e.repeat) expect = false; // a modifier or another key now owns the repeat
    if (['Space', 'KeyR', 'KeyV', 'KeyP'].includes(e.code)) {
      e.preventDefault();
      if (!e.repeat) onAction(e.code);
    }
  };
  const up = (e) => {
    fine = e.shiftKey && !e.code.startsWith('Shift');
    last = now();
    held.delete(e.code);
    held.delete(BY_KEY[(e.key || '').toLowerCase()]);
    expect = false; // the remaining keys no longer repeat
  };
  const lost = () => { clear(); touch.clear(); fine = false; };
  const vis = () => { if (document.hidden) lost(); };
  addEventListener('keydown', down);
  addEventListener('keyup', up);
  addEventListener('blur', lost);
  document.addEventListener('visibilitychange', vis);

  return {
    /** Duty for this frame, after the watchdog. */
    duty() {
      if (held.size && now() - last > (expect ? WATCHDOG_MS : HOLD_MS)) { clear(); tripped = now(); }
      const d = [0, 0, 0];
      const mag = fine ? FINE : FULL;
      for (const k of new Set([...held, ...touch])) if (MAP[k]) d[MAP[k][0]] += MAP[k][1] * mag[MAP[k][0]];
      return d;
    },
    /** 0..1 of the watchdog window used since the last key event (only while a key repeats). */
    watchdog: () => (held.size ? Math.min(1, (now() - last) / (expect ? WATCHDOG_MS : HOLD_MS)) : 0),
    tripped: () => tripped && now() - tripped < 2500,
    fine: () => fine,
    held: () => new Set([...held, ...touch]),
    /** Wire a touch pad: elements with data-key="KeyW" etc. */
    pad(root) {
      root.querySelectorAll('[data-key]').forEach((b) => {
        const k = b.dataset.key;
        const on = (e) => { e.preventDefault(); b.setPointerCapture?.(e.pointerId); if (MAP[k]) touch.add(k); else onAction(k); };
        const off = () => touch.delete(k);
        b.addEventListener('pointerdown', on);
        for (const ev of ['pointerup', 'pointercancel', 'lostpointercapture']) b.addEventListener(ev, off);
      });
    },
    reset: lost,
    dispose() {
      removeEventListener('keydown', down);
      removeEventListener('keyup', up);
      removeEventListener('blur', lost);
      document.removeEventListener('visibilitychange', vis);
    },
  };
}
