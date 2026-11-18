// Orbit camera around the car: drag to orbit, wheel or pinch to zoom. Zooming all the way in
// drops into the car's own camera (first person); zooming out from there leaves it again.
import { P } from './scene3d.js';

const MIN = 0.22, MAX = 4.5;

export function makeOrbit(el, onMode) {
  const o = { az: 0, el: 0.42, dist: 0.9, fpv: false };
  const pts = new Map();
  let pinch = 0;
  const setFpv = (on) => { if (o.fpv !== on) { o.fpv = on; onMode?.(on); } };
  const zoom = (f) => {
    if (o.fpv) { if (f > 1) { setFpv(false); o.dist = MIN * 1.6; } return; }
    o.dist = Math.min(MAX, o.dist * f);
    if (o.dist < MIN) setFpv(true);
  };
  const down = (e) => { if (e.button > 0) return; el.setPointerCapture(e.pointerId); pts.set(e.pointerId, [e.clientX, e.clientY]); };
  const move = (e) => {
    if (!pts.has(e.pointerId)) return;
    const [x0, y0] = pts.get(e.pointerId);
    pts.set(e.pointerId, [e.clientX, e.clientY]);
    if (pts.size === 2) {
      const [a, b] = [...pts.values()];
      const d = Math.hypot(a[0] - b[0], a[1] - b[1]);
      if (pinch) zoom(pinch / d);
      pinch = d;
      return;
    }
    if (o.fpv) return;
    o.az -= (e.clientX - x0) * 0.008;
    o.el = Math.max(0.05, Math.min(1.5, o.el + (e.clientY - y0) * 0.006));
  };
  const up = (e) => { pts.delete(e.pointerId); pinch = 0; };
  const wheel = (e) => { e.preventDefault(); zoom(Math.exp(e.deltaY * 0.0015)); };
  el.addEventListener('pointerdown', down);
  el.addEventListener('pointermove', move);
  for (const ev of ['pointerup', 'pointercancel']) el.addEventListener(ev, up);
  el.addEventListener('wheel', wheel, { passive: false });

  return {
    state: o,
    toggle() { if (o.fpv) { setFpv(false); o.dist = 0.9; } else setFpv(true); },
    /** Place `cam` on the orbit around the car pose q (airframe frame). */
    apply(cam, q) {
      const r = o.dist, h = Math.cos(o.el) * r, a = q[2] + o.az + Math.PI;
      const tx = q[0], ty = q[1], tz = 0.06;
      cam.fov = 50;
      cam.position.copy(P(tx + h * Math.cos(a), ty + h * Math.sin(a), tz + Math.sin(o.el) * r));
      cam.lookAt(P(tx, ty, tz));
      cam.updateProjectionMatrix();
    },
    dispose() {
      el.removeEventListener('pointerdown', down);
      el.removeEventListener('pointermove', move);
      el.removeEventListener('wheel', wheel);
    },
  };
}
