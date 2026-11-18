// Playground environments for the challenge, procedural (no assets): what is around the airframe
// changes what the camera faces. Each has its sensor effects on the server (ugv.lab.sim.ENVIRONMENTS).
import * as THREE from '/static/vendor/three.module.min.js';
import { P } from './scene3d.js';

export const ENVIRONMENTS = {
  studio: 'Studio',
  lab: 'Lab room',
  hangar: 'Hangar',
  workshop: 'Cluttered workshop',
  night: 'Night',
};

const mat = (color, o = {}) => new THREE.MeshStandardMaterial({ color, roughness: 0.85, ...o });
function box(g, [x, y, z], [l, w, h], m, yaw = 0) {
  const b = new THREE.Mesh(new THREE.BoxGeometry(l, h, w), m);
  b.position.copy(P(x, y, z + h / 2));
  b.rotation.y = yaw;
  b.castShadow = b.receiveShadow = true;
  g.add(b);
  return b;
}
function cyl(g, [x, y, z], r, h, m) {
  const c = new THREE.Mesh(new THREE.CylinderGeometry(r, r, h, 16), m);
  c.position.copy(P(x, y, z + h / 2));
  c.castShadow = true;
  g.add(c);
  return c;
}
function noiseTexture(base, spread, size = 256) {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const g = c.getContext('2d');
  const img = g.createImageData(size, size);
  for (let i = 0; i < size * size; i++) {
    const v = base + (Math.random() - 0.5) * spread;
    img.data.set([v, v, v * 0.98, 255], i * 4);
  }
  g.putImageData(img, 0, 0);
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(20, 20);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}
function walls(g, x0, x1, y0, y1, h, m) {
  const t = 0.18;
  box(g, [(x0 + x1) / 2, y0, 0], [x1 - x0, t, h], m);
  box(g, [(x0 + x1) / 2, y1, 0], [x1 - x0, t, h], m);
  box(g, [x0, (y0 + y1) / 2, 0], [t, y1 - y0, h], m);
  box(g, [x1, (y0 + y1) / 2, 0], [t, y1 - y0, h], m);
}
function table(g, x, y, yaw, top) {
  const leg = mat(0x6d6f72, { metalness: 0.5, roughness: 0.4 });
  box(g, [x, y, 0.72], [1.6, 0.8, 0.03], top, yaw);
  for (const [dx, dy] of [[0.75, 0.35], [-0.75, 0.35], [0.75, -0.35], [-0.75, -0.35]]) {
    const c = Math.cos(yaw), s = Math.sin(yaw);
    cyl(g, [x + c * dx - s * dy, y + s * dx + c * dy, 0], 0.02, 0.72, leg);
  }
}
function quadcopter(g, x, y) {
  const dark = mat(0x1d1f22, { roughness: 0.5 }), prop = mat(0x2a2c30, { transparent: true, opacity: 0.6 });
  box(g, [x, y, 0.06], [0.16, 0.16, 0.05], dark);
  for (const a of [0.785, 2.356, 3.927, 5.498]) {
    const ax = x + Math.cos(a) * 0.17, ay = y + Math.sin(a) * 0.17;
    box(g, [(x + ax) / 2, (y + ay) / 2, 0.075], [0.24, 0.025, 0.02], dark, -a);
    cyl(g, [ax, ay, 0], 0.012, 0.09, dark);
    const disc = new THREE.Mesh(new THREE.CylinderGeometry(0.11, 0.11, 0.004, 24), prop);
    disc.position.copy(P(ax, ay, 0.1));
    g.add(disc);
  }
}
function chair(g, x, y, yaw) {
  const m = mat(0x30353c);
  box(g, [x, y, 0.45], [0.45, 0.45, 0.04], m, yaw);
  box(g, [x - 0.2 * Math.cos(yaw), y - 0.2 * Math.sin(yaw), 0.49], [0.04, 0.45, 0.45], m, yaw);
  cyl(g, [x, y, 0], 0.025, 0.45, mat(0x777777, { metalness: 0.6 }));
}

/** Build an environment. Returns the group plus lighting/floor settings for scene3d. */
export function buildEnv(name, airframe) {
  const g = new THREE.Group();
  const out = { group: g, light: 1, floor: null, grid: name === 'studio' || name === 'night', bg: null, fog: [4, 14] };
  if (name === 'lab') {
    // The real lab, roughly: 12 x 24 m, teaching tables, lockers, whiteboards, a glazed long wall.
    walls(g, -3.5, 8.5, -9.5, 14.5, 3.8, mat(0xd9d6cf));
    const glass = new THREE.MeshStandardMaterial({ color: 0xcfe3f2, emissive: 0x9fbfd8, emissiveIntensity: 0.6 });
    for (let y = -8; y < 14; y += 2.6) box(g, [8.4, y, 0.4], [0.05, 2.2, 2.7], glass);
    const top = mat(0xe9e5dc);
    for (const [x, y, yaw] of [[-2.2, 3, 0], [-2.2, 5, 0], [4.5, 6, 1.57], [4.5, 8.5, 1.57], [2, -7, 0], [3.8, -7, 0]]) table(g, x, y, yaw, top);
    for (let i = 0; i < 6; i++) box(g, [-3.2, -6 + i * 0.6, 0], [0.5, 0.55, 1.9], mat(0x23262b, { metalness: 0.4 }));
    for (const y of [-2, 2.5]) box(g, [-3.38, y, 0.9], [0.03, 2.4, 1.2], mat(0xf4f4f1, { roughness: 0.3 }));
    chair(g, -1.5, 3.2, 3.14); chair(g, 3.6, 6.2, 1.57); chair(g, 2.4, -6.3, -1.57);
    out.floor = mat(0xb9bcc0, { map: noiseTexture(185, 8) });
    out.bg = 0xdedbd3; out.fog = [8, 30]; out.light = 1.15;
  } else if (name === 'hangar') {
    walls(g, -14, 16, -12, 12, 9, mat(0x8b9097, { metalness: 0.3 }));
    for (let x = -12; x < 16; x += 4) {
      const lamp = new THREE.Mesh(new THREE.BoxGeometry(1.2, 0.08, 0.4), new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xffffff, emissiveIntensity: 2 }));
      lamp.position.copy(P(x, 0, 8.6));
      g.add(lamp);
    }
    for (const [x, y] of [[4, 5], [9, -4]]) box(g, [x, y, 0], [0.6, 0.45, 0.9], mat(0xb3261e, { metalness: 0.3, roughness: 0.5 }));
    airframe.then((root) => {
      if (!root) return;
      for (const [x, y, yaw] of [[3.5, -3, 0.8], [-4, 4.5, -2.2]]) {
        const c = root.clone(true);
        c.position.copy(P(x, y, 0)); c.position.y = root.position.y;
        c.rotation.y = yaw;
        g.add(c);
      }
    });
    out.floor = mat(0x8e8f8c, { map: noiseTexture(150, 14) });
    out.bg = 0x9aa0a6; out.fog = [10, 40]; out.light = 1.5;
  } else if (name === 'workshop') {
    const brown = mat(0x9c7a52), grey = mat(0x5d6168);
    for (const [x, y, s] of [[1.2, 1.4, 0.4], [1.5, 1.7, 0.3], [-1.6, -1.3, 0.5], [2.2, -0.6, 0.35], [-2.4, 1.1, 0.45], [0.6, -1.9, 0.3]]) box(g, [x, y, 0], [s, s * 0.8, s * 0.7], Math.random() < 0.5 ? brown : grey, Math.random());
    quadcopter(g, 1.4, -1.1);
    quadcopter(g, -1.9, 0.6);
    chair(g, -1.2, 1.9, 0.6); chair(g, 2.6, 1.0, 2.4);
    const cable = new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.6 });
    const curve = new THREE.CatmullRomCurve3([P(-2.5, -0.3, 0.01), P(-1.2, -0.6, 0.01), P(-0.9, 0.4, 0.01), P(0.4, 0.9, 0.01), P(1.8, 0.3, 0.01)]);
    const tube = new THREE.Mesh(new THREE.TubeGeometry(curve, 80, 0.008, 6), cable);
    g.add(tube);
    const jeans = mat(0x2c3e5a), shoe = mat(0x1a1a1a);
    for (const dy of [-0.12, 0.12]) { cyl(g, [-1.0, -1.6 + dy, 0.08], 0.06, 0.8, jeans); box(g, [-0.95, -1.6 + dy, 0], [0.28, 0.1, 0.08], shoe); }
    out.floor = mat(0x6f6a62, { map: noiseTexture(120, 14) });
    out.bg = 0x4a4741; out.fog = [5, 16]; out.light = 0.95;
  } else if (name === 'night') {
    out.light = 0.22; out.bg = 0x050506; out.fog = [2.5, 9];
  }
  return out;
}
