// Synthetic camera view: a real perspective render of the scene from the car's camera.
// Pinhole intrinsics from simulation/gazebo/camera_profile.json (fy 968 px at 720 rows, hfov ~67 deg).
// Policy frame (x, y, z up) maps to three.js as X = x, Y = z, Z = -y. Yaw is counter-clockwise from -y.
import * as THREE from '/static/vendor/three.module.min.js';
import { loadStrlModel } from '/static/stl.js';

const CAM_FWD = 0.071;
const CAM_Z = 0.0645;
const VFOV = (2 * Math.atan(360 / 968) * 180) / Math.PI;
const markerCache = new Map();
let R = null;

const P = (x, y, z) => new THREE.Vector3(x, z, -y);

function gridTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 512;
  const g = c.getContext('2d');
  g.fillStyle = '#9aa1a9';
  g.fillRect(0, 0, 512, 512);
  for (let i = 0; i < 4000; i++) {
    g.fillStyle = `rgba(${i % 2 ? 255 : 0},${i % 2 ? 255 : 0},${i % 2 ? 255 : 0},0.03)`;
    g.fillRect(Math.random() * 512, Math.random() * 512, 2, 2);
  }
  g.strokeStyle = 'rgba(30,38,48,0.35)';
  g.lineWidth = 3;
  g.strokeRect(0, 0, 512, 512);
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(16, 16);
  t.anisotropy = 8;
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

async function markerTexture(id) {
  if (markerCache.has(id)) return markerCache.get(id);
  const c = document.createElement('canvas');
  c.width = c.height = 136;
  const g = c.getContext('2d');
  g.fillStyle = '#f2f0ea';
  g.fillRect(0, 0, 136, 136);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.magFilter = THREE.NearestFilter;
  tex.anisotropy = 8;
  markerCache.set(id, tex);
  try {
    const m = await fetchJSON(`/api/aruco/${id}`);
    const cell = 100 / m.n;
    m.rows.forEach((row, r) =>
      row.forEach((v, q) => {
        g.fillStyle = v ? '#f6f6f2' : '#101010';
        g.fillRect(18 + q * cell, 18 + r * cell, cell + 0.5, cell + 0.5);
      }),
    );
    tex.needsUpdate = true;
    if (R) R.dirty = true;
  } catch {
    g.fillStyle = '#101010';
    g.fillRect(18, 18, 100, 100);
    tex.needsUpdate = true;
  }
  return tex;
}

// Flat card facing `normal` ('+y' | '-y' | '+x' | '-x'), 68 mm paper around a 50 mm marker.
function cardMesh(id) {
  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(0.068, 0.068),
    new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide }),
  );
  markerTexture(id).then((t) => {
    mesh.material.map = t;
    mesh.material.needsUpdate = true;
    if (R) R.dirty = true;
  });
  return mesh;
}

function card(id, x, y, z, normal) {
  const mesh = cardMesh(id);
  mesh.position.copy(P(x, y, z));
  mesh.rotation.y = { '+y': Math.PI, '-y': 0, '+x': Math.PI / 2, '-x': -Math.PI / 2 }[normal];
  return mesh;
}

function tube(a, b, r, mat) {
  const d = new THREE.Vector3().subVectors(b, a);
  const m = new THREE.Mesh(new THREE.CylinderGeometry(r, r, d.length(), 16), mat);
  m.position.copy(a).addScaledVector(d, 0.5);
  m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), d.normalize());
  m.castShadow = true;
  return m;
}

function buildQuad(geo) {
  const g = new THREE.Group();
  const sx = geo.skid_x ?? 0.157;
  const len = geo.gear_len ?? 0.3048;
  const r = geo.gear_r ?? 0.03;
  const frame = new THREE.MeshStandardMaterial({ color: 0x23272d, roughness: 0.55, metalness: 0.2 });
  const foam = new THREE.MeshStandardMaterial({ color: 0x3b4048, roughness: 0.9 });
  [-sx, sx].forEach((x) => {
    const skid = tube(P(x, -len / 2, r), P(x, len / 2, r), r, foam);
    g.add(skid);
    g.add(tube(P(x, 0, r), P(x * 0.55, 0, 0.39), 0.009, frame));
  });
  const deck = new THREE.Mesh(new THREE.CylinderGeometry(0.1175, 0.1175, 0.028, 40), frame);
  deck.position.copy(P(0, 0, 0.42));
  deck.castShadow = true;
  g.add(deck);
  const mast = tube(P(0, 0, 0.43), P(0, 0, 0.69), 0.012, frame);
  g.add(mast);
  [45, 135, 225, 315].forEach((deg) => {
    const a = (deg * Math.PI) / 180;
    const tip = P(0.36 * Math.cos(a), 0.36 * Math.sin(a), 0.44);
    g.add(tube(P(0, 0, 0.43), tip, 0.011, frame));
    const motor = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, 0.04, 20), frame);
    motor.position.copy(tip);
    g.add(motor);
    const prop = new THREE.Mesh(
      new THREE.CylinderGeometry(0.19, 0.19, 0.002, 40),
      new THREE.MeshStandardMaterial({ color: 0x15181c, transparent: true, opacity: 0.18 }),
    );
    prop.position.copy(tip).add(new THREE.Vector3(0, 0.03, 0));
    g.add(prop);
  });
  const sides = { 14: '+y', 6: '+x', 3: '-x' };
  (geo.tags || [
    { id: 14, x: -0.028, y: 0, z: 0.14 },
    { id: 6, x: 0.06, y: -0.05, z: 0.15 },
    { id: 3, x: -0.105, y: -0.049, z: 0.16 },
  ]).forEach((t) => g.add(card(t.id, t.x, t.y, t.z ?? 0.15, sides[t.id] || '+y')));
  return g;
}

// Real STRL-001 geometry from /static/models/strl001: one group per link, loaded once
// and reused across target swaps. The group is returned empty and filled as meshes land.
let strlTarget = null;
let strlJoints = null;
const PROP_JOINTS = ['right_prop_joint', 'left_prop_joint', 'rear_prop_joint'];
const PROP_RATE = 18; // rad/s, so the props read as running

function buildStrl() {
  if (strlTarget) return strlTarget;
  strlTarget = new THREE.Group();
  loadStrlModel(() => {
    if (R) R.dirty = true;
  }, cardMesh)
    .then(({ root, joints }) => {
      strlTarget.add(root);
      strlJoints = joints;
      window.POV3D.joints = joints;
      if (R) R.dirty = true;
    })
    .catch((e) => console.warn('STRL-001 unavailable', e));
  return strlTarget;
}

/** Rotate one joint about its SDF axis, clamped to its SDF limits. */
function setJoint(joint, angle) {
  if (!joint) return;
  const a = Math.min(joint.max, Math.max(joint.min, angle));
  joint.angle = a;
  joint.pivot.quaternion.setFromAxisAngle(joint.axis, a);
}

// Props spin; the control surfaces and nacelle tilt stay at their SDF zero pose.
function animateStrl(now) {
  if (!strlJoints) return;
  const a = now * PROP_RATE;
  for (let i = 0; i < PROP_JOINTS.length; i++) setJoint(strlJoints[PROP_JOINTS[i]], a);
}

function init(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x6b727b);
  scene.fog = new THREE.Fog(0x6b727b, 3.5, 9);
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(14, 14), new THREE.MeshStandardMaterial({ map: gridTexture(), roughness: 0.95 }));
  floor.rotation.x = -Math.PI / 2;
  floor.receiveShadow = true;
  scene.add(floor);
  scene.add(new THREE.HemisphereLight(0xdfe7f1, 0x555a60, 1.0));
  const sun = new THREE.DirectionalLight(0xffffff, 1.6);
  sun.position.set(1.2, 3, 1.5);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -2.5, right: 2.5, top: 2.5, bottom: -2.5, near: 0.5, far: 8 });
  scene.add(sun);
  const camera = new THREE.PerspectiveCamera(VFOV, 16 / 9, 0.005, 20);
  R = { renderer, scene, camera, target: null, targetKey: '', dirty: true };
  return R;
}

function setTarget(kind, geo) {
  const key = kind + JSON.stringify(geo?.tags || '') + (geo?.skid_x ?? '');
  if (R.targetKey === key) return;
  if (R.target) R.scene.remove(R.target);
  R.target = kind === 'strl001_belly' ? buildStrl() : buildQuad(geo || {});
  R.scene.add(R.target);
  R.targetKey = key;
}

window.POV3D = {
  /** Render the car's view at pose {x, y, yaw}. Returns false when WebGL is unavailable. */
  render(canvas, pose, kind, geo) {
    try {
      const r = R || init(canvas);
      const w = Math.max(2, canvas.clientWidth);
      const h = Math.max(2, canvas.clientHeight);
      if (canvas.width !== Math.round(w * devicePixelRatio) || canvas.height !== Math.round(h * devicePixelRatio)) {
        r.renderer.setPixelRatio(devicePixelRatio);
        r.renderer.setSize(w, h, false);
      }
      r.camera.aspect = w / h;
      r.camera.updateProjectionMatrix();
      setTarget(kind, geo);
      if (kind === 'strl001_belly') animateStrl(performance.now() / 1000);
      const fx = Math.sin(pose.yaw);
      const fy = -Math.cos(pose.yaw);
      r.camera.position.copy(P(pose.x + CAM_FWD * fx, pose.y + CAM_FWD * fy, CAM_Z));
      r.camera.lookAt(P(pose.x + CAM_FWD * fx + fx, pose.y + CAM_FWD * fy + fy, CAM_Z));
      r.renderer.render(r.scene, r.camera);
      return true;
    } catch (e) {
      console.warn('POV3D unavailable', e);
      return false;
    }
  },
  redraw() {
    if (R?.dirty) {
      R.dirty = false;
      if (typeof drawPov === 'function') drawPov();
    }
  },
  /** Joint handles, filled once the meshes land: { pivot, axis, min, max, angle }. */
  joints: {},
  /** Pose one joint by name, clamped to its SDF limits. */
  setJoint(name, angle) {
    setJoint(strlJoints?.[name], angle);
  },
};
setInterval(() => window.POV3D.redraw(), 250);
