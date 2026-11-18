// The challenge's 3D world in the airframe frame: the STRL-001 mesh, the Freenove car built from
// the Gazebo assembly (simulation/gazebo/assembly.py via tools/build_car_assets.py), the parking
// bay, and the airframe skeleton where the car believes it is. Gazebo axes (x fwd, y left, z up)
// map to three.js as (x, z, -y).
import * as THREE from '/static/vendor/three.module.min.js';
import { css } from './app.js';
import { loadStrlModel } from './stl.js';

export const P = (x, y, z = 0) => new THREE.Vector3(x, z, -y);
const BONES = [[0, 1], [0, 9], [2, 4], [4, 5], [5, 3], [6, 1], [1, 7], [1, 8], [10, 11], [9, 10], [9, 11]];

async function buildCar() {
  const spec = await (await fetch('/static/models/freenove/assembly.json')).json();
  const mats = {};
  const mat = (name) => (mats[name] ||= new THREE.MeshStandardMaterial({
    color: new THREE.Color().setRGB(...spec.colors[name], THREE.LinearSRGBColorSpace),
    roughness: name === 'silver' || name === 'gold' ? 0.35 : 0.7, metalness: name === 'silver' || name === 'gold' ? 0.6 : 0.05,
  }));
  const gz = new THREE.Group(); // Gazebo axes inside
  const wheels = [];
  for (const [name, g] of Object.entries(spec.groups)) {
    const grp = new THREE.Group();
    grp.position.set(...g.origin);
    for (const p of g.parts) {
      let geo;
      if (p.shape === 'box') geo = new THREE.BoxGeometry(...p.size);
      else { geo = new THREE.CylinderGeometry(p.size[0], p.size[0], p.size[1], 20); geo.rotateX(Math.PI / 2); }
      const m = new THREE.Mesh(geo, mat(p.color));
      m.position.set(...p.pos);
      m.rotation.set(p.rot[0], p.rot[1], p.rot[2], 'ZYX');
      m.castShadow = true;
      grp.add(m);
    }
    gz.add(grp);
    if (name.endsWith('_wheel')) wheels.push({ grp, side: g.origin[1] > 0 ? 1 : -1, front: g.origin[0] > 0 ? 1 : -1 });
  }
  const car = new THREE.Group();
  gz.rotation.x = -Math.PI / 2;
  car.add(gz);
  return { car, wheels };
}

export async function makeWorld(canvas, pipCanvas, bay, skeleton) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  const pip = new THREE.WebGLRenderer({ canvas: pipCanvas, antialias: true });
  pip.setPixelRatio(Math.min(2, devicePixelRatio || 1));
  renderer.setPixelRatio(Math.min(2, devicePixelRatio || 1));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  const scene = new THREE.Scene();
  const hemi = new THREE.HemisphereLight(0xffffff, 0x404040, 1.0);
  scene.add(hemi);
  const sun = new THREE.DirectionalLight(0xffffff, 2.2);
  sun.position.set(1.2, 5, 1.8);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -3, right: 3, top: 3, bottom: -3, near: 0.5, far: 12 });
  sun.shadow.bias = -0.0004;
  scene.add(sun);
  const floorMat = new THREE.MeshStandardMaterial({ roughness: 0.92 });
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(80, 80), floorMat);
  floor.rotation.x = -Math.PI / 2;
  floor.receiveShadow = true;
  scene.add(floor);
  let grid = null;

  // bay: corner brackets of the tolerance box on the floor
  const t = bay.target, L = bay.car.length / 2 + bay.tolerance.x, W = bay.car.width / 2 + bay.tolerance.y, k = 0.04;
  const c = Math.cos(t[2]), s = Math.sin(t[2]);
  const Q = (u, v) => P(t[0] + c * u - s * v, t[1] + s * u + c * v, 0.002);
  const bayPts = [];
  for (const [sx, sy] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) bayPts.push(Q(sx * L, sy * (W - k)), Q(sx * L, sy * W), Q(sx * L, sy * W), Q(sx * (L - k), sy * W));
  const bayLines = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(bayPts), new THREE.LineBasicMaterial());
  scene.add(bayLines);

  // the airframe skeleton where the car believes it is (depth test off: an overlay)
  const skMat = new THREE.LineBasicMaterial({ depthTest: false, transparent: true });
  const skGeo = new THREE.BufferGeometry().setFromPoints(BONES.flatMap(() => [new THREE.Vector3(), new THREE.Vector3()]));
  const sk = new THREE.LineSegments(skGeo, skMat);
  sk.renderOrder = 10;
  sk.visible = false;
  scene.add(sk);
  const jointMat = new THREE.PointsMaterial({ size: 7, sizeAttenuation: false, depthTest: false, transparent: true });
  const joints = new THREE.Points(new THREE.BufferGeometry().setFromPoints(skeleton.points.map(() => new THREE.Vector3())), jointMat);
  joints.renderOrder = 11;
  joints.visible = false;
  scene.add(joints);

  const cam = new THREE.PerspectiveCamera(40, 16 / 9, 0.02, 40);
  const carCam = new THREE.PerspectiveCamera(40, 16 / 9, 0.02, 40);
  const { car, wheels } = await buildCar();
  car.traverse((o) => { o.castShadow = true; });
  scene.add(car);
  const airframe = loadStrlModel(() => {}).then(({ root }) => {
    root.rotation.y = 0; // the loader turns the nose to -y; this frame keeps it along +x
    root.traverse((o) => { o.castShadow = true; o.receiveShadow = true; });
    scene.add(root);
    return root;
  }).catch(() => null);

  const theme = () => {
    const bg = new THREE.Color(css('--stage'));
    scene.background = bg;
    scene.fog = new THREE.Fog(bg, 4, 14);
    floorMat.color.set(css('--stage-floor'));
    const night = document.documentElement.dataset.mode === 'night';
    hemi.intensity = night ? 0.7 : 1.25;
    sun.intensity = night ? 1.8 : 2.4;
    if (grid) scene.remove(grid);
    grid = new THREE.GridHelper(20, 80, new THREE.Color(css('--stage-grid')), new THREE.Color(css('--stage-grid')));
    grid.position.y = 0.0015;
    grid.material.transparent = true;
    grid.material.opacity = night ? 0.9 : 0.7;
    scene.add(grid);
    bayLines.material.color.set(css('--good'));
    skMat.color.set(css('--signal'));
    jointMat.color.set(css('--signal'));
  };
  theme();

  let spin = 0, sized = '';
  return {
    renderer, scene, cam, carCam, airframe, theme,
    /** Place the car (airframe frame pose) and spin its wheels by the body velocity. */
    setCar(q, vel, dt) {
      car.position.copy(P(q[0], q[1], 0));
      car.rotation.y = q[2];
      if (vel) {
        const R = 0.0325, lw = 0.16;
        for (const w of wheels) {
          const roll = (vel[0] - w.side * w.front * vel[1] - w.side * vel[2] * lw) / R;
          w.grp.rotation.y += roll * dt;
        }
        spin += dt;
      }
    },
    showCar(on) { car.visible = on; },
    /** See-through airframe, so the car stays visible under the wing from overhead. */
    xray(on) {
      airframe.then((root) => root?.traverse((o) => {
        if (!o.material || o.material.opacity === (on ? 0.35 : 1)) return;
        o.material.transparent = on; o.material.opacity = on ? 0.35 : 1; o.material.depthWrite = !on;
      }));
    },
    /** Draw the skeleton for the airframe at planar pose `a` (airframe frame), or hide it. */
    setSkeleton(a, opacity = 1) {
      sk.visible = joints.visible = !!a;
      if (!a) return;
      const ca = Math.cos(a[2]), sa = Math.sin(a[2]);
      const pts = skeleton.points.map(([x, y, z]) => P(a[0] + ca * x - sa * y, a[1] + sa * x + ca * y, z));
      const pos = skGeo.attributes.position;
      BONES.forEach(([i, j], b) => { pos.setXYZ(2 * b, pts[i].x, pts[i].y, pts[i].z); pos.setXYZ(2 * b + 1, pts[j].x, pts[j].y, pts[j].z); });
      pos.needsUpdate = true;
      const jp = joints.geometry.attributes.position;
      pts.forEach((p, i) => jp.setXYZ(i, p.x, p.y, p.z));
      jp.needsUpdate = true;
      skMat.opacity = jointMat.opacity = opacity;
    },
    resize(w, h, pw, ph) {
      const key = `${w}x${h}:${pw}x${ph}`;
      if (key === sized) return;
      sized = key;
      renderer.setSize(w, h, false); cam.aspect = w / h; cam.updateProjectionMatrix();
      if (pw) pip.setSize(pw, ph, false);
    },
    /** Main view from `cam` (car shown unless `fpv`), and the car camera into the inset. */
    render(fpv, inset) {
      car.visible = !fpv;
      renderer.render(scene, fpv ? carCam : cam);
      if (inset) { car.visible = false; pip.render(scene, carCam); car.visible = true; }
    },
    dispose() { renderer.dispose(); pip.dispose(); },
  };
}
