// The challenge's 3D world in the airframe frame: the STRL-001 mesh, the Freenove car built from
// the Gazebo assembly (simulation/gazebo/assembly.py via tools/build_car_assets.py), the parking
// bay, and the airframe skeleton where the car believes it is. Gazebo axes (x fwd, y left, z up)
// map to three.js as (x, z, -y).
import * as THREE from '/static/vendor/three.module.min.js';
import { css } from './app.js';
import { buildEnv } from './env3d.js';
import { loadHdr } from './hdr.js';
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
  renderer.setClearColor(0x000000, 1); // letterbox bars around the sensor frame
  pip.setClearColor(0x000000, 1);
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
  // One real ground surface for every scene: the HDRI view projects the panorama's ground half
  // onto this disc (planar deprojection, perspective-correct near the car), which is why it can
  // sit at y=0 all the way out to the dome's equator instead of floating above a photo blur.
  const floorMat = new THREE.MeshStandardMaterial({ roughness: 0.92 });
  const shadowMat = new THREE.ShadowMaterial({ color: 0x000000, opacity: 0.34 });
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(80, 80), floorMat);
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = 0.0008; // the photo ground lives at y=0, the shadow catcher rides just above it
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

  const cam = new THREE.PerspectiveCamera(40, 16 / 9, 0.02, 200);
  const carCam = new THREE.PerspectiveCamera(40, 16 / 9, 0.02, 200);
  const { car, wheels } = await buildCar();
  car.traverse((o) => { o.castShadow = true; });
  scene.add(car);
  const airframe = loadStrlModel(() => {}).then(({ root }) => {
    root.rotation.y = 0; // the loader turns the nose to -y; this frame keeps it along +x
    root.traverse((o) => { o.castShadow = true; o.receiveShadow = true; });
    scene.add(root);
    return root;
  }).catch(() => null);

  let env = buildEnv('studio', airframe);
  let hdriOn = false;
  let panoLight = null;
  let pmrem = null;
  const hdriCache = new Map(); // id -> { texture, environment }, last few only
  const theme = () => {
    const bg = new THREE.Color(env.bg ?? css('--stage'));
    if (!hdriOn) {
      scene.background = bg;
      scene.fog = new THREE.Fog(bg, ...env.fog);
    }
    floor.material = hdriOn ? shadowMat : env.floor || floorMat;
    floorMat.color.set(css('--stage-floor'));
    const night = document.documentElement.dataset.mode === 'night';
    if (hdriOn) {
      // the panorama is the lamp: IBL covers ambient, the derived sun adds direct light and shadow
      hemi.intensity = 0;
      sun.position.copy(panoLight?.pos ?? new THREE.Vector3(1.2, 5, 1.8));
      sun.color.copy(panoLight?.color ?? new THREE.Color(0xffffff));
      sun.intensity = panoLight?.intensity ?? 0.45;
    } else {
      hemi.intensity = (night ? 0.7 : 1.25) * env.light;
      sun.intensity = (night ? 1.8 : 2.4) * env.light;
      sun.position.set(1.2, 5, 1.8);
      sun.color.set(0xffffff);
    }
    sun.castShadow = sun.intensity > 0.3;
    if (grid) scene.remove(grid);
    grid = new THREE.GridHelper(20, 80, new THREE.Color(css('--stage-grid')), new THREE.Color(css('--stage-grid')));
    grid.position.y = 0.0015;
    grid.material.transparent = true;
    grid.material.opacity = night ? 0.9 : 0.7;
    grid.visible = !hdriOn && env.grid;
    scene.add(grid);
    bayLines.material.color.set(css('--good'));
    skMat.color.set(css('--signal'));
    jointMat.color.set(css('--signal'));
  };
  theme();

  // HDRI grounding: the panorama is a real room and `scene.background` is infinitely far away,
  // so the photo floor never lines up with the y=0 plane and the car looks pasted on. Instead
  // the equirect renders on a dome whose equator is exactly the floor plane, like a grounded
  // skybox: the photograph's horizon sits at y=0 at every azimuth.
  const equirectGround = (tex) => {
    const uHorizon = { value: 0 };
    const uRollB = { value: 0 };
    const uRollC = { value: 0 };
    const uEyeH = { value: 1.55 }; // capture height of the panorama camera, in scene metres
    const mat = new THREE.MeshBasicMaterial({ map: tex, side: THREE.FrontSide, fog: false });
    mat.userData.uHorizon = uHorizon;
    mat.userData.uRollB = uRollB;
    mat.userData.uRollC = uRollC;
    mat.userData.uEyeH = uEyeH;
    mat.onBeforeCompile = (shader) => {
      shader.uniforms.uHorizon = uHorizon;
      shader.uniforms.uRollB = uRollB;
      shader.uniforms.uRollC = uRollC;
      shader.uniforms.uEyeH = uEyeH;
      // Anchor the deprojection to the capture point at the panorama's own camera height, not
      // to the viewer's eye: eye-ray sampling squashes the photo floor by eyeH/captureH (25x at
      // the car camera's 6 cm) and runs out of photographed ground at grazing angles, turning
      // the mid-distance floor into stretched sky/wall pixels. Anchored, the disc is a real
      // photographic ground plane at true scale — each floor point shows the panorama pixel
      // that photographed that distance — and stays correct from orbit height down to FPV.
      shader.vertexShader = shader.vertexShader
        .replace('#include <common>', '#include <common>\nvarying vec3 vDir;')
        .replace('#include <project_vertex>', '#include <project_vertex>\nvDir = (modelMatrix * vec4(transformed, 1.0)).xyz;');
      shader.fragmentShader = shader.fragmentShader
        .replace('#include <common>', '#include <common>\nvarying vec3 vDir;\nuniform float uHorizon, uRollB, uRollC, uEyeH;')
        .replace('#include <map_fragment>', `#ifdef USE_MAP
  float az = atan(vDir.z, vDir.x);
  float u = az * 0.15915494309189535 + 0.5;
  float el = -atan(uEyeH, max(length(vDir.xz), 0.01));
  float v = el * 0.3183098861837907 + 0.5
    - uHorizon - uRollB * cos(az) - uRollC * sin(az);
  diffuseColor *= texture2D(map, vec2(u, clamp(v, 0.0, 1.0)));
#endif`);
    };
    const geo = new THREE.CircleGeometry(78, 96);
    geo.rotateX(-Math.PI / 2);
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.y = 0; // the car's floor plane; the shadow catcher rides just above
    if (tex) tex.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
    return mesh;
  };

  const equirectDome = (tex) => {
    const uHorizon = { value: 0 };
    const uRollB = { value: 0 };
    const uRollC = { value: 0 };
    const mat = new THREE.MeshBasicMaterial({ map: tex, side: THREE.BackSide, fog: false });
    mat.userData.uHorizon = uHorizon;
    mat.userData.uRollB = uRollB;
    mat.userData.uRollC = uRollC;
    mat.onBeforeCompile = (shader) => {
      shader.uniforms.uHorizon = uHorizon;
      shader.uniforms.uRollB = uRollB;
      shader.uniforms.uRollC = uRollC;
      shader.vertexShader = shader.vertexShader
        .replace('#include <common>', '#include <common>\nvarying vec3 vDir;')
        .replace('#include <project_vertex>', '#include <project_vertex>\nvDir = (modelMatrix * vec4(transformed, 1.0)).xyz - cameraPosition;');
      shader.fragmentShader = shader.fragmentShader
        .replace('#include <common>', '#include <common>\nvarying vec3 vDir;\nuniform float uHorizon, uRollB, uRollC;')
        .replace('#include <map_fragment>', `#ifdef USE_MAP
  vec3 dir = normalize(vDir);
  float az = atan(dir.z, dir.x);
  float u = az * 0.15915494309189535 + 0.5;
  float v = asin(clamp(dir.y, -1.0, 1.0)) * 0.3183098861837907 + 0.5
    - uHorizon - uRollB * cos(az) - uRollC * sin(az);
  diffuseColor *= texture2D(map, vec2(u, v));
#endif`);
    };
    return new THREE.Mesh(new THREE.SphereGeometry(38, 64, 32), mat);
  };
  let dome = null;
  let ground = null;

  // Lighting a photograph with studio lights looks wrong: a dusk sky cannot make a white fuselage
  // glow. The panorama itself says where its light comes from, so estimate the direct light from
  // the HDR data — the brightest sky lobe becomes a directional sun whose intensity follows the
  // captured luminance and whose colour is the lobe's tint; image-based lighting from the same
  // panorama provides everything else. Overcast panoramas collapse to a soft fill, as they should.
  const panoramaLighting = (texture) => {
    const { data, width: w, height: h } = texture.image;
    const lum = (i) => 0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2];
    // median sky luminance via a log2 histogram, and the brightest sky sample
    const bins = new Uint32Array(96); // log2 luminance from -10 to 14
    let maxL = 0, maxI = -1, count = 0;
    for (let y = 0; y < h / 2; y += 2) {
      for (let x = 0; x < w; x += 2) {
        const i = (y * w + x) * 4, L = lum(i);
        count++;
        const b = ((Math.log2(L + 1e-9) + 10) * 4) | 0;
        if (b >= 0 && b < 96) bins[b]++;
        if (L > maxL) { maxL = L; maxI = i; }
      }
    }
    if (maxI < 0) return null;
    let med = maxL;
    for (let a = count / 2, b = 0; b < 96; b++) {
      a -= bins[b];
      if (a <= 0) { med = Math.pow(2, -10 + ((b + 0.5) * 24) / 96); break; }
    }
    // pixel → world direction, matching the dome shader's equirect mapping (flipY data)
    const dirOf = (x, y) => {
      const az = ((x + 0.5) / w - 0.5) * Math.PI * 2;
      const el = (0.5 - (y + 0.5) / h) * Math.PI;
      const ce = Math.cos(el);
      return [ce * Math.cos(az), Math.sin(el), ce * Math.sin(az)];
    };
    // refine around the brightest sample: the direct-light disk and its near halo
    const cx = ((maxI / 4) % w), cy = Math.floor(maxI / 4 / w);
    const rad = Math.max(2, Math.round(w * 0.008));
    let wx = 0, wy = 0, wz = 0, wl = 0, wc = 0, cr = 0, cg = 0, cb2 = 0;
    for (let y = Math.max(0, cy - rad); y < Math.min(h, cy + rad + 1); y++) {
      for (let dx = -rad; dx <= rad; dx++) {
        const x = (cx + dx + w) % w;
        const i = (y * w + x) * 4, L = lum(i);
        if (L < maxL * 0.2) continue;
        const [X, Y, Z] = dirOf(x, y);
        wx += X * L; wy += Y * L; wz += Z * L; wl += L; wc++;
        cr += data[i]; cg += data[i + 1]; cb2 += data[i + 2];
      }
    }
    if (!wc) return null;
    const p = Math.hypot(wx, wy, wz) || 1;
    const elC = Math.asin(wy / p);
    const halo = wl / wc, contrast = halo / Math.max(med, 1e-6);
    const strong = contrast > 8 && elC > 0.02; // a distinct sun, not just the bright sky edge
    const intensity = strong
      ? Math.min(3.2, Math.max(0.45, 0.55 * Math.log10(1 + halo / 30)))
      : 0.22; // overcast / diffuse: keep a faint modelling fill, no hard shadows
    const c = [cr / wc, cg / wc, cb2 / wc].map((v) => v / Math.max(...[cr / wc, cg / wc, cb2 / wc]));
    const col = c.map((v) => 0.55 + 0.45 * v); // desaturate 45% toward white (sensor tint, not paint)
    return {
      pos: new THREE.Vector3(wx / p, wy / p, wz / p).multiplyScalar(6),
      intensity,
      color: new THREE.Color(col[0], col[1], col[2]),
    };
  };

  const useHdri = ({ texture, environment, lighting }, horizon = {}) => {
    hdriOn = true;
    panoLight = lighting;
    scene.environment = environment;
    scene.environmentIntensity = 1;
    scene.background = null; // the dome renders instead
    scene.fog = null;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 0.95;
    scene.remove(env.group);
    env = { group: new THREE.Group(), light: 1, floor: null, grid: false, bg: null, fog: [8, 30] };
    scene.add(env.group);
    if (dome) {
      dome.material.map = texture;
      dome.visible = true;
    } else {
      dome = equirectDome(texture);
      scene.add(dome);
      ground = equirectGround(texture);
      scene.add(ground);
    }
    if (ground) {
      ground.visible = true;
      ground.material.map = texture;
      ground.material.userData.uHorizon.value = horizon.offset ?? 0;
      ground.material.userData.uRollB.value = horizon.roll_b ?? 0;
      ground.material.userData.uRollC.value = horizon.roll_c ?? 0;
    }
    dome.material.userData.uHorizon.value = horizon.offset ?? 0; // uniforms read per frame, no recompile
    dome.material.userData.uRollB.value = horizon.roll_b ?? 0;
    dome.material.userData.uRollC.value = horizon.roll_c ?? 0;
    theme();
  };

  let spin = 0, sized = '';
  // The car camera is a mono 8-bit sensor with auto-exposure off. Its picture is the scene
  // rendered linearly into an HDR target, refracted by the true lens (radial distortion),
  // blurred in the lens (defocus / motion as an isotropic kernel), gained in stops, saturated
  // at full well, then crunched to 8-bit gray — the same frame the detector analyses.
  const makeSensorPass = (r) => {
    const rt = new THREE.WebGLRenderTarget(1, 1, {
      type: THREE.HalfFloatType, depthBuffer: true, samples: 4,
    });
    const material = new THREE.ShaderMaterial({
      uniforms: {
        uTex: { value: rt.texture },
        uStops: { value: 0 },
        uBlurPx: { value: 0 },
        uRes: { value: new THREE.Vector2(1280, 720) },
        uC: { value: new THREE.Vector2(640, 360) },
        uF: { value: 968 },
        uK1: { value: 0 },
        uK2: { value: 0 },
      },
      vertexShader: 'varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }',
      fragmentShader: `
        uniform sampler2D uTex;
        uniform float uStops; // stops over correct exposure; negative = under
        uniform float uBlurPx; // Gaussian sigma in sensor pixels; 0 = sharp
        uniform vec2 uRes, uC;
        uniform float uF, uK1, uK2; // focal and radial distortion (OpenCV model)
        varying vec2 vUv;
        vec3 srgb(vec3 c) {
          return mix(12.92 * c, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
        }
        // Distortion (OpenCV): p_d = c + F * (p - c)/|..| * (1 + k1 r^2 + k2 r^4). The render
        // target is the perfect pinhole image, so output pixels sample it at the inverse.
        vec2 unwarp(vec2 px) {
          vec2 nc = (px - uC) / uF;
          vec2 m = nc;
          for (int i = 0; i < 3; i++) {
            float s = 1.0 + uK1 * dot(m, m) + uK2 * dot(m, m) * dot(m, m);
            m = nc / s;
          }
          return uC + m * uF;
        }
        void main() {
          vec2 px = vUv * uRes;
          vec3 acc = vec3(0.0);
          float wsum = 0.0;
          if (uBlurPx > 0.35) {
            for (int i = 0; i < 12; i++) {
              float fi = float(i);
              float rr = (fi + 0.5) / 12.0;
              float a = fi * 2.399963; // golden angle
              vec2 off = vec2(cos(a), sin(a)) * (rr * 2.0 * uBlurPx);
              float w = exp(-0.5 * dot(off, off) / (uBlurPx * uBlurPx));
              acc += texture2D(uTex, unwarp(px + off) / uRes).rgb * w;
              wsum += w;
            }
            acc /= wsum;
          } else {
            acc = texture2D(uTex, unwarp(px) / uRes).rgb;
          }
          float l = dot(acc, vec3(0.2126, 0.7152, 0.0722)) * exp2(uStops);
          // full well saturation: soft knee into a hard clip
          float knee = 0.6;
          l = l <= knee ? l : min(1.0, 1.0 - (1.0 - knee) * exp(-(l - knee) / (1.0 - knee)));
          l = floor(clamp(l, 0.0, 1.0) * 255.0 + 0.5) / 255.0; // 8-bit quantization
          gl_FragColor = vec4(vec3(srgb(vec3(l))), 1.0);
        }`,
      depthTest: false, depthWrite: false,
    });
    const passScene = new THREE.Scene();
    const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material);
    quad.frustumCulled = false; // gl_Position ignores the camera; skip frustum culling
    passScene.add(quad);
    const passCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    return {
      /** Render the sensor image into the CSS-pixel region (x, y, w, h) of the canvas.
       * `sensor` carries this camera's state: exposure stops, blur in sensor px, and the lens cal. */
      draw(camera, sensor, x, y, w, h, cw, ch) {
        const pr = r.getPixelRatio();
        rt.setSize(Math.max(1, Math.floor(w * pr)), Math.max(1, Math.floor(h * pr)));
        const tone = r.toneMapping;
        r.toneMapping = THREE.NoToneMapping; // linear into the target
        r.setScissorTest(false);
        r.setRenderTarget(rt);
        r.render(scene, camera);
        r.setRenderTarget(null);
        r.toneMapping = tone;
        const u = material.uniforms;
        u.uStops.value = sensor.exposure || 0;
        u.uBlurPx.value = sensor.blurPx || 0;
        u.uRes.value.set(sensor.res[0], sensor.res[1]);
        u.uC.value.set(sensor.cx, sensor.cy);
        u.uF.value = sensor.f0 * sensor.focalScale;
        u.uK1.value = sensor.k1 || 0;
        u.uK2.value = sensor.k2 || 0;
        r.setViewport(x, ch - y - h, w, h);
        r.setScissor(x, ch - y - h, w, h);
        r.setScissorTest(true);
        r.render(passScene, passCam);
      },
    };
  };
  const mainSensor = makeSensorPass(renderer);
  const pipSensor = makeSensorPass(pip);
  return {
    renderer, scene, cam, carCam, airframe, theme,
    /** Swap the surroundings (env3d.ENVIRONMENTS). */
    setEnv(name) {
      hdriOn = false;
      scene.environment = null;
      scene.backgroundIntensity = 1;
      scene.environmentIntensity = 1;
      renderer.toneMapping = THREE.NoToneMapping;
      if (dome) dome.visible = false;
      if (ground) ground.visible = false;
      scene.remove(env.group);
      env = buildEnv(name, airframe);
      scene.add(env.group);
      theme();
    },
    /** A real panorama (Poly Haven): grounded dome plus image-based lighting. */
    async setHdri(id, horizon = {}) {
      let entry = hdriCache.get(id);
      if (!entry) {
        pmrem ||= new THREE.PMREMGenerator(renderer);
        const texture = await loadHdr(`/api/park/hdri/${id}.hdr`);
        entry = { texture, environment: pmrem.fromEquirectangular(texture).texture, lighting: panoramaLighting(texture) };
        hdriCache.set(id, entry);
        while (hdriCache.size > 3) {
          const [old, e] = hdriCache.entries().next().value;
          e.texture.dispose(); e.environment.dispose();
          hdriCache.delete(old);
        }
      }
      useHdri(entry, horizon);
    },
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
    /**
     * Main view from `cam` (car shown unless `fpv`), and the car camera into the inset.
     * `rect`/`pipRect` are the sensor frames in CSS pixels from the top-left; whatever lies
     * outside them is cleared black, so the picture has the camera's 16:9 field of view on
     * any window shape instead of a wider one. `sensor` is the camera state (see makeSensorPass);
     * the car camera renders through the same pixel pipeline the detector analyses.
     */
    render(fpv, inset, rect = null, pipRect = null, sensor = { exposure: 0, blurPx: 0 }) {
      car.visible = !fpv;
      const cw = renderer.domElement.clientWidth, ch = renderer.domElement.clientHeight;
      const draw = (r, camera) => {
        const x = r?.x ?? 0, y = r?.y ?? 0, w = r?.w ?? cw, h = r?.h ?? ch;
        renderer.setScissorTest(false);
        renderer.setViewport(0, 0, cw, ch);
        if (r) renderer.clear(); // the bars: the sensor frame does not cover the window
        renderer.setViewport(x, ch - y - h, w, h);
        renderer.setScissor(x, ch - y - h, w, h);
        renderer.setScissorTest(true);
        renderer.render(scene, camera);
      };
      if (fpv && rect) {
        renderer.setScissorTest(false);
        renderer.setViewport(0, 0, cw, ch);
        renderer.clear();
        mainSensor.draw(carCam, sensor, rect.x, rect.y, rect.w, rect.h, cw, ch);
      } else {
        draw(rect, fpv ? carCam : cam);
      }
      if (inset) {
        car.visible = false;
        const pw = pip.domElement.clientWidth, ph = pip.domElement.clientHeight;
        const px = pipRect?.x ?? 0, py = pipRect?.y ?? 0, pwo = pipRect?.w ?? pw, pho = pipRect?.h ?? ph;
        pip.setScissorTest(false);
        pip.setViewport(0, 0, pw, ph);
        pip.clear();
        pipSensor.draw(carCam, sensor, px, py, pwo, pho, pw, ph);
        car.visible = true;
      }
    },
    dispose() { renderer.dispose(); pip.dispose(); },
  };
}
