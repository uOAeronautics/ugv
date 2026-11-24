// The detector half of the strict camera protocol (docs/PARK_SIM.md): the same pixels the
// human is staring at are the pixels the pipeline reads. The scene is rendered by scene3d.js
// through the sensor chain; here we project the 15 keypoints into that frame, look at their
// patches, and decide what the detector saw — never the car's true pose.
const W0 = 1280, H0 = 720;
const mulberry32 = (seed) => {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};
const gauss = (rng) => {
  const u = Math.max(rng(), 1e-9), v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
};
const inverse = (a) => [-Math.cos(a[2]) * a[0] - Math.sin(a[2]) * a[1], Math.sin(a[2]) * a[0] - Math.cos(a[2]) * a[1], -a[2]];
const clamp01 = (x) => Math.min(1, Math.max(0, x));

/**
 * Project all model points into the image, exactly as the car's projection does
 * (camera_points + _true_projection + radial distortion, ports of ugv.park.perception).
 * `rel` is the airframe's pose in the car frame (x, y, yaw), `mount` the camera's yaw/pitch
 * and `fx` the true intrinsics (968 * focal_scale, principal point, k1, k2).
 * Returns [u, v, front] per point.
 */
export function projectPoints(points, rel, mount, fx) {
  const F = fx.f0 * fx.focalScale, cx = fx.cx, cy = fx.cy;
  const c = Math.cos(rel[2]), s = Math.sin(rel[2]);
  const cyw = Math.cos(mount.yaw), syw = Math.sin(mount.yaw);
  const cp = Math.cos(mount.pitch), sp = Math.sin(mount.pitch);
  const out = [];
  for (const [mx, my, mz] of points) {
    const x = rel[0] + c * mx - s * my - fx.camX;
    const y = rel[1] + s * mx + c * my;
    const z = mz - fx.camZ;
    const pc = [-y, -z, x];
    // q = pc @ (pitch @ yaw)^T, pitch about camera x, yaw about camera y
    const py = pc[0] * cyw + pc[2] * syw;
    const pyy = pc[0] * (-syw) + pc[2] * cyw;
    const qx = py;
    const qy = cp * pc[1] - sp * pyy;
    const qz = sp * pc[1] + cp * pyy;
    const front = qz > 0.05;
    const invZ = front ? 1 / qz : 1;
    let nx = qx * invZ, ny = qy * invZ;
    if (fx.k1 || fx.k2) {
      const r2 = nx * nx + ny * ny;
      const sc = 1 + fx.k1 * r2 + fx.k2 * r2 * r2;
      nx *= sc; ny *= sc;
    }
    out.push([F * nx + cx, F * ny + cy, front]);
  }
  return out;
}

export function makeDetector({ seed, sim, cal, points }) {
  const rng = mulberry32((seed * 2654435761) >>> 0);
  const state = {
    stops: sim.exposure_stops ?? 0,
    dropRate: sim.drop_rate ?? 0,
    burst: sim.burst ?? 0,
    exposureRate: sim.exposure_rate ?? 0,
    noisePx: sim.noise_px ?? 0,
    blurPx: sim.blur_px ?? 0,
    focusM: sim.focus_m ?? 0,
    blurCoc: sim.blur_coc ?? 0,
    shakeDeg: sim.shake_deg ?? 0,
    bumpDeg: sim.bump_deg ?? 0,
    bumpAtS: sim.bump_at_s ?? 0,
    mountYaw: sim.mount?.yaw ?? 0,
    mountPitch: sim.mount?.pitch ?? 0,
    occluder: sim.occluder ?? null,
    knocked: false,
    exposureLeft: 0,
    exposureSign: 1,
    dropping: false,
  };
  const cv = document.createElement('canvas');
  cv.width = W0; cv.height = H0;
  const ax = cv.getContext('2d', { willReadFrequently: true });

  /** Draw this frame's dice: drops, exposure spells, shake, the one-time knock. Called before render. */
  const beginFrame = (t, wRate, rangeM = 1.0) => {
    // frames that never arrive (Poisson drops with burst runs)
    let lost = false;
    if (state.dropRate > 0) {
      if (state.dropping) state.dropping = rng() < state.burst;
      else if (rng() < state.dropRate) state.dropping = rng() < state.burst;
      lost = state.dropping;
    }
    // exposure spells beyond the fixed setting
    if (state.exposureLeft > 0) state.exposureLeft -= 0.1;
    else if (state.exposureRate > 0 && rng() < state.exposureRate * 0.1) {
      state.exposureLeft = 0.3 + rng() * 1.7;
      state.exposureSign = rng() < 0.5 ? -1 : 1;
    }
    const spell = state.exposureLeft > 0 ? state.exposureSign : 0;
    // mount jitter (angular noise about yaw and pitch) and the one-time knock
    const sr = (state.shakeDeg * Math.PI) / 180;
    const shake = sr ? [gauss(rng) * sr, gauss(rng) * sr] : [0, 0];
    const knockDeg = (state.bumpDeg * Math.PI) / 180;
    let knockYaw = 0, knockPitch = 0, knockedNow = false;
    if (knockDeg && !state.knocked && t >= state.bumpAtS) {
      state.knocked = true;
      knockedNow = true; // the frame it happens is also smeared; the offset persists afterwards
    }
    if (state.knocked && knockDeg) { knockYaw = 0.6 * knockDeg; knockPitch = knockDeg; }
    // the knock persists as a mount offset; the frame it happens also smears
    const jump = Math.hypot(shake[0], shake[1]) + (knockedNow ? knockDeg : 0);
    const smear = state.blurPx * Math.abs(wRate) + 968 * jump * 0.5;
    const defocus = state.focusM
      ? state.blurCoc * Math.abs(1 / Math.max(rangeM, 0.1) - 1 / state.focusM)
      : 0;
    return {
      lost,
      exposure: state.stops + spell,
      mount: { yaw: state.mountYaw + shake[0] + knockYaw, pitch: state.mountPitch + shake[1] - knockPitch },
      blurPx: smear + defocus, // sensor px: drives the blur in the render and sigmaPx
      sigmaPx: state.noisePx + 0.5 * (smear + defocus),
    };
  };

  /** Project the 15 true keypoints and read their patches in the *rendered* frame. */
  const detect = (src, srcRect, fxDraw, carPose) => {
    const { x, y, w, h } = srcRect;
    ax.drawImage(src, x * devicePixelRatio, y * devicePixelRatio, w * devicePixelRatio, h * devicePixelRatio, 0, 0, W0, H0);
    if (state.occluder) {
      ax.fillStyle = '#000';
      ax.fillRect(state.occluder[0], state.occluder[1], state.occluder[2] - state.occluder[0], state.occluder[3] - state.occluder[1]);
    }
    const img = ax.getImageData(0, 0, W0, H0).data;
    const rel = inverse(carPose);
    const mount = fxDraw.mount;
    const proj = projectPoints(points, rel, mount, cal);
    const inFrame = proj.filter(([u, v, front]) => front && u >= 0 && u < W0 && v >= 0 && v < H0);
    if (inFrame.length < 2) return [];
    let uMin = Infinity, uMax = -Infinity, vMin = Infinity, vMax = -Infinity;
    for (const [u, v] of inFrame) { uMin = Math.min(uMin, u); uMax = Math.max(uMax, u); vMin = Math.min(vMin, v); vMax = Math.max(vMax, v); }
    if (Math.hypot(uMax - uMin, vMax - vMin) < 40) return []; // too small to lock on
    const dets = [];
    for (let k = 0; k < proj.length; k++) {
      const [u, v, front] = proj[k];
      if (!front || u < 0 || u >= W0 || v < 0 || v >= H0) continue;
      // patch stats in sensor px
      const r = 7;
      const u0 = Math.max(0, Math.floor(u - r)), u1 = Math.min(W0 - 1, Math.floor(u + r));
      const v0 = Math.max(0, Math.floor(v - r)), v1 = Math.min(H0 - 1, Math.floor(v + r));
      let n = 0, sum = 0, sum2 = 0, sat = 0, crush = 0;
      for (let yy = v0; yy <= v1; yy++) {
        for (let xx = u0; xx <= u1; xx++) {
          const i = ((yy * W0 + xx) * 4);
          const l = img[i] / 255; // the sensor is mono
          n += 1; sum += l; sum2 += l * l;
          if (l > 0.98) sat += 1; else if (l < 0.02) crush += 1;
        }
      }
      const mean = sum / n;
      const variance = Math.max(0, sum2 / n - mean * mean);
      const std = Math.sqrt(variance);
      const satFrac = sat / n, crushFrac = crush / n;
      // A learned matcher fires ~85-90% of the time on a clean frame; it is blinded by
      // saturation, clipping and featureless patches — so the prior sits high and the pixels
      // take it back down. Blown full-well and dead black patches score zero, honestly.
      let p = 0.88 - 1.0 * satFrac - 1.0 * crushFrac - 0.7 * clamp01((0.045 - std) / 0.045);
      p = Math.max(0, Math.min(0.95, p));
      if (rng() >= p) continue;
      const sigma = fxDraw.sigmaPx + satFrac * 3;
      let du = gauss(rng) * sigma, dv = gauss(rng) * sigma;
      if (rng() < 0.02) {
        const mag = 20 + rng() * 60;
        du = (rng() < 0.5 ? -1 : 1) * mag; dv = (rng() < 0.5 ? -1 : 1) * mag; // misfire
      }
      dets.push([k, Math.round((u + du) * 10) / 10, Math.round((v + dv) * 10) / 10]);
    }
    return dets;
  };

  return {
    setStops(v) { state.stops = v; },
    getStops: () => state.stops,
    beginFrame,
    detect,
  };
}
