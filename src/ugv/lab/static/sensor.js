// The car camera's sensor side of the strict camera protocol (docs/PARK_SIM.md): the dice that
// decide each frame's physical faults (drops, exposure spells, shake, the one-time knock, blur),
// and the capture of the exact rendered 1280x720 frame as mono bytes for the trained keypoint
// detector on the server. Nothing here knows or uses where the keypoints are.
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
export function makeSensor({ seed, sim }) {
  const rng = mulberry32((seed * 2654435761) >>> 0);
  const state = {
    stops: sim.exposure_stops ?? 0,
    dropRate: sim.drop_rate ?? 0,
    burst: sim.burst ?? 0,
    exposureRate: sim.exposure_rate ?? 0,
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
      blurPx: smear + defocus, // sensor px: blurs the rendered pixels
    };
  };

  /** The frame exactly as rendered (plus the physical occluder), as 1280x720 mono bytes. */
  const capture = (src, srcRect) => {
    const { x, y, w, h } = srcRect;
    const k = src.width / src.clientWidth; // CSS px to the canvas's own pixels
    ax.drawImage(src, x * k, y * k, w * k, h * k, 0, 0, W0, H0);
    if (state.occluder) {
      ax.fillStyle = '#000';
      ax.fillRect(state.occluder[0], state.occluder[1], state.occluder[2] - state.occluder[0], state.occluder[3] - state.occluder[1]);
    }
    const rgba = ax.getImageData(0, 0, W0, H0).data;
    const out = new Uint8Array(W0 * H0);
    for (let i = 0, j = 0; j < out.length; i += 4, j++) out[j] = (77 * rgba[i] + 150 * rgba[i + 1] + 29 * rgba[i + 2]) >> 8;
    return out;
  };

  return {
    setStops(v) { state.stops = v; },
    getStops: () => state.stops,
    beginFrame,
    capture,
  };
}
