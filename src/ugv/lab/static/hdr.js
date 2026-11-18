// Minimal Radiance RGBE (.hdr) reader for Poly Haven's 1k panoramas: enough for an equirect
// background and image-based lighting, no remote code. Returns a DataTexture.
import * as THREE from '/static/vendor/three.module.min.js';

const toFloat = (r, g, b, e) => {
  const f = e ? Math.pow(2, e - 128) / 256 : 0;
  return [r * f, g * f, b * f];
};

export async function loadHdr(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${url}`);
  const bytes = new Uint8Array(await res.arrayBuffer());
  let pos = 0;
  const line = () => {
    const start = pos;
    while (pos < bytes.length && bytes[pos] !== 10) pos++;
    const s = String.fromCharCode(...bytes.subarray(start, pos));
    pos++; // newline
    return s;
  };
  if (line() !== '#?RADIANCE') throw new Error('not a Radiance file');
  let format = '';
  for (let l = line(); l !== ''; l = line()) {
    if (l.startsWith('FORMAT=')) format = l.slice(7);
  }
  if (format !== '32-bit_rle_rgbe') throw new Error(`unsupported HDR format: ${format}`);
  const m = /^([+-])Y\s+(\d+)\s+([+-])X\s+(\d+)$/.exec(line());
  if (!m) throw new Error('bad HDR resolution line');
  const w = Number(m[4]), h = Number(m[2]);
  const data = new Float32Array(w * h * 4);
  const put = (i, r, g, b, e) => {
    const [R, G, B] = toFloat(r, g, b, e);
    data[i] = R; data[i + 1] = G; data[i + 2] = B; data[i + 3] = 1;
  };
  const row = w * 4;
  for (let y = 0; y < h; y++) {
    const scan = pos + 4 <= bytes.length && bytes[pos] === 2 && bytes[pos + 1] === 2 && (bytes[pos + 2] & 0x80) === 0;
    if (!scan) {
      // Old flat RGBE: a whole image of 4 bytes per pixel (rare in Poly Haven's library).
      for (let i = 0; i < w * h; i++, pos += 4) put(i * 4, bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]);
      break;
    }
    const sw = (bytes[pos + 2] << 8) | bytes[pos + 3];
    pos += 4;
    if (sw !== w) throw new Error('HDR scanline width mismatch');
    const rgbe = new Uint8Array(row);
    for (let c = 0; c < 4; c++) {
      let x = 0;
      while (x < w) {
        const n = bytes[pos++];
        if (n > 128) {
          const v = bytes[pos++];
          for (let i = 0; i < n - 128; i++) rgbe[(x++) * 4 + c] = v;
        } else if (n > 0) {
          for (let i = 0; i < n; i++) rgbe[(x++) * 4 + c] = bytes[pos++];
        } else throw new Error('bad HDR run length');
      }
    }
    const o = y * row;
    for (let x = 0; x < w; x++) {
      const i = o + x * 4, j = x * 4;
      put(i, rgbe[j], rgbe[j + 1], rgbe[j + 2], rgbe[j + 3]);
    }
  }
  const tex = new THREE.DataTexture(data, w, h, THREE.RGBAFormat, THREE.FloatType);
  tex.mapping = THREE.EquirectangularReflectionMapping;
  tex.minFilter = tex.magFilter = THREE.LinearFilter;
  tex.generateMipmaps = false;
  tex.flipY = true;
  tex.needsUpdate = true;
  return tex;
}
