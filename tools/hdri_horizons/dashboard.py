#!/usr/bin/env python3
"""Horizon labeling dashboard: review/correct the per-panorama floor junction.

The merged CC0 panorama set (OpenHDRI + ambientCG, see fetch_panos.py); this tool lets you scrub the grounded dome's
horizon offset (and roll) with a live preview and saves the result to
src/ugv/lab/static/hdri-horizons.json, which the lab consumes on next panorama select.

Run:  uv run python tools/hdri_horizons/dashboard.py [--port 8770]
Open: http://127.0.0.1:8770/
"""

import argparse
import json
import re
import shutil
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "src" / "ugv" / "lab" / "static"
CACHE = ROOT / "runs" / "cache" / "hdri"
CACHE_V2 = ROOT / "runs" / "cache" / "hdri_v2"
THUMBS = CACHE / "thumbs"
INDEX = STATIC / "hdri.json"
OUT = STATIC / "hdri-horizons.json"
REVIEWED = STATIC / "hdri-horizons-reviewed.json"

sys.path.insert(0, str(Path(__file__).parent))
from rgbe import load_hdr, tonemap_png  # noqa: E402

ID_RE = re.compile(r"^[a-z0-9_]+$")


def read_json(path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def reviewed_map():
    data = read_json(REVIEWED, {})
    return {k: bool(v) for k, v in data.items()} if isinstance(data, dict) else {}


def save_horizon(hid, offset, roll_b, roll_c):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    horizons = read_json(OUT, {})
    prev = horizons.get(hid, {})
    horizons[hid] = {
        "offset": round(float(offset), 4),
        "roll_b": round(float(roll_b), 4),
        "roll_c": round(float(roll_c), 4),
        "conf": prev.get("conf", 1.0),
    }
    if OUT.exists():
        backups = ROOT / "runs" / "cache" / "hdri" / "horizon_backups"
        backups.mkdir(parents=True, exist_ok=True)
        import time

        shutil.copy(OUT, backups / f"hdri-horizons-{int(time.time())}.json")
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(horizons, indent=1, sort_keys=True) + "\n")
    tmp.replace(OUT)


def save_reviewed(hid, reviewed):
    data = read_json(REVIEWED, {})
    if not isinstance(data, dict):
        data = {}
    if reviewed:
        data[hid] = True
    else:
        data.pop(hid, None)
    REVIEWED.write_text(json.dumps(dict(sorted(data.items())), indent=1) + "\n")


def thumb(hid):
    THUMBS.mkdir(parents=True, exist_ok=True)
    dst = THUMBS / f"{hid}.png"
    if dst.exists():
        return dst.read_bytes()
    src_v2 = CACHE_V2 / f"{hid}_2k.hdr" if hid.startswith(("acg_", "ohl_")) else None
    src = src_v2 if (src_v2 and src_v2.exists()) else CACHE / f"{hid}_1k.hdr"
    if not src.exists():
        return None
    w, h, rgb = load_hdr(src)
    png = tonemap_png(rgb, w, h)
    dst.write_bytes(png)
    return png


PAGE = r"""<!doctype html>
<html lang="en" data-mode="night">
<head><meta charset="utf-8"><title>Horizon labeling · CC0 panoramas</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 :root { color-scheme: dark; }
 * { box-sizing: border-box; }
 body { margin:0; font:14px/1.4 system-ui, sans-serif; background:#121211; color:#eeebe3; }
 header { position:sticky; top:0; display:flex; gap:12px; align-items:center; padding:8px 12px; background:#1a1a18; border-bottom:1px solid #634936; z-index:5; }
 header h1 { font-size:13px; letter-spacing:.04em; text-transform:uppercase; margin:0; white-space:nowrap; color:#eeebe3; }
 header .progress { font-family:ui-monospace, monospace; color:#ffb27d; white-space:nowrap; }
 header input { flex:1; min-width:0; background:#232320; color:inherit; border:1px solid #634936; border-radius:0; padding:6px 8px; }
 header select { background:#232320; color:inherit; border:1px solid #634936; padding:6px 8px; }
 main { display:grid; grid-template-columns:1fr 340px; height:calc(100vh - 49px); }
 #grid { overflow-y:auto; padding:10px; display:grid; grid-template-columns:repeat(auto-fill, minmax(150px, 1fr)); gap:8px; align-content:start; }
 .card { background:#1a1a18; border:1px solid #333; padding:6px; cursor:pointer; }
 .card:hover, .card.sel { border-color:#ff6a2b; }
 .card img { width:100%; aspect-ratio:2/1; object-fit:cover; background:#000; display:block; }
 .card .row { display:flex; justify-content:space-between; font-family:ui-monospace, monospace; font-size:10px; margin-top:4px; color:#b8b4a8; }
 .card .rev { color:#7dde8c; }
 aside { border-left:1px solid #634936; display:flex; flex-direction:column; min-width:0; }
 aside .view { position:relative; border-bottom:1px solid #634936; background:#000; }
 aside canvas { width:100%; display:block; cursor:crosshair; }
 aside .bar { padding:10px 12px; display:flex; flex-direction:column; gap:10px; overflow-y:auto; }
 aside h2 { font-family:ui-monospace, monospace; font-size:13px; margin:0; word-break:break-all; }
 label { display:flex; justify-content:space-between; gap:8px; font-size:11px; color:#b8b4a8; }
 label output { font-family:ui-monospace, monospace; color:#eeebe3; }
 input[type=range] { width:100%; accent-color:#ff6a2b; }
 .row2 { display:flex; gap:8px; }
 button { background:#232320; color:#eeebe3; border:1px solid #634936; padding:8px 10px; cursor:pointer; font:inherit; }
 button.primary { background:#b83a0a; border-color:#b83a0a; color:#fff; font-weight:600; }
 button:focus-visible, .card:focus-visible { outline:2px solid #ff6a2b; outline-offset:1px; }
 kbd { font-family:ui-monospace, monospace; background:#000; padding:1px 4px; border:1px solid #444; font-size:10px; }
 .hint { font-size:11px; color:#8f8b7e; }
 :focus { outline:2px solid #ff6a2b; }
</style></head>
<body>
<header>
  <h1>Horizon labeling</h1>
  <span class="progress" id="progress">… reviewed</span>
  <input id="q" type="search" placeholder="Search panoramas…" autocomplete="off">
  <select id="sort" aria-label="Sort"><option value="name">Sort: name</option><option value="conf">Sort: confidence</option><option value="unreviewed">Unreviewed first</option></select>
</header>
<main>
  <section id="grid" aria-label="Panoramas"></section>
  <aside aria-label="Selected panorama">
    <div class="view"><canvas id="cv" width="512" height="256" role="img" aria-label="Panorama with horizon overlay"></canvas></div>
    <div class="bar">
      <h2 id="sel-name">Select a panorama</h2>
      <p class="hint" id="sel-src"></p>
      <label>Pitch offset <output id="o-v">0.000</output></label>
      <input type="range" id="pitch" min="-0.4" max="0.4" step="0.005" value="0">
      <label>Roll amplitude <output id="ra-v">0.000</output></label>
      <input type="range" id="roll-a" min="0" max="0.15" step="0.002" value="0">
      <label>Roll phase <output id="rp-v">0°</output></label>
      <input type="range" id="roll-p" min="0" max="360" step="2" value="0">
      <label class="reviewed-toggle"><span><input type="checkbox" id="reviewed"> Mark reviewed</span><output id="conf-v"></output></label>
      <div class="row2"><button id="reset">Reset to 0</button><button id="save-next" class="primary">Save &amp; next <kbd>Enter</kbd></button></div>
      <p class="hint">Drag the green sine on the image or use the sliders. The line should sit exactly where floor meets wall. <kbd>↑</kbd>/<kbd>↓</kbd> pitch · <kbd>←</kbd>/<kbd>→</kbd> roll amp · <kbd>Q</kbd>/<kbd>E</kbd> phase · <kbd>Enter</kbd> save &amp; next · <kbd>/</kbd> focus search.</p>
    </div>
  </aside>
</main>
<script>
const $ = (s) => document.querySelector(s);
let list = [], sel = null, img = null;
const state = { offset: 0, rollA: 0, rollP: 0, reviewed: false };

function fmt(n) { return (n >= 0 ? '+' : '') + n.toFixed(3); }

async function load() {
  const r = await fetch('/api/list');
  list = await r.json();
  renderGrid();
  updateProgress();
}
function updateProgress() {
  $('#progress').textContent = `${list.filter(x => x.reviewed).length}/${list.length} reviewed`;
}
function visible() {
  const q = $('#q').value.trim().toLowerCase();
  const sort = $('#sort').value;
  let v = list.filter(x => !q || x.id.includes(q) || (x.name || '').toLowerCase().includes(q));
  if (sort === 'conf') v.sort((a, b) => (a.conf ?? 1) - (b.conf ?? 1));
  else if (sort === 'unreviewed') v.sort((a, b) => (a.reviewed ? 1 : 0) - (b.reviewed ? 1 : 0) || a.id.localeCompare(b.id));
  else v.sort((a, b) => a.id.localeCompare(b.id));
  return v;
}
function renderGrid() {
  $('#grid').innerHTML = visible().map(x => `
    <div class="card" tabindex="0" data-id="${x.id}" role="button" aria-pressed="false">
      <img loading="lazy" src="/api/img/${x.id}" alt="${x.name}" onerror="this.replaceWith(Object.assign(document.createElement('div'), {textContent:'no image', style:'aspect-ratio:2/1;display:grid;place-items:center;color:#777'}))">
      <div class="row"><span>offset ${fmt(x.offset)}${x.conf != null ? ` · conf ${x.conf.toFixed(2)}` : ''}</span>${x.reviewed ? '<span class="rev">✓</span>' : ''}</div>
      <div class="row"><span>${(x.name || x.id)}</span><span>${x.source || ''}</span></div>
    </div>`).join('');
}
$('#grid').addEventListener('click', (e) => { const c = e.target.closest('.card'); if (c) select(c.dataset.id); });
$('#grid').addEventListener('keydown', (e) => { const c = e.target.closest('.card'); if (c && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); select(c.dataset.id); } });
$('#q').addEventListener('input', renderGrid);
$('#sort').addEventListener('change', renderGrid);

async function select(id) {
  sel = list.find(x => x.id === id);
  if (!sel) return;
  state.offset = sel.offset ?? 0;
  state.rollA = Math.hypot(sel.roll_b ?? 0, sel.roll_c ?? 0);
  state.rollP = ((Math.atan2(sel.roll_c ?? 0, sel.roll_b ?? 0) * 180 / Math.PI) + 360) % 360;
  state.reviewed = !!sel.reviewed;
  $('#sel-name').textContent = sel.name || sel.id;
  $('#sel-src').textContent = [sel.source, sel.license ? `license ${sel.license}` : ''].filter(Boolean).join(' · ');
  $('#pitch').value = state.offset; $('#roll-a').value = state.rollA; $('#roll-p').value = state.rollP;
  $('#reviewed').checked = state.reviewed;
  $('#conf-v').textContent = sel.conf != null ? `detector conf ${sel.conf.toFixed(2)}` : '';
  document.querySelectorAll('.card').forEach(c => c.classList.toggle('sel', c.dataset.id === id));
  img = new Image();
  img.onload = draw;
  img.src = '/api/img/' + id;
  syncLabels();
}
function syncLabels() {
  $('#o-v').textContent = fmt(state.offset);
  $('#ra-v').textContent = state.rollA.toFixed(3);
  $('#rp-v').textContent = Math.round(state.rollP) + '°';
}
function draw() {
  const cv = $('#cv'), ctx = cv.getContext('2d');
  if (!img) return;
  cv.width = img.width; cv.height = img.height;
  ctx.drawImage(img, 0, 0);
  const W = cv.width, H = cv.height;
  ctx.strokeStyle = '#4ade80';
  ctx.lineWidth = Math.max(2, H / 128);
  ctx.shadowColor = 'rgba(0,0,0,.8)';
  ctx.shadowBlur = 3;
  ctx.beginPath();
  for (let x = 0; x <= W; x++) {
    const az = (x / W) * 2 * Math.PI - Math.PI;
    // file v of junction per azimuth; shader subtracts offset + roll
    const phi = az + Math.PI;
    const b = Math.cos(state.rollP * Math.PI / 180) * state.rollA;
    const c = Math.sin(state.rollP * Math.PI / 180) * state.rollA;
    const v = 0.5 - state.offset - b * Math.cos(az) - c * Math.sin(az);
    const y = (1 - v) * H;
    x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.shadowBlur = 0;
}
$('#pitch').addEventListener('input', e => { state.offset = +e.target.value; syncLabels(); draw(); });
$('#roll-a').addEventListener('input', e => { state.rollA = +e.target.value; syncLabels(); draw(); });
$('#roll-p').addEventListener('input', e => { state.rollP = +e.target.value; syncLabels(); draw(); });
$('#reviewed').addEventListener('change', e => { state.reviewed = e.target.checked; });
$('#reset').addEventListener('click', () => { state.offset = 0; state.rollA = 0; syncLabels(); $('#pitch').value = 0; $('#roll-a').value = 0; $('#roll-p').value = 0; draw(); });

// drag the sine vertically to set pitch; horizontal drag rotates phase
let drag = null;
$('#cv').addEventListener('pointerdown', e => { drag = { x: e.clientX, y: e.clientY, off: state.offset, ph: state.rollP }; $('#cv').setPointerCapture(e.pointerId); });
$('#cv').addEventListener('pointermove', e => {
  if (!drag) return;
  const H = $('#cv').clientHeight;
  state.offset = Math.max(-0.4, Math.min(0.4, drag.off + (drag.y - e.clientY) / H * 1.6));
  state.rollP = (drag.ph + (e.clientX - drag.x) / 4 + 360) % 360;
  $('#pitch').value = state.offset; $('#roll-p').value = state.rollP;
  syncLabels(); draw();
});
$('#cv').addEventListener('pointerup', () => drag = null);

async function save(advance) {
  if (!sel) return;
  const b = Math.cos(state.rollP * Math.PI / 180) * state.rollA;
  const c = Math.sin(state.rollP * Math.PI / 180) * state.rollA;
  await fetch('/api/save', { method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ id: sel.id, offset: state.offset, roll_b: b, roll_c: c, reviewed: state.reviewed }) });
  const rec = list.find(x => x.id === sel.id);
  Object.assign(rec, { offset: state.offset, roll_b: b, roll_c: c, reviewed: state.reviewed });
  updateProgress(); renderGrid();
  if (advance) {
    const v = visible(); const i = v.findIndex(x => x.id === sel.id);
    const next = v[(i + 1) % v.length];
    if (next) select(next.id);
  }
}
$('#save-next').addEventListener('click', () => save(true));
document.addEventListener('keydown', e => {
  if (e.key === '/' && document.activeElement.tagName !== 'INPUT') { e.preventDefault(); $('#q').focus(); return; }
  if (document.activeElement.tagName === 'INPUT' || document.activeElement.tagName === 'SELECT') return;
  const fine = e.shiftKey ? 0.02 : 0.005;
  if (e.key === 'ArrowUp') { state.offset = Math.min(0.4, state.offset + fine); }
  else if (e.key === 'ArrowDown') { state.offset = Math.max(-0.4, state.offset - fine); }
  else if (e.key === 'ArrowLeft') { state.rollA = Math.max(0, state.rollA - 0.005); }
  else if (e.key === 'ArrowRight') { state.rollA = Math.min(0.15, state.rollA + 0.005); }
  else if (e.key === 'q') { state.rollP = (state.rollP - 4 + 360) % 360; }
  else if (e.key === 'e') { state.rollP = (state.rollP + 4) % 360; }
  else if (e.key === 'Enter' && sel) { e.preventDefault(); save(true); return; }
  else return;
  e.preventDefault();
  $('#pitch').value = state.offset; $('#roll-a').value = state.rollA; $('#roll-p').value = state.rollP;
  syncLabels(); draw();
});
load();
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="text/html; charset=utf-8"):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send(200, PAGE)
            return
        if path == "/api/list":
            hdris = read_json(INDEX, {"assets": []}).get("assets", [])
            horizons = read_json(OUT, {})
            reviewed = reviewed_map()
            out = []
            for a in hdris:
                hid = a["id"]
                h = horizons.get(hid) or {}
                offset = h.get("offset", 0.0) if isinstance(h, dict) else float(h)
                out.append(
                    {
                        "id": hid,
                        "name": a.get("name", hid),
                        "env": a.get("env"),
                        "cat": a.get("cat"),
                        "source": a.get("source"),
                        "license": a.get("license"),
                        "offset": offset,
                        "roll_b": h.get("roll_b", 0.0) if isinstance(h, dict) else 0.0,
                        "roll_c": h.get("roll_c", 0.0) if isinstance(h, dict) else 0.0,
                        "conf": h.get("conf") if isinstance(h, dict) else None,
                        "reviewed": reviewed.get(hid, False),
                    }
                )
            self._send(200, json.dumps(out), "application/json")
            return
        m = re.match(r"^/api/img/([a-z0-9_]+)$", path)
        if m:
            data = thumb(m.group(1))
            if data is None:
                self._send(
                    404, " panorama not cached — open it once in the lab, then reload", "text/plain"
                )
            else:
                self._send(200, data, "image/png")
            return
        self._send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/save":
            self._send(404, "not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length))
            hid = str(body["id"])
            offset = float(body["offset"])
            roll_b = float(body.get("roll_b", 0.0))
            roll_c = float(body.get("roll_c", 0.0))
            assert ID_RE.match(hid)
            assert abs(offset) <= 0.5 and abs(roll_b) <= 0.3 and abs(roll_c) <= 0.3
        except Exception:
            self._send(400, "bad request", "text/plain")
            return
        save_horizon(hid, offset, roll_b, roll_c)
        if "reviewed" in body:
            save_reviewed(hid, bool(body["reviewed"]))
        self._send(200, json.dumps({"ok": True}), "application/json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    args = ap.parse_args()
    print(f"Horizon labeling dashboard: http://127.0.0.1:{args.port}/")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
