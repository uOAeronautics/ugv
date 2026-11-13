#!/usr/bin/env python3
"""Training & Experiment Observatory (stdlib + matplotlib only, no torch).

Reads rl/runs/archive.jsonl + rl/runs/*.json + docs/AGENT_BOARD.md and
prints (or serves) a Pareto summary, per-level L0-L6 breakdown, and links
to trajectory visualizations (rl/runs/vis_*).

Usage:
  python3 tools/training_ui.py --summary            # Pareto + counts
  python3 tools/training_ui.py --summary --plot out.png
  python3 tools/training_ui.py --serve [--port 8765] # tiny HTML page
"""
import argparse
import glob
import html
import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "rl", "runs")
ARCHIVE = os.path.join(RUNS, "archive.jsonl")
BOARD = os.path.join(ROOT, "docs", "AGENT_BOARD.md")
PI_BENCH = os.path.join(RUNS, "pi_inference_bench.json")
LEVELS = ["L0", "L1", "L2", "L3", "L4", "L5", "L6"]
# (min success, max collisions) per 1000-ep eval; L2+ are report-only.
GATES = {"L0": (980, 5), "L1": (980, 5)}
LOG_LINE = re.compile(
    r"iter\s+(\d+)\s+pol\s+(\S+)\s+val\s+(\S+)"
    r"(?:\s+aux\s+(\S+))?\s+std\s+(\S+)\s+succ\s+(\S+)\s+coll\s+(\S+)")


def load_entries():
    """Merge archive.jsonl (preferred) with runs/*.json eval files."""
    entries = {}
    if os.path.exists(ARCHIVE):
        with open(ARCHIVE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                m = r.get("eval_metrics", {}) or {}
                entries[r.get("run_id", "unknown")] = {
                    "run_id": r.get("run_id", "?"),
                    "track": r.get("track", "?"),
                    "level": str(r.get("level", m.get("level", "?"))),
                    "success": m.get("success"),
                    "collisions": m.get("collisions"),
                    "timeouts": m.get("timeouts"),
                    "episodes": m.get("episodes"),
                    "time_median": m.get("time_median"),
                    "pos_p95": m.get("pos_p95"),
                    "yaw_p95_deg": m.get("yaw_p95_deg"),
                    "source": "archive.jsonl",
                }
    for path in sorted(glob.glob(os.path.join(RUNS, "*.json"))):
        run_id = os.path.splitext(os.path.basename(path))[0]
        if run_id in entries:
            continue  # archive is canonical
        try:
            with open(path) as f:
                m = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(m, dict) or "success" not in m:
            continue
        entries[run_id] = {
            "run_id": run_id,
            "track": "eval-json",
            "level": str(m.get("level", "?")),
            "success": m.get("success"),
            "collisions": m.get("collisions"),
            "timeouts": m.get("timeouts"),
            "episodes": m.get("episodes"),
            "time_median": m.get("time_median"),
            "pos_p95": m.get("pos_p95"),
            "yaw_p95_deg": m.get("yaw_p95_deg"),
            "source": os.path.basename(path),
        }
    return [e for e in entries.values()
            if isinstance(e.get("success"), int)]


def pareto_frontier(entries):
    """Non-dominated on (success max, collisions min), per level.

    Time is reported, not gated: including it admits degenerate
    rows (0 succ, fastest time) onto the frontier.
    """
    front = []
    by_level = {}
    for e in entries:
        by_level.setdefault(e["level"], []).append(e)
    for lvl, group in by_level.items():
        for a in group:
            dominated = False
            for b in group:
                if b is a:
                    continue
                bs = b["success"] or 0
                as_ = a["success"] or 0
                bc = b["collisions"] if b["collisions"] is not None else 10**9
                ac = a["collisions"] if a["collisions"] is not None else 10**9
                if (bs, -bc) > (as_, -ac) and bs >= as_ and bc <= ac:
                    dominated = True
                    break
            if not dominated:
                front.append(a)
    return sorted(front, key=lambda e: (e["level"], -(e["success"] or 0)))


def level_breakdown(entries):
    rows = []
    for lvl in LEVELS:
        g = [e for e in entries if e["level"] == lvl]
        if not g:
            rows.append((lvl, 0, None, None, None))
            continue
        best = max(g, key=lambda e: (e["success"] or 0,
                                     -(e["collisions"] or 0)))
        tot_s = sum(e["success"] or 0 for e in g)
        tot_c = sum(e["collisions"] or 0 for e in g
                    if e["collisions"] is not None)
        tot_t = sum(e["timeouts"] or 0 for e in g
                    if e["timeouts"] is not None)
        rows.append((lvl, len(g), tot_s, tot_c, tot_t, best))
    return rows


def viz_links():
    out = {}
    for d in sorted(glob.glob(os.path.join(RUNS, "vis_*"))):
        name = os.path.basename(d)
        try:
            files = sorted(os.listdir(d))
        except OSError:
            files = []
        imgs = [f for f in files if f.endswith((".png", ".mp4"))]
        out[name] = imgs[:6]
    return out


def load_archive_full():
    """Raw archive.jsonl rows (with parent_id, track, config_delta)."""
    rows = []
    if os.path.exists(ARCHIVE):
        with open(ARCHIVE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def lineage(rows):
    """parent_id -> [children]; roots = rows with no known parent."""
    ids = {r.get("run_id") for r in rows}
    children = {}
    for r in rows:
        p = r.get("parent_id")
        if p:
            children.setdefault(p, []).append(r.get("run_id"))
    roots = [r.get("run_id") for r in rows
             if not r.get("parent_id") or r.get("parent_id") not in ids]
    return children, roots


def run_tags(rows, logs):
    """Badge runs: Pareto frontier, JEPA/LeWM aux-head, curriculum."""
    entries = load_entries()
    frontier = {(e["level"], e["run_id"]) for e in pareto_frontier(entries)}
    tags = {}
    for r in rows:
        t = []
        if (str(r.get("level")), r.get("run_id")) in frontier:
            t.append("Pareto")
        rid = str(r.get("run_id", ""))
        if "jepa" in rid or "lew" in rid:
            t.append("JEPA")
        elif rid in logs and any(p["aux"] is not None for p in logs[rid]):
            t.append("JEPA")
        if "curric" in rid or "curric" in str(r.get("config_delta", {})):
            t.append("curric")
        tags[r.get("run_id")] = t
    return tags


def load_logs():
    """rl/runs/*.log -> {run_id: [dict(iter,pol,val,aux,std,succ,coll)]}."""
    out = {}
    for path in sorted(glob.glob(os.path.join(RUNS, "*.log"))):
        run_id = os.path.splitext(os.path.basename(path))[0]
        pts = []
        try:
            with open(path) as f:
                for line in f:
                    m = LOG_LINE.search(line)
                    if not m:
                        continue
                    g = m.groups()
                    pts.append({
                        "iter": int(g[0]),
                        "pol": _f(g[1]), "val": _f(g[2]),
                        "aux": _f(g[3]) if g[3] is not None else None,
                        "std": _f(g[4]), "succ": _f(g[5]),
                        "coll": _f(g[6]),
                    })
        except OSError:
            continue
        if pts:
            out[run_id] = pts
    return out


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_pi_bench():
    try:
        with open(PI_BENCH) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def readiness_rows(entries, bench):
    """(level, best_id, succ, coll, gate, verdict) for L0-L6 + Pi row."""
    rows = []
    for lvl in LEVELS:
        g = [e for e in entries if e["level"] == lvl]
        if not g:
            rows.append((lvl, None, None, None, GATES.get(lvl),
                         "NO DATA"))
            continue
        best = max(g, key=lambda e: (e["success"] or 0,
                                     -(e["collisions"] or 0)))
        gate = GATES.get(lvl)
        if gate is None:
            verdict = "REPORT"
        else:
            need_s, max_c = gate
            n_coll = best["collisions"] if best["collisions"] is not None \
                else 10**9
            ok = (best["success"] or 0) >= need_s and n_coll <= max_c
            verdict = "PASS" if ok else "FAIL"
        rows.append((lvl, best["run_id"], best["success"],
                     best["collisions"], gate, verdict))
    p50 = (((bench.get("latency_ms_single_core") or {})
            .get("stateful_step_plus_shield") or {}).get("p50"))
    proj = (bench.get("pi4b_projection") or {}) \
        .get("stateful_plus_shield_p50_ms", "?")
    verdict = "PASS" if bench and "PASS" in str(
        (bench.get("pi4b_projection") or {}).get("verdict", "")) else \
        ("NO DATA" if not bench else "FAIL")
    rows.append(("Pi4B", f"p50 {p50}ms host -> {proj}ms Pi",
                 None, None, "budget 5.0ms", verdict))
    return rows


def load_board_feed(want=None):
    """Parse AGENT_BOARD.md tables -> [(type, id, mid, evidence)]."""
    feed = []
    try:
        with open(BOARD) as f:
            lines = f.read().splitlines()
    except OSError:
        return feed
    for line in lines:
        m = re.match(r"\|\s*\[(SCOUT|EXP|ADVERSARY|GATE)\]\s*([^|]*)\|"
                     r"([^|]*)\|([^|]*)\|?", line.strip())
        if not m:
            continue
        typ, rid, mid, ev = m.groups()
        if want and typ != want:
            continue
        feed.append((typ, rid.strip(), mid.strip(), ev.strip()[:160]))
    return feed


def fmt(v, spec="{:.2f}"):
    return spec.format(v) if isinstance(v, (int, float)) else "-"


def dag_text(rows):
    """Lineage DAG as indented parent -> child chains with badges."""
    children, roots = lineage(rows)
    by_id = {r.get("run_id"): r for r in rows}
    try:
        logs = load_logs()
    except OSError:
        logs = {}
    tags = run_tags(rows, logs)
    L = []
    seen = set()

    def walk(rid, depth):
        r = by_id.get(rid, {})
        m = r.get("eval_metrics", {}) or {}
        badge = "".join(f" [{t}]" for t in tags.get(rid, []))
        L.append(f"{'  ' * depth}{rid}{badge} "
                 f"{r.get('level', '?')} "
                 f"succ={m.get('success', '-')} "
                 f"coll={m.get('collisions', '-')}")
        for c in sorted(children.get(rid, [])):
            if c not in seen:
                seen.add(c)
                walk(c, depth + 1)

    for root in sorted(roots):
        if root not in seen:
            seen.add(root)
            walk(root, 0)
    orphans = [rid for rid in by_id if rid not in seen]
    for rid in sorted(orphans):
        seen.add(rid)
        walk(rid, 0)
    return "\n".join(L) if L else "(empty archive)"


def readiness_text(entries):
    bench = load_pi_bench()
    L = []
    L.append(f"{'lvl':<5} {'gate':<16} {'verdict':<8} best(run=succ/coll)")
    for lvl, best, s, c, gate, verdict in readiness_rows(entries, bench):
        if isinstance(gate, tuple):
            g = f">={gate[0]}s/<={gate[1]}c"
        else:
            g = gate if isinstance(gate, str) else "report-only"
        if best is None:
            L.append(f"{lvl:<5} {g:<16} {verdict:<8} no data")
        elif lvl == "Pi4B":
            L.append(f"{lvl:<5} {g:<16} {verdict:<8} {best}")
        else:
            L.append(f"{lvl:<5} {g:<16} {verdict:<8} {best}={s}/{c}")
    shield = ((bench.get("shield_stress") or {}).get("status", "")
              if bench else "")
    if shield:
        L.append(f"shield: {shield[:120]}")
    return "\n".join(L)


def board_text(feed, limit=25):
    L = []
    for typ, rid, mid, ev in feed[-limit:]:
        L.append(f"[{typ}] {rid} | {mid} | {ev}")
    return "\n".join(L) if L else "(no board rows)"


def curves_tail_text(logs):
    L = []
    L.append(f"{'run':<28} {'iters':>5} last(pol val aux std succ coll)")
    for rid in sorted(logs):
        pts = logs[rid]
        last = pts[-1]
        aux = fmt(last['aux']) if last['aux'] is not None else "-"
        L.append(f"{rid:<28} {len(pts):>5} "
                 f"{fmt(last['pol'])} {fmt(last['val'])} {aux} "
                 f"{fmt(last['std'])} {fmt(last['succ'], '{:.0f}')} "
                 f"{fmt(last['coll'], '{:.0f}')}")
    return "\n".join(L) if L else "(no training logs)"


def summary_text(entries):
    L = []
    L.append(f"entries: {len(entries)} "
             f"({sum(1 for e in entries if e['source'] == 'archive.jsonl')} "
             f"from archive.jsonl, "
             f"{sum(1 for e in entries if e['source'] != 'archive.jsonl')} "
             f"from runs/*.json only)")
    L.append("")
    L.append("== Pareto frontier (per level: max success, min coll) ==")
    L.append(f"{'level':<5} {'run_id':<28} {'succ':>5} {'coll':>5} "
             f"{'t-out':>5} {'med-s':>7} track")
    for e in pareto_frontier(entries):
        L.append(f"{e['level']:<5} {e['run_id']:<28} {e['success'] or 0:>5} "
                 f"{e['collisions'] if e['collisions'] is not None else '-':>5} "
                 f"{e['timeouts'] if e['timeouts'] is not None else '-':>5} "
                 f"{fmt(e['time_median'], '{:.1f}'):>7} {e['track']}")
    L.append("")
    L.append("== L0-L6 success/collision/timeout breakdown ==")
    L.append(f"{'lvl':<4} {'nruns':>5} {'tot_succ':>8} {'tot_coll':>8} "
             f"{'tot_tout':>8} best(run_id=succ/coll)")
    for row in level_breakdown(entries):
        if len(row) == 5:
            L.append(f"{row[0]:<4} {row[1]:>5} {'-':>8} {'-':>8} {'-':>8} "
                     f"no data")
        else:
            lvl, n, s, c, t, b = row
            L.append(f"{lvl:<4} {n:>5} {s:>8} {c:>8} {t:>8} "
                     f"{b['run_id']}={b['success']}/{b['collisions']}")
    L.append("")
    L.append("== Discovery tree / lineage DAG (parent -> child) ==")
    L.append(dag_text(load_archive_full()))
    L.append("")
    L.append("== Readiness scorecard (L0-L6 + Pi 4B inference) ==")
    L.append(readiness_text(entries))
    L.append("")
    L.append("== Board feed (latest; --board TYPE to filter) ==")
    L.append(board_text(load_board_feed()))
    L.append("")
    L.append("== Live training curves (last iter per log) ==")
    try:
        L.append(curves_tail_text(load_logs()))
    except OSError:
        L.append("(no training logs)")
    L.append("")
    L.append("== trajectory visualizations (rl/runs/vis_*) ==")
    viz = viz_links()
    if not viz:
        L.append("none found")
    for d, files in viz.items():
        L.append(f"{d}/: " + (", ".join(files) if files else "(empty)"))
    if os.path.exists(BOARD):
        L.append("")
        L.append(f"board: {BOARD}")
    return "\n".join(L)


def make_plot(entries, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for lvl in LEVELS:
        g = [e for e in entries if e["level"] == lvl]
        if not g:
            continue
        ax.scatter([e["collisions"] or 0 for e in g],
                   [e["success"] or 0 for e in g], label=lvl)
        for e in g:
            ax.annotate(e["run_id"][:18], ((e["collisions"] or 0),
                                           (e["success"] or 0)), fontsize=6)
    ax.set_xlabel("collisions (per eval)")
    ax.set_ylabel("success (per eval)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    return out


def latest_log_run(logs):
    """Most recently modified log run (active training)."""
    best, best_mtime = None, -1
    for rid in logs:
        p = os.path.join(RUNS, rid + ".log")
        try:
            mt = os.path.getmtime(p)
        except OSError:
            continue
        if mt > best_mtime:
            best, best_mtime = rid, mt
    return best or (sorted(logs)[0] if logs else None)


def make_curves(logs, run_id, out):
    """Per-iter (pol, val, aux, std, succ, coll) panels for one run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts = logs.get(run_id)
    if not pts:
        raise SystemExit(f"no log data for run {run_id}")
    keys = ["pol", "val", "aux", "std", "succ", "coll"]
    have = [k for k in keys if any(p[k] is not None for p in pts)]
    fig, axs = plt.subplots(2, 3, figsize=(12, 6), squeeze=False)
    it = [p["iter"] for p in pts]
    for ax, k in zip([a for row in axs for a in row], keys):
        if k in have:
            ax.plot(it, [p[k] if p[k] is not None else float("nan")
                         for p in pts])
        ax.set_title(f"{run_id} {k}")
        ax.set_xlabel("iter")
    fig.tight_layout()
    fig.savefig(out)
    return out


def curve_svg(pts, key, w=300, h=70):
    """Inline SVG polyline for one metric (stdlib, no JS deps)."""
    ys = [p[key] for p in pts if p[key] is not None]
    if len(ys) < 2:
        return "<span>no data</span>"
    lo, hi = min(ys), max(ys)
    span = (hi - lo) or 1.0
    step = w / (len(ys) - 1)
    d = " ".join(f"{i * step:.1f},{h - 4 - (y - lo) / span * (h - 10):.1f}"
                 for i, y in enumerate(ys))
    last = ys[-1]
    return (f"<svg width='{w}' height='{h}'><polyline points='{d}' "
            f"fill='none' stroke='#4da3ff' stroke-width='1.5'/>"
            f"<text x='4' y='12' font-size='10' fill='#888'>"
            f"{html.escape(key)} last={last:.3f}</text></svg>")


def html_page(entries, board_filter=None):
    rows = load_archive_full()
    logs = load_logs()
    tags = run_tags(rows, logs)
    children, _ = lineage(rows)
    front = {(e["level"], e["run_id"]) for e in pareto_frontier(entries)}
    bench = load_pi_bench()
    feed = load_board_feed(board_filter)
    active = sorted(logs, key=lambda r: os.path.getmtime(
        os.path.join(RUNS, r + ".log")))[:-5:-1]
    H = ["<html><head><meta charset='utf-8'>",
         "<style>body{font-family:sans-serif;margin:2em;max-width:1100px}"
         "table{border-collapse:collapse}td,th{border:1px solid #ccc;"
         "padding:2px 8px;font-size:13px}.badge{background:#eef;border-radius:"
         "4px;padding:0 4px;font-size:11px}ul{line-height:1.5}</style></head>",
         "<body><h1>Training Observatory</h1>"]
    H.append(f"<p>{len(entries)} eval entries, {len(rows)} archive rows, "
             f"{len(logs)} training logs.</p>")

    H.append("<h2>Pareto frontier</h2><table><tr><th>level</th><th>run</th>"
             "<th>succ</th><th>coll</th><th>med-s</th></tr>")
    for e in pareto_frontier(entries):
        H.append(f"<tr><td>{html.escape(e['level'])}</td>"
                 f"<td>{html.escape(e['run_id'])}</td>"
                 f"<td>{e['success']}</td><td>{e['collisions']}</td>"
                 f"<td>{fmt(e['time_median'], '{:.1f}')}</td></tr>")
    H.append("</table>")

    H.append("<h2>Discovery tree / lineage DAG</h2><ul>")
    by_id = {r.get("run_id"): r for r in rows}
    for r in rows:
        rid = r.get("run_id")
        kids = sorted(children.get(rid, []))
        badge = "".join(f" <span class='badge'>{t}</span>"
                        for t in tags.get(rid, []))
        link = (f" -&gt; {', '.join(html.escape(k) for k in kids)}"
                if kids else "")
        star = " *" if (str(r.get("level")), rid) in front else ""
        H.append(f"<li><b>{html.escape(str(rid))}</b>{star}{badge} "
                 f"L{html.escape(str(r.get('level', '?')))} "
                 f"parent={html.escape(str(r.get('parent_id')))}{link}</li>")
    H.append("</ul><p>* Pareto-dominant. Badges: JEPA = aux-head run, "
             "curric = multi-axis curriculum.</p>")

    H.append("<h2>Readiness scorecard (L0-L6 + Pi 4B)</h2>")
    H.append("<table><tr><th>level</th><th>gate</th><th>verdict</th>"
             "<th>best</th></tr>")
    for lvl, best, s, c, gate, verdict in readiness_rows(entries, bench):
        g = f"&ge;{gate[0]}s/&le;{gate[1]}c" if isinstance(gate, tuple) \
            else html.escape(str(gate or "report-only"))
        H.append(f"<tr><td>{lvl}</td><td>{g}</td><td><b>{verdict}</b></td>"
                 f"<td>{html.escape(str(best))}"
                 + (f"={s}/{c}" if s is not None else "") + "</td></tr>")
    H.append("</table>")

    H.append("<h2>Board feed</h2>")
    H.append("<div>" + "".join(
        f"<button onclick=\"f('{t}')\">{t}</button>"
        for t in ["ALL", "GATE", "EXP", "SCOUT", "ADVERSARY"]) + "</div>")
    H.append("<table id='feed'><tr><th>type</th><th>id</th><th>mid</th>"
             "<th>evidence</th></tr>")
    for typ, rid, mid, ev in feed:
        H.append(f"<tr data-t='{typ}'><td>{typ}</td>"
                 f"<td>{html.escape(rid)}</td><td>{html.escape(mid)}</td>"
                 f"<td>{html.escape(ev)}</td></tr>")
    H.append("</table><script>function f(t){for(const r of "
             "document.querySelectorAll('#feed tr[data-t]'))"
             "r.style.display=(t==='ALL'||r.dataset.t===t)?'':'none';}"
             "</script>")

    H.append("<h2>Live training curves</h2>")
    for rid in active:
        pts = logs[rid]
        H.append(f"<h3>{html.escape(rid)} ({len(pts)} iters)</h3><div>")
        for k in ["pol", "val", "aux", "std", "succ", "coll"]:
            H.append(curve_svg(pts, k))
        H.append("</div>")
    H.append("</body></html>")
    return "".join(H)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Training & Experiment Observatory")
    ap.add_argument("--summary", action="store_true",
                    help="print Pareto + counts")
    ap.add_argument("--plot", default=None,
                    help="save Pareto scatter to PNG (needs --summary)")
    ap.add_argument("--serve", action="store_true",
                    help="serve HTML summary page")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--board", default=None,
                    choices=["GATE", "EXP", "SCOUT", "ADVERSARY"],
                    help="print filterable board feed for one type")
    ap.add_argument("--lineage", action="store_true",
                    help="print only the discovery tree / lineage DAG")
    ap.add_argument("--run", default=None,
                    help="run id for --curves (default: latest log)")
    ap.add_argument("--curves", default=None,
                    help="save per-iter (pol,val,aux,std,succ,coll) "
                    "panels to PNG")
    a = ap.parse_args(argv)
    entries = load_entries()
    if a.lineage:
        print(dag_text(load_archive_full()))
        return 0
    if a.board:
        print(board_text(load_board_feed(a.board), limit=1000))
        return 0
    if a.curves:
        logs = load_logs()
        run = a.run or latest_log_run(logs)
        if not run:
            raise SystemExit("no training logs found")
        print("curves: " + make_curves(logs, run, a.curves))
        return 0
    if a.serve:
        page = html_page(entries).encode()

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)

            def log_message(self, *args):
                pass
        print(f"serving on http://127.0.0.1:{a.port}/ ({len(entries)} entries)")
        HTTPServer(("127.0.0.1", a.port), H).serve_forever()
        return 0
    # default action: print summary
    print(summary_text(entries))
    if a.plot:
        print("plot: " + make_plot(entries, a.plot))
    return 0


if __name__ == "__main__":
    sys.exit(main())
