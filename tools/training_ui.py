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
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "rl", "runs")
ARCHIVE = os.path.join(RUNS, "archive.jsonl")
BOARD = os.path.join(ROOT, "docs", "AGENT_BOARD.md")
LEVELS = ["L0", "L1", "L2", "L3", "L4", "L5", "L6"]


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


def fmt(v, spec="{:.2f}"):
    return spec.format(v) if isinstance(v, (int, float)) else "-"


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


def main(argv=None):
    ap = argparse.ArgumentParser(description="Training & Experiment Observatory")
    ap.add_argument("--summary", action="store_true",
                    help="print Pareto + counts")
    ap.add_argument("--plot", default=None,
                    help="save Pareto scatter to PNG (needs --summary)")
    ap.add_argument("--serve", action="store_true",
                    help="serve HTML summary page")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args(argv)
    entries = load_entries()
    if a.serve:
        text = summary_text(entries)

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                body = ("<html><body><h1>Training Observatory</h1><pre>"
                        + text.replace("&", "&amp;").replace("<", "&lt;")
                        + "</pre></body></html>").encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

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
