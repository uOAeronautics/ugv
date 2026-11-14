#!/usr/bin/env python3
"""Zero-dependency Shared KB & Deterministic RAG / Symbol-Slice tool."""

import ast, json, pathlib, re, sys

KB_DIR = pathlib.Path(__file__).resolve().parent
ROOT = KB_DIR.parent
INDEX_PATH, FACTS_PATH = KB_DIR / "index.json", KB_DIR / "FACTS.md"
PY_DIRS = ["src/ugv", "tools/l7", "rl", "tests"]


def _route(dec):
    if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
        p = (
            dec.args[0].value
            if dec.args
            and isinstance(dec.args[0], ast.Constant)
            and isinstance(dec.args[0].value, str)
            else None
        )
        for kw in dec.keywords:
            if (
                kw.arg == "path"
                and isinstance(kw.value, ast.Constant)
                and isinstance(kw.value.value, str)
            ):
                p = kw.value.value
        if p and (p.startswith("/api/") or p == "/api" or p.startswith("/ws")):
            return f"{dec.func.attr.upper()} {p}"
    return None


def _index_py(path: pathlib.Path, rel: str):
    syms, tree = [], ast.parse(path.read_text("utf-8", "replace"), filename=rel)
    for n in tree.body:
        end = getattr(n, "end_lineno", n.lineno)
        if isinstance(n, ast.ClassDef):
            syms.append(
                {"path": rel, "kind": "class", "name": n.name, "start": n.lineno, "end": end}
            )
            for m in n.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    syms.append(
                        {
                            "path": rel,
                            "kind": "method",
                            "name": f"{n.name}.{m.name}",
                            "start": m.lineno,
                            "end": getattr(m, "end_lineno", m.lineno),
                        }
                    )
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            syms.append(
                {"path": rel, "kind": "func", "name": n.name, "start": n.lineno, "end": end}
            )
            for d in n.decorator_list:
                r = _route(d)
                if r:
                    syms.append(
                        {
                            "path": rel,
                            "kind": "route",
                            "name": f"{r} -> {n.name}",
                            "start": min([x.lineno for x in n.decorator_list] + [n.lineno]),
                            "end": end,
                        }
                    )
    return syms


def _index_static(path: pathlib.Path, rel: str):
    syms, lines, ext = [], path.read_text("utf-8", "replace").splitlines(), path.suffix.lower()
    if ext == ".html":
        for i, ln in enumerate(lines, 1):
            for m in re.finditer(r'\bid=["\']([A-Za-z0-9_-]+)["\']', ln):
                syms.append(
                    {"path": rel, "kind": "html_id", "name": f"#{m.group(1)}", "start": i, "end": i}
                )
    elif ext == ".css":
        i = 0
        while i < len(lines):
            m = re.match(r'^\s*([#.][A-Za-z0-9_.,\s:>~+*\[\]="\'-]+)\s*\{', lines[i])
            if m:
                depth, j = lines[i].count("{") - lines[i].count("}"), i
                while depth > 0 and j + 1 < len(lines):
                    j += 1
                    depth += lines[j].count("{") - lines[j].count("}")
                syms.append(
                    {
                        "path": rel,
                        "kind": "css",
                        "name": " ".join(m.group(1).split()),
                        "start": i + 1,
                        "end": j + 1,
                    }
                )
                i = j + 1
                continue
            i += 1
    elif ext == ".js":
        for i, ln in enumerate(lines, 1):
            for m in re.finditer(
                r'(?:getElementById\(["\']([A-Za-z0-9_-]+)["\']|querySelector\(["\']#([A-Za-z0-9_-]+))',
                ln,
            ):
                syms.append(
                    {
                        "path": rel,
                        "kind": "js_id",
                        "name": f"#{m.group(1) or m.group(2)}",
                        "start": i,
                        "end": i,
                    }
                )
            fm = re.match(
                r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z0-9_$]+)\s*\(|^\s*(?:const|let|var)\s+([A-Za-z0-9_$]+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>",
                ln,
            )
            if fm:
                syms.append(
                    {
                        "path": rel,
                        "kind": "js_func",
                        "name": fm.group(1) or fm.group(2),
                        "start": i,
                        "end": i,
                    }
                )
    return syms


def cmd_index():
    symbols, runs = [], []
    for d in PY_DIRS:
        for p in sorted((ROOT / d).glob("*.py")):
            symbols.extend(_index_py(p, str(p.relative_to(ROOT))))
    for p in sorted((ROOT / "src/ugv/static").glob("*")):
        if (
            p.is_file()
            and p.suffix.lower() in {".html", ".css", ".js"}
            and not p.name.endswith(".min.js")
        ):
            symbols.extend(_index_static(p, str(p.relative_to(ROOT))))
    for p in sorted((ROOT / "calibration/runs/l7").glob("*/summary.json")):
        d = json.loads(p.read_text("utf-8", "replace"))
        runs.append(
            {
                "path": str(p.relative_to(ROOT)),
                "run_id": d.get("run_id") or p.parent.name,
                "outcome": d.get("outcome")
                or d.get("status")
                or d.get("reason")
                or d.get("result")
                or d.get("mode"),
                "start": d.get("start_pose") or d.get("start"),
                "end": d.get("end_pose") or d.get("end"),
                "duration": d.get("duration_s") or d.get("duration"),
                "target": d.get("target") or d.get("target_pose") or d.get("goal"),
            }
        )
    INDEX_PATH.write_text(
        json.dumps({"symbols": symbols, "runs": runs}, indent=1), encoding="utf-8"
    )
    print(f"Indexed {len(symbols)} symbols, {len(runs)} runs -> {INDEX_PATH}")


def _print_slice(rel: str, start: int, end: int):
    p = pathlib.Path(rel) if pathlib.Path(rel).is_absolute() else ROOT / rel
    lines = p.read_text("utf-8", "replace").splitlines()
    for ln in range(max(1, start), min(len(lines), end) + 1):
        print(f"{ln:4d}: {lines[ln - 1]}")


def cmd_slice(spec: str):
    path_str, rng = spec.rsplit(":", 1)
    s, *e = rng.split("-", 1)
    _print_slice(path_str, int(s), int(e[0]) if e else int(s))


def cmd_facts(section: str = ""):
    text = FACTS_PATH.read_text("utf-8")
    if not section:
        print(text.rstrip())
        return
    q, cur_h, cur, out = section.lower(), "", [], []
    for ln in text.splitlines():
        if ln.startswith("## "):
            if cur_h and q in cur_h.lower():
                out.extend(cur)
            cur_h, cur = ln, [ln]
        elif cur_h:
            cur.append(ln)
    if cur_h and q in cur_h.lower():
        out.extend(cur)
    print("\n".join(out) if out else f"No FACTS section matching '{section}'")


def cmd_query(argv):
    slice_n, terms, i = 0, [], 0
    while i < len(argv):
        if argv[i] == "--slice" and i + 1 < len(argv):
            slice_n = int(argv[i + 1])
            i += 2
        elif argv[i].startswith("--slice="):
            slice_n = int(argv[i].split("=", 1)[1])
            i += 1
        else:
            terms.append(argv[i].lower())
            i += 1
    data, hits = json.loads(INDEX_PATH.read_text("utf-8")), []
    for s in data["symbols"]:
        if all(t in f"{s['path']} {s['kind']} {s['name']}".lower() for t in terms):
            nm = s["name"].lower()
            exact = any(nm == t or nm.endswith(f".{t}") for t in terms)
            hits.append(
                (0 if exact else 1, s["path"], s["start"], s["end"], s["kind"], s["name"], True)
            )
    if FACTS_PATH.exists():
        for ln_no, ln in enumerate(FACTS_PATH.read_text("utf-8").splitlines(), 1):
            if ln.strip() and all(t in ln.lower() for t in terms):
                hits.append((2, ".agent_kb/FACTS.md", ln_no, ln_no, "fact", ln.strip(), False))
    for r in data["runs"]:
        if all(t in json.dumps(r).lower() for t in terms):
            hits.append(
                (
                    3,
                    r["path"],
                    1,
                    1,
                    "run",
                    f"{r['run_id']} outcome={r['outcome']} dur={r['duration']} target={r['target']}",
                    False,
                )
            )
    hits.sort(key=lambda x: (x[0], x[1], x[2]))
    for _, path, st, en, kind, name, _ in hits[:25]:
        print(f"{path}:{st}-{en} | {kind} | {name}")
    if slice_n > 0:
        shown = 0
        for _, path, st, en, _, _, is_code in hits:
            if not is_code:
                continue
            end_s = min(en, st + slice_n - 1)
            print(f"--- {path}:{st}-{end_s} ---")
            _print_slice(path, st, end_s)
            shown += 1
            if shown >= 3:
                break


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(
            "usage: kb.py {index|query <terms> [--slice N]|slice <file:start-end>|facts [section]}"
        )
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "index":
        cmd_index()
    elif cmd == "query":
        cmd_query(rest)
    elif cmd == "slice" and rest:
        cmd_slice(rest[0])
    elif cmd == "facts":
        cmd_facts(" ".join(rest))
    else:
        sys.exit(2)
