"""Read and update research/: decisions, attempts and tasks (one Markdown file each).

uv run python tools/research/ledger.py brief                      # computed cold-start state (default model, active decisions, tasks, lessons)
uv run python tools/research/ledger.py status                     # live tasks, blocked first
uv run python tools/research/ledger.py status --as noah           # noah's inbox: his and unassigned ready tasks
uv run python tools/research/ledger.py roadmap                    # milestones and what each waits on
uv run python tools/research/ledger.py list --kind attempts --grep motion
uv run python tools/research/ledger.py new task <slug> --author fernando
uv run python tools/research/ledger.py claim <task-id> --as fernando
uv run python tools/research/ledger.py done <task-id> --as fernando --result <record-id|commit>
uv run python tools/research/ledger.py check
uv run python tools/research/ledger.py export > .cache/research.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys

from ugv.research import RESEARCH, as_row, board_ids, load, uncovered_board_records, validate
from ugv.research.tasks import (
    TRANSITIONS,
    after_cycles,
    brief_lines,
    roadmap_lines,
    status_lines,
    tasks_by_id,
    today,
    transition,
)

TEMPLATES = {
    "decision": (
        "status: active\nsupersedes: []\nrefs: []\ntags: []\n",
        "## Decision\n\n\n## Why\n\n\n## Revisit if\n\n",
    ),
    "attempt": (
        "question: \nparent: none\noutcome: open\nfailure_mode: none\nrefs: []\ntags: []\n",
        "## Tried\n\n\n## Result\n\n\n## Lesson\n\n\n## Re-run only if\n\n",
    ),
    "task": (
        "owner: unassigned\nstate: open\nupdated: {today}\ndue: none\nafter: []\ncheck: none\nrefs: []\ntags: []\n",
        "## Goal\n\n\n## Done when\n\n",
    ),
}


def _list(kind: str | None, grep: str | None, state: str | None) -> None:
    for r in load():
        if kind and r.kind != kind:
            continue
        if state and r.meta.get("state") != state:
            continue
        if grep and grep.lower() not in r.path.read_text().lower():
            continue
        what = (
            r.meta.get("status")
            or r.meta.get("state")
            or f"{r.meta.get('outcome')}/{r.meta.get('failure_mode')}"
        )
        print(f"{r.kind[:-1]:8s} {r.id:52s} {what}")


def _check() -> int:
    records, board = load(), board_ids()
    problems = validate(records, board)
    problems += [
        f"archived hypothesis {i} is not cited by any attempt"
        for i in uncovered_board_records(records, board)
    ]
    problems += [f"task {i} waits on itself through `after`" for i in after_cycles(records)]
    print("\n".join(problems) or f"ok: {len(records)} records")
    return 1 if problems else 0


def _new(kind: str, slug: str, author: str) -> None:
    rid = f"{today()}-{slug}"
    path = RESEARCH / f"{kind}s" / f"{rid}.md"
    if path.exists():
        sys.exit(f"{path} exists; pick another slug")
    meta, body = TEMPLATES[kind]
    path.write_text(
        f"---\nid: {rid}\ndate: {rid[:10]}\nauthor: {author}\n{meta.format(today=today())}---\n\n{body}"
    )
    print(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_brief = sub.add_parser("brief")
    p_brief.add_argument("--as", dest="who", help="filter live tasks to your inbox")
    p_brief.add_argument("-n", type=int, default=5, help="number of latest kept attempts")
    p_status = sub.add_parser("status")
    p_status.add_argument("--as", dest="who", help="only your tasks and unassigned ready ones")
    sub.add_parser("roadmap")
    p_list = sub.add_parser("list")
    p_list.add_argument("--kind", choices=["decisions", "attempts", "tasks"])
    p_list.add_argument("--state")
    p_list.add_argument("--grep")
    sub.add_parser("check")
    sub.add_parser("export")
    p_new = sub.add_parser("new")
    p_new.add_argument("kind", choices=list(TEMPLATES))
    p_new.add_argument("slug")
    p_new.add_argument(
        "--author", required=True, help="who you are: fernando, noah, or an agent id"
    )
    for action in TRANSITIONS:
        p = sub.add_parser(action)
        p.add_argument("task")
        p.add_argument("--as", dest="who", required=True)
        p.add_argument("--note", default="")
        p.add_argument("--result", nargs="*", default=None)
    args = parser.parse_args()
    if args.cmd == "brief":
        from ugv.pose.detector import DEFAULT

        print(
            "\n".join(
                brief_lines(load(), detector_default=DEFAULT, n_attempts=args.n, owner=args.who)
            )
        )
    elif args.cmd == "status":
        print("\n".join(status_lines(load(), owner=args.who)) or "no live tasks")
    elif args.cmd == "roadmap":
        print("\n".join(roadmap_lines(load())))
    elif args.cmd == "list":
        _list(args.kind, args.grep, args.state)
    elif args.cmd == "check":
        sys.exit(_check())
    elif args.cmd == "export":
        for r in load():
            print(json.dumps(as_row(r)))
    elif args.cmd == "new":
        _new(args.kind, args.slug, args.author)
    else:
        tasks = tasks_by_id(load())
        if args.task not in tasks:
            sys.exit(f"no task {args.task!r}; see `ledger.py status`")
        try:
            transition(tasks[args.task], args.cmd, args.who, args.note, args.result, tasks)
        except ValueError as e:
            sys.exit(str(e))
        print(f"{args.task}: {args.cmd} ok")


if __name__ == "__main__":
    main()
