"""Task state transitions and the status view for `research/tasks/`.

A task file is edited by its owner only, so two collaborators never conflict on one file. The
transitions below are the only ones the CLI performs; each rewrites `state`, `owner` and
`updated` in the front matter and appends a dated line under `## Log`.

    open -> claimed -> done | blocked | dropped      claimed or blocked -> open (release)
    blocked -> claimed                               done needs a result; a task's `check`
                                                     command must exit 0 before done

`after` lists the tasks one waits on. A task is ready when every one of them is done; claim
refuses a task that is not ready. The roadmap and each person's inbox are views computed from
the task files, never files of their own, so there is one source of truth and nothing to merge.
"""

from __future__ import annotations

import re
import subprocess
from datetime import UTC, date, datetime

from ugv.research.records import ROOT, Record

STALE_DAYS = 7
TRANSITIONS = {
    "claim": ({"open", "blocked"}, "claimed"),
    "block": ({"claimed"}, "blocked"),
    "release": ({"claimed", "blocked"}, "open"),
    "done": ({"claimed"}, "done"),
    "drop": ({"open", "claimed", "blocked"}, "dropped"),
}


def today() -> str:
    return f"{datetime.now(UTC):%Y-%m-%d}"


def _set(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^{key}:.*$", re.MULTILINE)
    head, sep, body = text.partition("\n---\n")
    head = (
        pattern.sub(f"{key}: {value}", head) if pattern.search(head) else f"{head}\n{key}: {value}"
    )
    return head + sep + body


LIVE = ("open", "claimed", "blocked")


def tasks_by_id(records: list[Record]) -> dict[str, Record]:
    return {r.id: r for r in records if r.kind == "tasks"}


def waiting_on(task: Record, tasks: dict[str, Record]) -> list[str]:
    """Ids in `task`'s `after` that are not done. Empty means the task is ready."""
    return [
        d
        for d in task.get_list("after")
        if tasks.get(d) is None or tasks[d].meta.get("state") != "done"
    ]


def transition(
    task: Record,
    action: str,
    who: str,
    note: str,
    result: list[str] | None = None,
    tasks: dict[str, Record] | None = None,
) -> None:
    """Apply `action` to `task` on disk. Raises ValueError on an illegal step, with what to do instead.

    `tasks` (all tasks by id) lets claim check `after`; without it the dependency gate is skipped.
    """
    sources, target = TRANSITIONS[action]
    state = task.meta.get("state")
    if state not in sources:
        raise ValueError(f"{task.id} is {state}; {action} needs one of {sorted(sources)}")
    if action == "claim" and tasks is not None and (pending := waiting_on(task, tasks)):
        raise ValueError(
            f"{task.id} waits on {', '.join(pending)}; claim those or ask their owners"
        )
    owner = task.meta.get("owner")
    if action != "claim" and owner not in (who, "unassigned"):
        raise ValueError(
            f"{task.id} belongs to {owner}; ask them in a task or claim it after release"
        )
    if action == "done":
        if not result:
            raise ValueError("done needs --result: the attempt, decision or commit that shows it")
        check = task.meta.get("check")
        if isinstance(check, str) and check and check != "none":
            run = subprocess.run(check, shell=True, cwd=ROOT, capture_output=True, text=True)
            if run.returncode:
                raise ValueError(
                    f"check failed ({run.returncode}): {check}\n{run.stdout[-800:]}{run.stderr[-800:]}"
                )
    if action == "block" and not note:
        raise ValueError("block needs --note saying what or whom it waits on")
    text = task.path.read_text()
    text = _set(text, "state", target)
    text = _set(
        text,
        "owner",
        "unassigned" if action == "release" else (who if action == "claim" else str(owner)),
    )
    text = _set(text, "updated", today())
    if result:
        text = _set(text, "result", "[" + ", ".join(result) + "]")
    if action == "block" and "## Blocked on" not in text:
        text = text.rstrip("\n") + f"\n\n## Blocked on\n\n{note}\n"
    if "## Log" not in text:
        text = text.rstrip("\n") + "\n\n## Log\n"
    text = text.rstrip("\n") + f"\n- {today()} {who}: {action}{': ' + note if note else ''}\n"
    task.path.write_text(text)


def _line(r: Record, tasks: dict[str, Record], now: date) -> tuple[tuple[int, str], str]:
    state = str(r.meta.get("state"))
    age = (now - date.fromisoformat(str(r.meta.get("updated")))).days
    stale = state == "claimed" and age > STALE_DAYS
    pending = waiting_on(r, tasks)
    label = "waiting" if state == "open" and pending else state
    rank = (
        0
        if state == "blocked"
        else 1
        if stale
        else 2
        if state == "claimed"
        else 4
        if pending
        else 3,
        r.id,
    )
    line = f"{'STALE ' if stale else ''}{label:8s} {r.meta.get('owner', '?'):10s} {r.id}"
    if r.meta.get("due") not in (None, "", "none"):
        line += f"  due {r.meta['due']}"
    if pending:
        line += f"  after {', '.join(pending)}"
    return rank, line


def status_lines(
    records: list[Record], now: date | None = None, owner: str | None = None
) -> list[str]:
    """One line per live task, blocked and stale first, tasks waiting on others last.

    With `owner`, only that person's tasks plus unassigned ready ones (their inbox).
    """
    now = now or datetime.now(UTC).date()
    tasks = tasks_by_id(records)
    rows = []
    for r in tasks.values():
        if r.meta.get("state") not in LIVE:
            continue
        mine = r.meta.get("owner") in (owner, "unassigned") if owner else True
        if mine and not (owner and r.meta.get("owner") == "unassigned" and waiting_on(r, tasks)):
            rows.append(_line(r, tasks, now))
    return [line for _, line in sorted(rows)]


def roadmap_lines(records: list[Record]) -> list[str]:
    """Markdown roadmap: each live task tagged `milestone`, by due date, with the live tasks it
    transitively waits on (deepest first); then live tasks under no milestone."""
    tasks = tasks_by_id(records)
    milestones = sorted(
        (
            r
            for r in tasks.values()
            if "milestone" in r.get_list("tags") and r.meta.get("state") in LIVE
        ),
        key=lambda r: (str(r.meta.get("due")), r.id),
    )
    out, seen = [], set()
    for m in milestones:
        out += [
            "",
            f"## {m.id} (due {m.meta.get('due')}, {m.meta.get('state')}, {m.meta.get('owner')})",
            "",
        ]
        order: list[str] = []
        _deps_first(m.id, tasks, order, set())
        steps = [t for t in (tasks[tid] for tid in order[:-1]) if t.meta.get("state") in LIVE]
        for t in steps:
            pending = waiting_on(t, tasks)
            when = f"after {', '.join(pending)}" if pending else "ready"
            out.append(f"- {t.id}: {t.meta.get('state')}, {t.meta.get('owner')}, {when}")
        if not steps:
            out.append("- waits on nothing; ready")
        seen.update(order)
    rest = [r.id for r in tasks.values() if r.meta.get("state") in LIVE and r.id not in seen]
    if rest:
        out += [
            "",
            "## Not on a milestone",
            "",
            *(f"- {tid} ({tasks[tid].meta.get('owner')})" for tid in sorted(rest)),
        ]
    return ["# Roadmap (generated from research/tasks; do not edit)", *out]


def _deps_first(tid: str, tasks: dict[str, Record], order: list[str], visiting: set[str]) -> None:
    if tid in order or tid in visiting or tid not in tasks:
        return
    visiting.add(tid)
    for d in tasks[tid].get_list("after"):
        _deps_first(d, tasks, order, visiting)
    order.append(tid)


def after_cycles(records: list[Record]) -> list[str]:
    """Task ids that sit on a cycle of `after` links (a cycle can never become ready)."""
    tasks = tasks_by_id(records)
    on_cycle = []
    for start in tasks:
        stack, seen = list(tasks[start].get_list("after")), set()
        while stack:
            cur = stack.pop()
            if cur == start:
                on_cycle.append(start)
                break
            if cur in seen or cur not in tasks:
                continue
            seen.add(cur)
            stack += tasks[cur].get_list("after")
    return sorted(on_cycle)
