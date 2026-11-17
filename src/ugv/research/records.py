"""Project memory and work queue: one Markdown file per decision, attempt or task under `research/`.

Each file starts with a front-matter block of `key: value` lines between `---` fences. A value
is a scalar or a `[a, b]` list. The file stem is the record id, `<YYYY-MM-DD>-<slug>`, so two
collaborators adding records never touch the same file. The body is free Markdown with the
sections named in `SECTIONS`.

Decisions record what was chosen and why; attempts record what was tried and how it ended,
shaped like a Dream-RSI discovery tree (`parent` links an attempt to the one it built on); tasks
are the open work, with an owner and a state. Collaborators who work at different times read
the same files instead of replaying a message log. The previous hypothesis log
(`explore/HYPOTHESES.jsonl`, 2026-11-17) is frozen in `explore-archive.jsonl`; attempts cite its
`H-NNNN` ids, and every id there is distilled into at least one attempt.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESEARCH = ROOT / "research"
BOARD = RESEARCH / "explore-archive.jsonl"

KINDS = ("decisions", "attempts", "tasks")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ID_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-[a-z0-9]+(?:-[a-z0-9]+)*$")
BOARD_REF_RE = re.compile(r"^H-\d{4}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")

DECISION_STATUS = ("active", "superseded", "reversed")
ATTEMPT_OUTCOME = ("kept", "negative", "invalid", "retracted", "abandoned", "open")
TASK_STATE = ("open", "claimed", "blocked", "done", "dropped")
# Failure modes: ML red flags (from the fmri-prh `ml-faux-pas` skill), process failures, and
# robotics failures below.
FAILURE_MODES = (
    "none",
    "synthetic-data",
    "filled-input",
    "row-count",
    "out-of-distribution",
    "stale-cache",
    "shared-units",
    "wrong-unit",
    "preprocessing-leak",
    "broken-baseline",
    "saturation",
    "incomplete-controls",
    "null-without-input-baseline",
    "one-readout",
    "non-zero-floor",
    "bad-interval",
    "underpowered",
    "no-op-step",
    "dead-units",
    "multiple-comparisons",
    "overlapping-views",
    "unreproducible",
    "verb-outruns-test",
    "fabricated-artifact",
    "wrong-attribution",
    "scope-drift",
    # robotics and sim-to-real
    "plant-mismatch",
    "sensor-noise",
    "loop-latency",
    "frame-convention",
    "eval-mismatch",
    "blind-zone",
    "seed-bifurcation",
    "hardware-fault",
    "safety-abort",
)
REQUIRED = {
    "decisions": ("id", "date", "author", "status", "refs"),
    "attempts": ("id", "date", "author", "question", "outcome", "failure_mode", "refs"),
    "tasks": ("id", "date", "author", "owner", "state", "updated", "refs"),
}
SECTIONS = {
    "decisions": ("## Decision", "## Why"),
    "attempts": ("## Tried", "## Result", "## Lesson"),
    "tasks": ("## Goal", "## Done when"),
}


@dataclass
class Record:
    """One parsed research file: `meta` from front matter, `body` the Markdown after it."""

    kind: str
    path: Path
    meta: dict[str, str | list[str]]
    body: str
    errors: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.path.stem

    def get_list(self, key: str) -> list[str]:
        value = self.meta.get(key, [])
        return value if isinstance(value, list) else ([value] if value and value != "none" else [])


def _parse_value(raw: str) -> str | list[str]:
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        return [v.strip() for v in raw[1:-1].split(",") if v.strip()]
    return raw


def parse(path: Path, kind: str) -> Record:
    """Parse one record. Malformed front matter becomes an error on the record, not an exception."""
    text = path.read_text()
    parts = text.split("---\n", 2)
    if not text.startswith("---\n") or len(parts) < 3:
        return Record(kind, path, {}, text, ["missing front matter fenced by ---"])
    meta: dict[str, str | list[str]] = {}
    errors = []
    for line in parts[1].splitlines():
        key, sep, value = line.partition(":")
        if not sep:
            errors.append(f"front matter line without ':': {line!r}")
            continue
        meta[key.strip()] = _parse_value(value)
    return Record(kind, path, meta, parts[2], errors)


def load(root: Path = RESEARCH) -> list[Record]:
    """Every record under `root/decisions` and `root/attempts`, sorted by id."""
    return [parse(p, kind) for kind in KINDS for p in sorted((root / kind).glob("*.md"))]


def board_ids(board: Path = BOARD) -> dict[str, str]:
    """Archived hypothesis id -> "H" (one entry per proposed hypothesis)."""
    out = {}
    for line in board.read_text().splitlines():
        record = json.loads(line)
        if record.get("event") == "propose":
            out[record["id"]] = "H"
    return out


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def validate(records: Iterable[Record], board: dict[str, str]) -> list[str]:
    """Return every schema and link error, as `<path>: <message>` strings. Empty means valid."""
    records = list(records)
    ids = {r.id: r for r in records}
    problems = []
    for r in records:
        err = list(r.errors)
        if not ID_RE.match(r.id):
            err.append("file name must be <YYYY-MM-DD>-<lowercase-slug>.md")
        if r.meta.get("id") != r.id:
            err.append(f"id {r.meta.get('id')!r} differs from file name {r.id!r}")
        if r.meta.get("date") != r.id[:10]:
            err.append("date must equal the id's date prefix")
        err += [f"missing key {k!r}" for k in REQUIRED[r.kind] if k not in r.meta]
        err += [f"missing section {s!r}" for s in SECTIONS[r.kind] if s not in r.body]
        if r.kind == "decisions":
            if r.meta.get("status") not in DECISION_STATUS:
                err.append(f"status must be one of {DECISION_STATUS}")
            if r.meta.get("status") == "superseded" and not r.get_list("superseded_by"):
                err.append("a superseded decision names its successor in superseded_by")
            for key in ("supersedes", "superseded_by"):
                err += [f"{key} {t!r} is not a decision" for t in r.get_list(key) if t not in ids]
        elif r.kind == "tasks":
            err += _task_errors(r, ids)
        else:
            if r.meta.get("outcome") not in ATTEMPT_OUTCOME:
                err.append(f"outcome must be one of {ATTEMPT_OUTCOME}")
            if r.meta.get("failure_mode") not in FAILURE_MODES:
                err.append(f"failure_mode must be one of FAILURE_MODES in {Path(__file__).name}")
            err += [f"parent {p!r} is not an attempt" for p in r.get_list("parent") if p not in ids]
        for ref in r.get_list("refs"):
            if BOARD_REF_RE.match(ref):
                if ref not in board:
                    err.append(f"hypothesis {ref!r} is not in research/explore-archive.jsonl")
            elif not COMMIT_RE.match(ref) and not (ROOT / ref).exists():
                err.append(f"ref {ref!r} is not a hypothesis id, commit hash or existing path")
        problems += [f"{_rel(r.path)}: {e}" for e in err]
    for a in records:
        if a.kind == "decisions" and a.meta.get("status") == "active":
            for b in records:
                if a.id in b.get_list("supersedes") and b.meta.get("status") == "active":
                    problems.append(f"{_rel(a.path)}: active, but superseded by {b.id}")
    return problems


def _known(ref: str, ids: dict[str, Record]) -> bool:
    return ref in ids or bool(COMMIT_RE.match(ref))


def _task_errors(r: Record, ids: dict[str, Record]) -> list[str]:
    """State rules: an owned state has an owner, blocked says on what, done names its result."""
    err = []
    state, owner = r.meta.get("state"), r.meta.get("owner")
    if state not in TASK_STATE:
        err.append(f"state must be one of {TASK_STATE}")
    if state in ("claimed", "blocked") and owner in (None, "", "unassigned"):
        err.append(f"a {state} task has an owner")
    if state == "blocked" and "## Blocked on" not in r.body:
        err.append("a blocked task has a '## Blocked on' section")
    if state == "done" and not r.get_list("result"):
        err.append("a done task lists its result (record ids or commits) in result")
    if not DATE_RE.match(str(r.meta.get("updated", ""))) or str(r.meta.get("updated")) < r.id[:10]:
        err.append("updated is a YYYY-MM-DD date on or after the task's date")
    err += [
        f"result {x!r} is not a record id or commit hash"
        for x in r.get_list("result")
        if not _known(x, ids)
    ]
    err += [
        f"after {d!r} is not a task"
        for d in r.get_list("after")
        if d not in ids or ids[d].kind != "tasks"
    ]
    return err


def uncovered_board_records(records: Iterable[Record], board: dict[str, str]) -> list[str]:
    """Archived hypothesis ids that no attempt cites in `refs`."""
    cited = {ref for r in records if r.kind == "attempts" for ref in r.get_list("refs")}
    return [i for i in board if i not in cited]


def as_row(r: Record) -> dict[str, object]:
    """Flat dict for export: front matter plus kind and the first line of the Lesson/Decision."""
    head = {"attempts": "## Lesson", "decisions": "## Decision", "tasks": "## Goal"}[r.kind]
    summary = (
        r.body.split(head, 1)[-1].strip().split("\n\n", 1)[0].replace("\n", " ")
        if head in r.body
        else ""
    )
    return {"kind": r.kind[:-1], **r.meta, "summary": summary}
