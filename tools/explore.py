#!/usr/bin/env python3
"""Explore-loop state machine: propose, predict, start, close, retro, list, next.

The ledger explore/HYPOTHESES.jsonl is append-only. Illegal steps exit 2 with a reason; nothing is
written. Stdlib only. Cycle and rules: explore/LOOP.md, .steward/GOAL.md.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LEDGER = ROOT / "explore" / "HYPOTHESES.jsonl"
FAMILIES = (
    "classical-control",
    "rl",
    "perception",
    "plant-calibration",
    "task-geometry",
    "evaluation-validity",
    "hardware",
)
OUTCOMES = ("confirmed", "refuted", "inconclusive")
FAMILY_REFUTE_LIMIT = 3
RETRO_EVERY = 5
RUN_ID = re.compile(r"^\d{8}-\d{6}")


class Refusal(Exception):
    """An illegal step. The message says what to do instead."""


@dataclass
class Hyp:
    id: str
    family: str
    statement: str
    cost: str
    status: str = "proposed"  # proposed -> predicted -> running -> closed
    prediction: str = ""
    falsifier: str = ""
    outcome: str = ""
    evidence: str = ""


@dataclass
class State:
    hyps: dict[str, Hyp] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    retros: int = 0
    last_family: str = ""
    running: str = ""

    def closed(self) -> list[Hyp]:
        return [h for h in self.hyps.values() if h.status == "closed"]

    def refuted(self, family: str) -> int:
        return sum(1 for h in self.closed() if h.family == family and h.outcome == "refuted")

    def retro_due(self) -> bool:
        return len(self.closed()) // RETRO_EVERY > self.retros


def replay(events: list[dict]) -> State:
    st = State()
    for ev in events:
        kind = ev["event"]
        if kind == "propose":
            hid = ev["id"]
            st.hyps[hid] = Hyp(hid, ev["family"], ev["statement"], ev["cost"])
            st.order.append(hid)
            st.last_family = ev["family"]
        elif kind == "predict":
            h = st.hyps[ev["id"]]
            h.prediction, h.falsifier, h.status = ev["prediction"], ev["falsifier"], "predicted"
        elif kind == "start":
            st.hyps[ev["id"]].status = "running"
            st.running = ev["id"]
        elif kind == "close":
            h = st.hyps[ev["id"]]
            h.status, h.outcome, h.evidence = "closed", ev["outcome"], ev["evidence"]
            st.running = ""
        elif kind == "retro":
            st.retros += 1
    return st


def read_events(path: Path) -> list[dict]:
    return (
        []
        if not path.exists()
        else [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    )


def append(path: Path, event: dict) -> None:
    event = {
        "seq": len(read_events(path)) + 1,
        "ts": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        **event,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")


def need(text: str, what: str) -> str:
    if not (text or "").strip():
        raise Refusal(f"{what} is required")
    return text.strip()


def get(st: State, hid: str) -> Hyp:
    if hid not in st.hyps:
        raise Refusal(f"no hypothesis {hid!r} in the ledger (see: explore.py list)")
    return st.hyps[hid]


def check_retro(st: State) -> None:
    if st.retro_due():
        raise Refusal(
            f"retro due: {len(st.closed())} closed hypotheses, {st.retros} retro "
            "entries. Record one first: explore.py retro --text "
            '"what did we assume that no run tested?"'
        )


def valid_evidence(ev: str) -> bool:
    p = Path(ev)
    return bool(RUN_ID.match(ev)) or (p if p.is_absolute() else ROOT / p).exists()


def cmd_propose(a, st: State) -> str:
    check_retro(st)
    if a.family not in FAMILIES:
        raise Refusal(f"family must be one of: {', '.join(FAMILIES)}")
    statement = need(a.statement, "--statement")
    cost = need(a.cost, "--cost")
    if a.changed and not a.retry_of:
        raise Refusal("--changed only applies with --retry-of")
    if a.retry_of:
        dead = get(st, a.retry_of)
        if not (dead.status == "closed" and dead.outcome == "refuted"):
            raise Refusal(
                f"{a.retry_of} is not a dead end (only refuted hypotheses can be retried)"
            )
        need(a.changed, "--changed (what is different now)")
    else:
        for h in st.closed():
            if h.outcome == "refuted" and h.statement.strip().lower() == statement.lower():
                raise Refusal(
                    f'identical to dead end {h.id}; retry it with --retry-of {h.id} --changed "..."'
                )
    if (
        st.last_family
        and st.refuted(st.last_family) >= FAMILY_REFUTE_LIMIT
        and a.family == st.last_family
        and not a.escalate
    ):
        raise Refusal(
            f"{st.refuted(st.last_family)} refuted in {st.last_family}: the next "
            'proposal must be another family. Pass --escalate "reason" to stay'
        )
    hid = f"H-{len(st.order) + 1:04d}"
    event = {
        "event": "propose",
        "id": hid,
        "family": a.family,
        "statement": statement,
        "cost": cost,
        "retry_of": a.retry_of or "",
        "changed": a.changed or "",
        "escalate": a.escalate or "",
    }
    append(a.ledger, event)
    return f"{hid} proposed ({a.family}); next: explore.py predict --id {hid} ..."


def cmd_predict(a, st: State) -> str:
    check_retro(st)
    h = get(st, a.id)
    if h.status != "proposed":
        raise Refusal(f"{a.id} is {h.status}; predict needs a proposed hypothesis")
    prediction = need(a.prediction, "--prediction")
    falsifier = need(a.falsifier, "--falsifier")
    append(
        a.ledger, {"event": "predict", "id": a.id, "prediction": prediction, "falsifier": falsifier}
    )
    return f"{a.id} predicted; next: explore.py start --id {a.id}"


def cmd_start(a, st: State) -> str:
    check_retro(st)
    h = get(st, a.id)
    if h.status != "predicted":
        raise Refusal(f"{a.id} is {h.status}; start needs a predicted hypothesis")
    if st.running:
        raise Refusal(
            f"{st.running} is running; close it before starting {a.id} (one job at a time)"
        )
    append(a.ledger, {"event": "start", "id": a.id})
    return f"{a.id} started; close it with a run id or evidence path"


def cmd_close(a, st: State) -> str:
    h = get(st, a.id)
    if h.status != "running":
        raise Refusal(f"{a.id} is {h.status}; close needs a started hypothesis")
    if a.outcome not in OUTCOMES:
        raise Refusal(f"--outcome must be one of: {', '.join(OUTCOMES)}")
    evidence = need(a.evidence, "--evidence")
    if not valid_evidence(evidence):
        raise Refusal("--evidence must be a run id (YYYYMMDD-HHMMSS...) or an existing path")
    append(a.ledger, {"event": "close", "id": a.id, "outcome": a.outcome, "evidence": evidence})
    return f"{a.id} closed as {a.outcome}"


def cmd_retro(a, st: State) -> str:
    text = need(a.text, "--text")
    append(a.ledger, {"event": "retro", "text": text})
    return f"retro recorded ({st.retros + 1} total)"


def cmd_list(_a, st: State) -> str:
    rows = [
        f"{h.id} {h.status:<9} {h.family:<20} {h.outcome or '-':<12} {h.statement}"
        for h in map(st.hyps.get, st.order)
    ]
    return "\n".join(rows) or "(ledger empty)"


def cmd_next(_a, st: State) -> str:
    if st.retro_due():
        return 'retro (required): explore.py retro --text "..."'
    if st.running:
        return (
            f"close {st.running}: explore.py close --id {st.running} --outcome ... --evidence ..."
        )
    for hid in st.order:
        h = st.hyps[hid]
        if h.status == "predicted":
            return f"start {hid}: explore.py start --id {hid}"
        if h.status == "proposed":
            return f"predict {hid}: explore.py predict --id {hid} --prediction ... --falsifier ..."
    lock = ""
    if st.last_family and st.refuted(st.last_family) >= FAMILY_REFUTE_LIMIT:
        lock = f" (family must not be {st.last_family}; or --escalate)"
    return f"propose: explore.py propose --family <family> --statement ... --cost ...{lock}"


COMMANDS = {
    "propose": cmd_propose,
    "predict": cmd_predict,
    "start": cmd_start,
    "close": cmd_close,
    "retro": cmd_retro,
    "list": cmd_list,
    "next": cmd_next,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    sub = p.add_subparsers(dest="cmd", required=True)
    fields = {
        "propose": ("family", "statement", "cost", "retry-of", "changed", "escalate"),
        "predict": ("id", "prediction", "falsifier"),
        "start": ("id",),
        "close": ("id", "outcome", "evidence"),
        "retro": ("text",),
        "list": (),
        "next": (),
    }
    for name, names in fields.items():
        sp = sub.add_parser(name)
        for f in names:
            sp.add_argument(f"--{f}", default="")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        print(COMMANDS[a.cmd](a, replay(read_events(a.ledger))))
    except Refusal as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
