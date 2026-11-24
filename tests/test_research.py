"""Research records parse, link and distil every archived explore hypothesis."""

from __future__ import annotations

from pathlib import Path

import pytest

from ugv.research import board_ids, load, parse, uncovered_board_records, validate
from ugv.research.tasks import (
    after_cycles,
    brief_lines,
    roadmap_lines,
    status_lines,
    tasks_by_id,
    transition,
)


def test_records_are_valid() -> None:
    problems = validate(load(), board_ids())
    assert not problems, "\n".join(problems)


def test_every_archived_hypothesis_is_distilled() -> None:
    """Every H-NNNN in research/explore-archive.jsonl is cited by an attempt."""
    missing = uncovered_board_records(load(), board_ids())
    assert not missing, f"cite these in research/: {missing}"


def test_validate_catches_broken_record(tmp_path: Path) -> None:
    (tmp_path / "attempts").mkdir()
    path = tmp_path / "attempts" / "2026-01-01-x.md"
    path.write_text(
        "---\nid: 2026-01-01-y\ndate: 2026-01-01\nauthor: a\nquestion: q\noutcome: fine\n"
        "failure_mode: none\nparent: 2026-01-01-gone\nrefs: [H-9999]\n---\n\n## Tried\n"
    )
    problems = " ".join(validate([parse(path, "attempts")], {}))
    for expected in (
        "differs from file name",
        "outcome must be",
        "parent",
        "hypothesis",
        "## Result",
    ):
        assert expected in problems


def _task(
    tmp_path: Path, state: str = "open", owner: str = "unassigned", check: str = "none"
) -> Path:
    (tmp_path / "tasks").mkdir(exist_ok=True)
    path = tmp_path / "tasks" / "2026-01-01-t.md"
    path.write_text(
        f"---\nid: 2026-01-01-t\ndate: 2026-01-01\nauthor: a\nowner: {owner}\nstate: {state}\n"
        f"updated: 2026-01-01\ncheck: {check}\nrefs: []\n---\n\n## Goal\ng\n\n## Done when\nd\n"
    )
    return path


def test_task_lifecycle(tmp_path: Path) -> None:
    path = _task(tmp_path)
    transition(parse(path, "tasks"), "claim", "fernando", "")
    task = parse(path, "tasks")
    assert task.meta["state"] == "claimed" and task.meta["owner"] == "fernando"
    with pytest.raises(ValueError, match="belongs to fernando"):
        transition(task, "done", "noah", "", ["abc1234"])
    with pytest.raises(ValueError, match="needs --result"):
        transition(task, "done", "fernando", "")
    transition(task, "done", "fernando", "shipped", ["abc1234"])
    done = parse(path, "tasks")
    assert done.meta["state"] == "done" and done.get_list("result") == ["abc1234"]
    assert "fernando: done: shipped" in done.body
    assert not validate([done], {})


def test_task_check_gates_done(tmp_path: Path) -> None:
    path = _task(tmp_path, state="claimed", owner="noah", check="exit 3")
    with pytest.raises(ValueError, match="check failed"):
        transition(parse(path, "tasks"), "done", "noah", "", ["abc1234"])


def test_blocked_task_needs_reason(tmp_path: Path) -> None:
    path = _task(tmp_path, state="blocked", owner="noah")
    assert any("Blocked on" in p for p in validate([parse(path, "tasks")], {}))


def _dep_task(
    root: Path, slug: str, after: str = "", owner: str = "unassigned", tags: str = ""
) -> Path:
    (root / "tasks").mkdir(exist_ok=True)
    path = root / "tasks" / f"2026-01-01-{slug}.md"
    path.write_text(
        f"---\nid: 2026-01-01-{slug}\ndate: 2026-01-01\nauthor: a\nowner: {owner}\nstate: open\n"
        f"updated: 2026-01-01\ndue: none\nafter: [{after}]\ncheck: none\nrefs: []\ntags: [{tags}]\n---\n\n"
        "## Goal\ng\n\n## Done when\nd\n"
    )
    return path


def test_claim_waits_for_after(tmp_path: Path) -> None:
    _dep_task(tmp_path, "data")
    _dep_task(tmp_path, "paper", after="2026-01-01-data", tags="milestone")
    tasks = tasks_by_id(load(tmp_path))
    with pytest.raises(ValueError, match="waits on 2026-01-01-data"):
        transition(tasks["2026-01-01-paper"], "claim", "noah", "", tasks=tasks)
    roadmap = "\n".join(roadmap_lines(load(tmp_path)))
    assert (
        "## 2026-01-01-paper" in roadmap and "- 2026-01-01-data: open, unassigned, ready" in roadmap
    )
    inbox = status_lines(load(tmp_path), owner="noah")
    assert len(inbox) == 1 and "2026-01-01-data" in inbox[0], (
        "a blocked unassigned task stays out of inboxes"
    )


def test_after_cycle_is_reported(tmp_path: Path) -> None:
    _dep_task(tmp_path, "a", after="2026-01-01-b")
    _dep_task(tmp_path, "b", after="2026-01-01-a")
    assert after_cycles(load(tmp_path)) == ["2026-01-01-a", "2026-01-01-b"]


def test_live_tasks_have_no_after_cycle() -> None:
    assert not after_cycles(load())


def test_brief_lines_reports_active_state() -> None:
    from ugv.pose.detector import DEFAULT

    text = "\n".join(brief_lines(load(), detector_default=DEFAULT, n_attempts=3))
    assert f"{DEFAULT[0]}, {DEFAULT[1]}" in text
    assert "## Active decisions" in text
    assert "## Live tasks" in text
    assert "## Latest 3 kept attempts" in text
