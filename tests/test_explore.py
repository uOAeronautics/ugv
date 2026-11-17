"""Tests for tools/explore.py: the explore-loop state machine refuses illegal steps."""

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "explore", Path(__file__).resolve().parent.parent / "tools" / "explore.py"
)
explore = importlib.util.module_from_spec(_SPEC)
sys.modules["explore"] = explore
_SPEC.loader.exec_module(explore)

RUN = "20261116-101500-probe"
FAMILIES = ["classical-control", "rl", "perception", "plant-calibration", "task-geometry"]


@pytest.fixture
def ledger(tmp_path):
    return tmp_path / "HYPOTHESES.jsonl"


def run(ledger, *args):
    return explore.main(["--ledger", str(ledger), *args])


def propose(ledger, family="classical-control", statement="gain X fixes overshoot", **kw):
    args = ["propose", "--family", family, "--statement", statement, "--cost", "1 probe"]
    for k, v in kw.items():
        args += [f"--{k.replace('_', '-')}", v]
    assert run(ledger, *args) == 0


def cycle(ledger, hid, outcome="confirmed", family=None):
    assert run(ledger, "predict", "--id", hid, "--prediction", "p", "--falsifier", "f") == 0
    assert run(ledger, "start", "--id", hid) == 0
    assert run(ledger, "close", "--id", hid, "--outcome", outcome, "--evidence", RUN) == 0


def refute_in(ledger, family, n):
    for i in range(n):
        propose(ledger, family=family, statement=f"dead {family} {i}")
        cycle(ledger, f"H-{_count(ledger):04d}", "refuted")


def _count(ledger):
    return sum(1 for e in explore.read_events(ledger) if e["event"] == "propose")


def test_happy_path_full_cycle(ledger, capsys):
    propose(ledger)
    assert (
        run(
            ledger,
            "predict",
            "--id",
            "H-0001",
            "--prediction",
            "rms < 2 deg",
            "--falsifier",
            "rms >= 2 deg",
        )
        == 0
    )
    assert run(ledger, "start", "--id", "H-0001") == 0
    assert run(ledger, "close", "--id", "H-0001", "--outcome", "confirmed", "--evidence", RUN) == 0
    capsys.readouterr()
    assert run(ledger, "list") == 0
    out = capsys.readouterr().out
    assert "H-0001" in out and "confirmed" in out
    assert run(ledger, "next") == 0
    assert "propose" in capsys.readouterr().out


def test_evidence_path_accepted(ledger, tmp_path):
    propose(ledger)
    run(ledger, "predict", "--id", "H-0001", "--prediction", "p", "--falsifier", "f")
    run(ledger, "start", "--id", "H-0001")
    log = tmp_path / "summary.json"
    log.write_text("{}")
    assert (
        run(ledger, "close", "--id", "H-0001", "--outcome", "refuted", "--evidence", str(log)) == 0
    )


def test_ledger_starts_empty_and_is_append_only(ledger):
    assert explore.read_events(ledger) == []
    propose(ledger)
    first = ledger.read_text()
    propose(ledger, statement="another idea")
    assert ledger.read_text().startswith(first)


def test_unknown_family_refused(ledger, capsys):
    assert run(ledger, "propose", "--family", "vibes", "--statement", "s", "--cost", "c") == 2
    assert "family must be one of" in capsys.readouterr().err
    assert not ledger.exists()


def test_missing_fields_refused(ledger, capsys):
    assert run(ledger, "propose", "--family", "rl", "--statement", "", "--cost", "c") == 2
    assert "--statement is required" in capsys.readouterr().err


def test_predict_requires_proposed(ledger, capsys):
    assert run(ledger, "predict", "--id", "H-0009", "--prediction", "p", "--falsifier", "f") == 2
    assert "no hypothesis" in capsys.readouterr().err


def test_start_requires_prediction(ledger, capsys):
    propose(ledger)
    assert run(ledger, "start", "--id", "H-0001") == 2
    assert "needs a predicted hypothesis" in capsys.readouterr().err


def test_one_running_at_a_time(ledger, capsys):
    propose(ledger, statement="first")
    propose(ledger, statement="second")
    for hid in ("H-0001", "H-0002"):
        run(ledger, "predict", "--id", hid, "--prediction", "p", "--falsifier", "f")
    assert run(ledger, "start", "--id", "H-0001") == 0
    assert run(ledger, "start", "--id", "H-0002") == 2
    assert "one job at a time" in capsys.readouterr().err


def test_close_requires_running_and_valid_evidence(ledger, capsys):
    propose(ledger)
    run(ledger, "predict", "--id", "H-0001", "--prediction", "p", "--falsifier", "f")
    assert run(ledger, "close", "--id", "H-0001", "--outcome", "refuted", "--evidence", RUN) == 2
    assert "needs a started hypothesis" in capsys.readouterr().err
    run(ledger, "start", "--id", "H-0001")
    assert (
        run(ledger, "close", "--id", "H-0001", "--outcome", "refuted", "--evidence", "trust me")
        == 2
    )
    assert "run id" in capsys.readouterr().err
    assert run(ledger, "close", "--id", "H-0001", "--outcome", "maybe", "--evidence", RUN) == 2


def test_refuted_statement_cannot_be_reproposed_bare(ledger, capsys):
    propose(ledger, statement="lower kp")
    cycle(ledger, "H-0001", "refuted")
    assert run(ledger, "propose", "--family", "rl", "--statement", "Lower KP", "--cost", "c") == 2
    assert "--retry-of H-0001" in capsys.readouterr().err


def test_retry_needs_dead_end_and_changed(ledger, capsys):
    propose(ledger, statement="lower kp")
    cycle(ledger, "H-0001", "refuted")
    assert (
        run(
            ledger,
            "propose",
            "--family",
            "classical-control",
            "--statement",
            "lower kp v2",
            "--cost",
            "c",
            "--retry-of",
            "H-0001",
        )
        == 2
    )
    assert "--changed" in capsys.readouterr().err
    assert (
        run(
            ledger,
            "propose",
            "--family",
            "classical-control",
            "--statement",
            "lower kp v2",
            "--cost",
            "c",
            "--retry-of",
            "H-0001",
            "--changed",
            "new deadzone",
        )
        == 0
    )
    propose(ledger, statement="unrelated", family="rl")
    cycle(ledger, "H-0003", "confirmed")
    assert (
        run(
            ledger,
            "propose",
            "--family",
            "rl",
            "--statement",
            "x",
            "--cost",
            "c",
            "--retry-of",
            "H-0003",
            "--changed",
            "y",
        )
        == 2
    )
    assert "not a dead end" in capsys.readouterr().err


def test_three_refuted_forces_family_switch(ledger, capsys):
    refute_in(ledger, "rl", 3)
    assert (
        run(ledger, "propose", "--family", "rl", "--statement", "fresh rl idea", "--cost", "c") == 2
    )
    assert "must be another family" in capsys.readouterr().err
    assert (
        run(
            ledger,
            "propose",
            "--family",
            "perception",
            "--statement",
            "fresh vision",
            "--cost",
            "c",
        )
        == 0
    )


def test_family_switch_lock_can_be_escalated(ledger):
    refute_in(ledger, "rl", 3)
    assert (
        run(
            ledger,
            "propose",
            "--family",
            "rl",
            "--statement",
            "still rl",
            "--cost",
            "c",
            "--escalate",
            "only rl sees the failure mode",
        )
        == 0
    )
    assert "only rl sees" in ledger.read_text()


def test_retro_forced_after_five_closed(ledger, capsys):
    for i in range(5):
        propose(ledger, family=FAMILIES[i], statement=f"idea {i}")
        cycle(ledger, f"H-{i + 1:04d}", "confirmed")
    assert run(ledger, "propose", "--family", "hardware", "--statement", "h", "--cost", "c") == 2
    assert "retro due" in capsys.readouterr().err
    assert run(ledger, "next") == 0
    assert "retro" in capsys.readouterr().out
    assert run(ledger, "retro", "--text", "") == 2
    assert run(ledger, "retro", "--text", "we assumed the plant gain is symmetric") == 0
    assert run(ledger, "propose", "--family", "hardware", "--statement", "h", "--cost", "c") == 0


def test_refusal_writes_nothing(ledger):
    propose(ledger)
    before = ledger.read_text()
    run(ledger, "start", "--id", "H-0001")
    run(ledger, "close", "--id", "H-0001", "--outcome", "refuted", "--evidence", RUN)
    assert ledger.read_text() == before


def test_list_empty_ledger(ledger, capsys):
    assert run(ledger, "list") == 0
    assert "(ledger empty)" in capsys.readouterr().out
