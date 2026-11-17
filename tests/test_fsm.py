import pytest

from ugv.fsm import IllegalTransition, Machine

TABLE = {"a": ("b",), "b": ("c", "a"), "c": ()}


def test_declared_moves_are_recorded_and_age_uses_the_clock():
    t = [0.0]
    m = Machine("m", "a", TABLE, clock=lambda: t[0])
    t[0] = 2.0
    assert m.age() == 2.0
    assert m.go("b", "seen tag") is True and m.go("b") is False
    assert m.history[-1] == (2.0, "a", "b", "seen tag")


def test_undeclared_and_unknown_moves_raise():
    m = Machine("m", "a", TABLE)
    with pytest.raises(IllegalTransition):
        m.go("c")
    with pytest.raises(IllegalTransition):
        m.go("zzz")
    assert m.state == "a" and not m.can("c") and m.can("b")


def test_terminal_and_force():
    m = Machine("m", "c", TABLE)
    assert m.terminal
    m.force("a")
    assert m.state == "a" and not m.terminal
    with pytest.raises(ValueError):
        Machine("m", "nope", TABLE)
