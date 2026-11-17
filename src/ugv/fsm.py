"""A small explicit state machine: a declared transition table, checked on every change.

Implicit state machines (a string field mutated across branches) hide which transitions
exist. This one makes the table data: an undeclared transition raises instead of silently
moving the car into a stage nobody designed. Pure Python, no I/O, safe on the Pi.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping


class IllegalTransition(RuntimeError):
    """Raised when `go` is asked for a transition the table does not declare."""


class Machine:
    """States, who may follow whom, and a bounded history of what happened and why.

    `table` maps each state to the states it may move to. A state with no entry (or an
    empty set) is terminal. `go(state)` to the current state is a no-op, so a stage can
    be re-asserted every tick without noise.
    """

    def __init__(
        self,
        name: str,
        initial: str,
        table: Mapping[str, Iterable[str]],
        *,
        clock: Callable[[], float] = time.monotonic,
        history: int = 32,
    ) -> None:
        self.name = name
        self.table = {state: frozenset(nexts) for state, nexts in table.items()}
        self.states = frozenset(self.table) | frozenset().union(*self.table.values())
        if initial not in self.states:
            raise ValueError(f"{name}: unknown initial state {initial!r}")
        self._clock = clock
        self.state = initial
        self.since = clock()
        self.history: deque[tuple[float, str, str, str]] = deque(maxlen=history)

    @property
    def terminal(self) -> bool:
        return not self.table.get(self.state)

    def age(self) -> float:
        """Seconds spent in the current state."""
        return self._clock() - self.since

    def can(self, state: str) -> bool:
        return state == self.state or state in self.table.get(self.state, frozenset())

    def go(self, state: str, why: str = "") -> bool:
        """Move to `state`. Returns True if the state changed. Raises if undeclared."""
        if state == self.state:
            return False
        if state not in self.states:
            raise IllegalTransition(f"{self.name}: unknown state {state!r}")
        if state not in self.table.get(self.state, frozenset()):
            raise IllegalTransition(f"{self.name}: {self.state} -> {state} is not declared")
        now = self._clock()
        self.history.append((now, self.state, state, why))
        self.state, self.since = state, now
        return True

    def force(self, state: str, why: str = "forced") -> None:
        """Jump anywhere. For tests and operator resets, never for control logic."""
        if state not in self.states:
            raise ValueError(f"{self.name}: unknown state {state!r}")
        self.history.append((self._clock(), self.state, state, why))
        self.state, self.since = state, self._clock()
