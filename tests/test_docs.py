"""Live documents name only paths that exist, so a deleted file cannot linger in the docs."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIVE = [ROOT / "AGENTS.md", ROOT / "README.md", ROOT / "research" / "README.md"]
LIVE += sorted((ROOT / "docs").glob("*.md"))
PATH = re.compile(
    r"`((?:src|tools|docs|research|simulation|tests|config|calibration|rl|explore)/[^`\s<>*|]+)`"
)
GONE = ("explore/", "rl/", ".agent_kb", "AGENT_BOARD", "HANDOFF.md", "tools/explore.py")


HISTORY = ("retired", "pre-pose-20261117")  # a line or document that marks itself as history


def _paths(doc: Path) -> list[str]:
    text = doc.read_text()
    paragraphs = text.split("\n\n")
    if len(paragraphs) > 1 and any(h in paragraphs[1] for h in HISTORY):
        return []  # the document declares itself a historical record in its first paragraph
    return [
        m.rstrip(".,:;)")
        for line in text.splitlines()
        if not any(h in line for h in HISTORY)
        for m in PATH.findall(line)
    ]


def test_live_docs_name_existing_paths() -> None:
    missing = [
        f"{doc.relative_to(ROOT)}: {p}"
        for doc in LIVE
        for p in _paths(doc)
        if not any(c in p for c in "{}")
        and not (ROOT / p.split(":")[0]).exists()
        and not (ROOT / "docs" / p).exists()
    ]
    assert not missing, "\n".join(missing)


def test_retired_paths_are_not_pointed_at() -> None:
    hits = [
        f"{doc.relative_to(ROOT)}: {p}" for doc in LIVE for p in _paths(doc) if p.startswith(GONE)
    ]
    assert not hits, "\n".join(hits)
