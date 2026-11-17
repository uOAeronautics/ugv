---
id: 2026-11-17-research-ledger
date: 2026-11-17
author: fer-claude
status: active
supersedes: []
refs: [research/README.md]
tags: [coordination]
---

## Decision

`research/` (decisions, attempts, tasks; one Markdown file each, `tools/research/ledger.py`)
is the only memory and work queue. It replaces `explore/` (frozen as
`research/explore-archive.jsonl`), `.agent_kb/`, `docs/AGENT_BOARD.md` and `HANDOFF.md`.
Attempts form a tree through `parent`, as in a Dream-RSI discovery tree (arXiv:2609.14858), so
a later search can replay which branches failed and why. The roadmap is `ledger.py roadmap`,
computed from task `after` links; no roadmap file is kept.

## Why

Four parallel memories (board, handoff, knowledge base, explore ledger) disagreed with each
other and with the code; one file per record never conflicts between collaborators working at
different times, and `tests/test_research.py` checks every link.

## Revisit if

Records stop being written; then the cost of the format is too high and it should shrink.
