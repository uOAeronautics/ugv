# Research ledger

The project's memory and work queue, one Markdown file per record. On 2026-11-17 it replaced
`explore/`, `.agent_kb/`, the agent board and `HANDOFF.md`: collaborators who work at different
times read current state from files instead of replaying a message log. Procedure for people and agents: the `research-ledger` skill.

- `decisions/`: a choice others must follow (model set, protocol, venue, direction).
- `attempts/`: one experiment or artifact and how it ended. `parent` links it to the attempt it
  built on, so the files form a tree like a Dream-RSI discovery tree (arXiv:2609.14858): a
  later search policy can replay which branches failed and why.
- `tasks/`: open work with an owner and a state. A question for a collaborator is a task they own.
  `after` lists the tasks one waits on; tasks tagged `milestone` are the roadmap.
- `explore-archive.jsonl`: the frozen explore hypothesis log (H-0001 to H-0010, 2026-11-17).
  Every hypothesis in it is cited by an attempt. The dead ends of the 2D RL track
  (`explore/HISTORY.md` at tag `pre-pose-20261117`) are attempts under
  `2026-11-17-track-2d-rl`; records distilled that day carry the distillation date, not the
  date of the original run.

One file per record means two people never edit the same file. The roadmap and each person's
inbox are computed from the task files on demand, never stored, so there is no shared file to
merge and no second copy to drift. The id is the file stem,
`<YYYY-MM-DD>-<slug>`.

## Commands

```sh
uv run python tools/research/ledger.py status                   # live tasks
uv run python tools/research/ledger.py status --as <you>        # your inbox
uv run python tools/research/ledger.py roadmap                  # milestones and their dependencies
uv run python tools/research/ledger.py list --grep tda          # prior work on a topic
uv run python tools/research/ledger.py new task <slug> --author <you>
uv run python tools/research/ledger.py claim|block|release|done|drop <id> --as <you>
uv run python tools/research/ledger.py check                    # same rules as tests/test_research.py
uv run python tools/research/ledger.py export > .cache/research.jsonl   # one row per record
```

## Schema

Front matter is `key: value` lines between `---` fences; lists are `[a, b]`.

| Key | Decisions | Attempts | Tasks |
|---|---|---|---|
| `id`, `date`, `author` | required | required | required |
| `status` | `active`, `superseded`, `reversed` | | |
| `supersedes`, `superseded_by` | decision ids | | |
| `question` | | one sentence | |
| `parent` | | attempt id or `none` | |
| `outcome` | | `kept`, `negative`, `invalid`, `retracted`, `abandoned`, `open` | |
| `failure_mode` | | one of `FAILURE_MODES` in `src/ugv/research/records.py` | |
| `owner`, `state`, `updated` | | | owner or `unassigned`; `open`, `claimed`, `blocked`, `done`, `dropped`; last change |
| `due`, `after`, `check`, `result` | | | date; task ids it waits on; shell command that must pass before `done`; record ids or commits |
| `refs`, `tags` | archived `H-NNNN` ids, commits or repo paths; free tags | same | same |

Body sections: decisions `## Decision`, `## Why` (and `## Revisit if`); attempts `## Tried`,
`## Result`, `## Lesson` (and `## Re-run only if`); tasks `## Goal`, `## Done when` (and
`## Blocked on`, `## Log`, which the CLI appends to).

Outcomes: `negative`, sound experiment, the answer is no; `invalid`, the design was wrong;
`retracted`, a reported result was withdrawn; `abandoned`, stopped without an answer; `kept`,
the result stands; `open`, not finished.

## Rules checked by `tests/test_research.py`

- File name, `id` and `date` agree; required keys and sections exist; enums are valid.
- Links resolve: `parent`, `supersedes`, `superseded_by`, `after`, `result`, archived hypothesis ids.
- No active decision is superseded by an active one.
- A claimed or blocked task has an owner, a blocked one says on what, a done one names its result.
- Every archived hypothesis is cited by an attempt.

The CLI also refuses `done` without a result or with a failing `check`, `claim` while an `after`
task is not done, and moves on a task owned by someone else; `check` also rejects `after` cycles. Bodies describe history and may name deleted files;
`tests/test_docs.py` does not read this folder.
