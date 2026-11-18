---
name: research-ledger
description: Read and update research/ - decisions, attempts and tasks, one Markdown file each - the only coordination channel between collaborators. Use at the start of any work session, before designing or re-running an experiment, when claiming or finishing work, when asking a collaborator something, and after any decision, correction or failed run.
---

# Research ledger

`research/` replaces `explore/`, the agent board and `HANDOFF.md`. Collaborators work at
different times, so state is kept in files you can read cold, not in a message log you replay.
Schema: `research/README.md`. The frozen hypothesis log is `research/explore-archive.jsonl`;
read it only to trace an old `H-NNNN`.

## Start of a session

```sh
git pull --ff-only
uv run python tools/research/ledger.py status --as <you> # your inbox: your tasks and unassigned ready ones
uv run python tools/research/ledger.py roadmap           # milestones and what each waits on
uv run python tools/research/ledger.py list --grep <topic>   # prior decisions and attempts
```

Answer every open task whose `owner` is you or your human before starting new work.

## Before designing or running

- Attempt `retracted`, `invalid` or `negative`: closed. Re-run only if its "Re-run only if"
  holds; say which in the task you claim.
- Attempt `open`: build on it with `parent: <its id>`; do not start a sibling.
- Active decision: follow it. To change it, write a new decision with `supersedes: [<id>]`
  and set the old one to `superseded` with `superseded_by`.

## Work through tasks

```sh
uv run python tools/research/ledger.py new task <slug> --author <you>
uv run python tools/research/ledger.py claim <id> --as <you>
uv run python tools/research/ledger.py block <id> --as <you> --note "waits on noah: ..."
uv run python tools/research/ledger.py done <id> --as <you> --result <attempt-id|commit>
uv run python tools/research/ledger.py release <id> --as <you>
```

`done` refuses without a result and refuses when the task's `check` command fails. `claim`
refuses while a task in `after` is not done: claim the blocker, or ask its owner. A new
deadline or deliverable is a task tagged `milestone` whose `after` lists the work it needs;
there is no roadmap file to edit. To ask a
collaborator, open a task with `owner: <them>` and the question as its Goal. Commit and push
the task file at once; it is the message.

| Event | Write |
|---|---|
| Starting shared work | claim a task (or `new task`, then claim) |
| A run or artifact ended | attempt; then `done` the task with that attempt as result |
| A posted or committed result was wrong | attempt with `outcome: retracted` citing what it corrects |
| A choice others must follow | decision |
| A question for someone | task owned by them |

An attempt names the flag that sank it in `failure_mode` (the list in `src/ugv/research/records.py`). Its
`## Lesson` is one sentence another agent can act on. Every number names its run id or command.

- **An invalid run is recorded, not erased.** A mis-set magnitude or a contaminated host makes
  the numbers invalid, not private: mark `outcome: invalid`, state what was wrong, keep the
  table, and link the re-run that replaces it. Silent re-runs make every other number suspect.
- **`negative` means removed.** If the mechanism lost, delete it from the tree and say where it
  was; a losing mechanism left behind a flag will be switched on by someone who reads the
  flag's name and not the record. A flag is for a winner that is not yet default.

## Rules

- One record per file; edit only records you own (owner or author). Exception: mark a
  decision superseded.
- A claimed task untouched for 7 days shows `STALE`; ask its owner in a task before taking it.
- Push small and often: a task file is how the other person learns what you are doing.
