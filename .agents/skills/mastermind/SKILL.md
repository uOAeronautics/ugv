---
name: mastermind
description: Fan a job out to cheap subagents and gate what comes back. Use when the user says /mastermind, asks to delegate, parallelise or "use subagents", or when a task splits into independent pieces.
---

# Mastermind

The orchestrator plans, briefs, verifies and decides. Subagents do the reading, computing and
editing.

## Loop

1. **Split.** One deliverable per child, checkable without reading the child's transcript: a
   file, a JSON answer, a passing command. Pieces that touch the same file or the same
   exclusive resource (a device, a GPU, a deploy target) are one piece with one owner.
2. **Route** each piece by tier (below). Start at the cheapest tier that fits; escalate only
   after a failed check.
3. **Brief.** Write the brief to a file and tell the child to read it. State the working
   directory, what it may and may not touch, the exact output file and format, how it must
   verify itself, and a report length cap. Give paths and line ranges, not "find the code".
4. **Launch async**, all independent children in one message. Do not poll.
5. **Gate.** Check the deliverable with a command or grader you control. A child's "all tests
   pass" counts its own tests, not yours. For a diagnosis or a risky change, give the same
   evidence to a second child with the opposite prior and compare.
6. **Report** what was checked and how; say what was not.

## Routing

### Standing override (user, 2026-11-14)

All delegates in this project run on Qwen4-Max-Preview (`qmodel_latest`) via
`acpRegistry_qoder_cli` with `reasoning_effort: "high"`. The tier table below is fallback
guidance for when that provider is unavailable.

| Tier | Use for | Models |
|---|---|---|
| Fast / Scoped | Scouting, log and data queries, edits with a complete spec | Inkling 2 Small (`opencode/inkling-2-small-free`), Claude Haiku 5.5 (`medium`) |
| High-Intelligence / Skeptical | Review, diagnosis, experiments, architecture, any brief that may itself be wrong | Gemini 4 Hydrogen (`medium` preferred over `low`), Claude Haiku 5.5 (`medium`/`high`) |

### Model Blacklist (Do NOT spawn for technical, coding, or research council tasks)
- **NEVER use**: `gemini-3.8-flash`, `opencode/nemotron-3.5-lightning-free`, `opencode/beam-free`, `opencode/iceman-free`, `opencode/deepseek-v4.1-flash-free`, `opencode-go/muse-spark-1.3-contributor` (low), or other weak free-tier models when deep technical analysis, mathematical rigor, or reliable code execution is required.
- Default to **Gemini 4 Hydrogen (`thinking_level: "medium"`)**, **Claude Haiku 5.5 (`effort: "medium"`)**, or **Inkling 2 Small (`opencode/inkling-2-small-free`)**.

- Fast-tier models execute a brief as written, including its mistakes. Skeptical/High-intelligence models
  report a flawed brief instead of implementing it silently.
- Prefer `thinking_level: "medium"` on Gemini 4 Hydrogen as the default baseline.
- Above these tiers, do the work yourself or use a top-tier model. Do not assume the table
  holds for tasks longer than about 10 minutes, large repositories, hardware or vision.

## Traps

- **Sandboxed harnesses.** Some deny paths outside the project. Put fixtures and briefs inside
  an allowed path before launching, and treat "access denied" as a brief error, not a model
  failure.
- **Approval modes.** A child in an interactive permission mode waits forever. Set the
  non-interactive mode when launching.
- **Spawn cost.** A fresh child pays for its whole prompt once. Run a one-command question
  yourself.
- **Children may commit or push against the brief.** Check `git log`, `origin/main` and the
  reflog before committing; gate the deliverable on disk with your own commands and treat
  the child's report as a claim, not evidence.
- **Token counts differ per harness.** Compare wall time and outcomes across harnesses, not
  reported tokens.

## Keeping the table honest

Re-measure when a model is added or the table looks stale: give every candidate the same
fixtures, grade with hidden checks and known answers, and record correctness, wall time and
whether a planted flaw in the brief was flagged. If every candidate passes, the task is too
easy to rank them; add a harder one before changing the routing.
