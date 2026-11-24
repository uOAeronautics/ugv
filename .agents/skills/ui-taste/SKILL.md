---
name: ui-taste
description: Design, review or fix any web UI in this repo (the Lab dashboard, the car dashboard, a report page) without the tells of AI-generated interfaces. Use before writing UI code, when the owner calls a page "slop", "SaaS-looking" or low effort, and to run a screenshot-and-verify loop with reviewer subagents.
---

# UI taste

The owner rejects interfaces that look generated: card grids with padding everywhere, grey
explanatory text, a logo glyph plus tagline, a sidebar with a coloured left border, a plain
"Theme" button. The Lab dashboard (`src/ugv/lab/static/`) is the reference for what was accepted
instead. Read `references/tells.md` before designing; it is the checklist reviewers grade against.

## The house direction (Lab dashboard, since 2026-11-24)

A handheld-console system UI ("like a Nintendo product", owner, 2026-11-24). It replaced the
2026-11-18 flight-line console; keep new pages in it unless the owner asks for another direction.

- **Palette.** Soft grey shell (`#ececee` day, `#1c1c1f` night), white/graphite tiles, one red for
  the action you can take (`#d0021b` day, `#ff5a6a` night), a cyan ring (`--ring`) for what is
  selected or focused, green/amber/red for state only. Tokens live in `styles.css`; never
  hard-code a colour in a page.
- **Type.** Nunito (vendored, OFL) for everything people read, sentence case, heavy weights for
  numbers; JetBrains Mono for numeric table cells and ids only. No letter-spaced uppercase labels.
- **Structure.** Rounded tiles (`.ruled` children, `--r`) with gaps and a soft shadow, pill
  buttons and segmented controls that press in, a sliding pill for the section tabs. One
  memorable element per page (the 98 % hero, the full-bleed 3D stage, the pose viewer).
- **Theme switch.** Moon/sun pill with a sliding knob and the circular View Transition wipe.
  Pages redraw canvases in `retheme()`, they never re-render.

## Rules that caught real problems here

1. Every sentence on screen must carry a fact. Explanations go in a tooltip or are deleted.
2. Contrast is computed, not eyeballed: ≥4.5:1 for text in both themes. No opacity on text.
3. No number appears on two pages. If two pages repeat each other, merge them.
4. Synthetic and simulation numbers never stand alone as headlines; say what has not been tested
   (the Park page's status row: simulation / real car / detector on real frames).
5. A page that is a run log is a section, not a tab. Four tabs beat six.
6. HUDs over 3D get floating tiles (`color-mix` of the tile colour, blurred), never bare text, and nothing below
   11 px.
7. Phone (390 px): bottom dock with labels, no hover-only information, touch controls for anything
   driven by keys, nothing wider than the screen.
8. Input that moves something (even simulated) has a watchdog and releases on blur, hidden tab
   and lost keyup (`drive.js`).

## Loop: build, screenshot, verify

1. Plan the page in the house direction; name the one memorable element and what you delete.
2. Build. Then screenshot every page at 1600×1000 and 390 px, both themes, with headless
   Chromium (Playwright; launch with `--use-angle=swiftshader` so WebGL renders), plus scripted
   interactions for anything stateful (drive, autopilot, scrub).
3. Look at the screenshots yourself first. Fix the obvious.
4. Verify with 2–3 reviewer subagents with opposite priors (an AI-slop hunter, a game-UI art
   director, the engineer who uses the page daily). Brief them in a file: product, owner's
   complaints, screenshot paths, the checklist, and a demand for exact fixes with screenshot
   citations. They must not edit files. See `mastermind` for briefing and gating.
5. Apply what survives your judgement; a reviewer's claim is not evidence until you see it in a
   screenshot. Repeat 2–4 until a round finds nothing above cosmetic.
