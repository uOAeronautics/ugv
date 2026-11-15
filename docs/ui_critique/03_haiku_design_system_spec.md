# 03 — Definitive Design Critique & Research-Grade Dashboard Architecture

**Status:** Pass 1 (lead critic and design-system spec). Spec only: `tools/observatory/` is not edited.
**Scope:** UGV Research Observatory (`tools/observatory/`, port 8768, `docs/COMMANDS.md:113`).
**Evidence:** the 12 PNGs in this folder (the request said 10; there are 12, see §1) and the working tree at `9838b8b` plus the uncommitted observatory edits. Some screenshots may predate fixes already in the working tree, for example the STRL001 selector and 2.5D toggle now in `index.html:94–99`.
**Snapshot caveat:** `tools/observatory/data.py` was modified at 11:36 and `server.py` at 11:38 during this review, after I had read them. I did not write to either. Line numbers may be off by a few lines, and the observed behaviour should be re-checked against the current tree before a fix lands.
**Method:** code read by hand and grep-verified. The server was not run and no browser session was driven. No rollout was started, because a rollout uses the GPU and writes into `rl/runs/`.
**Critics 01 and 02:** not yet combined. §13 lists the points they should challenge.

---

## 0. Direct answers

### 0.1 "Is Gazebo on the dashboard?" Not usable yet. The data side is partly plumbed.

| Layer | What exists | What is missing | Evidence |
|---|---|---|---|
| Leaderboard ingest | Gazebo summary JSON in `rl/runs/gazebo_l6/*.json` is merged into the table. | The `engine` and `sensing` fields are never read. | `server.py:151–205`; `grep engine tools/observatory/server.py` returns nothing. |
| L6 status card | Card exists. | It filters on `level === 'L6'`, but Gazebo runs carry `level: L1`, so the card is always `—`. | `app.js:108–109`; `rl/runs/gazebo_l6/baseline_gazebo_L1.json` (`summary.level = "L1"`). |
| Replay | Gazebo bundle prefix `gazebo_l6:` is wired in the server. | `rl/runs/gazebo_l6/` has **zero** `*.traces.json`, so no Gazebo episode can be replayed. The only per-run data is a summary JSON, a `.log`, and a `_trace.npy`. | `server.py:391–396`; directory listing. |
| Rollout trigger | Runs `rl/evaluate.py` (fast VecSim). | No Gazebo engine option. Level options stop at `L_real` / `L_chaos`. | `server.py:465–472`; `index.html:75–76`. |
| Simulator honesty | The Gazebo summary says `sensing: "emulated from Gazebo pose (rl/sim.py model)"`. | No badge in the UI says "emulated". | `rl/runs/gazebo_l6/baseline_gazebo_L1.json`. |
| Live bridge | Runner exists (`simulation/gazebo/run_policy_episodes.py`, +159/−10 uncommitted). | `docs/RL_PARKING.md:101,104` describes the L6 plumbing as a **fake bridge**: "live Gazebo bridge not built". | `docs/RL_PARKING.md:101,104`. |
| STRL001 belly | Airframe selector and hard-coded drawing exist. | Geometry is hard-coded in `app.js:459–612`. Its source documents are in untracked `docs/council_v2/`, which I did not review. | `index.html:94–98`. |

**Verdict:** Gazebo L6 is not on the dashboard in any form a researcher can trust. The next step is the live bridge, not more UI. Until it exists, every L6 number must carry an `EMULATED` badge.

### 0.2 "Is the 97% baseline the L6 result?" No.

`rl/runs/gazebo_l6/baseline_gazebo_L1.json`: policy `ekf_staged`, **level L1**, 97/100 episodes, seed 11, sensing emulated. It is a Gazebo run in the **L1 configuration**. The L6 gate is `≥90% of L3/L4 rate over ≥100 episodes` (`docs/RL_PARKING.md:39`), and this run is not compared against L3/L4. The spec labels it `Gazebo · L1 config · emulated · N=100`.

### 0.3 The pipeline labels in the brief do not match the source of truth

`CLAUDE.md` makes `docs/RL_PARKING.md` the source for RL plan and gates, so the UI must use its definitions.

| Brief said | Source says | Correction |
|---|---|---|
| L0 Analytic | L0: aligned 0.75 m ahead, posterior mean, perfect state (`RL_PARKING.md:26`) | "L0 · perfect state (teacher gate)" |
| L1 Sensor Latency | L1: 0.5–1.5 m, ±30° bearing, ±60° yaw, tags visible, noise, 80 ms latency (`:28`) | "L1 · nominal noise + latency" |
| L2 60-Particle Dynamics | L2: full 60-draw posterior plus extra DR (yaw coupling, deadzone, lag, gain, sag) (`:30`) | "L2 · 60-draw posterior + DR". Use "draw", not "particle". |
| L3 Vision Dropouts | L3: drops 25–45%, outliers, 3 s blackout, tag loss under drone (`:32`) | Correct as stated. |
| L_real in the chain between L3 and L6 | `L_real` is a plant-matched **config** (`rl/sim.py:144`): dist 0.5–1.5 m, 30°/60°, latency 50–100 ms, no drops, **min tag range 0.40 m** (the "blind ring"), duty cap 0.44. It is not a sequential step. | Show as a side lane "L_real · real-plant match", not in the main ladder. |
| L_chaos (absent from brief) | `rl/sim.py:112`: bursty drops 10–30%, blackouts p=0.008, wheel faults 10%, floor slip 1.5%, action delay 50%. | Side lane "L_chaos · stress". |
| L6 Gazebo | Sim-to-sim: native rollers, rendered camera, real `ParkingEstimator`. Gate ≥90% of L3/L4 over ≥100 eps (`:38–39`). | As stated, plus the sensing badge in §5.3. |
| L7 Real Pi | Operator-authorised, drone powered off. Gate: staged ≥8/10 L1 starts, then ≥18/20 L4 starts, 0 contacts (`:40–41`). | As stated. The dashboard never starts motion (§8.4). |

---

## 1. Evidence inventory

The folder has **12** PNGs, not 10. One is off-topic. The `#` column is a content index I assigned, not the filename order, and the "screenshot N" references elsewhere in this spec use it.

| # | Screenshot (content) | What it proves |
|---|---|---|
| 1 | External RC sim (RealFlight 9.5) with STRL001 loaded. Error dialog, and warnings `Plane file refers to missing Visual Frame '~CS_LMVS'` / `'~CS_Pivot1'`. | **Not a dashboard screen.** Relevant only as a warning that the STRL001 model asset is incomplete. Do not use it as the belly-geometry source. |
| 2 | Training crop: checklist with `[x]` at line ends; `dr_scale & succ/coll rate` and `aux_loss & dyn_loss` with legends printed over each other. | §2 A9, D5 |
| 3 | HuggingFace "Cosmos3-Examples" collection page. | **Unrelated to the observatory.** Excluded from the critique. If it was meant as a design reference, say so. |
| 4 | Rollout "Level" dropdown: L0–L5, L_real, L_chaos. No L6, no L7, no engine choice. | §2 B5 |
| 5 | Leaderboard: L-cards `L0 100%`, `L1 97% (baseline_gazebo_L1)`, `L6 —`, `L7 —`; table with Pareto badges; Real Pi L7 runs collapsed at the bottom. | §2 A4, A11, B1–B3 |
| 6 | Replay sidebar lower half: Transport, `t=0.00s step 0/479`, nine layer checkboxes, a native scroll bar. | §2 D3, D4 |
| 7 | 2D Layout tab: Bundle and Rollout forms, `job job_1794673578 : failed` with no reason, empty "No frames" strips, "No pose in this frame" on the arena. | §2 C1–C2, D8 |
| 8 | Training tab, 4-chart grid: overprinted legends; SPS `171916` and std `−12734` on one axis; chart titles that name series not drawn. | §2 A6–A9 |
| 9 | Bundle dropdown: mixed `student_…`, `obs_…`, `l7:…` items, no grouping, no provenance. | §2 E3 |
| 10 | Policy dropdown: ~20 teacher and student checkpoints, no grouping or gate status. | §2 E3 |
| 11 | Sim-vs-Real Contract: red `INFO` banner while 5+ checks are red; `null / N/A` cards with green borders; `real μ=0.000` for vision Hz. | §2 A1–A3 |
| 12 | Leaderboard filter row: two rows of chips, duplicate `ALL`, no group labels. | §2 D10 |

---

## 2. Diagnosis: why it reads as "visual diarrhea"

Severity: **S1** = the dashboard can mislead about the science. **S2** = a researcher cannot tell what they are looking at. **S3** = visual quality or maintainability.

### A. Scientific correctness

| ID | Sev | Defect | Evidence |
|---|---|---|---|
| A1 | S1 | **Fail-open contract.** A check with no `pass` key counts as PASS, and a check with an unknown status counts as PASS. Screenshot 11 shows `action_sat_act_0 null / N/A`, `gazebo_l6_gate null / N/A` and `plant_checks` with green borders. | `server.py:563` (`v.get("pass", True)`), `server.py:576` (`passed = status != "FAIL"`) |
| A2 | S1 | **Banner contradicts the checks.** The banner reads `INFO` in red. Its colour is set by whether the status string contains "pass", so anything else is red, and the red does not match the checks beneath it. | `app.js:1065–1066`; screenshot 11 |
| A3 | S1 | **Null shown as zero.** `vision_hz real μ=0.000`, `dropout_rate real μ=1.000`, but `loop_ms real μ=—`. "No data" and "zero" are not distinguished. The selected run's name is `…watch-shutter4ms`, so a camera test may explain the zeros. The UI cannot tell. | Screenshot 11; `server.py:632–700` |
| A4 | S1 | **Mixed sample sizes ranked together.** Gate runs use N=1000, Gazebo uses N=100, and rollouts use N=5–20. Cards take the maximum success rate across tracks, so `L1 97%` is a 100-episode Gazebo run compared against 1000-episode sim. | `app.js:110`; `baseline_gazebo_L1.json` (`episodes: 100`) |
| A5 | S1 | **Smoke and debug runs enter the leaderboard.** Rollout JSONs go to the `rl/runs/` root and are ranked like gate runs. The Gazebo folder has the same problem: `gazebo_l6/` holds 1-episode and debug summaries (`smoke_emu.json`, `dbg_cam2.json`) and a 5-episode student run at 0/5, all counted in the "gz 13" pill and shown as Gazebo rows. | `server.py:467–503`; `_counts` at `server.py:37–62`; `rl/runs/gazebo_l6/*.json` (`episodes` 1 or 5) |
| A6 | S2 | **Chart titles name series that are not drawn.** Chart 4 is titled "std & entropy / mean_return & sps" but draws only `std` and `sps`. Chart 1 omits `timeout_rate` and `false_park`. `anchor_frac`, `entropy` and `mean_return` are parsed but never drawn. | `index.html:60–63`; `app.js:207–211`; `data.py:52–67` |
| A7 | S1 | **Incompatible units on one axis.** SPS (~10⁵) and `std` (~10⁴ negative) share chart 4. `std` also appears in chart 3, so one series is shown in two charts. | `app.js:209–210`; screenshot 8 |
| A8 | S2 | **Colour means different things in different charts.** Colour is `CHART_COLORS[(ri*spec.length+si) % 6]`: six colours reused across up to 15 lines, and no run keeps a stable colour. | `app.js:227` |
| A9 | S2 | **Legends overprint.** The legend wraps by resetting `x` but not `y`, so every wrapped entry is drawn on the same baseline. This is the "unreadable garbage" in screenshots 2 and 8. | `app.js:267–273` (wrap at `:272`) |
| A10 | S1 | **No engine or sensing provenance in the table.** Fast VecSim, Gazebo, and real L7 rows look identical, and `engine` is ignored. | `server.py` (no `engine` reference) |
| A11 | S2 | **Undefined labels.** "Pareto" (no definition), "gz 13" in a pill, "Pi4B 0.136 ms/step" (the fallback chain in `app.js:45–47` does not say which metric). | `app.js:42–47`; screenshots 5, 8 |
| A12 | S2 | **Frame and tag conventions absent.** The arena does not state its origin, axis directions, or yaw sign. `rl/sim.py` and `autopark.py` are CCW-positive; `parking.Pose2` is CW-positive. ArUco layouts (14/6/3 vs 9/12/13/14) are not identified. | `CLAUDE.md` "Frames and tags"; `app.js` arena drawing |

### B. Gazebo, L6 and L7 visibility

| ID | Sev | Defect | Evidence |
|---|---|---|---|
| B1 | S1 | L6 card is always `—` because it keys on `level === 'L6'`, which no run has. | `app.js:108–109` |
| B2 | S2 | L7 card is always `—` because L7 runs are in a separate array and never reach the cards. | `app.js:108–109`, `:54` |
| B3 | S2 | Real L7 field runs, the only real-robot evidence, sit in a collapsed `<details>` under the table. | `index.html:46–47` |
| B4 | S1 | No Gazebo episode can be replayed. The replay bundle list silently omits Gazebo runs that have no trace. | Zero `gazebo_l6/*.traces.json`; `server.py:391–396` |
| B5 | S1 | Rollout has no Gazebo engine. | `server.py:465–472`; `index.html:75–76` |
| B6 | S2 | STRL001 belly geometry is hard-coded in the UI. | `app.js:459–612` |
| B7 | S2 | "Open in Replay" for L7 falls back to `end.jpg` silently when frames are missing. | `app.js:163–175`; `server.py:442–462` |

### C. Rollout failure (functional, S1)

| ID | Sev | Defect | Evidence |
|---|---|---|---|
| C1 | S1 | **Likely cause of the `FileNotFoundError`.** The form sends a checkpoint **basename** (`p.name`). The job runs `evaluate.py` with `cwd=rl/`, and `load_policy` calls `torch.load(spec)`. The checkpoints live in `rl/runs/` (39 `.pt` files there, none in `rl/`). Read from code; **not executed**. Confirm with one run while the server log is visible. | `server.py:467`, `server.py:505`, `rl/evaluate.py:35`; `ls rl/*.pt` returns nothing |
| C2 | S1 | **Failure reason discarded.** The subprocess captures no stdout or stderr. The UI shows `job … : failed`, and the server stores only `exit N`. | `server.py:505`, `server.py:528`; `app.js:1102–1103` |
| C3 | S1 | **No GPU or job lock.** `CLAUDE.md` requires one job to own the GPU at a time. The endpoint spawns unconditionally, so two clicks means two GPU jobs. | `server.py:475–517` |
| C4 | S2 | Policy and level are hard-coded in the HTML (`student`, `L1`). Episodes are capped at 20, while the gate is 1000 episodes, so the UI cannot express a gate run. | `index.html:75–77`; `server.py:479` |
| C5 | S3 | Job state lives in an in-memory dict and is lost on restart. | `server.py:JOBS` |

### D. Visual system (S3, but the most visible)

| ID | Sev | Defect | Evidence |
|---|---|---|---|
| D1 | S3 | Dark by default, with no light theme, for a research tool whose figures must be read in daylight and in print. | `style.css:1` (`--bg:#0b0f14`) |
| D2 | S3 | **Browser-default controls.** No rules for `button`, `input[type=checkbox]`, `input[type=range]` or `select`. No `color-scheme`, so the browser draws light buttons, light scrollbars and light native popups. Screenshots 6, 7, 9 and 10 show this. | `index.html:79, 87–89, 92`; `style.css` (no `color-scheme`) |
| D3 | S3 | **Nested scroll.** `.controls` scrolls (`style.css:41`), `#episode-list` scrolls (`:77`), `.run-selector` scrolls (`:36`), `.table-wrap` scrolls (`:25`), and native `<select>` popups scroll. Four or more scroll regions in one view. | as cited |
| D4 | S3 | Replay sidebar is 260 px (`style.css:35`) with about 25 controls stacked vertically. | `index.html:70–102` |
| D5 | S2 | **Checkbox labels wrap inline.** Each checkbox sits at the end of the previous line. `.controls label{display:block}` applies only inside `.controls` (`style.css:43`), but the run list is in `.run-selector`. | `style.css:36`, `:43`; `app.js:186–187`; screenshot 2 |
| D6 | S3 | **Blurry canvases.** No `devicePixelRatio` handling, so all canvases render at 1× and scale up. | `app.js:237`, `:615`, `:794`, `:952` |
| D7 | S3 | No type scale. Body is 13 px, tables 12 px, labels 10–11 px. Numbers are not tabular, so columns do not align. | `style.css:3`, `:24`; `app.js:53–58` |
| D8 | S3 | Three empty "No frames" strips take about 240 px of height and give no reason. | `index.html:119–123`; `style.css:56`; screenshot 7 |
| D9 | S2 | A red `INFO` banner reads as an error. | `app.js:1066`; screenshot 11 |
| D10 | S3 | Two rows of chips with identical styling, duplicate `ALL`, and no group labels. | `app.js:23–24`, `:29–31`; `index.html:31–35`; screenshot 12 |
| D11 | S3 | Inconsistent notation: `|dm|`, `σ=—`, `null / N/A`, `real μ=` with no units. | Screenshot 11; `server.py:600–700` |
| D12 | S2 | No hierarchy. The header pills and the card row repeat the same facts. Nothing states the project's current answer. | `index.html:13–18`, `:37` |

### E. Information architecture and explanation

| ID | Sev | Defect | Evidence |
|---|---|---|---|
| E1 | S2 | The pipeline is never explained. L0–L_chaos appear with no definitions, tooltips, or links to `RL_PARKING.md`. | Screenshots 4, 5 |
| E2 | S2 | Cryptic series names taken from log tokens: `pol_loss & val_loss`, `dr_scale & succ/coll rate`. | `index.html:60–63` |
| E3 | S2 | Run ID is the only identity: `student_Lreal_adr_v4_s0_Lreal_dr025`. Nothing says what changed. `config_delta` exists in the data but is not shown. | Screenshots 9, 10 |
| E4 | S3 | "(legacy .log)" versus "(metrics.jsonl)" is printed on every run. It is a provenance flag, not part of the name. | `app.js:187` |
| E5 | S2 | The replay studio has about nine control groups and no preset or intent. Tab names describe a place, not a question. | `index.html:23–26` |
| E6 | S2 | No view states the question it answers. | All tabs |
| E7 | S2 | A bundle with zero frames shows an empty arena with no explanation. | Screenshot 7 |

### F. Code shape and governance

| ID | Sev | Defect | Evidence |
|---|---|---|---|
| F1 | S3 | `app.js` is 1,139 lines in one file. `CLAUDE.md` targets about 300 lines per file. | `wc -l` |
| F2 | S3 | `server.py` is 780 lines and mixes ingest, metrics, contract logic, rollout, and routing. | `server.py` |
| F3 | S3 | `parse_log` and `parse_metrics_jsonl` both emit every metric under two names, so every consumer has to know both. | `data.py:50–75`, `:96–131` |
| F4 | S3 | Global mutable `state` object, with no modules and no tests for rendering logic. | `app.js:2` |

---

## 3. Root causes

1. **No provenance in the data model.** The UI cannot say which engine, sensing mode, N, seed or gate produced a number, so it puts them side by side as if they were comparable. Most of A1–A10 follow from this.
2. **Unknown is coded as pass or zero.** The contract and channel code fills missing data with a default that looks like a result. That is the most dangerous defect in the current build.
3. **Visual layer is the browser default plus ad hoc CSS.** There are no tokens, no theme, and no control styling, so every control looks different and the page reads as unfinished.
4. **No question per view.** Each tab shows everything it can, with no hierarchy and no statement of what it answers.
5. **Scientific framing is missing.** The dashboard never says which stage a result belongs to, which gate it passes or fails, or what remains unproven.

---

## 4. Design principles

1. **Every number carries provenance** in the same place: `engine · sensing · N · seed · gate`.
2. **Unknown is not zero and not pass.** Null renders as `no data` with a reason. A contract cannot be PASS while any required check is missing.
3. **One question per view**, stated in one sentence at the top, answered by the first screen.
4. **One unit per axis.** No dual axes. Mixed-unit metrics get stacked sub-plots sharing x.
5. **Colour encodes identity or status, never both.** Status always has an icon and a word. Run identity uses a stable colour per run.
6. **Daylight is the default.** Dark is a parallel token set that meets the same contrast targets, not an inversion.
7. **The dashboard is read-only with respect to the car.** It may start compute jobs on the desktop. It never starts motion, and it never contacts the Pi (`SAFETY.md`, level S0).

---

## 5. Pipeline model

### 5.1 Stage stepper (top of Leaderboard)

Main ladder: `L0 → L1 → L2 → L3 → L4 → L5 → L6 → L7`. Side lanes: `L_real` and `L_chaos`.

| Stage | Label on UI | Tests (from `RL_PARKING.md`) | Gate (as written) | Status chip shows |
|---|---|---|---|---|
| L0 | perfect state | aligned 0.75 m ahead, posterior mean | teacher 1000/1000 | best gate-eligible run, N, CI |
| L1 | nominal noise + latency | 0.5–1.5 m starts, 80 ms latency | — | same |
| L2 | 60-draw posterior + DR | full posterior, extra DR (yaw coupling, deadzone, lag, gain, sag) | — | same |
| L3 | drops, outliers, blackouts | drops 25–45%, outliers, 3 s blackout | — | same |
| L4 | open 4×4 m | any heading, drone may start out of view | — | same |
| L5 | full 24×12 m lab | lab + furniture | — | same |
| L6 | Gazebo sim-to-sim | native rollers, rendered camera, real estimator | ≥90% of L3/L4 over ≥100 eps | **sensing badge** (§5.3) |
| L7 | real robot | operator-authorised, drone off | staged ≥8/10 L1, ≥18/20 L4, 0 contacts | runs collected, gate status |
| L_real (side) | real-plant match | 0.40 m blind ring, duty cap 0.44, no drops | — | best run |
| L_chaos (side) | stress | drops 10–30%, blackouts, wheel faults 10%, action delay 50% | — | best run |

Each stage cell links to its row in `docs/RL_PARKING.md`. The status chip shows **gate-eligible** runs only. Non-eligible runs appear as "best any, not gate-eligible" in grey.

### 5.2 Gate eligibility rule

- Sim levels L0–L5: `episodes ≥ 1000` **and** `seed == 11` **and** the run is not a rollout smoke run (`CLAUDE.md`: `rl/evaluate.py`, 1000 episodes, seed 11, is the only success gate).
- L6: its own rule (`≥100` episodes, comparison to L3/L4), flagged `not compared yet` until that comparison is computed.
- L7: the staged rule above, from calibration runs.
- Everything else is `smoke` or `exploratory`, with a chip, and is excluded from gate chips.

The per-level thresholds are defined **once**, in a table in the observatory backend that cites `RL_PARKING.md` line by line. A test fails if the table drifts from the citations.

### 5.3 Sensing badge (non-negotiable)

Every run carries `sensing ∈ {analytic, emulated, live_bridge, real}`. The table and replay show it as a badge:

- `EMULATED` (amber): Gazebo sensing computed from simulated pose, the current state of L6 (`RL_PARKING.md:101,104`).
- `LIVE BRIDGE` (cobalt): only after the live Gazebo bridge exists and the run used it.
- `REAL` (emerald): L7 field data.

An L6 number may not be displayed without its badge.

---

## 6. Design system

### 6.1 Colour tokens (light, default)

Contrast was measured with the WCAG formula on `#ffffff` unless stated. Two of the brief's colours fail AA as **text**. They stay as fills and icons. The text variants below are used for words.

| Token | Hex | Role | Contrast |
|---|---|---|---|
| `--bg-app` | `#f8fafc` | app background | — |
| `--bg-surface` | `#ffffff` | cards, panels | — |
| `--border` | `#e2e8f0` | 1px card and divider border | — |
| `--shadow-card` | `0 1px 2px rgba(15,23,42,.06)` | card shadow | — |
| `--text` | `#0f172a` | primary text | 17.9 |
| `--text-2` | `#475569` | secondary text | 7.6 |
| `--text-3` | `#64748b` | captions, axis labels | 4.8 |
| `--accent` | `#2563eb` | cobalt: primary action, selected, `LIVE BRIDGE` | 5.2 |
| `--success` | `#059669` | emerald fill and icon | 3.8 (**not for text**) |
| `--success-text` | `#047857` | emerald text | 5.5 |
| `--warning` | `#d97706` | amber fill and icon | 3.2 (**not for text**) |
| `--warning-text` | `#b45309` | amber text (`EMULATED`) | 5.0 |
| `--danger` | `#dc2626` | crimson fill and icon, FAIL | 4.8 |
| `--danger-text` | `#b91c1c` | crimson text | 6.5 |
| `--neutral-chip` | `#f1f5f9` background, `--text-2` text | `INCOMPLETE`, `no data` | — |

Status chips (icon + word, never colour alone): `✓ PASS` emerald-text on emerald-50; `✕ FAIL` crimson-text on red-50; `◌ INCOMPLETE` slate; `◇ EMULATED` amber-text on amber-50; `● GATE-ELIGIBLE` cobalt; `○ SMOKE` slate outline.

**Dark theme (parallel, same roles):**

| Token | Hex | Contrast on `--bg-app-dark` (`#0b1120`) |
|---|---|---|
| `--bg-app-dark` | `#0b1120` | — |
| `--bg-surface-dark` | `#111827` | — |
| `--border-dark` | `#1f2937` | — |
| `--text-dark` | `#e5e7eb` | 15.2 |
| `--text-2-dark` | `#9ca3af` | 7.4 |
| `--accent-dark` | `#60a5fa` | 7.4 |
| `--success-dark` | `#34d399` | 9.8 |
| `--warning-dark` | `#fbbf24` | 11.3 |
| `--danger-dark` | `#f87171` | 6.8 |

Switching: `:root { color-scheme: light; }`, `[data-theme="dark"] { color-scheme: dark; }`. The choice is a per-viewer preference stored in `localStorage` (allowed for a local tool), and it does not change any data. Scrollbars use `scrollbar-width: thin` and `scrollbar-color` from the tokens in both themes.

### 6.2 Chart categorical palette

Okabe–Ito, which is colour-blind safe: `#0072B2`, `#D55E00`, `#009E73`, `#CC79A7`, `#E69F00`, `#56B4E9`. Yellow `#F0E442` is excluded because it fails on white. **Colour identity is per run.** Series within a run are distinguished by line style: solid for policy, dashed for value or auxiliary, dotted for ADR anchors. Identity is never encoded by colour alone.

### 6.3 Type

- System UI stack (no CDN, so the lab can run offline). Numbers use `font-variant-numeric: tabular-nums`. IDs and paths use `ui-monospace`.
- Scale: 12 / 13 / 14 / 16 / 20 / 24 px, with line-heights 1.4 for body and 1.2 for headings. Body is 14 px, never below 12 px.
- Muted text must meet AA. `#7a8699` on `#121821` is 4.8 and only passes at 11 px. Do not go lower than 12 px for any text in the spec.

### 6.4 Spacing, radius, elevation

4 px grid. Cards: 8 px radius, 16 px padding. Controls: 6 px radius, 32 px height. One shadow token. No gradients. No glassmorphism.

### 6.5 Controls

- Buttons: primary (cobalt fill, white text), secondary (1px `--border`, `--text`), ghost (text only). Focus ring: 2 px cobalt, 2 px offset, visible on every control.
- Checkboxes and range inputs: native elements with `accent-color: var(--accent)`, so keyboard and screen-reader behaviour is kept.
- Selects: custom trigger, native popup kept for accessibility. Group labels inside the popup.
- Scrollbars: one scroll container per panel at most. The panel body scrolls; the page does not.

### 6.6 Motion

Transitions 150 ms on hover and focus only. No animation of data. `prefers-reduced-motion` removes all transitions.

---

## 7. Views

Each tab opens with its question, one sentence, then the answer.

| Tab | Question |
|---|---|
| Leaderboard | Which policy passes which stage, and how sure are we? |
| Training | Is the curriculum learning, and are the losses behaving? |
| Replay Studio | What did this episode do, frame by frame? |
| Sim-vs-Real | Do the sim channels match the car's measured channels? |

### 7.1 Leaderboard

1. **Header strip** (one line): last gate-eligible run, Gazebo L6 status, L7 runs, Pi bench (labelled with its metric and date). Replaces the five header pills.
2. **Stage stepper** (§5.1), with the Gazebo L6 status card to its right: `Gazebo · L1 config · 97/100 · EMULATED · N=100 · student raw vs shielded: not yet compared`. Student raw and shielded are compared only when both summary JSONs exist. In `rl/runs/gazebo_l6/` there is a `policy_gazebo_student-shield-none_L1.gz.log` log but **no summary JSON** for either student shield variant. The only student Gazebo summaries are 1–5 episode runs (0 successes in the ones I read).
3. **Filters** as segmented controls with group labels: `Engine` (Sim · Gazebo · Real), `Level`, `Kind` (Teacher · Student · Baseline · Oracle), toggle `Gate-eligible only`, search. Duplicate `ALL` is removed.
4. **Table** (default sort: stage, then CI lower bound). Columns: Run (copy button), Engine, Kind, Level, **N**, **Seed**, Success with a small forest plot of the Wilson interval on a 0–1 axis, Collisions %, Timeouts %, Median time (s), Pos p95 (mm), Yaw p95 (deg), Gate chip, Sensing badge. Rows with N < 1000 are shown with the `SMOKE` or `NOT GATE` chip. Pareto is a column with a tooltip that gives its definition.
5. **Real L7 runs**: a visible section below the table, not collapsed.
6. **Diff**: select two rows, and a drawer shows deltas with units, plus `config_delta` as a list of changed keys.

### 7.2 Training & ADR curves

- **Run picker** (left, 260 px): grouped by engine and level, searchable, with a colour swatch per run. Up to four runs selected. Provenance (`metrics.jsonl` or legacy `.log`) appears as a small flag, not as part of the name.
- **Six-cell grid** (3 × 2 at ≥1280 px). Each cell has a title with its unit, a chart, and an HTML legend bar beneath it. The legend wraps in normal layout and never draws on the canvas. Hover shows one tooltip that scrubs all six cells together (shared cursor).

| Cell | Title | Series | Axis |
|---|---|---|---|
| 1 | Success & collision [0–1] | `succ_rate`, `coll_rate`, `timeout_rate`; `false_park` optional | fixed 0–1 |
| 2 | ADR curriculum scale | `dr_scale`, `anchor_frac` | 0–1 |
| 3 | Policy & value loss | `pol_loss`, `val_loss` | log |
| 4 | Auxiliary pose & extrinsics loss | `aux_loss`, `dyn_loss` | log |
| 5 | Exploration | top: `std` (action units); bottom: `entropy` (nats) | separate sub-plots |
| 6 | Throughput & return | top: `sps` (steps/s); bottom: `mean_return` | separate sub-plots |

Notes:
- The brief asked for SPS and mean return on one chart. That mixes units, so cell 6 becomes two stacked sub-plots sharing x. The cell count stays at six.
- Series a run did not log show in the legend as `not logged`, not as a missing line.
- Chart titles are generated from the series list, so they cannot name something that is not drawn (test in §10).

### 7.3 Replay Studio

Three columns at ≥1280 px, stacking below that.

**Left rail (280 px, collapsible):**
- Source tabs: `Sim (VecSim)` · `Gazebo L6` · `Real Pi L7`. Each shows its provenance banner: engine, sensing badge, N, seed.
- Bundle list grouped by source, showing N, seed and outcome. A bundle with no traces says `summary only`, with the reason.
- Episode chips with counts: All · Success · Collision · Timeout · False-park.
- Rollout trigger, collapsed by default:
  - Engine: Fast VecSim · Gazebo L6 (badge shown).
  - Policy: grouped by kind, showing the checkpoint's path. The request sends a **policy id**, and the server resolves it to an absolute path under `rl/runs/` and returns a 400 naming the resolved path if the file is absent (fixes C1).
  - Level: L0–L5, L_real, L_chaos, plus L6 when Gazebo is selected.
  - Episodes: presets `5 smoke` · `100 (L6 minimum)` · `1000 gate`, with a live gate chip.
  - Seed: 11 by default, with a gate note.
  - Start: disabled while another GPU job is running. The button shows the owner job id (C3).
- Job panel: queued / running / done / failed; exit code; last 20 lines of stdout and stderr; link to the full log (C2).

**Centre (arena):**
- Canvas at 1:1 metres, DPR-correct, up to 720 px wide. The frame label top-left shows `t`, `step`, `frame`. Top-right shows the engine and sensing badge.
- Frame convention drawn on the arena: origin, axis directions, and the yaw sign (CCW positive, as in `rl/sim.py`).
- Airframe switch: `Quadcopter skids` | `STRL001 belly`, as a segmented control. Geometry loaded from a versioned file (§8.5), not from the JS.
- 2D / 2.5D toggle. Layers in a popover with swatches.
- Blind ring drawn at the configured `min_tag_range` (0.40 m for L_real) and labelled with its value.
- Under the arena: transport bar (⏮ · ◀ −1 · ▶/❚❚ · +1 · ▶| · speed · scrubber). The scrubber shows event ticks for blackouts and collisions.

**Right rail (360 px):**
- **Camera POV**, aspect-correct. Real L7 shows `end.jpg` with its timestamp. Sim shows the synthetic pinhole view. Each is labelled with its source.
- **Step inspector**: table with unit column: true pose (x, y, yaw, CCW), estimated pose, tag measured, delay (ms), valid flag, blackout flag, duty L/R, phase.
  - The arrival state is shown as **two separate fields**: `arrived (estimate)` and `parked (end frame confirmed)`. Only the second counts as parked (`CLAUDE.md`: an estimate-based "arrived" is not a parked car).
- **Telemetry strips** (three, stacked, shared x, cursor-synced):
  1. Pose error: position (mm) and yaw (deg), as two sub-plots.
  2. IMU gyro (rad/s).
  3. Duty L/R with the cap line at 0.44. This equals 1800/4095 = 0.4396, the car's PWM cap, so the line is labelled with both forms.
- **Empty state:** if the bundle has no frames, the arena and strips say why and what would produce them. No blank panels.

### 7.4 Sim-vs-Real Contract

- **Banner** computed from the checks, with three states: `PASS` only if every required check has a measured value and passes; `FAIL` if any check fails; `INCOMPLETE` if any is missing. The banner reads `k of n checks evaluated · m missing · f failing`.
- **Checks** grouped: Timing · Tags & vision · Gyro/IMU · Actuation · Gazebo gate. Each row has its measured value, limit, unit, and status chip.
- **Channels** as a table: channel · real (μ, σ, n) · sim (μ, σ, n) · delta in σ · status. Missing values show `no data`, never zero.
- **Source run** picker grouped by engine. The sim reference (`sim_reference_L1.npz`) is shown with a one-line description of what it is.

---

## 8. Data contract and backend

### 8.1 Run record

Every run, from any source, normalises to:

```
RunRecord {
  run_id, source_file,
  engine:    "fast_vecsim" | "gazebo_l6" | "real_l7" | "log_only",
  sensing:   "analytic" | "emulated" | "camera" | "live_bridge" | "real",   // "camera" = rendered-camera pipeline (seen in gazebo_l6/dbg_cam2.json)
  kind:      "teacher" | "student" | "baseline" | "oracle" | "unknown",
  level:     "L0".."L5" | "L_real" | "L_chaos" | "L6" | "L7" | null,
  episodes:  int | null,
  seed:      int | null,
  success, wilson_lo, wilson_hi,
  collisions, timeouts, false_park,
  gate: { name, status: "pass"|"fail"|"not_run"|"not_eligible", reason }
}
```

Unknown fields are `null`, never a default. `track` is no longer inferred from the run name (`server.py:86–99`).

### 8.2 Responses

Every response carries `schema_version` and a `provenance` block (sources read, file counts, last modified). Errors use one envelope: `{ error: { code, message, hint, log_tail } }`.

### 8.3 Jobs

- `POST /api/jobs` takes `{ engine, policy_id, level, episodes, seed }`. Validates the policy id against the checkpoint index, not against the basename.
- One job at a time for the GPU. A second request returns `409` naming the owner job.
- stdout and stderr go to a per-job log under `rl/runs/obs_jobs/` (already gitignored, `RL_PARKING.md:93`).
- Job state persists to disk, so a restart does not lose it.

### 8.4 Read-only with respect to the car

No endpoint contacts the Pi, opens SSH, sends serial, or starts motion. The observatory reads local records and starts only desktop evaluation jobs. This keeps it at S0 (`docs/SAFETY.md`).

### 8.5 Airframe geometry

STRL001 and quadcopter geometry move from `app.js` to one versioned JSON under `calibration/` (curated fits), with its source cited. Do not hard-code geometry until `docs/council_v2/` has been reviewed and reconciled, and until the RealFlight asset warnings (missing `~CS_LMVS`, `~CS_Pivot1`) are resolved or ruled out as the geometry source.

### 8.6 Server split

Routers by resource (overview, leaderboard, curves, replay, jobs, contract). Ingest modules per source (eval JSON, Gazebo summaries, training logs, L7 runs, contract). One schema for training logs, so `parse_log` and `parse_metrics_jsonl` stop emitting every key twice (F3).

---

## 9. Module layout and size budget

Target: no file over ~300 lines. A data table or a file that does one thing may be longer, but say so in the file header.

```
tools/observatory/
  server.py                      thin: app factory + main()             (< 80)
  api/
    overview.py  leaderboard.py  curves.py  replay.py  jobs.py  contract.py
  ingest/
    eval_json.py  gazebo.py  training_logs.py  l7.py  contract.py
  gates.py                       per-level gate table, cites RL_PARKING.md
  wilson.py                      (existing)
static/
  index.html                     shell only
  css/tokens.css  base.css  components.css  layout.css
  js/
    main.js  router.js  store.js  api.js  format.js  units.js
    leaderboard/  table.js  forest.js  stage_stepper.js  filters.js  diff.js
    training/     run_picker.js  grid.js  cells.js  legend.js  sync.js
    replay/       rail.js  rollout.js  arena.js  transport.js  pov.js  inspector.js  strips.js
    contract/     banner.js  checks.js  channels.js
    jobs/         panel.js
  vendor/uplot.min.js            vendored, version pinned, licence checked before vendoring
```

Use ES modules, no build step. `uPlot` (canvas, cursor sync, small) is the recommended chart library, vendored with its licence and version recorded. The alternative is a small custom canvas renderer with DPR handling and HTML legends. Either is acceptable. Do not add a framework.

---

## 10. Acceptance criteria

Each is testable. Phase gates in §12 use these.

1. **Chart titles match series.** A unit test builds the title from the cell config and checks it against the drawn series list.
2. **No canvas legends.** Legends are HTML elements. A test fails if `fillText` is used for legend entries.
3. **Missing never passes.** A fixture with a null `pass` or an unknown status renders `INCOMPLETE`, never `PASS`.
4. **Banner matches checks.** Fixture with one failing check renders `FAIL`. Banner colour and word come from the same computed value.
5. **Policy path bug is closed.** A basename policy returns 400 naming the resolved path. A valid policy id runs.
6. **Failures show their cause.** A deliberately failing job shows its last 20 stderr lines in the panel.
7. **L6 is populated and badged.** With `baseline_gazebo_L1.json`, the L6 card shows the run with `EMULATED`, N=100, and the gate status `not compared yet`.
8. **No browser-default control** is visible in either theme (screenshot diff at 1440 px).
9. **One scroll container per panel**, checked by a DOM test that counts scrollable ancestors.
10. **Contrast.** Every text token pair on its background is ≥ 4.5:1 (automated over the token table in §6.1).
11. **Fits the viewport.** At 1280×800, the Leaderboard shows the header, stage stepper, filters and first eight rows with no page scroll. At 1440×900, Replay Studio shows arena, transport and inspector with no page scroll.
12. **Every row shows N and seed.** Rows with N < 1000 show the SMOKE or NOT GATE chip and are excluded from gate chips.
13. **Null renders as "no data"** with a reason, never `0` and never a bar of zero length.
14. **Each view states its question** in one sentence at the top.
15. **No motion endpoint.** A grep over the backend finds no SSH, serial, or Pi-address code.
16. **File size.** No file over 300 lines except those whose header says why.
17. **Gate table drift.** A test checks each threshold in `gates.py` against the line cited in `RL_PARKING.md`.

---

## 11. Gazebo L6 and STRL001: what is needed

**Gazebo L6 on the dashboard requires, in this order:**

1. The **live Gazebo bridge** (`docs/RL_PARKING.md:104`: "live Gazebo bridge not built"). Until then, the UI shows `EMULATED` on every L6 number. This is the blocking dependency.
2. **Trace output** from `simulation/gazebo/run_policy_episodes.py`, so episodes can be replayed. Today there are zero `*.traces.json` in `rl/runs/gazebo_l6/`. This runner is uncommitted (+159/−10). Review it before relying on it.
3. **Raw vs shielded** student summaries in the same schema, so the L6 status card can compare them.
4. **The L6 comparison to L3/L4** (`RL_PARKING.md:39`), computed by `gates.py`, not by hand.

**STRL001 belly docking requires:**

1. Its geometry in `calibration/` with a citation, after review of `docs/council_v2/03_strix_plane_geometry_model.md` and `09_lr_disambiguation_and_belly_shield.md` (I have not read these).
2. A check that the RealFlight asset is not the geometry source, given the missing visual frames in screenshot 1.
3. Frame labels that say which airframe the origin refers to (`CLAUDE.md` frame rules).

Until these are done, the STRL001 option is shown in the UI with a `DRAFT GEOMETRY` badge and no claim about belly-contact accuracy.

---

## 12. Phase plan

Each phase ends with its acceptance criteria passing.

| Phase | Scope | Fixes | Criteria |
|---|---|---|---|
| **P0 · correctness hot-fixes** (no redesign) | Policy id resolution; stderr capture; contract fail-open and banner; L6 keying; chart titles; legend overlap | C1, C2, A1–A2, B1, A6, A9 | 1, 2, 3, 4, 5, 6, 7 |
| **P1 · data model and provenance** | `RunRecord`, `gates.py`, engine and sensing fields, N and seed everywhere | A4, A5, A10, F3, B2 | 12, 13, 17 |
| **P2 · tokens, controls, themes** | `tokens.css`, control styling, `color-scheme`, dark set, type scale, DPR canvases | D1, D2, D6, D7, D9, D10 | 8, 10 |
| **P3 · layouts** | Leaderboard stepper and table; Training 6-cell grid with uPlot; Sim-vs-Real checks and channels | A7, A8, B3, D3, D5, D8, D11, D12, E1–E4 | 1, 9, 11, 14 |
| **P4 · Replay Studio** | Left rail, arena with frame convention, inspector with separate arrived/parked, three strips, empty states | B4, B5, B6, B7, C3, C4, C5, D4, E5, E7 | 11, 15 |
| **P5 · Gazebo L6 live** | Blocked on live bridge (§11.1) | B4 full, B5 full, L6 comparison | 7, with `LIVE BRIDGE` badge only after bridge exists |

P0 is the priority. It fixes the three defects that can mislead a researcher: the fail-open contract, the rollout that cannot run, and the L6 card that always shows `—`. The visual redesign (P2–P4) should not ship before P0.

---

## 13. Self-critique log (pass 1)

What this pass may have wrong, or did not do:

1. **Nothing was run.** The server was not started and no browser was driven. Visual claims come from the screenshots and the code. The `FileNotFoundError` cause (C1) is inferred from the code path and must be confirmed with one run while the server log is visible.
2. **Screenshot mapping.** I cited screenshots by content, not by filename, because I could not confirm which file is which in every case. The folder has 12 files, not 10.
3. **Working tree is dirty.** Some screenshot defects may already be partly fixed (the STRL001 selector and 2.5D toggle are in `index.html` but not in the screenshots). Re-capture before judging P0 complete.
4. **Council docs not read.** `docs/council_v2/` is untracked. §8.5 and §11 depend on it, and it may contradict my STRL001 claims.
5. **`rl/PLAN.md` not read.** It may define a newer ladder than `RL_PARKING.md`. If it does, it wins for the labels in §5, since `CLAUDE.md` names it as the plan of record. Check before P1.
6. **Palette changed.** The brief's `#059669` and `#d97706` fail AA as text. I substituted text variants (§6.1). This is a deliberate deviation.
7. **Cell 6 changed.** The brief asked for SPS and mean return on one chart. I split them into stacked sub-plots (§7.2). This is a deliberate deviation.
8. **Library choice is a recommendation, not a decision.** uPlot is recommended. Its licence must be checked before vendoring.
9. **Agents.** The brief asked for several Haiku agents and "inkling" agents. I did this pass inline. "Inkling" is not an agent type available here. The two other critics are the next step.

---

## 14. Questions for the user

1. Is the live Gazebo bridge the next milestone? The L6 card's status depends on it. Until it exists, the spec only allows `EMULATED`.
2. Should 5-episode rollouts appear in the leaderboard as `SMOKE` rows (my recommendation), or be hidden behind a toggle?
3. The Cosmos3 HuggingFace screenshot (#3 in §1): was it meant as a design reference? If so, which part should the design take from it?
